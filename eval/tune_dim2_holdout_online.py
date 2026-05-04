import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.assistments_dataset import AssistmentsDataset
from data.splits import split_by_student_holdout
from eval.evaluate_mirt import evaluate_mirt, evaluate_mirt_student_holdout_online
from models.mirt import MIRTModel


def _parse_float_list(text):
    vals = []
    for s in str(text).split(","):
        s = s.strip()
        if s:
            vals.append(float(s))
    if not vals:
        raise ValueError("Expected at least one numeric value.")
    return vals


def _train_one_epoch(model, loader, optimizer, device, student_l2):
    model.train()
    criterion = torch.nn.BCELoss()
    losses = []
    for batch in loader:
        student_idx = batch["student_idx"].long().to(device)
        item_idx = batch["item_idx"].long().to(device)
        correct = batch["correct"].float().to(device)

        optimizer.zero_grad()
        pred = model(student_idx, item_idx)
        loss = criterion(pred, correct)
        if student_l2 > 0.0:
            theta = model.student_emb(student_idx)
            loss = loss + student_l2 * theta.pow(2).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="Tune dim=2 MIRT on student-holdout with online-theta test evaluation."
    )
    parser.add_argument("--data", type=str, default="pix_mapping/pix_irt_outcome.csv")
    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--emb-dim", type=int, default=2)
    parser.add_argument("--lrs", type=str, default="0.005,0.01,0.02")
    parser.add_argument("--student-l2s", type=str, default="0.001,0.01")
    parser.add_argument("--theta-lr", type=float, default=0.1)
    parser.add_argument("--theta-prior-mean", type=float, default=0.0)
    parser.add_argument("--theta-prior-std", type=float, default=1.0)
    parser.add_argument("--theta-seed", type=int, default=42)
    parser.add_argument(
        "--out-csv",
        type=str,
        default="pix_mapping/mirt_dim2_hparam_tuning_history.csv",
    )
    parser.add_argument(
        "--out-summary-csv",
        type=str,
        default="pix_mapping/mirt_dim2_hparam_tuning_summary.csv",
    )
    args = parser.parse_args()

    lrs = _parse_float_list(args.lrs)
    student_l2s = _parse_float_list(args.student_l2s)
    combos = [(lr, l2) for lr in lrs for l2 in student_l2s]

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = str(device).startswith("cuda")

    dataset = AssistmentsDataset(args.data, cache_path=(args.cache_path or None))
    train_set, test_set = split_by_student_holdout(dataset, test_size=args.test_size, seed=args.seed)
    base = train_set.dataset if hasattr(train_set, "dataset") else train_set
    num_students = len(base.students)
    num_items = len(base.items)

    print(
        f"rows_total={len(dataset)} train_rows={len(train_set)} test_rows={len(test_set)} "
        f"num_students={num_students} num_items={num_items}"
    )
    print(
        f"emb_dim={args.emb_dim} epochs={args.epochs} batch_size={args.batch_size} "
        f"combos={len(combos)} lrs={lrs} student_l2s={student_l2s}"
    )

    history_rows = []
    summary_rows = []

    for run_idx, (lr, student_l2) in enumerate(combos, start=1):
        print(f"\n=== run {run_idx}/{len(combos)}: lr={lr}, student_l2={student_l2} ===", flush=True)
        model = MIRTModel(num_students, num_items, emb_dim=args.emb_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        run_rows = []
        for epoch in range(1, args.epochs + 1):
            mean_loss = _train_one_epoch(model, loader, optimizer, device, student_l2)
            train_metrics = evaluate_mirt(
                model,
                train_set,
                batch_size=args.eval_batch_size,
                device=device,
                num_workers=args.num_workers,
            )
            test_metrics = evaluate_mirt_student_holdout_online(
                model,
                test_set,
                theta_lr=args.theta_lr,
                theta_prior_mean=args.theta_prior_mean,
                theta_prior_std=args.theta_prior_std,
                seed=args.theta_seed,
            )
            row = {
                "lr": lr,
                "student_l2": student_l2,
                "epoch": epoch,
                "loss": mean_loss,
                "train_acc": float(train_metrics["accuracy"]),
                "train_auc": float(train_metrics["auc"]),
                "train_nll": float(train_metrics["nll"]),
                "test_acc": float(test_metrics["accuracy"]),
                "test_auc": float(test_metrics["auc"]),
                "test_nll": float(test_metrics["nll"]),
            }
            run_rows.append(row)
            history_rows.append(row)
            print(
                f"epoch={epoch} loss={mean_loss:.4f} "
                f"train_auc={row['train_auc']:.4f} test_auc={row['test_auc']:.4f}",
                flush=True,
            )

        run_df = pd.DataFrame(run_rows).sort_values("epoch")
        best_idx = run_df["test_auc"].idxmax()
        best_row = run_df.loc[best_idx]
        final_row = run_df.iloc[-1]
        summary_rows.append(
            {
                "lr": lr,
                "student_l2": student_l2,
                "best_epoch": int(best_row["epoch"]),
                "best_test_auc": float(best_row["test_auc"]),
                "best_test_acc": float(best_row["test_acc"]),
                "final_epoch": int(final_row["epoch"]),
                "final_test_auc": float(final_row["test_auc"]),
                "final_train_auc": float(final_row["train_auc"]),
            }
        )

    hist_df = pd.DataFrame(history_rows).sort_values(["lr", "student_l2", "epoch"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("best_test_auc", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    hist_df.to_csv(args.out_csv, index=False)
    summary_df.to_csv(args.out_summary_csv, index=False)

    print("\nSummary (sorted by best_test_auc):")
    print(summary_df.to_string(index=False))
    print(f"\nsaved_history={os.path.abspath(args.out_csv)}")
    print(f"saved_summary={os.path.abspath(args.out_summary_csv)}")


if __name__ == "__main__":
    main()
