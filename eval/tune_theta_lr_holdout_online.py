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
    out = []
    for s in str(text).split(","):
        s = s.strip()
        if s:
            out.append(float(s))
    if not out:
        raise ValueError("Expected at least one float value.")
    return out


def _parse_int_list(text):
    out = []
    for s in str(text).split(","):
        s = s.strip()
        if s:
            out.append(int(s))
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


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
        description="Tune theta_lr for student-holdout online evaluation with fixed dim=2 training."
    )
    parser.add_argument("--data", type=str, default="pix_mapping/pix_irt_outcome.csv")
    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=16384)

    parser.add_argument("--emb-dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--student-l2", type=float, default=0.001)

    parser.add_argument("--theta-lrs", type=str, default="0.02,0.05,0.1,0.2,0.3")
    parser.add_argument("--theta-prior-mean", type=float, default=0.0)
    parser.add_argument("--theta-prior-std", type=float, default=1.0)
    parser.add_argument("--theta-seed", type=int, default=42)
    parser.add_argument(
        "--eval-epochs",
        type=str,
        default="2,5",
        help="Epoch checkpoints (1-based) at which theta_lr is evaluated.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="pix_mapping/mirt_dim2_theta_lr_sweep.csv",
    )
    parser.add_argument(
        "--out-summary-csv",
        type=str,
        default="pix_mapping/mirt_dim2_theta_lr_sweep_summary.csv",
    )
    args = parser.parse_args()

    theta_lrs = _parse_float_list(args.theta_lrs)
    eval_epochs = sorted(set(_parse_int_list(args.eval_epochs)))
    if min(eval_epochs) < 1 or max(eval_epochs) > args.epochs:
        raise ValueError("eval_epochs must be between 1 and epochs.")

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = str(device).startswith("cuda")

    dataset = AssistmentsDataset(args.data, cache_path=(args.cache_path or None))
    train_set, test_set = split_by_student_holdout(dataset, test_size=args.test_size, seed=args.seed)
    base = train_set.dataset if hasattr(train_set, "dataset") else train_set
    num_students = len(base.students)
    num_items = len(base.items)

    print(
        f"rows_total={len(dataset)} train_rows={len(train_set)} test_rows={len(test_set)} "
        f"num_students={num_students} num_items={num_items}",
        flush=True,
    )
    print(
        f"train_cfg emb_dim={args.emb_dim} epochs={args.epochs} lr={args.lr} "
        f"batch_size={args.batch_size} student_l2={args.student_l2}",
        flush=True,
    )
    print(
        f"theta_cfg theta_lrs={theta_lrs} eval_epochs={eval_epochs} "
        f"prior_mean={args.theta_prior_mean} prior_std={args.theta_prior_std}",
        flush=True,
    )

    model = MIRTModel(num_students, num_items, emb_dim=args.emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    checkpoints = {}
    for epoch in range(1, args.epochs + 1):
        loss = _train_one_epoch(model, loader, optimizer, device, args.student_l2)
        print(f"train epoch={epoch} loss={loss:.4f}", flush=True)
        if epoch in eval_epochs:
            checkpoints[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    rows = []
    for epoch in eval_epochs:
        print(f"\n=== evaluating checkpoint epoch={epoch} ===", flush=True)
        model.load_state_dict(checkpoints[epoch])
        train_metrics = evaluate_mirt(
            model,
            train_set,
            batch_size=args.eval_batch_size,
            device=device,
            num_workers=args.num_workers,
        )
        for theta_lr in theta_lrs:
            test_metrics = evaluate_mirt_student_holdout_online(
                model,
                test_set,
                theta_lr=theta_lr,
                theta_prior_mean=args.theta_prior_mean,
                theta_prior_std=args.theta_prior_std,
                seed=args.theta_seed,
            )
            rows.append(
                {
                    "checkpoint_epoch": int(epoch),
                    "theta_lr": float(theta_lr),
                    "train_auc": float(train_metrics["auc"]),
                    "train_acc": float(train_metrics["accuracy"]),
                    "test_auc": float(test_metrics["auc"]),
                    "test_acc": float(test_metrics["accuracy"]),
                    "test_nll": float(test_metrics["nll"]),
                }
            )

    df = pd.DataFrame(rows).sort_values(["checkpoint_epoch", "theta_lr"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    summary = (
        df.sort_values("test_auc", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(args.out_summary_csv, index=False)

    print("\nTop settings by test_auc:")
    print(summary.head(10).to_string(index=False))
    print(f"\nsaved_csv={os.path.abspath(args.out_csv)}")
    print(f"saved_summary={os.path.abspath(args.out_summary_csv)}")


if __name__ == "__main__":
    main()
