import argparse
import os
import sys

import matplotlib.pyplot as plt
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


def _parse_emb_dims(text):
    vals = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals.append(int(chunk))
    if not vals:
        raise ValueError("No embedding dimensions provided.")
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


def _plot_auc_curves(df, out_path):
    dims = sorted(df["emb_dim"].unique().tolist())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)

    for d in dims:
        sub = df[df["emb_dim"] == d].sort_values("epoch")
        axes[0].plot(sub["epoch"], sub["train_auc"], marker="o", label=f"d={d}")
        axes[1].plot(sub["epoch"], sub["test_auc"], marker="o", label=f"d={d}")

    axes[0].set_title("Train AUC vs Epoch")
    axes[1].set_title("Test AUC vs Epoch (Holdout Online Theta)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Grid search on emb_dim with student-holdout online-theta evaluation."
    )
    parser.add_argument("--data", type=str, default="pix_mapping/pix_irt_outcome.csv")
    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emb-dims", type=str, default="2,3,5,8")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--student-l2", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--theta-lr", type=float, default=0.1)
    parser.add_argument("--theta-prior-mean", type=float, default=0.0)
    parser.add_argument("--theta-prior-std", type=float, default=1.0)
    parser.add_argument("--theta-seed", type=int, default=42)
    parser.add_argument(
        "--out-csv",
        type=str,
        default="pix_mapping/mirt_holdout_online_auc_grid.csv",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="pix_mapping/mirt_holdout_online_auc_grid.png",
    )
    args = parser.parse_args()

    emb_dims = _parse_emb_dims(args.emb_dims)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AssistmentsDataset(args.data, cache_path=(args.cache_path or None))
    train_set, test_set = split_by_student_holdout(
        dataset,
        test_size=args.test_size,
        seed=args.seed,
    )
    base = train_set.dataset if hasattr(train_set, "dataset") else train_set
    num_students = len(base.students)
    num_items = len(base.items)

    print(
        f"rows_total={len(dataset)} train_rows={len(train_set)} test_rows={len(test_set)} "
        f"num_students={num_students} num_items={num_items}"
    )
    print(
        f"grid emb_dims={emb_dims} epochs={args.epochs} lr={args.lr} "
        f"batch_size={args.batch_size} student_l2={args.student_l2}"
    )

    all_rows = []
    for emb_dim in emb_dims:
        print(f"\n=== emb_dim={emb_dim} ===")
        model = MIRTModel(num_students, num_items, emb_dim=emb_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=str(device).startswith("cuda"),
        )

        for epoch in range(1, args.epochs + 1):
            mean_loss = _train_one_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device=device,
                student_l2=args.student_l2,
            )
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
                "emb_dim": int(emb_dim),
                "epoch": int(epoch),
                "loss": float(mean_loss),
                "train_acc": float(train_metrics["accuracy"]),
                "train_auc": float(train_metrics["auc"]),
                "train_nll": float(train_metrics["nll"]),
                "test_acc": float(test_metrics["accuracy"]),
                "test_auc": float(test_metrics["auc"]),
                "test_nll": float(test_metrics["nll"]),
            }
            all_rows.append(row)
            print(
                f"epoch={epoch} loss={mean_loss:.4f} "
                f"train_auc={row['train_auc']:.4f} test_auc={row['test_auc']:.4f}"
            )

    df = pd.DataFrame(all_rows).sort_values(["emb_dim", "epoch"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    _plot_auc_curves(df, args.out_plot)

    summary = (
        df.sort_values(["emb_dim", "epoch"])
        .groupby("emb_dim", as_index=False)
        .tail(1)[["emb_dim", "epoch", "train_auc", "test_auc", "train_acc", "test_acc", "train_nll", "test_nll"]]
        .sort_values("emb_dim")
        .reset_index(drop=True)
    )
    print("\nFinal-epoch summary:")
    print(summary.to_string(index=False))
    print(f"\nsaved_csv={os.path.abspath(args.out_csv)}")
    print(f"saved_plot={os.path.abspath(args.out_plot)}")


if __name__ == "__main__":
    main()
