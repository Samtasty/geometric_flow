import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.assistments_dataset import AssistmentsDataset
from data.splits import IndexedSubset, split_temporal_by_student
from train.train_mirt import train_mirt


def eval_batched(model, dataset, batch_size=8192):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            s = batch["student_idx"].long()
            i = batch["item_idx"].long()
            y = batch["correct"].float().numpy()
            p = model(s, i).numpy()
            y_true.extend(y.tolist())
            y_pred.extend(p.tolist())

    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)
    acc = accuracy_score(y_true, y_pred >= 0.5)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    nll = log_loss(y_true, y_pred, labels=[0, 1]) if len(np.unique(y_true)) > 1 else float("nan")
    return {"accuracy": float(acc), "auc": float(auc), "nll": float(nll)}


def parse_csv_list(text, cast):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Grid search MIRT on PIX data.")
    parser.add_argument("--data", type=str, default="pix_mapping/pix_irt_outcome.csv")
    parser.add_argument("--sample", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emb-dim", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio-in-train", type=float, default=0.9)
    parser.add_argument("--lrs", type=str, default="0.001,0.003,0.01")
    parser.add_argument("--batch-sizes", type=str, default="1024,2048,4096")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--renorm-options", type=str, default="false,true")
    parser.add_argument("--out-csv", type=str, default="pix_mapping/mirt_grid_emb2_results.csv")
    args = parser.parse_args()

    lrs = parse_csv_list(args.lrs, float)
    batch_sizes = parse_csv_list(args.batch_sizes, int)
    renorm_options = [x.strip().lower() == "true" for x in args.renorm_options.split(",")]

    full = AssistmentsDataset(args.data)
    idxs = np.arange(len(full))
    if args.sample > 0 and args.sample < len(full):
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(idxs, size=args.sample, replace=False)
        work_ds = IndexedSubset(full, idxs)
    else:
        work_ds = full

    # Temporal split per student: train+val vs test
    trainval_set, test_set = split_temporal_by_student(work_ds, train_ratio=args.train_ratio)
    # Within trainval, temporal split again to get validation tail
    train_set, val_set = split_temporal_by_student(trainval_set, train_ratio=args.val_ratio_in_train)

    print(f"Rows total source: {len(full)}")
    print(f"Rows used: {len(work_ds)}")
    print(f"Train rows: {len(train_set)}")
    print(f"Val rows: {len(val_set)}")
    print(f"Test rows: {len(test_set)}")

    rows = []
    for lr in lrs:
        for batch_size in batch_sizes:
            for renorm in renorm_options:
                print(
                    f"\n=== emb_dim={args.emb_dim} lr={lr} batch={batch_size} renorm={renorm} ==="
                )
                model, _ = train_mirt(
                    train_set,
                    emb_dim=args.emb_dim,
                    batch_size=batch_size,
                    lr=lr,
                    epochs=args.epochs,
                    val_set=val_set,
                    early_stopping=True,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    renorm=renorm,
                )
                tr = eval_batched(model, train_set)
                va = eval_batched(model, val_set)
                te = eval_batched(model, test_set)
                row = {
                    "emb_dim": args.emb_dim,
                    "lr": lr,
                    "batch_size": batch_size,
                    "renorm": renorm,
                    "train_acc": tr["accuracy"],
                    "train_auc": tr["auc"],
                    "train_nll": tr["nll"],
                    "val_acc": va["accuracy"],
                    "val_auc": va["auc"],
                    "val_nll": va["nll"],
                    "test_acc": te["accuracy"],
                    "test_auc": te["auc"],
                    "test_nll": te["nll"],
                }
                rows.append(row)
                print(
                    f"val_acc={row['val_acc']:.4f} test_acc={row['test_acc']:.4f} "
                    f"val_nll={row['val_nll']:.4f} test_nll={row['test_nll']:.4f}"
                )

    results = pd.DataFrame(rows).sort_values(["val_nll", "test_nll", "test_acc"])
    results.to_csv(args.out_csv, index=False)
    print("\nBest by val_nll:")
    print(results.head(5).to_string(index=False))
    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
