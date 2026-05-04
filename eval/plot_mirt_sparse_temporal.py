import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.irt_tensor_dataset import IRTTensorDataset
from train.train_mirt_sparse import temporal_split_by_student, train_sparse_mirt


def main():
    parser = argparse.ArgumentParser(description="Train sparse MIRT and plot accuracy vs epochs.")
    parser.add_argument("--data", type=str, default="pix_mapping/pix_irt_outcome.csv")
    parser.add_argument("--cache-path", type=str, default="pix_mapping/pix_irt_outcome.pt")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--emb-dim", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--student-l2", type=float, default=0.001)
    parser.add_argument("--item-l2", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--tqdm", action="store_true", help="Show tqdm progress bars per epoch.")
    parser.add_argument(
        "--out-csv",
        type=str,
        default="pix_mapping/mirt_sparse_temporal_accuracy_vs_epoch.csv",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="pix_mapping/mirt_sparse_temporal_accuracy_vs_epoch.png",
    )
    args = parser.parse_args()

    dataset = IRTTensorDataset(
        args.data,
        cache_path=(args.cache_path or None),
        build_cache=args.build_cache,
    )
    train_set, test_set = temporal_split_by_student(dataset, train_ratio=args.train_ratio)
    device = args.device if args.device else None

    print(f"rows_total={len(dataset)} train_rows={len(train_set)} test_rows={len(test_set)}")

    _, history = train_sparse_mirt(
        train_set,
        emb_dim=args.emb_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        student_l2=args.student_l2,
        item_l2=args.item_l2,
        device=device,
        num_workers=args.num_workers,
        eval_train_set=train_set,
        eval_test_set=test_set,
        eval_every=1,
        eval_batch_size=args.eval_batch_size,
        show_progress=args.tqdm,
    )

    df = pd.DataFrame(history)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(df["epoch"], df["test_acc"], marker="o", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Sparse MIRT (Temporal Split): Accuracy vs Epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out_plot), exist_ok=True)
    plt.savefig(args.out_plot, dpi=140)
    plt.close()

    print(f"saved_csv={os.path.abspath(args.out_csv)}")
    print(f"saved_plot={os.path.abspath(args.out_plot)}")


if __name__ == "__main__":
    main()
