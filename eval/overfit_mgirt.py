import argparse
import os
import sys
# Ensure project root is on PYTHONPATH when running as a script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import Subset

from data.assistments_dataset import AssistmentsDataset
from train.train_mgirt import train_mgirt_mle
from eval.evaluate_mgirt import evaluate_mgirt


def main():
    parser = argparse.ArgumentParser(description="Overfit test for MG-IRT (MLE)")
    parser.add_argument("--data", type=str, default="data/data.csv")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--emb-dim", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    args = parser.parse_args()

    dataset = AssistmentsDataset(args.data)
    n = min(args.n, len(dataset))
    subset = Subset(dataset, list(range(n)))

    model, _ = train_mgirt_mle(
        subset,
        emb_dim=args.emb_dim,
        alpha=args.alpha,
        beta=args.beta,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        val_split=0.0,
        early_stopping=False,
    )

    print("Overfit evaluation (same subset):")
    evaluate_mgirt(model, subset)


if __name__ == "__main__":
    main()
