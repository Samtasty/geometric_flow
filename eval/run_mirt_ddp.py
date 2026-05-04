import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train.train_mirt_ddp import train_mirt_ddp


def main():
    parser = argparse.ArgumentParser(
        description="Train MIRT with DDP (item params synchronized, student embeddings local)."
    )
    parser.add_argument("--data", type=str, default="pix_mapping/pix_irt_outcome.csv")
    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument("--emb-dim", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--student-l2", type=float, default=0.0)
    parser.add_argument("--max-students", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--world-size", type=int, default=0)
    parser.add_argument("--backend", type=str, default="")
    args = parser.parse_args()

    metrics = train_mirt_ddp(
        data_path=args.data,
        cache_path=args.cache_path or None,
        emb_dim=args.emb_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        train_ratio=args.train_ratio,
        student_l2=args.student_l2,
        max_students=args.max_students,
        seed=args.seed,
        num_workers=args.num_workers,
        eval_batch_size=args.eval_batch_size,
        world_size=(None if args.world_size <= 0 else args.world_size),
        backend=(None if not args.backend else args.backend),
    )
    print("DDP metrics:", metrics)


if __name__ == "__main__":
    main()
