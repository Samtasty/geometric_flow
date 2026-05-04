import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.assistments_dataset import AssistmentsDataset
from data.splits import split_by_student_holdout, split_temporal_by_student
from eval.evaluate_mirt import (
    evaluate_mirt,
    evaluate_mirt_item_only,
    evaluate_mirt_student_holdout_online,
)
from train.train_mirt import train_mirt


def main():
    parser = argparse.ArgumentParser(description="Run MIRT with configurable split strategies.")
    parser.add_argument("--data", type=str, default="pix_mapping/pix_irt_outcome.csv")
    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument(
        "--split",
        type=str,
        default="temporal_per_student",
        choices=["temporal_per_student", "student_holdout"],
    )
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emb-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--batch-mode",
        type=str,
        default="random",
        choices=["random", "student", "student_mix"],
        help="random: row-shuffled, student: one student/batch, student_mix: many students/batch",
    )
    parser.add_argument("--students-per-batch", type=int, default=64)
    parser.add_argument("--interactions-per-student", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--student-l2", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="", help="cpu|cuda, empty=auto")
    parser.add_argument("--sample", type=int, default=0, help="Optional number of rows to sample.")
    parser.add_argument(
        "--holdout-eval",
        type=str,
        default="item_only",
        choices=["item_only", "online_theta"],
        help="Evaluation mode when split=student_holdout.",
    )
    parser.add_argument("--theta-lr", type=float, default=0.1)
    parser.add_argument("--theta-prior-mean", type=float, default=0.0)
    parser.add_argument("--theta-prior-std", type=float, default=1.0)
    parser.add_argument("--theta-seed", type=int, default=42)
    args = parser.parse_args()

    dataset = AssistmentsDataset(args.data, cache_path=(args.cache_path or None))
    if args.sample > 0 and args.sample < len(dataset):
        rng = np.random.default_rng(args.seed)
        sampled = rng.choice(np.arange(len(dataset)), size=args.sample, replace=False)
        from data.splits import IndexedSubset

        dataset = IndexedSubset(dataset, sampled)

    if args.split == "temporal_per_student":
        train_set, test_set = split_temporal_by_student(dataset, train_ratio=args.train_ratio)
        split_desc = (
            f"Temporal per-student split: first {args.train_ratio:.2f} of each student sequence in train, "
            "rest in test."
        )
    else:
        train_set, test_set = split_by_student_holdout(
            dataset, test_size=args.test_size, seed=args.seed
        )
        split_desc = (
            f"Student holdout split: {args.test_size:.2f} of students in test only; "
            "no student overlap between train and test."
        )

    print(split_desc)
    print(f"Train rows: {len(train_set)}")
    print(f"Test rows: {len(test_set)}")

    model, _ = train_mirt(
        train_set,
        emb_dim=args.emb_dim,
        batch_size=args.batch_size,
        batch_mode=args.batch_mode,
        students_per_batch=args.students_per_batch,
        interactions_per_student=(
            None if args.interactions_per_student <= 0 else args.interactions_per_student
        ),
        lr=args.lr,
        student_l2=args.student_l2,
        epochs=args.epochs,
        seed=args.seed,
        num_workers=args.num_workers,
        device=(None if not args.device else args.device),
    )

    print("Train metrics:")
    evaluate_mirt(
        model,
        train_set,
        batch_size=args.eval_batch_size,
        device=(None if not args.device else args.device),
        num_workers=args.num_workers,
    )

    print("Test metrics:")
    if args.split == "student_holdout":
        if args.holdout_eval == "online_theta":
            evaluate_mirt_student_holdout_online(
                model,
                test_set,
                theta_lr=args.theta_lr,
                theta_prior_mean=args.theta_prior_mean,
                theta_prior_std=args.theta_prior_std,
                seed=args.theta_seed,
            )
        else:
            evaluate_mirt_item_only(
                model,
                test_set,
                batch_size=args.eval_batch_size,
                device=(None if not args.device else args.device),
                num_workers=args.num_workers,
            )
    else:
        evaluate_mirt(
            model,
            test_set,
            batch_size=args.eval_batch_size,
            device=(None if not args.device else args.device),
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
