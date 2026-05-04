import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.assistments_dataset import AssistmentsDataset
from data.splits import IndexedSubset, split_by_student_holdout, split_temporal_by_student
from eval.evaluate_mirt import evaluate_mirt, evaluate_mirt_item_only
from models.mirt import MIRTModelEM


def main():
    parser = argparse.ArgumentParser(description="Run asynchronous/online EM-MIRT.")
    parser.add_argument("--data", type=str, default="pix_mapping/pix_irt_outcome.csv")
    parser.add_argument(
        "--split",
        type=str,
        default="temporal_per_student",
        choices=["temporal_per_student", "student_holdout"],
    )
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=0)

    parser.add_argument("--emb-dim", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num-em-iter", type=int, default=3)
    parser.add_argument("--data-workers", type=int, default=0)
    parser.add_argument("--e-workers", type=int, default=12)
    parser.add_argument("--student-chunk-size", type=int, default=256)
    parser.add_argument("--participation-rate", type=float, default=1.0)
    parser.add_argument("--m-step-batches-per-round", type=int, default=1)
    args = parser.parse_args()

    dataset = AssistmentsDataset(args.data)
    if args.sample > 0 and args.sample < len(dataset):
        rng = np.random.default_rng(args.seed)
        sampled = rng.choice(np.arange(len(dataset)), size=args.sample, replace=False)
        dataset = IndexedSubset(dataset, sampled)

    if args.split == "temporal_per_student":
        train_set, test_set = split_temporal_by_student(dataset, train_ratio=args.train_ratio)
        split_desc = (
            f"Temporal per-student split: first {args.train_ratio:.2f} per student in train, rest in test."
        )
    else:
        train_set, test_set = split_by_student_holdout(
            dataset,
            test_size=args.test_size,
            seed=args.seed,
        )
        split_desc = (
            f"Student holdout split: {args.test_size:.2f} students in test only, no overlap."
        )

    base = train_set.dataset if hasattr(train_set, "dataset") else train_set
    model = MIRTModelEM(
        num_students=len(base.students),
        num_items=len(base.items),
        emb_dim=args.emb_dim,
        num_em_iter=args.num_em_iter,
    )

    print(split_desc)
    print(f"Train rows: {len(train_set)}")
    print(f"Test rows: {len(test_set)}")
    print(
        "EM async config:"
        f" emb_dim={args.emb_dim}, batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs},"
        f" num_em_iter={args.num_em_iter}, e_workers={args.e_workers},"
        f" student_chunk_size={args.student_chunk_size}, participation_rate={args.participation_rate},"
        f" m_step_batches_per_round={args.m_step_batches_per_round}"
    )

    model.em_train_async_online(
        train_set,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        num_workers=args.data_workers,
        e_workers=args.e_workers,
        student_chunk_size=args.student_chunk_size,
        participation_rate=args.participation_rate,
        m_step_batches_per_round=args.m_step_batches_per_round,
        seed=args.seed,
    )

    print("Train metrics:")
    evaluate_mirt(model, train_set)
    print("Test metrics:")
    if args.split == "student_holdout":
        evaluate_mirt_item_only(model, test_set)
    else:
        evaluate_mirt(model, test_set)


if __name__ == "__main__":
    main()
