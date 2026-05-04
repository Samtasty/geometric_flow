import argparse
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.assistments_dataset import AssistmentsDataset
from data.splits import IndexedSubset, split_temporal_by_student
from eval.evaluate_mirt import evaluate_mirt
from train.train_mirt import train_mirt
from train.train_mirt_ddp import train_mirt_ddp


def run_single(
    data_path,
    cache_path,
    max_students,
    seed,
    emb_dim,
    batch_size,
    lr,
    epochs,
    train_ratio,
    student_l2,
    num_workers,
    eval_batch_size,
):
    dataset = AssistmentsDataset(data_path, cache_path=cache_path)
    if max_students > 0:
        rng = np.random.default_rng(seed)
        all_students = np.array(dataset.data["student_idx"].unique())
        keep = set(rng.choice(all_students, size=min(max_students, len(all_students)), replace=False).tolist())
        keep_rows = dataset.data.index[dataset.data["student_idx"].isin(keep)].to_numpy()
        dataset = IndexedSubset(dataset, keep_rows)

    train_set, test_set = split_temporal_by_student(dataset, train_ratio=train_ratio)

    t0 = time.perf_counter()
    model, _ = train_mirt(
        train_set,
        emb_dim=emb_dim,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        student_l2=student_l2,
        num_workers=num_workers,
    )
    train_metrics = evaluate_mirt(
        model, train_set, batch_size=eval_batch_size, num_workers=num_workers
    )
    test_metrics = evaluate_mirt(
        model, test_set, batch_size=eval_batch_size, num_workers=num_workers
    )
    dt = time.perf_counter() - t0
    return {
        "seconds_total": dt,
        "train": train_metrics,
        "test": test_metrics,
        "train_rows": len(train_set),
        "test_rows": len(test_set),
    }


def run_ddp(
    data_path,
    cache_path,
    max_students,
    seed,
    emb_dim,
    batch_size,
    lr,
    epochs,
    train_ratio,
    student_l2,
    num_workers,
    eval_batch_size,
    world_size,
):
    t0 = time.perf_counter()
    metrics = train_mirt_ddp(
        data_path=data_path,
        cache_path=cache_path,
        emb_dim=emb_dim,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        train_ratio=train_ratio,
        student_l2=student_l2,
        max_students=max_students,
        seed=seed,
        num_workers=num_workers,
        eval_batch_size=eval_batch_size,
        world_size=world_size,
    )
    dt = time.perf_counter() - t0
    return {"seconds_total": dt, **metrics}


def main():
    parser = argparse.ArgumentParser(description="Benchmark single vs DDP MIRT.")
    parser.add_argument("--data", type=str, default="pix_mapping/pix_irt_outcome.csv")
    parser.add_argument("--cache-path", type=str, default="pix_mapping/pix_irt_outcome.pt")
    parser.add_argument("--max-students", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emb-dim", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--student-l2", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--ddp-world-size", type=int, default=2)
    parser.add_argument(
        "--out",
        type=str,
        default="pix_mapping/benchmark_ddp_vs_single.json",
    )
    args = parser.parse_args()

    if args.cache_path and not os.path.exists(args.cache_path):
        print(f"Cache not found at {args.cache_path}, using raw data loading.")
        cache_path = None
    else:
        cache_path = args.cache_path if args.cache_path else None

    print("Running single-process baseline...")
    single = run_single(
        data_path=args.data,
        cache_path=cache_path,
        max_students=args.max_students,
        seed=args.seed,
        emb_dim=args.emb_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        train_ratio=args.train_ratio,
        student_l2=args.student_l2,
        num_workers=args.num_workers,
        eval_batch_size=args.eval_batch_size,
    )

    print("Running DDP...")
    ddp = run_ddp(
        data_path=args.data,
        cache_path=cache_path,
        max_students=args.max_students,
        seed=args.seed,
        emb_dim=args.emb_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        train_ratio=args.train_ratio,
        student_l2=args.student_l2,
        num_workers=args.num_workers,
        eval_batch_size=args.eval_batch_size,
        world_size=args.ddp_world_size,
    )

    speedup = single["seconds_total"] / ddp["seconds_total"] if ddp["seconds_total"] > 0 else float("nan")
    payload = {
        "config": vars(args),
        "single": single,
        "ddp": ddp,
        "speedup_single_over_ddp": speedup,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"saved={os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
