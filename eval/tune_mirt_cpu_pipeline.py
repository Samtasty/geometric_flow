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


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Tune single-process CPU MIRT pipeline.")
    parser.add_argument("--data", type=str, default="pix_mapping/pix_irt_outcome.csv")
    parser.add_argument("--cache-path", type=str, default="pix_mapping/pix_irt_outcome.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=150000)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--emb-dim", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--student-l2", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-sizes", type=str, default="1024,2048,4096")
    parser.add_argument("--num-workers-list", type=str, default="0,4,8")
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--out", type=str, default="pix_mapping/mirt_cpu_tuning.json")
    args = parser.parse_args()

    batch_sizes = parse_int_list(args.batch_sizes)
    workers_list = parse_int_list(args.num_workers_list)

    cache_path = args.cache_path if (args.cache_path and os.path.exists(args.cache_path)) else None
    dataset = AssistmentsDataset(args.data, cache_path=cache_path)

    if args.sample > 0 and args.sample < len(dataset):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(np.arange(len(dataset)), size=args.sample, replace=False)
        dataset = IndexedSubset(dataset, idx)

    train_set, test_set = split_temporal_by_student(dataset, train_ratio=args.train_ratio)

    results = []
    for batch_size in batch_sizes:
        for num_workers in workers_list:
            try:
                t0 = time.perf_counter()
                model, _ = train_mirt(
                    train_set,
                    emb_dim=args.emb_dim,
                    batch_size=batch_size,
                    lr=args.lr,
                    epochs=args.epochs,
                    student_l2=args.student_l2,
                    num_workers=num_workers,
                    batch_mode="random",
                    device="cpu",
                )
                train_metrics = evaluate_mirt(
                    model,
                    train_set,
                    batch_size=args.eval_batch_size,
                    device="cpu",
                    num_workers=num_workers,
                )
                test_metrics = evaluate_mirt(
                    model,
                    test_set,
                    batch_size=args.eval_batch_size,
                    device="cpu",
                    num_workers=num_workers,
                )
                elapsed = time.perf_counter() - t0
                result = {
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "seconds_total": elapsed,
                    "train": train_metrics,
                    "test": test_metrics,
                    "status": "ok",
                }
            except Exception as exc:
                result = {
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "status": "failed",
                    "error": str(exc),
                }
            print(result)
            results.append(result)

    valid = [r for r in results if r.get("status") == "ok"]
    best = sorted(valid, key=lambda x: x["seconds_total"])[0] if valid else None
    payload = {
        "config": vars(args),
        "best_by_time": best,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"saved={os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
