
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def _calibration_gap_quantile_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[float, float]:
    q = np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
    q[0] = 0.0
    q[-1] = 1.0
    for i in range(1, len(q)):
        if q[i] <= q[i - 1]:
            q[i] = min(1.0, q[i - 1] + 1e-12)
    idx = np.digitize(y_prob, q[1:-1], right=False)

    gaps: list[float] = []
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        pred = float(np.mean(y_prob[mask]))
        obs = float(np.mean(y_true[mask]))
        gap = abs(pred - obs)
        gaps.append(gap)
        ece += gap * float(np.mean(mask))
    return float(np.mean(gaps)), float(ece)


def compute_rasch_calibration(
    *,
    run_dir: Path,
    ktm_csv: Path,
    split_name: str,
) -> dict[str, float | int | str]:
    run_summary = pd.read_csv(run_dir / "run_summary.csv").iloc[0]
    split_assign = pd.read_csv(run_dir / "user_split_assignments.csv")
    split_map = dict(zip(split_assign["user"], split_assign["split"]))

    usecols = ["user", "answer_number", "proficiency", "difficulties", "correct"]
    df = pd.read_csv(ktm_csv, usecols=usecols)
    truncate_at_round = int(run_summary["truncate_at_round"])
    if truncate_at_round > 0:
        df = df[df["answer_number"] <= truncate_at_round].copy()

    df = df[df["user"].map(split_map) == split_name].copy()
    theta = df["proficiency"].to_numpy(dtype=float)
    delta = df["difficulties"].to_numpy(dtype=float)
    y_true = df["correct"].to_numpy(dtype=int)

    y_prob = 1.0 / (1.0 + np.exp(-(theta - delta)))
    y_prob = np.clip(y_prob, 1e-12, 1.0 - 1e-12)

    cal_gap, ece = _calibration_gap_quantile_bins(y_true, y_prob, n_bins=10)
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "dataset": str(run_dir.name),
        "split": split_name,
        "n_rows": int(len(df)),
        "logloss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "auc": auc,
        "cal_gap_q10": cal_gap,
        "ece_q10": ece,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Rasch calibration diagnostics on a user-split run.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--ktm-csv", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-csv", type=str, default="")
    args = parser.parse_args()

    row = compute_rasch_calibration(
        run_dir=Path(args.run_dir).resolve(),
        ktm_csv=Path(args.ktm_csv).resolve(),
        split_name=str(args.split),
    )
    df = pd.DataFrame([row])
    print(df.to_string(index=False))
    if args.out_csv:
        out_path = Path(args.out_csv).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"saved_csv={out_path}")


if __name__ == "__main__":
    main()
