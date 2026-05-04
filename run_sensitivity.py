"""Sensitivity analysis: perturb proficiency (context theta) with Gaussian noise.

Reuses the preprocessed ktm_dataframe.csv from a base run, adds N(0, sigma) noise
to the proficiency column, then runs the full pipeline (behavior fitting + training).

Usage:
    python run_sensitivity.py \
        --base-ktm results/attempts_trunc30/ktm_dataframe.csv \
        --noise-stds 0.1 0.2 \
        --output-prefix results/attempts_noise \
        [extra args forwarded to run_last_round_simple.py]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_SCRIPT = PROJECT_ROOT / "offpolicy_ktm_pipeline" / "run_last_round_simple.py"
PYTHON = sys.executable


def _add_noise(df: pd.DataFrame, std: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["proficiency"] = out["proficiency"] + rng.normal(0.0, std, size=len(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-ktm",
        type=str,
        required=True,
        help="Path to an existing ktm_dataframe.csv (e.g. results/attempts_trunc30/ktm_dataframe.csv).",
    )
    parser.add_argument(
        "--noise-stds",
        type=float,
        nargs="+",
        default=[0.1, 0.2],
        help="List of Gaussian noise std values to test.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output directory prefix. Each run saves to <prefix>_noise<std> (e.g. results/attempts_noise_0p1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (each noise level gets seed + i).",
    )
    # Pipeline passthrough args (same settings as the base run)
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument("--round-cap-strategy", type=str, default="quantile")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dm-delta-grid", type=int, default=121)
    parser.add_argument("--objectives", type=str, nargs="+", default=["ips", "snips", "dr", "mis"])
    parser.add_argument("--min-obs-per-order", type=int, default=200)
    parser.add_argument("--context-bins", type=int, default=120)
    parser.add_argument("--context-ratio-clip", type=float, default=20.0)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    base_ktm_path = Path(args.base_ktm).resolve()
    if not base_ktm_path.exists():
        print(f"ERROR: base ktm not found: {base_ktm_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading base ktm: {base_ktm_path}")
    df_base = pd.read_csv(base_ktm_path)
    print(f"  rows={len(df_base)}, order_sequence range: {df_base['order_sequence'].min()}–{df_base['order_sequence'].max()}")

    for i, std in enumerate(args.noise_stds):
        std_tag = f"{std:.2f}".replace(".", "p")
        out_dir = f"{args.output_prefix}_{std_tag}"
        print(f"\n{'='*60}")
        print(f"Noise std={std}  ->  {out_dir}")
        print(f"{'='*60}")

        df_noisy = _add_noise(df_base, std=std, seed=args.seed + i)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, prefix=f"ktm_noise_{std_tag}_") as f:
            noisy_csv = f.name
        df_noisy.to_csv(noisy_csv, index=False)
        print(f"Saved noisy ktm ({len(df_noisy)} rows) to tmp: {noisy_csv}")

        cmd = [
            PYTHON, str(PIPELINE_SCRIPT),
            "--input-csv", noisy_csv,
            "--input-is-ktm",
            "--output-dir", out_dir,
            "--seed", str(args.seed + i),
            "--max-rounds", str(args.max_rounds),
            "--round-cap-strategy", args.round_cap_strategy,
            "--truncate-at-round", "0",
            "--epochs", str(args.epochs),
            "--dm-delta-grid", str(args.dm_delta_grid),
            "--objectives", *args.objectives,
            "--min-obs-per-order", str(args.min_obs_per_order),
            "--context-bins", str(args.context_bins),
            "--context-ratio-clip", str(args.context_ratio_clip),
            "--max-weight", str(args.max_weight),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
        ]
        if args.device:
            cmd += ["--device", args.device]

        print("Running:", " ".join(cmd))
        ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if ret.returncode != 0:
            print(f"ERROR: pipeline failed for std={std} (exit code {ret.returncode})", file=sys.stderr)
            sys.exit(ret.returncode)

        Path(noisy_csv).unlink(missing_ok=True)
        print(f"Done: std={std} -> {out_dir}")

    print(f"\nAll noise levels complete. Compare results in:")
    for std in args.noise_stds:
        std_tag = f"{std:.2f}".replace(".", "p")
        print(f"  {args.output_prefix}_{std_tag}/")
    print(f"  (baseline: {Path(args.base_ktm).parent}/)")


if __name__ == "__main__":
    main()
