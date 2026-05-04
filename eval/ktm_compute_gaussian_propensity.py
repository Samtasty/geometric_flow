import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ktm import processing_data


def _validate_coef_columns(df_fit: pd.DataFrame, mu_degree: int, sigma_degree: int) -> None:
    missing = []
    for d in range(mu_degree + 1):
        c = f"beta_mu_{d}"
        if c not in df_fit.columns:
            missing.append(c)
    for d in range(sigma_degree + 1):
        c = f"beta_sigma_{d}"
        if c not in df_fit.columns:
            missing.append(c)
    if missing:
        raise ValueError(f"Missing coefficient columns in fit CSV: {missing}")


def _nearest_available_order(x: np.ndarray, available: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(available, x)
    pos = np.clip(pos, 0, len(available) - 1)
    left = np.maximum(pos - 1, 0)
    right = pos
    choose_left = np.abs(x - available[left]) <= np.abs(x - available[right])
    out = np.where(choose_left, available[left], available[right])
    return out


def _attach_subset_params(
    df_proc: pd.DataFrame,
    df_fit: pd.DataFrame,
    *,
    fill_missing_nearest_order: bool = True,
) -> pd.DataFrame:
    work = df_proc.copy()
    fit = df_fit.copy()

    if (fit["order_min"] == fit["order_max"]).all():
        fit = fit.rename(columns={"order_min": "order_sequence"})
        keep_cols = ["subset", "order_sequence"] + [c for c in fit.columns if c.startswith("beta_")]
        merged = work.merge(fit[keep_cols], on="order_sequence", how="left")
        if fill_missing_nearest_order:
            miss = merged["subset"].isna()
            if miss.any():
                available = np.sort(fit["order_sequence"].unique().astype(int))
                target = merged.loc[miss, "order_sequence"].to_numpy(dtype=int)
                nearest = _nearest_available_order(target, available)
                fallback = fit[keep_cols].rename(columns={"order_sequence": "nearest_order"})
                miss_df = pd.DataFrame(
                    {
                        "order_sequence": target,
                        "nearest_order": nearest,
                    },
                    index=merged.index[miss],
                )
                miss_df = miss_df.merge(fallback, on="nearest_order", how="left").set_index(miss_df.index)
                merged.loc[miss, "subset"] = miss_df["subset"].to_numpy()
                for c in [x for x in keep_cols if x.startswith("beta_")]:
                    merged.loc[miss, c] = miss_df[c].to_numpy()
        return merged

    work["subset"] = np.nan
    beta_cols = [c for c in fit.columns if c.startswith("beta_")]
    for c in beta_cols:
        work[c] = np.nan
    for _, row in fit.iterrows():
        lo = int(row["order_min"])
        hi = int(row["order_max"])
        m = (work["order_sequence"] >= lo) & (work["order_sequence"] <= hi)
        work.loc[m, "subset"] = row["subset"]
        for c in beta_cols:
            work.loc[m, c] = row[c]
    return work


def _compute_propensity(
    df: pd.DataFrame,
    *,
    mu_degree: int,
    sigma_degree: int,
    sigma_floor: float,
) -> pd.DataFrame:
    out = df.copy()
    theta = out["proficiency"].to_numpy(dtype=float)
    delta = out["difficulties"].to_numpy(dtype=float)

    mu = np.zeros_like(theta, dtype=float)
    log_sigma = np.zeros_like(theta, dtype=float)

    for d in range(mu_degree + 1):
        mu += out[f"beta_mu_{d}"].to_numpy(dtype=float) * (theta ** d)
    for d in range(sigma_degree + 1):
        log_sigma += out[f"beta_sigma_{d}"].to_numpy(dtype=float) * (theta ** d)

    sigma = np.maximum(np.exp(log_sigma), sigma_floor)
    z = (delta - mu) / sigma
    log_prop = -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * (z ** 2)
    prop = np.exp(log_prop)

    out["mu_hat"] = mu
    out["sigma_hat"] = sigma
    out["log_propensity"] = log_prop
    out["propensity"] = prop
    # Reward upweights correct answers on harder items.
    difficulty_min = float(np.nanmin(delta))
    correct = out["correct"].to_numpy(dtype=float)
    out["reward"] = correct * (delta - difficulty_min)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-tuple Gaussian propensity p(delta | theta, order)."
    )
    parser.add_argument("--data", type=str, default="pix_mapping/pix_processed.csv")
    parser.add_argument("--fit-csv", type=str, required=True)
    parser.add_argument("--user-col", type=str, default="user_id")
    parser.add_argument("--item-col", type=str, default="challenge_id")
    parser.add_argument("--correct-col", type=str, default="outcome")
    parser.add_argument("--skill-col", type=str, default="skill_id")
    parser.add_argument("--answer-order-col", type=str, default="answer_number")
    parser.add_argument("--sample-users", type=int, default=0)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma-floor", type=float, default=1e-4)
    parser.add_argument("--no-nearest-fallback", action="store_true")
    parser.add_argument(
        "--out-csv",
        type=str,
        default="pix_mapping/ktm_gaussian_propensity.csv",
    )
    args = parser.parse_args()

    df_fit = pd.read_csv(args.fit_csv)
    if df_fit.empty:
        raise ValueError("Fit CSV is empty.")
    mu_degree = int(df_fit["mu_degree"].iloc[0])
    sigma_degree = int(df_fit["sigma_degree"].iloc[0])
    _validate_coef_columns(df_fit, mu_degree=mu_degree, sigma_degree=sigma_degree)

    usecols = [args.user_col, args.item_col, args.correct_col, args.answer_order_col]
    if args.skill_col:
        usecols.append(args.skill_col)
    df = pd.read_csv(args.data, usecols=usecols).dropna(
        subset=[args.user_col, args.item_col, args.correct_col]
    )

    if args.sample_users > 0 and args.sample_users < df[args.user_col].nunique():
        rng = np.random.default_rng(args.seed)
        users = df[args.user_col].dropna().unique()
        keep = set(rng.choice(users, size=args.sample_users, replace=False).tolist())
        df = df[df[args.user_col].isin(keep)].copy()
    if args.sample_rows > 0 and args.sample_rows < len(df):
        df = df.sample(n=args.sample_rows, random_state=args.seed).copy()

    df = df.rename(
        columns={
            args.user_col: "user",
            args.item_col: "item",
            args.correct_col: "correct",
            args.answer_order_col: "answer_number",
        }
    )
    if args.skill_col and args.skill_col in df.columns:
        df = df.rename(columns={args.skill_col: "skill"})

    df_proc = processing_data(
        df,
        user_col="user",
        item_col="item",
        correct_col="correct",
        skill_col="skill",
        order_cols=["answer_number"],
        reduce=False,
        rank_start=0,
    )

    merged = _attach_subset_params(
        df_proc,
        df_fit,
        fill_missing_nearest_order=not args.no_nearest_fallback,
    )
    out = _compute_propensity(
        merged,
        mu_degree=mu_degree,
        sigma_degree=sigma_degree,
        sigma_floor=args.sigma_floor,
    )

    out["sequence"] = out["user"]
    cols = [
        "sequence",
        "user",
        "item",
        "correct",
        "reward",
        "answer_number",
        "sequence_position",
        "order_sequence",
        "sequence_length",
        "proficiency",
        "difficulties",
        "subset",
        "mu_hat",
        "sigma_hat",
        "log_propensity",
        "propensity",
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    matched = int(out["subset"].notna().sum()) if "subset" in out.columns else 0
    print(f"rows={len(out)} matched_subsets={matched}")
    print(f"mu_degree={mu_degree} sigma_degree={sigma_degree}")
    print(f"saved_csv={os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
