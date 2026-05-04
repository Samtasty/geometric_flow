from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import animation
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def hist_density_ratio(
    theta_ref: np.ndarray,
    theta_cur: np.ndarray,
    theta_eval: np.ndarray,
    *,
    n_bins: int = 120,
    ratio_clip: float = 20.0,
    eps: float = 1e-8,
) -> np.ndarray:
    lo = float(min(theta_ref.min(), theta_cur.min(), theta_eval.min()))
    hi = float(max(theta_ref.max(), theta_cur.max(), theta_eval.max()))
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, int(max(4, n_bins)) + 1)
    q_hist, _ = np.histogram(theta_ref, bins=edges, density=True)
    p_hist, _ = np.histogram(theta_cur, bins=edges, density=True)
    ratio_bins = (q_hist + eps) / (p_hist + eps)
    if ratio_clip > 1.0:
        ratio_bins = np.clip(ratio_bins, 1.0 / ratio_clip, ratio_clip)
    idx = np.clip(np.digitize(theta_eval, bins=edges[1:-1], right=False), 0, len(ratio_bins) - 1)
    out = ratio_bins[idx].astype(np.float64)
    return out / np.clip(float(out.mean()), 1e-12, None)


def select_target_theta(
    df: pd.DataFrame,
    *,
    theta_col: str,
    order_col: str,
    rounds_sorted: list[int],
    target_dist: str,
    target_k: int,
) -> np.ndarray:
    if target_dist == "round1":
        return df.loc[df[order_col] == rounds_sorted[0], theta_col].to_numpy(dtype=float)
    if target_dist == "global":
        return df[theta_col].to_numpy(dtype=float)
    if target_dist == "pooled_first_k":
        keep = set(rounds_sorted[: max(1, int(target_k))])
        return df.loc[df[order_col].isin(keep), theta_col].to_numpy(dtype=float)
    raise ValueError(f"Unknown target_dist={target_dist}")


def build_model(model_name: str, seed: int) -> object:
    if model_name == "hist_gb_regressor":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_depth=4,
            min_samples_leaf=64,
            random_state=seed,
        )
    if model_name == "decision_tree_classifier":
        return DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=64,
            random_state=seed,
        )
    if model_name == "random_forest_classifier":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=16,
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "xgb_classifier":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise ImportError(
                "xgboost is required for model='xgb_classifier'. "
                "Install with: pip install xgboost"
            ) from exc
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=0.05,
            n_estimators=250,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )
    raise ValueError(f"Unknown model={model_name}")


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Animate Direct Method policy curve delta*(theta) over cumulative rounds.",
    )
    p.add_argument("--csv", type=str, default="pix_mapping/pix_behavior_30rounds/ktm_dataframe_round30.csv")
    p.add_argument("--theta-col", type=str, default="proficiency")
    p.add_argument("--delta-col", type=str, default="difficulties")
    p.add_argument("--correct-col", type=str, default="correct")
    p.add_argument("--reward-col", type=str, default="")
    p.add_argument("--order-col", type=str, default="order_sequence")
    p.add_argument("--max-rounds", type=int, default=10)
    p.add_argument(
        "--target-dist",
        type=str,
        default="round1",
        choices=["round1", "global", "pooled_first_k"],
    )
    p.add_argument("--target-k", type=int, default=5)
    p.add_argument("--n-bins", type=int, default=120)
    p.add_argument("--ratio-clip", type=float, default=20.0)
    p.add_argument("--max-rows-per-round", type=int, default=60000)
    p.add_argument("--theta-q-low", type=float, default=0.01)
    p.add_argument("--theta-q-high", type=float, default=0.99)
    p.add_argument("--n-theta", type=int, default=220)
    p.add_argument("--n-delta", type=int, default=240)
    p.add_argument("--max-scatter", type=int, default=5000)
    p.add_argument(
        "--model",
        type=str,
        default="hist_gb_regressor",
        choices=[
            "hist_gb_regressor",
            "decision_tree_classifier",
            "random_forest_classifier",
            "xgb_classifier",
        ],
    )
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-gif",
        type=str,
        default="pix_mapping/pix_behavior_30rounds/direct_method_policy_by_round.gif",
    )
    p.add_argument(
        "--out-curves-csv",
        type=str,
        default="pix_mapping/pix_behavior_30rounds/direct_method_policy_by_round_curves.csv",
    )
    args = p.parse_args()

    cols = [args.theta_col, args.delta_col, args.correct_col, args.order_col]
    if args.reward_col:
        cols.append(args.reward_col)
    df = pd.read_csv(args.csv, usecols=cols).dropna().copy()
    df[args.correct_col] = df[args.correct_col].astype(float)
    df[args.order_col] = df[args.order_col].astype(int)

    delta_min = float(df[args.delta_col].min())
    if args.reward_col and args.reward_col in df.columns:
        df["_reward"] = df[args.reward_col].astype(float)
    else:
        df["_reward"] = df[args.correct_col] * (df[args.delta_col] - delta_min)

    rounds_sorted = sorted(df[args.order_col].unique().tolist())
    rounds = rounds_sorted[: max(1, int(args.max_rounds))]
    if not rounds:
        raise ValueError("No rounds found in input file.")

    theta_ref = select_target_theta(
        df,
        theta_col=args.theta_col,
        order_col=args.order_col,
        rounds_sorted=rounds_sorted,
        target_dist=args.target_dist,
        target_k=args.target_k,
    )

    theta_lo = float(df[args.theta_col].quantile(args.theta_q_low))
    theta_hi = float(df[args.theta_col].quantile(args.theta_q_high))
    if theta_hi <= theta_lo:
        theta_hi = theta_lo + 1e-6
    theta_grid = np.linspace(theta_lo, theta_hi, int(max(20, args.n_theta)))

    delta_lo = float(df[args.delta_col].min())
    delta_hi = float(df[args.delta_col].max())
    if delta_hi <= delta_lo:
        delta_hi = delta_lo + 1e-6
    delta_grid = np.linspace(delta_lo, delta_hi, int(max(30, args.n_delta)))

    n_th = len(theta_grid)
    n_de = len(delta_grid)
    tg = np.repeat(theta_grid, n_de)
    dg = np.tile(delta_grid, n_th)
    x_eval = np.column_stack([tg, dg])

    frames: list[dict[str, object]] = []
    curve_rows: list[dict[str, float | int]] = []
    for ridx, r in enumerate(rounds, start=1):
        cur = df[df[args.order_col] <= r].copy()
        if args.max_rows_per_round > 0 and len(cur) > args.max_rows_per_round:
            cur = cur.sample(n=int(args.max_rows_per_round), random_state=int(args.seed + ridx)).copy()

        x = cur[[args.theta_col, args.delta_col]].to_numpy(dtype=float)
        theta_cur = cur[args.theta_col].to_numpy(dtype=float)
        w_ctx = hist_density_ratio(
            theta_ref,
            theta_cur,
            theta_cur,
            n_bins=args.n_bins,
            ratio_clip=args.ratio_clip,
        )

        model = build_model(args.model, seed=int(args.seed + ridx))
        if args.model in {"xgb_classifier", "decision_tree_classifier", "random_forest_classifier"}:
            y = cur[args.correct_col].to_numpy(dtype=int)
            model.fit(x, y, sample_weight=w_ctx)
            p_eval = model.predict_proba(x_eval)[:, 1].reshape(n_th, n_de)
            reward_pred = p_eval * (delta_grid[None, :] - float(delta_min))
        else:
            y = cur["_reward"].to_numpy(dtype=float)
            model.fit(x, y, sample_weight=w_ctx)
            reward_pred = model.predict(x_eval).reshape(n_th, n_de)

        best_idx = np.argmax(reward_pred, axis=1)
        delta_star = delta_grid[best_idx]
        reward_star = reward_pred[np.arange(n_th), best_idx]

        ess = float((np.sum(w_ctx) ** 2) / np.clip(np.sum(w_ctx ** 2), 1e-12, None))
        view = cur
        if len(view) > args.max_scatter:
            view = view.sample(n=int(args.max_scatter), random_state=int(args.seed + ridx))
        x_sc = view[args.theta_col].to_numpy(dtype=float)
        y_sc = view[args.delta_col].to_numpy(dtype=float)

        frames.append(
            {
                "round_index": int(ridx),
                "round_value": int(r),
                "n_rows": int(len(cur)),
                "ess": ess,
                "scatter_x": x_sc,
                "scatter_y": y_sc,
                "delta_star": delta_star,
            }
        )
        for j in range(n_th):
            curve_rows.append(
                {
                    "round_index": int(ridx),
                    "round_value": int(r),
                    "theta": float(theta_grid[j]),
                    "delta_star": float(delta_star[j]),
                    "predicted_reward_star": float(reward_star[j]),
                    "n_rows": int(len(cur)),
                    "ess_ctx": ess,
                }
            )

    curves_df = pd.DataFrame(curve_rows)
    ensure_parent(args.out_curves_csv)
    curves_df.to_csv(args.out_curves_csv, index=False)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    def _draw(k: int) -> None:
        fr = frames[k]
        ax.clear()
        ax.scatter(
            fr["scatter_x"],
            fr["scatter_y"],
            s=10,
            alpha=0.08,
            color="#7f7f7f",
            label="cumulative logged (theta, delta)",
        )
        ax.plot(theta_grid, fr["delta_star"], color="#1f77b4", lw=2.7, label="DM policy: delta*(theta)")
        ax.set_xlim(theta_lo, theta_hi)
        ax.set_ylim(delta_lo, delta_hi)
        ax.set_xlabel("proficiency (theta)")
        ax.set_ylabel("difficulty (delta)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        ax.set_title(
            f"Round {fr['round_index']}/{len(frames)} (order={fr['round_value']})  "
            f"n={fr['n_rows']}  ess={fr['ess']:.0f}"
        )

    ani = animation.FuncAnimation(
        fig,
        _draw,
        frames=len(frames),
        interval=max(40, int(1000.0 / max(0.1, args.fps))),
        repeat=False,
    )
    ensure_parent(args.out_gif)
    ani.save(args.out_gif, writer=animation.PillowWriter(fps=max(0.5, args.fps)))
    plt.close(fig)

    print(f"rounds={len(rounds)} target_dist={args.target_dist} model={args.model}")
    print(f"saved_curves_csv={os.path.abspath(args.out_curves_csv)}")
    print(f"saved_gif={os.path.abspath(args.out_gif)}")


if __name__ == "__main__":
    main()
