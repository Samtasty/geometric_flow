import argparse
import os
from dataclasses import dataclass

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class DirectMethodModel:
    family: str
    model: object
    delta_min: float

    def predict_reward(self, x: np.ndarray) -> np.ndarray:
        if self.family == "classifier":
            p = self.model.predict_proba(x)[:, 1]
            return p * (x[:, 1] - self.delta_min)
        return np.asarray(self.model.predict(x), dtype=float)


def build_model(family: str, model_name: str, random_state: int) -> object:
    if family == "classifier":
        if model_name == "decision_tree_classifier":
            return DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=64,
                random_state=random_state,
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
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
            )
        if model_name == "hist_gb":
            return HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_depth=3,
                min_samples_leaf=64,
                random_state=random_state,
            )
        if model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=16,
                n_jobs=-1,
                random_state=random_state,
            )
        if model_name == "logistic_poly2":
            return Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    (
                        "lr",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=1000,
                            C=1.0,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
        raise ValueError(f"Unknown classifier model: {model_name}")

    if family == "regressor":
        if model_name == "hist_gb":
            return HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                max_depth=3,
                min_samples_leaf=64,
                random_state=random_state,
            )
        if model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=16,
                n_jobs=-1,
                random_state=random_state,
            )
        if model_name == "ridge_poly2":
            return Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("ridge", Ridge(alpha=1.0, random_state=random_state)),
                ]
            )
        raise ValueError(f"Unknown regressor model: {model_name}")

    raise ValueError(f"Unknown family: {family}")


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


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
    round_col: str,
    rounds_sorted: list[int],
    target_dist: str,
    target_k: int,
) -> np.ndarray:
    if target_dist == "round1":
        return df.loc[df[round_col] == rounds_sorted[0], theta_col].to_numpy(dtype=float)
    if target_dist == "global":
        return df[theta_col].to_numpy(dtype=float)
    if target_dist == "pooled_first_k":
        keep = set(rounds_sorted[: max(1, int(target_k))])
        return df.loc[df[round_col].isin(keep), theta_col].to_numpy(dtype=float)
    raise ValueError(f"Unknown target_dist={target_dist}")


def fit_model_with_weights(model: object, family: str, model_name: str, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> object:
    if family == "classifier" and model_name == "logistic_poly2":
        return model.fit(x, y, lr__sample_weight=w)
    if family == "regressor" and model_name == "ridge_poly2":
        return model.fit(x, y, ridge__sample_weight=w)
    return model.fit(x, y, sample_weight=w)


def compute_irt_optimal_delta(
    theta_grid: np.ndarray,
    delta_grid: np.ndarray,
    delta_min: float,
) -> np.ndarray:
    out = np.zeros_like(theta_grid, dtype=float)
    for i, th in enumerate(theta_grid):
        r = (delta_grid - delta_min) * sigmoid(th - delta_grid)
        out[i] = delta_grid[int(np.argmax(r))]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a Direct Method model and plot policy delta*(theta)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="pix_mapping/pix_behavior_30rounds/ktm_dataframe_round30.csv",
    )
    parser.add_argument("--theta-col", type=str, default="proficiency")
    parser.add_argument("--delta-col", type=str, default="difficulties")
    parser.add_argument("--correct-col", type=str, default="correct")
    parser.add_argument("--reward-col", type=str, default="")
    parser.add_argument("--round-col", type=str, default="order_sequence")
    parser.add_argument("--train-max-round", type=int, default=19)
    parser.add_argument("--family", type=str, choices=["classifier", "regressor"], default="regressor")
    parser.add_argument(
        "--model",
        type=str,
        default="hist_gb",
        help="classifier: decision_tree_classifier|xgb_classifier|hist_gb|random_forest|logistic_poly2 ; regressor: hist_gb|random_forest|ridge_poly2",
    )
    parser.add_argument("--n-theta", type=int, default=220)
    parser.add_argument("--n-delta", type=int, default=280)
    parser.add_argument("--theta-q-low", type=float, default=0.01)
    parser.add_argument("--theta-q-high", type=float, default=0.99)
    parser.add_argument(
        "--target-dist",
        type=str,
        default="round1",
        choices=["round1", "global", "pooled_first_k"],
    )
    parser.add_argument("--target-k", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=120)
    parser.add_argument("--ratio-clip", type=float, default=20.0)
    parser.add_argument("--max-train", type=int, default=450000)
    parser.add_argument("--max-scatter", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-irt-optimal", action="store_true")
    parser.add_argument(
        "--out-curve-csv",
        type=str,
        default="pix_mapping/pix_behavior_30rounds/direct_method_policy_curve.csv",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="pix_mapping/pix_behavior_30rounds/direct_method_policy_curve.png",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    cols_needed = [args.theta_col, args.delta_col, args.correct_col, args.round_col]
    if args.reward_col:
        cols_needed.append(args.reward_col)
    df = df[cols_needed].dropna().copy()
    df[args.correct_col] = df[args.correct_col].astype(float)
    df[args.round_col] = df[args.round_col].astype(int)

    delta_min = float(df[args.delta_col].min())
    if args.reward_col and args.reward_col in df.columns:
        df["_reward"] = df[args.reward_col].astype(float)
    else:
        df["_reward"] = df[args.correct_col] * (df[args.delta_col] - delta_min)

    train_df = df[df[args.round_col] <= args.train_max_round].copy()
    if len(train_df) == 0:
        raise ValueError("No rows in train split. Check --train-max-round and --round-col.")
    if len(train_df) > args.max_train:
        train_df = train_df.sample(n=args.max_train, random_state=args.seed)

    rounds_sorted = sorted(df[args.round_col].unique().tolist())
    theta_ref = select_target_theta(
        df,
        theta_col=args.theta_col,
        round_col=args.round_col,
        rounds_sorted=rounds_sorted,
        target_dist=args.target_dist,
        target_k=args.target_k,
    )
    theta_train = train_df[args.theta_col].to_numpy(dtype=float)
    w_ctx = hist_density_ratio(
        theta_ref,
        theta_train,
        theta_train,
        n_bins=args.n_bins,
        ratio_clip=args.ratio_clip,
    )

    x_train = train_df[[args.theta_col, args.delta_col]].to_numpy(dtype=float)
    y_train = (
        train_df[args.correct_col].to_numpy(dtype=int)
        if args.family == "classifier"
        else train_df["_reward"].to_numpy(dtype=float)
    )

    base_model = build_model(args.family, args.model, args.seed)
    base_model = fit_model_with_weights(base_model, args.family, args.model, x_train, y_train, w_ctx)
    dm_model = DirectMethodModel(family=args.family, model=base_model, delta_min=delta_min)

    theta_low = float(df[args.theta_col].quantile(args.theta_q_low))
    theta_high = float(df[args.theta_col].quantile(args.theta_q_high))
    delta_low = float(df[args.delta_col].min())
    delta_high = float(df[args.delta_col].max())

    theta_grid = np.linspace(theta_low, theta_high, args.n_theta)
    delta_grid = np.linspace(delta_low, delta_high, args.n_delta)

    policy_delta = np.zeros_like(theta_grid, dtype=float)
    policy_reward = np.zeros_like(theta_grid, dtype=float)
    for i, th in enumerate(theta_grid):
        xg = np.column_stack(
            [np.full(args.n_delta, th, dtype=float), delta_grid.astype(float)]
        )
        rg = dm_model.predict_reward(xg)
        j = int(np.argmax(rg))
        policy_delta[i] = delta_grid[j]
        policy_reward[i] = rg[j]

    curve_df = pd.DataFrame(
        {
            "theta": theta_grid,
            "delta_star": policy_delta,
            "predicted_reward_star": policy_reward,
            "family": args.family,
            "model": args.model,
        }
    )

    irt_opt = None
    if args.plot_irt_optimal:
        irt_opt = compute_irt_optimal_delta(theta_grid, delta_grid, delta_min)
        curve_df["irt_optimal_delta_star"] = irt_opt

    os.makedirs(os.path.dirname(args.out_curve_csv), exist_ok=True)
    curve_df.to_csv(args.out_curve_csv, index=False)

    scatter_df = df
    if len(scatter_df) > args.max_scatter:
        scatter_df = scatter_df.sample(n=args.max_scatter, random_state=args.seed)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.scatter(
        scatter_df[args.theta_col],
        scatter_df[args.delta_col],
        s=9,
        alpha=0.1,
        color="#7f7f7f",
        label="logged (theta, delta)",
    )
    ax.plot(theta_grid, policy_delta, color="#1f77b4", lw=2.6, label="DM policy: delta*(theta)")
    if irt_opt is not None:
        ax.plot(theta_grid, irt_opt, color="#2ca02c", lw=2.0, ls="--", label="IRT-optimal delta*(theta)")
    ax.set_xlabel("proficiency (theta)")
    ax.set_ylabel("difficulty (delta)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_plot), exist_ok=True)
    fig.savefig(args.out_plot, dpi=180)
    plt.close(fig)

    ess = float((np.sum(w_ctx) ** 2) / np.clip(np.sum(w_ctx ** 2), 1e-12, None))
    print(f"train_rows={len(train_df)} family={args.family} model={args.model}")
    print(
        f"context_weights: mean={float(np.mean(w_ctx)):.4f} std={float(np.std(w_ctx)):.4f} "
        f"min={float(np.min(w_ctx)):.4f} max={float(np.max(w_ctx)):.4f} ess={ess:.1f}"
    )
    print(f"saved_curve_csv={os.path.abspath(args.out_curve_csv)}")
    print(f"saved_plot={os.path.abspath(args.out_plot)}")


if __name__ == "__main__":
    main()
