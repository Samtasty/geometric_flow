from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier


def _hist_density_ratio(
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


def _select_target_theta(
    df: pd.DataFrame,
    *,
    theta_col: str,
    order_col: str,
    rounds: list[int],
    target_dist: str,
    target_k: int,
) -> np.ndarray:
    if target_dist == "round1":
        return df.loc[df[order_col] == rounds[0], theta_col].to_numpy(dtype=float)
    if target_dist == "global":
        return df[theta_col].to_numpy(dtype=float)
    if target_dist == "pooled_first_k":
        keep = set(rounds[: max(1, int(target_k))])
        return df.loc[df[order_col].isin(keep), theta_col].to_numpy(dtype=float)
    raise ValueError(f"Unknown target_dist={target_dist}")


@dataclass
class ModelSpec:
    name: str
    kind: str
    est: object


def _model_specs(seed: int) -> list[ModelSpec]:
    specs: list[ModelSpec] = [
        ModelSpec(
            name="logistic_l2",
            kind="plain",
            est=LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, random_state=seed),
        ),
        ModelSpec(
            name="logistic_poly2",
            kind="pipeline_lr",
            est=Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("lr", LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, random_state=seed)),
                ]
            ),
        ),
        ModelSpec(
            name="random_forest",
            kind="plain",
            est=RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=16,
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        ModelSpec(
            name="decision_tree_classifier",
            kind="plain",
            est=DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=64,
                random_state=seed,
            ),
        ),
        ModelSpec(
            name="hist_gb",
            kind="plain",
            est=HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_depth=3,
                min_samples_leaf=64,
                random_state=seed,
            ),
        ),
    ]
    try:
        from xgboost import XGBClassifier

        specs.append(
            ModelSpec(
                name="xgb_classifier",
                kind="plain",
                est=XGBClassifier(
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
                ),
            )
        )
    except Exception:
        # Keep script runnable when xgboost is not installed.
        pass
    return specs


def _fit_with_weights(spec: ModelSpec, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> object:
    if spec.kind == "pipeline_lr":
        return spec.est.fit(x, y, lr__sample_weight=w)
    return spec.est.fit(x, y, sample_weight=w)


def _cv_auc_weighted(
    *,
    spec: ModelSpec,
    x: np.ndarray,
    y: np.ndarray,
    w_ctx: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[float, float, int]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), 0
    skf = StratifiedKFold(n_splits=max(2, int(folds)), shuffle=True, random_state=int(seed))
    aucs: list[float] = []
    used = 0
    for tr_idx, va_idx in skf.split(x, y):
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue
        x_tr = x[tr_idx]
        x_va = x[va_idx]
        w_tr = w_ctx[tr_idx]
        w_va = w_ctx[va_idx]
        m = _fit_with_weights(spec, x_tr, y_tr, w_tr)
        p = m.predict_proba(x_va)[:, 1]
        auc = roc_auc_score(y_va, p, sample_weight=w_va)
        aucs.append(float(auc))
        used += 1
    if not aucs:
        return float("nan"), float("nan"), 0
    return float(np.mean(aucs)), float(np.std(aucs)), int(used)


def run_benchmark(
    *,
    csv_path: str,
    theta_col: str,
    action_col: str,
    label_col: str,
    order_col: str,
    max_rounds: int,
    target_dist: str,
    target_k: int,
    n_bins: int,
    ratio_clip: float,
    cv_folds: int,
    max_rows_per_round: int,
    seed: int,
    out_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, usecols=[theta_col, action_col, label_col, order_col]).dropna().copy()
    df[label_col] = df[label_col].astype(int)
    df[order_col] = df[order_col].astype(int)

    rounds = sorted(df[order_col].unique().tolist())[: max(1, int(max_rounds))]
    target_theta = _select_target_theta(
        df,
        theta_col=theta_col,
        order_col=order_col,
        rounds=rounds,
        target_dist=target_dist,
        target_k=target_k,
    )
    specs = _model_specs(seed)
    rows: list[dict[str, float | int | str]] = []
    for ridx, r in enumerate(rounds, start=1):
        cur = df[df[order_col] <= r].copy()
        if max_rows_per_round > 0 and len(cur) > max_rows_per_round:
            cur = cur.sample(n=int(max_rows_per_round), random_state=int(seed)).copy()
        x = cur[[theta_col, action_col]].to_numpy(dtype=float)
        y = cur[label_col].to_numpy(dtype=int)
        theta_cur = cur[theta_col].to_numpy(dtype=float)
        w_ctx = _hist_density_ratio(
            target_theta,
            theta_cur,
            theta_cur,
            n_bins=n_bins,
            ratio_clip=ratio_clip,
        )
        for spec in specs:
            auc_mean, auc_std, folds_used = _cv_auc_weighted(
                spec=spec,
                x=x,
                y=y,
                w_ctx=w_ctx,
                folds=cv_folds,
                seed=seed + ridx,
            )
            rows.append(
                {
                    "round_index": int(ridx),
                    "round_value": int(r),
                    "n_rows": int(len(cur)),
                    "positive_rate": float(np.mean(y)),
                    "model": spec.name,
                    "auc_cv_weighted_mean": auc_mean,
                    "auc_cv_weighted_std": auc_std,
                    "folds_used": int(folds_used),
                }
            )

    long_df = pd.DataFrame(rows)
    summary_df = (
        long_df.sort_values(
            by=["round_index", "auc_cv_weighted_mean"],
            ascending=[True, False],
            na_position="last",
        )
        .groupby("round_index", as_index=False)
        .head(1)
        .rename(columns={"model": "best_model", "auc_cv_weighted_mean": "best_auc_cv_weighted_mean"})
        [["round_index", "round_value", "n_rows", "positive_rate", "best_model", "best_auc_cv_weighted_mean"]]
        .reset_index(drop=True)
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out / "direct_method_reweighted_cv_auc_by_round.csv", index=False)
    summary_df.to_csv(out / "direct_method_reweighted_cv_best_model_by_round.csv", index=False)
    return long_df, summary_df


def main() -> None:
    p = argparse.ArgumentParser(
        description="Round-by-round cumulative Direct Method model selection with context-shift reweighting."
    )
    p.add_argument("--csv", type=str, default="pix_mapping/pix_behavior_30rounds/ktm_dataframe_round30.csv")
    p.add_argument("--theta-col", type=str, default="proficiency")
    p.add_argument("--action-col", type=str, default="difficulties")
    p.add_argument("--label-col", type=str, default="correct")
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
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--max-rows-per-round", type=int, default=500000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="pix_mapping/pix_behavior_30rounds")
    args = p.parse_args()

    long_df, summary_df = run_benchmark(
        csv_path=args.csv,
        theta_col=args.theta_col,
        action_col=args.action_col,
        label_col=args.label_col,
        order_col=args.order_col,
        max_rounds=args.max_rounds,
        target_dist=args.target_dist,
        target_k=args.target_k,
        n_bins=args.n_bins,
        ratio_clip=args.ratio_clip,
        cv_folds=args.cv_folds,
        max_rows_per_round=args.max_rows_per_round,
        seed=args.seed,
        out_dir=args.out_dir,
    )

    with pd.option_context("display.max_rows", 50, "display.width", 180):
        print("\nPer-round best model:")
        print(summary_df.to_string(index=False))
        print("\nSaved:")
        print(str(Path(args.out_dir, "direct_method_reweighted_cv_auc_by_round.csv").resolve()))
        print(str(Path(args.out_dir, "direct_method_reweighted_cv_best_model_by_round.csv").resolve()))


if __name__ == "__main__":
    main()
