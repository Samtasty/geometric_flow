import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ktm import processing_data


@dataclass
class FitResult:
    beta_mu: np.ndarray
    beta_sigma: np.ndarray
    nll: float
    rmse: float
    mae: float
    success: bool
    message: str


def _poly_design(x: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return np.column_stack(cols)


def _fit_gaussian_regression(
    theta: np.ndarray,
    delta: np.ndarray,
    mu_degree: int,
    sigma_degree: int,
    sigma_floor: float = 1e-4,
) -> FitResult:
    x_mu = _poly_design(theta, mu_degree)
    x_sig = _poly_design(theta, sigma_degree)

    beta_mu0 = np.linalg.lstsq(x_mu, delta, rcond=None)[0]
    resid0 = delta - x_mu @ beta_mu0
    init_log_std = float(np.log(np.std(resid0) + 1e-3))
    beta_sig0 = np.zeros(x_sig.shape[1], dtype=float)
    beta_sig0[0] = init_log_std
    p0 = np.concatenate([beta_mu0, beta_sig0])

    k_mu = x_mu.shape[1]
    cst = 0.5 * np.log(2.0 * np.pi)

    def nll(params: np.ndarray) -> float:
        b_mu = params[:k_mu]
        b_sig = params[k_mu:]
        mu = x_mu @ b_mu
        log_sigma = x_sig @ b_sig
        sigma = np.exp(log_sigma)
        sigma = np.maximum(sigma, sigma_floor)
        z = (delta - mu) / sigma
        return float(np.mean(cst + np.log(sigma) + 0.5 * (z ** 2)))

    out = minimize(nll, p0, method="L-BFGS-B")
    p = out.x if out.success else p0
    beta_mu = p[:k_mu]
    beta_sigma = p[k_mu:]
    mu_hat = x_mu @ beta_mu
    rmse = float(np.sqrt(np.mean((delta - mu_hat) ** 2)))
    mae = float(np.mean(np.abs(delta - mu_hat)))
    return FitResult(
        beta_mu=beta_mu,
        beta_sigma=beta_sigma,
        nll=float(nll(p)),
        rmse=rmse,
        mae=mae,
        success=bool(out.success),
        message=str(out.message),
    )


def _build_subsets_by_bins(df: pd.DataFrame, n_bins: int) -> list[tuple[str, pd.DataFrame]]:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    work = df.copy()
    work["order_bin"] = pd.qcut(
        work["order_sequence"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )
    subsets: list[tuple[str, pd.DataFrame]] = []
    for b, g in work.groupby("order_bin", sort=True):
        lo = int(g["order_sequence"].min())
        hi = int(g["order_sequence"].max())
        label = f"bin_{int(b)}_order_{lo}_{hi}"
        subsets.append((label, g.copy()))
    return subsets


def _build_subsets_by_order(
    df: pd.DataFrame,
    *,
    min_obs_per_subset: int = 1,
) -> list[tuple[str, pd.DataFrame]]:
    subsets: list[tuple[str, pd.DataFrame]] = []
    for order_val, g in df.groupby("order_sequence", sort=True):
        if len(g) < min_obs_per_subset:
            continue
        label = f"order_{int(order_val)}"
        subsets.append((label, g.copy()))
    return subsets


def _plot_subset_fits(
    subset_results: list[dict],
    out_plot: str,
    max_points_scatter: int = 4000,
):
    n = len(subset_results)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows), squeeze=False)
    rng = np.random.default_rng(42)

    for idx, rec in enumerate(subset_results):
        ax = axes[idx // ncols, idx % ncols]
        theta = rec["theta"]
        delta = rec["delta"]
        beta_mu = rec["beta_mu"]
        beta_sigma = rec["beta_sigma"]
        mu_degree = rec["mu_degree"]
        sigma_degree = rec["sigma_degree"]

        if len(theta) > max_points_scatter:
            take = rng.choice(len(theta), size=max_points_scatter, replace=False)
            xs = theta[take]
            ys = delta[take]
        else:
            xs = theta
            ys = delta

        ax.scatter(xs, ys, s=8, alpha=0.25, color="#4477AA")

        x_grid = np.linspace(theta.min(), theta.max(), 300)
        x_mu = _poly_design(x_grid, mu_degree)
        x_sig = _poly_design(x_grid, sigma_degree)
        mu = x_mu @ beta_mu
        sigma = np.exp(x_sig @ beta_sigma)
        sigma = np.maximum(sigma, 1e-4)

        ax.plot(x_grid, mu, color="#CC3311", linewidth=2.0, label="mu(theta)")
        ax.plot(x_grid, mu + 2.0 * sigma, color="#228833", linewidth=1.2, linestyle="--")
        ax.plot(x_grid, mu - 2.0 * sigma, color="#228833", linewidth=1.2, linestyle="--", label="mu ± 2sigma")

        ax.set_title(
            f"{rec['subset']} | n={rec['n_obs']} | nll={rec['nll']:.3f}"
        )
        ax.set_xlabel("theta (proficiency)")
        ax.set_ylabel("delta (difficulty)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="best")

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    plt.savefig(out_plot, dpi=150)
    plt.close(fig)


def _animate_subset_fits(
    subset_results: list[dict],
    out_gif: str,
    *,
    fps: int = 2,
    max_points_scatter: int = 5000,
):
    if not subset_results:
        return

    rng = np.random.default_rng(42)
    global_theta_min = min(float(np.min(rec["theta"])) for rec in subset_results)
    global_theta_max = max(float(np.max(rec["theta"])) for rec in subset_results)
    global_delta_min = min(float(np.min(rec["delta"])) for rec in subset_results)
    global_delta_max = max(float(np.max(rec["delta"])) for rec in subset_results)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    def draw_frame(i: int):
        ax.clear()
        rec = subset_results[i]
        theta = rec["theta"]
        delta = rec["delta"]
        beta_mu = rec["beta_mu"]
        beta_sigma = rec["beta_sigma"]
        mu_degree = rec["mu_degree"]
        sigma_degree = rec["sigma_degree"]

        if len(theta) > max_points_scatter:
            take = rng.choice(len(theta), size=max_points_scatter, replace=False)
            xs = theta[take]
            ys = delta[take]
        else:
            xs = theta
            ys = delta
        ax.scatter(xs, ys, s=10, alpha=0.25, color="#4477AA")

        x_grid = np.linspace(global_theta_min, global_theta_max, 350)
        mu = _poly_design(x_grid, mu_degree) @ beta_mu
        sigma = np.exp(_poly_design(x_grid, sigma_degree) @ beta_sigma)
        sigma = np.maximum(sigma, 1e-4)
        ax.plot(x_grid, mu, color="#CC3311", linewidth=2.0, label="mu(theta)")
        ax.plot(x_grid, mu + 2.0 * sigma, color="#228833", linewidth=1.2, linestyle="--", label="mu ± 2sigma")
        ax.plot(x_grid, mu - 2.0 * sigma, color="#228833", linewidth=1.2, linestyle="--")

        ax.set_xlim(global_theta_min, global_theta_max)
        ax.set_ylim(global_delta_min, global_delta_max)
        ax.set_xlabel("theta (proficiency)")
        ax.set_ylabel("delta (difficulty)")
        ax.set_title(
            f"Frame {i + 1}/{len(subset_results)} | {rec['subset']} | "
            f"n={rec['n_obs']} | nll={rec['nll']:.3f}"
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(subset_results), interval=1000 / max(fps, 1))
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    writer = animation.PillowWriter(fps=max(fps, 1))
    ani.save(out_gif, writer=writer)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="KTM probabilistic Gaussian regression: delta | theta ~ N(mu(theta), sigma(theta)^2)."
    )
    parser.add_argument("--data", type=str, default="pix_mapping/pix_processed.csv")
    parser.add_argument("--user-col", type=str, default="user_id")
    parser.add_argument("--item-col", type=str, default="challenge_id")
    parser.add_argument("--correct-col", type=str, default="outcome")
    parser.add_argument("--skill-col", type=str, default="skill_id")
    parser.add_argument("--answer-order-col", type=str, default="answer_number")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--sample-users", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=6)
    parser.add_argument(
        "--subset-mode",
        type=str,
        default="bin",
        choices=["bin", "order"],
        help="bin: qcut bins on order_sequence; order: one subset per exact order_sequence.",
    )
    parser.add_argument("--mu-degree", type=int, default=1)
    parser.add_argument("--sigma-degree", type=int, default=1)
    parser.add_argument("--min-obs-per-subset", type=int, default=200)
    parser.add_argument(
        "--out-csv",
        type=str,
        default="pix_mapping/ktm_gaussian_regression_subsets.csv",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="pix_mapping/ktm_gaussian_regression_subsets.png",
    )
    parser.add_argument(
        "--out-gif",
        type=str,
        default="pix_mapping/ktm_gaussian_regression_subsets.gif",
    )
    parser.add_argument("--make-animation", action="store_true")
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()
    if args.mu_degree < 1:
        raise ValueError("--mu-degree must be >= 1")
    if args.sigma_degree < 1:
        raise ValueError("--sigma-degree must be >= 1")

    usecols = [args.user_col, args.item_col, args.correct_col, args.answer_order_col]
    if args.skill_col:
        usecols.append(args.skill_col)
    df = pd.read_csv(args.data, usecols=usecols)
    df = df.dropna(subset=[args.user_col, args.item_col, args.correct_col]).copy()

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
    if args.subset_mode == "order":
        subsets = _build_subsets_by_order(
            df_proc,
            min_obs_per_subset=args.min_obs_per_subset,
        )
    else:
        subsets = _build_subsets_by_bins(df_proc, n_bins=args.n_bins)

    rows = []
    plot_payload = []
    for subset_name, g in subsets:
        theta = g["proficiency"].to_numpy(dtype=float)
        delta = g["difficulties"].to_numpy(dtype=float)
        n_obs = len(g)
        if n_obs < args.min_obs_per_subset:
            continue

        fit = _fit_gaussian_regression(
            theta=theta,
            delta=delta,
            mu_degree=args.mu_degree,
            sigma_degree=args.sigma_degree,
        )

        rec = {
            "subset": subset_name,
            "n_obs": n_obs,
            "order_min": int(g["order_sequence"].min()),
            "order_max": int(g["order_sequence"].max()),
            "mu_degree": args.mu_degree,
            "sigma_degree": args.sigma_degree,
            "nll": fit.nll,
            "rmse": fit.rmse,
            "mae": fit.mae,
            "success": fit.success,
            "message": fit.message,
        }
        for i, v in enumerate(fit.beta_mu):
            rec[f"beta_mu_{i}"] = float(v)
        for i, v in enumerate(fit.beta_sigma):
            rec[f"beta_sigma_{i}"] = float(v)
        rows.append(rec)
        plot_payload.append(
            {
                "subset": subset_name,
                "n_obs": n_obs,
                "nll": fit.nll,
                "theta": theta,
                "delta": delta,
                "beta_mu": fit.beta_mu,
                "beta_sigma": fit.beta_sigma,
                "mu_degree": args.mu_degree,
                "sigma_degree": args.sigma_degree,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("subset").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    if plot_payload:
        _plot_subset_fits(plot_payload, args.out_plot)
        if args.make_animation:
            _animate_subset_fits(
                plot_payload,
                args.out_gif,
                fps=args.fps,
            )

    print(
        f"rows_input={len(df)} rows_processed={len(df_proc)} "
        f"subsets_total={len(subsets)} subsets_fit={len(out_df)} mode={args.subset_mode}"
    )
    print(f"saved_csv={os.path.abspath(args.out_csv)}")
    print(f"saved_plot={os.path.abspath(args.out_plot)}")
    if args.make_animation:
        print(f"saved_gif={os.path.abspath(args.out_gif)}")


if __name__ == "__main__":
    main()
