import argparse
import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ktm import processing_data


def _poly_design(x: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return np.column_stack(cols)


def _build_subsets_by_order(df: pd.DataFrame, min_obs_per_subset: int) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for order_val, g in df.groupby("order_sequence", sort=True):
        if len(g) < min_obs_per_subset:
            continue
        out.append((f"order_{int(order_val)}", g.copy()))
    return out


def _component_sigma(
    x_sigma: np.ndarray,
    gamma: np.ndarray,
    sigma_floor: float,
) -> np.ndarray:
    s = np.exp(x_sigma @ gamma)
    return np.maximum(s, sigma_floor)


def _extract_params(
    row: pd.Series,
    k: int,
    mu_degree: int,
    sigma_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_mu = mu_degree + 1
    p_sig = sigma_degree + 1
    pi = np.array([float(row[f"pi_{j}"]) for j in range(k)], dtype=float)
    beta_mu = np.zeros((k, p_mu), dtype=float)
    gamma_sigma = np.zeros((k, p_sig), dtype=float)
    for j in range(k):
        for m in range(p_mu):
            beta_mu[j, m] = float(row[f"beta_mu_{j}_{m}"])
        for m in range(p_sig):
            gamma_sigma[j, m] = float(row[f"gamma_sigma_{j}_{m}"])
    pi = np.maximum(pi, 1e-12)
    pi = pi / pi.sum()
    return pi, beta_mu, gamma_sigma


def main():
    parser = argparse.ArgumentParser(description="Animate KTM-GMM fit for a selected K.")
    parser.add_argument("--data", type=str, default="pix_mapping/pix_processed.csv")
    parser.add_argument("--results-csv", type=str, required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--mu-degree", type=int, default=1)
    parser.add_argument("--sigma-degree", type=int, default=1)
    parser.add_argument("--sigma-floor", type=float, default=0.1)
    parser.add_argument("--sample-users", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-obs-per-subset", type=int, default=500)
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument(
        "--out-gif",
        type=str,
        default="pix_mapping/ktm_gmm_k2_animation.gif",
    )
    args = parser.parse_args()

    # Load and preprocess data exactly like the GMM training script.
    usecols = ["user_id", "challenge_id", "outcome", "answer_number", "skill_id"]
    df = pd.read_csv(args.data, usecols=usecols).dropna(
        subset=["user_id", "challenge_id", "outcome"]
    )
    if args.sample_users > 0 and args.sample_users < df["user_id"].nunique():
        rng = np.random.default_rng(args.seed)
        users = df["user_id"].dropna().unique()
        keep = set(rng.choice(users, size=args.sample_users, replace=False).tolist())
        df = df[df["user_id"].isin(keep)].copy()

    df = df.rename(
        columns={
            "user_id": "user",
            "challenge_id": "item",
            "outcome": "correct",
            "answer_number": "answer_number",
            "skill_id": "skill",
        }
    )
    df_proc = processing_data(
        df,
        user_col="user",
        item_col="item",
        correct_col="correct",
        skill_col="skill",
        order_cols=["answer_number"],
        reduce=False,
        rank_start=0,
        fit_intercept=False,
        center_latents=True,
        target_std=1.0,
        clip_quantiles=(0.01, 0.99),
    )
    subsets = _build_subsets_by_order(df_proc, min_obs_per_subset=args.min_obs_per_subset)

    fit_df = pd.read_csv(args.results_csv)
    fit_df = fit_df[fit_df["n_components"] == args.k].copy()
    fit_df = fit_df.set_index("subset")

    frames = []
    for subset_name, g in subsets:
        if subset_name not in fit_df.index:
            continue
        row = fit_df.loc[subset_name]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        theta = g["proficiency"].to_numpy(dtype=float)
        delta = g["difficulties"].to_numpy(dtype=float)
        pi, beta_mu, gamma_sigma = _extract_params(
            row,
            k=args.k,
            mu_degree=args.mu_degree,
            sigma_degree=args.sigma_degree,
        )
        frames.append(
            {
                "subset": subset_name,
                "theta": theta,
                "delta": delta,
                "pi": pi,
                "beta_mu": beta_mu,
                "gamma_sigma": gamma_sigma,
                "n_obs": len(g),
                "val_nll": float(row["val_nll"]) if "val_nll" in row else float(row["nll"]),
            }
        )

    if not frames:
        raise ValueError("No frames were built. Check --results-csv, --k, and subset filtering.")

    global_theta_min = min(float(np.min(fr["theta"])) for fr in frames)
    global_theta_max = max(float(np.max(fr["theta"])) for fr in frames)
    global_delta_min = min(float(np.min(fr["delta"])) for fr in frames)
    global_delta_max = max(float(np.max(fr["delta"])) for fr in frames)
    rng = np.random.default_rng(args.seed)

    fig, ax = plt.subplots(figsize=(8.8, 6.4))

    def draw(i: int):
        ax.clear()
        fr = frames[i]
        theta = fr["theta"]
        delta = fr["delta"]

        if len(theta) > args.max_points:
            idx = rng.choice(len(theta), size=args.max_points, replace=False)
            xs = theta[idx]
            ys = delta[idx]
        else:
            xs = theta
            ys = delta
        ax.scatter(xs, ys, s=10, alpha=0.22, color="#4477AA")

        x_grid = np.linspace(global_theta_min, global_theta_max, 360)
        x_mu = _poly_design(x_grid, args.mu_degree)
        x_sigma = _poly_design(x_grid, args.sigma_degree)
        comp_colors = plt.cm.tab10(np.linspace(0, 1, max(args.k, 3)))
        for j in range(args.k):
            mu_j = x_mu @ fr["beta_mu"][j]
            sigma_j = _component_sigma(x_sigma, fr["gamma_sigma"][j], sigma_floor=args.sigma_floor)
            c = comp_colors[j]
            ax.plot(
                x_grid,
                mu_j,
                color=c,
                linewidth=2.0,
                label=f"Gaussian {j+1} mean (pi={fr['pi'][j]:.2f})",
            )
            ax.fill_between(
                x_grid,
                mu_j - 2.0 * sigma_j,
                mu_j + 2.0 * sigma_j,
                color=c,
                alpha=0.15,
            )

        ax.set_xlim(global_theta_min, global_theta_max)
        ax.set_ylim(global_delta_min, global_delta_max)
        ax.set_xlabel("theta (proficiency)")
        ax.set_ylabel("delta (difficulty)")
        ax.set_title(
            f"K={args.k} | Frame {i+1}/{len(frames)} | {fr['subset']} "
            f"| n={fr['n_obs']} | val_nll={fr['val_nll']:.3f}"
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / max(args.fps, 1))
    os.makedirs(os.path.dirname(args.out_gif), exist_ok=True)
    ani.save(args.out_gif, writer=animation.PillowWriter(fps=max(args.fps, 1)))
    plt.close(fig)

    print(f"frames={len(frames)}")
    print(f"saved_gif={os.path.abspath(args.out_gif)}")


if __name__ == "__main__":
    main()
