import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _poly_eval(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    for d, c in enumerate(coeffs):
        y += c * (x ** d)
    return y


def _extract_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def _global_curve(
    policy_df: pd.DataFrame,
    theta_grid: np.ndarray,
    order_weights: pd.Series,
    sigma_floor: float,
    band_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    mu_cols = _extract_cols(policy_df, "beta_mu_")
    sig_cols = _extract_cols(policy_df, "beta_sigma_")
    if not mu_cols or not sig_cols:
        raise ValueError("Policy file missing beta_mu_* or beta_sigma_* columns.")

    work = policy_df.copy()
    work["order_value"] = work["order_value"].astype(int)
    work["weight"] = work["order_value"].map(order_weights).fillna(0.0)
    if work["weight"].sum() <= 0:
        raise ValueError("No overlapping orders between policy and tuples data.")
    work["weight"] = work["weight"] / work["weight"].sum()

    mu_stack = []
    sig_stack = []
    for _, r in work.iterrows():
        bmu = r[mu_cols].to_numpy(dtype=float)
        bsig = r[sig_cols].to_numpy(dtype=float)
        mu = _poly_eval(theta_grid, bmu)
        sig = np.exp(_poly_eval(theta_grid, bsig))
        sig = np.maximum(sig, sigma_floor)
        mu_stack.append(mu)
        sig_stack.append(sig)
    mu_stack = np.vstack(mu_stack)
    sig_stack = np.vstack(sig_stack)
    w = work["weight"].to_numpy(dtype=float)[:, None]

    mu_global = np.sum(w * mu_stack, axis=0)
    if band_type == "learned_sigma":
        # Band reflects only learned noise scale sigma(theta), not between-order spread.
        sigma_global = np.sum(w * sig_stack, axis=0)
        sigma_global = np.maximum(sigma_global, sigma_floor)
        return mu_global, sigma_global
    if band_type == "total_variability":
        second = np.sum(w * (sig_stack ** 2 + mu_stack ** 2), axis=0)
        var_global = np.maximum(second - mu_global ** 2, 1e-12)
        std_global = np.sqrt(var_global)
        return mu_global, std_global
    raise ValueError(f"Unknown band_type: {band_type}")


def _optimal_curve_irt(
    theta_grid: np.ndarray,
    *,
    delta_min: float,
    delta_max: float,
    n_delta_grid: int = 1200,
) -> np.ndarray:
    # 1PL expected reward under action delta:
    # E[r | theta, delta] = sigmoid(theta - delta) * (delta - delta_min)
    d_grid = np.linspace(delta_min, delta_max, n_delta_grid)
    z = theta_grid[:, None] - d_grid[None, :]
    p = 1.0 / (1.0 + np.exp(-z))
    exp_reward = p * (d_grid[None, :] - delta_min)
    best_idx = np.argmax(exp_reward, axis=1)
    return d_grid[best_idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot learned policies and IRT-optimal policy on same (theta, delta) chart."
    )
    parser.add_argument("--ips-policy-csv", type=str, required=True)
    parser.add_argument("--snips-policy-csv", type=str, required=True)
    parser.add_argument("--cips-policy-csv", type=str, default="")
    parser.add_argument(
        "--tuples-csv",
        type=str,
        default="pix_mapping/ktm_offpolicy_tuples_notebook.csv",
    )
    parser.add_argument("--theta-col", type=str, default="proficiency")
    parser.add_argument("--delta-col", type=str, default="difficulties")
    parser.add_argument("--order-col", type=str, default="order_model")
    parser.add_argument("--sigma-floor", type=float, default=1e-4)
    parser.add_argument(
        "--band-type",
        type=str,
        default="learned_sigma",
        choices=["learned_sigma", "total_variability"],
    )
    parser.add_argument("--band-scale", type=float, default=1.0)
    parser.add_argument("--max-scatter", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument(
        "--out-plot",
        type=str,
        default="pix_mapping/policy_ips_snips_optimal_comparison.png",
    )
    args = parser.parse_args()

    tup = pd.read_csv(
        args.tuples_csv,
        usecols=[args.theta_col, args.delta_col, args.order_col],
    ).dropna()
    ips_df = pd.read_csv(args.ips_policy_csv)
    snips_df = pd.read_csv(args.snips_policy_csv)
    cips_df = pd.read_csv(args.cips_policy_csv) if args.cips_policy_csv else None

    theta_min = float(tup[args.theta_col].min())
    theta_max = float(tup[args.theta_col].max())
    delta_min = float(tup[args.delta_col].min())
    delta_max = float(tup[args.delta_col].max())
    theta_grid = np.linspace(theta_min, theta_max, 500)

    order_weights = tup[args.order_col].astype(int).value_counts().sort_index()

    ips_mu, ips_std = _global_curve(
        ips_df,
        theta_grid=theta_grid,
        order_weights=order_weights,
        sigma_floor=args.sigma_floor,
        band_type=args.band_type,
    )
    snips_mu, snips_std = _global_curve(
        snips_df,
        theta_grid=theta_grid,
        order_weights=order_weights,
        sigma_floor=args.sigma_floor,
        band_type=args.band_type,
    )
    if cips_df is not None:
        cips_mu, cips_std = _global_curve(
            cips_df,
            theta_grid=theta_grid,
            order_weights=order_weights,
            sigma_floor=args.sigma_floor,
            band_type=args.band_type,
        )
    opt_delta = _optimal_curve_irt(
        theta_grid,
        delta_min=delta_min,
        delta_max=delta_max,
    )

    rng = np.random.default_rng(args.seed)
    if len(tup) > args.max_scatter:
        idx = rng.choice(len(tup), size=args.max_scatter, replace=False)
        xs = tup.iloc[idx][args.theta_col].to_numpy(dtype=float)
        ys = tup.iloc[idx][args.delta_col].to_numpy(dtype=float)
    else:
        xs = tup[args.theta_col].to_numpy(dtype=float)
        ys = tup[args.delta_col].to_numpy(dtype=float)

    os.makedirs(os.path.dirname(args.out_plot), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.scatter(xs, ys, s=8, alpha=0.08, color="#888888", label="logged tuples")

    ax.plot(theta_grid, ips_mu, color="#1f77b4", lw=2.3, label="IPS policy mean")
    ax.fill_between(
        theta_grid,
        ips_mu - args.band_scale * ips_std,
        ips_mu + args.band_scale * ips_std,
        color="#1f77b4",
        alpha=0.15,
        label=f"IPS ± {args.band_scale:g} band",
    )

    ax.plot(theta_grid, snips_mu, color="#d62728", lw=2.3, label="SNIPS policy mean")
    ax.fill_between(
        theta_grid,
        snips_mu - args.band_scale * snips_std,
        snips_mu + args.band_scale * snips_std,
        color="#d62728",
        alpha=0.15,
        label=f"SNIPS ± {args.band_scale:g} band",
    )

    if cips_df is not None:
        ax.plot(theta_grid, cips_mu, color="#9467bd", lw=2.3, label="CIPS policy mean")
        ax.fill_between(
            theta_grid,
            cips_mu - args.band_scale * cips_std,
            cips_mu + args.band_scale * cips_std,
            color="#9467bd",
            alpha=0.15,
            label=f"CIPS ± {args.band_scale:g} band",
        )

    ax.plot(
        theta_grid,
        opt_delta,
        color="#2ca02c",
        lw=2.4,
        linestyle="--",
        label="IRT-optimal policy (argmax expected reward)",
    )

    ax.set_xlabel("theta")
    ax.set_ylabel("delta")
    ax.set_title("Policy Comparison in (theta, delta)")
    if args.y_min is not None and args.y_max is not None:
        ax.set_ylim(float(args.y_min), float(args.y_max))
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=180)
    plt.close(fig)

    print(f"saved_plot={os.path.abspath(args.out_plot)}")


if __name__ == "__main__":
    main()
