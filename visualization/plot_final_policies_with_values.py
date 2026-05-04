import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def poly_eval(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    for d, c in enumerate(coeffs):
        y += c * (x ** d)
    return y


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def optimal_curve(theta_grid: np.ndarray, delta_min: float, delta_max: float, n_grid: int = 1200) -> np.ndarray:
    d_grid = np.linspace(delta_min, delta_max, n_grid)
    z = theta_grid[:, None] - d_grid[None, :]
    p = sigmoid(z)
    exp_reward = p * (d_grid[None, :] - delta_min)
    return d_grid[np.argmax(exp_reward, axis=1)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot behavior, IPS, SNIPS and optimal policies together with expected values annotations."
    )
    parser.add_argument(
        "--tuples-csv",
        type=str,
        default="pix_mapping/ktm_gaussian_propensity_order_u10k_mu1_sigma2.csv",
    )
    parser.add_argument(
        "--behavior-fit-csv",
        type=str,
        default="pix_mapping/ktm_gaussian_regression_order_u10k_mu1_sigma2.csv",
    )
    parser.add_argument(
        "--ips-summary-csv",
        type=str,
        default="pix_mapping/sequential_single_policy_mu1_sigma2_ips_ep5/sequential_round_summary.csv",
    )
    parser.add_argument(
        "--snips-summary-csv",
        type=str,
        default="pix_mapping/sequential_single_policy_mu1_sigma2_snips_ep5/sequential_round_summary.csv",
    )
    parser.add_argument(
        "--ips-coef-csv",
        type=str,
        default="pix_mapping/sequential_single_policy_mu1_sigma2_ips_ep5/sequential_policy_coefficients.csv",
    )
    parser.add_argument(
        "--snips-coef-csv",
        type=str,
        default="pix_mapping/sequential_single_policy_mu1_sigma2_snips_ep5/sequential_policy_coefficients.csv",
    )
    parser.add_argument("--order-col", type=str, default="order_sequence")
    parser.add_argument("--theta-col", type=str, default="proficiency")
    parser.add_argument("--delta-col", type=str, default="difficulties")
    parser.add_argument("--sigma-floor", type=float, default=1e-4)
    parser.add_argument("--band-scale", type=float, default=1.0)
    parser.add_argument("--max-scatter", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--y-min", type=float, default=-4.0)
    parser.add_argument("--y-max", type=float, default=4.0)
    parser.add_argument(
        "--out-plot",
        type=str,
        default="pix_mapping/final_policies_behavior_ips_snips_optimal_with_values.png",
    )
    args = parser.parse_args()

    tup = pd.read_csv(args.tuples_csv, usecols=[args.theta_col, args.delta_col]).dropna()
    theta_min = float(tup[args.theta_col].min())
    theta_max = float(tup[args.theta_col].max())
    delta_min = float(tup[args.delta_col].min())
    delta_max = float(tup[args.delta_col].max())
    theta_grid = np.linspace(theta_min, theta_max, 500)

    ips_sum = pd.read_csv(args.ips_summary_csv).sort_values("round").iloc[-1]
    snips_sum = pd.read_csv(args.snips_summary_csv).sort_values("round").iloc[-1]
    final_order = int(ips_sum["order_sequence"])

    fit = pd.read_csv(args.behavior_fit_csv)
    if "order_sequence" in fit.columns:
        b_row = fit[fit["order_sequence"].astype(int) == final_order].iloc[0]
    else:
        b_row = fit[
            (fit["order_min"].astype(int) == final_order)
            & (fit["order_max"].astype(int) == final_order)
        ].iloc[0]
    b_mu_c = np.array([float(b_row.get("beta_mu_0", 0.0)), float(b_row.get("beta_mu_1", 0.0))], dtype=float)
    b_sig_c = np.array(
        [
            float(b_row.get("beta_sigma_0", 0.0)),
            float(b_row.get("beta_sigma_1", 0.0)),
            float(b_row.get("beta_sigma_2", 0.0)),
        ],
        dtype=float,
    )

    ips_coef = pd.read_csv(args.ips_coef_csv).sort_values("round").iloc[-1]
    snips_coef = pd.read_csv(args.snips_coef_csv).sort_values("round").iloc[-1]
    ips_mu_c = np.array([float(ips_coef["beta_mu_0"]), float(ips_coef["beta_mu_1"])], dtype=float)
    ips_sig_c = np.array(
        [float(ips_coef["beta_sigma_0"]), float(ips_coef["beta_sigma_1"]), float(ips_coef["beta_sigma_2"])],
        dtype=float,
    )
    snips_mu_c = np.array([float(snips_coef["beta_mu_0"]), float(snips_coef["beta_mu_1"])], dtype=float)
    snips_sig_c = np.array(
        [float(snips_coef["beta_sigma_0"]), float(snips_coef["beta_sigma_1"]), float(snips_coef["beta_sigma_2"])],
        dtype=float,
    )

    b_mu = poly_eval(theta_grid, b_mu_c)
    b_sig = np.maximum(np.exp(poly_eval(theta_grid, b_sig_c)), args.sigma_floor)
    ips_mu = poly_eval(theta_grid, ips_mu_c)
    ips_sig = np.maximum(np.exp(poly_eval(theta_grid, ips_sig_c)), args.sigma_floor)
    snips_mu = poly_eval(theta_grid, snips_mu_c)
    snips_sig = np.maximum(np.exp(poly_eval(theta_grid, snips_sig_c)), args.sigma_floor)
    opt = optimal_curve(theta_grid, delta_min=delta_min, delta_max=delta_max)

    rng = np.random.default_rng(args.seed)
    if len(tup) > args.max_scatter:
        idx = rng.choice(len(tup), size=args.max_scatter, replace=False)
        xs = tup.iloc[idx][args.theta_col].to_numpy(dtype=float)
        ys = tup.iloc[idx][args.delta_col].to_numpy(dtype=float)
    else:
        xs = tup[args.theta_col].to_numpy(dtype=float)
        ys = tup[args.delta_col].to_numpy(dtype=float)

    os.makedirs(os.path.dirname(args.out_plot), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.scatter(xs, ys, s=8, alpha=0.08, color="#999999")

    ax.plot(theta_grid, b_mu, color="#ff7f0e", lw=2.0, label="Behavior")
    ax.fill_between(theta_grid, b_mu - args.band_scale * b_sig, b_mu + args.band_scale * b_sig, color="#ff7f0e", alpha=0.14)

    ax.plot(theta_grid, ips_mu, color="#1f77b4", lw=2.2, label="Learned IPS")
    ax.fill_between(theta_grid, ips_mu - args.band_scale * ips_sig, ips_mu + args.band_scale * ips_sig, color="#1f77b4", alpha=0.14)

    ax.plot(theta_grid, snips_mu, color="#d62728", lw=2.2, label="Learned SNIPS")
    ax.fill_between(theta_grid, snips_mu - args.band_scale * snips_sig, snips_mu + args.band_scale * snips_sig, color="#d62728", alpha=0.14)

    ax.plot(theta_grid, opt, color="#2ca02c", lw=2.2, linestyle="--", label="Optimal (IRT)")

    x_anno = theta_grid[-1]
    ax.text(
        x_anno,
        b_mu[-1],
        f" behavior V={float(ips_sum['behavior_irt_value']):.3f}",
        color="#ff7f0e",
        fontsize=9,
        va="center",
        ha="left",
    )
    ax.text(
        x_anno,
        ips_mu[-1],
        f" IPS V={float(ips_sum['learned_irt_value']):.3f}",
        color="#1f77b4",
        fontsize=9,
        va="center",
        ha="left",
    )
    ax.text(
        x_anno,
        snips_mu[-1],
        f" SNIPS V={float(snips_sum['learned_irt_value']):.3f}",
        color="#d62728",
        fontsize=9,
        va="center",
        ha="left",
    )
    ax.text(
        x_anno,
        opt[-1],
        f" optimal V={float(ips_sum['optimal_irt_value']):.3f}",
        color="#2ca02c",
        fontsize=9,
        va="center",
        ha="left",
    )

    ax.set_xlabel("theta")
    ax.set_ylabel("delta")
    ax.set_ylim(args.y_min, args.y_max)
    ax.set_title("Behavior vs Learned Policies (IPS/SNIPS) vs Optimal")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=180)
    plt.close(fig)

    print(f"saved_plot={os.path.abspath(args.out_plot)}")


if __name__ == "__main__":
    main()
