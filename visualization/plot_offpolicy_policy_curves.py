import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _poly_eval(theta: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    y = np.zeros_like(theta, dtype=float)
    for d, c in enumerate(coeffs):
        y += c * (theta ** d)
    return y


def _extract_degree_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    cols = sorted(cols, key=lambda x: int(x.split("_")[-1]))
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot learned off-policy Gaussian curves in (theta, delta) space."
    )
    parser.add_argument(
        "--policy-csv",
        type=str,
        default="pix_mapping/offpolicy_ips_policy_mu2_sigma2.csv",
    )
    parser.add_argument(
        "--tuples-csv",
        type=str,
        default="pix_mapping/ktm_offpolicy_tuples_notebook.csv",
    )
    parser.add_argument("--order-col", type=str, default="order_model")
    parser.add_argument("--theta-col", type=str, default="proficiency")
    parser.add_argument("--delta-col", type=str, default="difficulties")
    parser.add_argument("--sigma-floor", type=float, default=1e-4)
    parser.add_argument("--max-scatter", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-orders", type=int, default=6)
    parser.add_argument(
        "--out-global-plot",
        type=str,
        default="pix_mapping/offpolicy_policy_curve_global.png",
    )
    parser.add_argument(
        "--out-orders-plot",
        type=str,
        default="pix_mapping/offpolicy_policy_curve_top_orders.png",
    )
    args = parser.parse_args()

    pol = pd.read_csv(args.policy_csv)
    tup = pd.read_csv(
        args.tuples_csv,
        usecols=[args.theta_col, args.delta_col, args.order_col],
    ).dropna()

    mu_cols = _extract_degree_columns(pol, "beta_mu_")
    sig_cols = _extract_degree_columns(pol, "beta_sigma_")
    if not mu_cols or not sig_cols:
        raise ValueError("Policy CSV missing beta_mu_* or beta_sigma_* columns.")

    freq = tup[args.order_col].astype(int).value_counts().sort_index()
    pol = pol.copy()
    pol["order_value"] = pol["order_value"].astype(int)
    pol["weight"] = pol["order_value"].map(freq).fillna(0.0)
    if pol["weight"].sum() <= 0:
        raise ValueError("No overlapping order values between policy and tuples data.")
    pol["weight"] = pol["weight"] / pol["weight"].sum()

    theta_min = float(tup[args.theta_col].min())
    theta_max = float(tup[args.theta_col].max())
    theta_grid = np.linspace(theta_min, theta_max, 400)

    mu_stack = []
    sig_stack = []
    w = pol["weight"].to_numpy(dtype=float)
    for _, r in pol.iterrows():
        bmu = r[mu_cols].to_numpy(dtype=float)
        bsig = r[sig_cols].to_numpy(dtype=float)
        mu = _poly_eval(theta_grid, bmu)
        sig = np.exp(_poly_eval(theta_grid, bsig))
        sig = np.maximum(sig, args.sigma_floor)
        mu_stack.append(mu)
        sig_stack.append(sig)
    mu_stack = np.vstack(mu_stack)  # [n_orders, n_grid]
    sig_stack = np.vstack(sig_stack)

    mu_global = (w[:, None] * mu_stack).sum(axis=0)
    second = (w[:, None] * (sig_stack ** 2 + mu_stack ** 2)).sum(axis=0)
    var_global = np.maximum(second - mu_global ** 2, 1e-12)
    std_global = np.sqrt(var_global)

    rng = np.random.default_rng(args.seed)
    if len(tup) > args.max_scatter:
        idx = rng.choice(len(tup), size=args.max_scatter, replace=False)
        xs = tup.iloc[idx][args.theta_col].to_numpy(dtype=float)
        ys = tup.iloc[idx][args.delta_col].to_numpy(dtype=float)
    else:
        xs = tup[args.theta_col].to_numpy(dtype=float)
        ys = tup[args.delta_col].to_numpy(dtype=float)

    os.makedirs(os.path.dirname(args.out_global_plot), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.scatter(xs, ys, s=8, alpha=0.12, color="#7f7f7f", label="logged tuples")
    ax.plot(theta_grid, mu_global, color="#1f77b4", lw=2.4, label="policy mean")
    ax.fill_between(
        theta_grid,
        mu_global - 2.0 * std_global,
        mu_global + 2.0 * std_global,
        color="#1f77b4",
        alpha=0.18,
        label="policy mean ± 2 std",
    )
    ax.set_xlabel("theta")
    ax.set_ylabel("delta")
    ax.set_title("Learned Policy Curve (order-marginal)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_global_plot, dpi=170)
    plt.close(fig)

    top = freq.sort_values(ascending=False).head(max(args.top_orders, 1)).index.tolist()
    sub = pol[pol["order_value"].isin(top)].copy()
    sub = sub.sort_values("order_value")
    cmap = plt.cm.get_cmap("tab10", len(sub))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(xs, ys, s=8, alpha=0.08, color="#888888")
    for i, (_, r) in enumerate(sub.iterrows()):
        bmu = r[mu_cols].to_numpy(dtype=float)
        bsig = r[sig_cols].to_numpy(dtype=float)
        mu = _poly_eval(theta_grid, bmu)
        sig = np.exp(_poly_eval(theta_grid, bsig))
        sig = np.maximum(sig, args.sigma_floor)
        c = cmap(i)
        order_v = int(r["order_value"])
        ax.plot(theta_grid, mu, lw=2.0, color=c, label=f"order {order_v} mean")
        ax.fill_between(theta_grid, mu - 2 * sig, mu + 2 * sig, color=c, alpha=0.12)

    ax.set_xlabel("theta")
    ax.set_ylabel("delta")
    ax.set_title(f"Learned Policy Curves by Order (top {len(sub)})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.out_orders_plot, dpi=170)
    plt.close(fig)

    print(f"saved_global={os.path.abspath(args.out_global_plot)}")
    print(f"saved_orders={os.path.abspath(args.out_orders_plot)}")


if __name__ == "__main__":
    main()
