import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.offpolicy_gaussian_policy import GlobalGaussianPolicy


def poly_eval(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    for d, c in enumerate(coeffs):
        y += c * (x ** d)
    return y


def optimal_curve_irt(theta_grid: np.ndarray, delta_min: float, delta_max: float) -> np.ndarray:
    d_grid = np.linspace(delta_min, delta_max, 1200)
    z = theta_grid[:, None] - d_grid[None, :]
    p = 1.0 / (1.0 + np.exp(-z))
    exp_reward = p * (d_grid[None, :] - delta_min)
    best_idx = np.argmax(exp_reward, axis=1)
    return d_grid[best_idx]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def policy_value_irt_mc(
    theta: np.ndarray,
    mu_coeffs: np.ndarray,
    sigma_coeffs: np.ndarray,
    *,
    delta_min: float,
    sigma_floor: float,
    n_mc: int = 64,
    seed: int = 42,
) -> float:
    if len(theta) == 0:
        return float("nan")
    mu = poly_eval(theta, mu_coeffs)
    sigma = np.maximum(np.exp(poly_eval(theta, sigma_coeffs)), sigma_floor)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(size=n_mc)[:, None]  # [n_mc, 1]
    delta = mu[None, :] + sigma[None, :] * eps
    prob = _sigmoid(theta[None, :] - delta)
    rew = prob * (delta - delta_min)
    return float(np.mean(rew))


def optimal_value_irt(
    theta: np.ndarray,
    *,
    delta_min: float,
    delta_max: float,
    n_delta_grid: int = 1200,
) -> float:
    if len(theta) == 0:
        return float("nan")
    d_grid = np.linspace(delta_min, delta_max, n_delta_grid)
    z = theta[:, None] - d_grid[None, :]
    p = _sigmoid(z)
    exp_reward = p * (d_grid[None, :] - delta_min)
    best = np.max(exp_reward, axis=1)
    return float(np.mean(best))


def _log_gaussian_pdf(delta: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (delta - mu) / sigma
    return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * (z ** 2)


def build_behavior_lookup(
    fit_df: pd.DataFrame,
    *,
    mu_degree: int,
    sigma_degree: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for _, row in fit_df.iterrows():
        order_val = int(row["order_sequence"])
        b_mu = np.array([float(row.get(f"beta_mu_{d}", 0.0)) for d in range(mu_degree + 1)], dtype=np.float64)
        b_sig = np.array([float(row.get(f"beta_sigma_{d}", 0.0)) for d in range(sigma_degree + 1)], dtype=np.float64)
        lookup[order_val] = (b_mu, b_sig)
    return lookup


def compute_log_mixture_propensity(
    theta: np.ndarray,
    delta: np.ndarray,
    *,
    component_orders: np.ndarray,
    component_weights: np.ndarray,
    behavior_lookup: dict[int, tuple[np.ndarray, np.ndarray]],
    sigma_floor: float,
) -> np.ndarray:
    if len(theta) == 0:
        return np.empty((0,), dtype=np.float32)

    log_mix = np.full(theta.shape[0], -np.inf, dtype=np.float64)
    for order_val, rho in zip(component_orders, component_weights):
        if rho <= 0:
            continue
        order_int = int(order_val)
        if order_int not in behavior_lookup:
            raise ValueError(f"Order {order_int} not found in behavior-fit lookup.")
        b_mu, b_sig = behavior_lookup[order_int]
        mu = poly_eval(theta, b_mu)
        sigma = np.maximum(np.exp(poly_eval(theta, b_sig)), sigma_floor)
        log_p = _log_gaussian_pdf(delta=delta, mu=mu, sigma=sigma)
        log_mix = np.logaddexp(log_mix, np.log(float(rho)) + log_p)
    return log_mix.astype(np.float32)


def compute_metrics(
    policy: GlobalGaussianPolicy,
    theta: torch.Tensor,
    delta: torch.Tensor,
    reward: torch.Tensor,
    log_prop_b: torch.Tensor,
    *,
    max_weight: float,
    clip_c: float,
    device: torch.device,
) -> dict[str, float]:
    with torch.no_grad():
        th = theta.to(device)
        de = delta.to(device)
        rw = reward.to(device)
        lb = log_prop_b.to(device)
        log_p_new = policy.log_prob(delta=de, theta=th)
        log_ratio = torch.clamp(log_p_new - lb, min=-20.0, max=float(np.log(max_weight)))
        w = torch.exp(log_ratio)
        w_clip = torch.clamp(w, max=float(clip_c))

        ips = torch.mean(w * rw)
        snips = torch.sum(w * rw) / torch.sum(w).clamp_min(1e-12)
        ess = (torch.sum(w) ** 2) / torch.sum(w ** 2).clamp_min(1e-12)

        cips = torch.mean(w_clip * rw)
        csnips = torch.sum(w_clip * rw) / torch.sum(w_clip).clamp_min(1e-12)
        ess_clip = (torch.sum(w_clip) ** 2) / torch.sum(w_clip ** 2).clamp_min(1e-12)

    return {
        "ips": float(ips.item()),
        "snips": float(snips.item()),
        "ess": float(ess.item()),
        "cips": float(cips.item()),
        "csnips": float(csnips.item()),
        "ess_clip": float(ess_clip.item()),
    }


def train_one_round(
    policy: GlobalGaussianPolicy,
    theta: torch.Tensor,
    delta: torch.Tensor,
    reward: torch.Tensor,
    log_prop_b: torch.Tensor,
    *,
    objective: str,
    epochs: int,
    batch_size: int,
    lr: float,
    l2_coef: float,
    max_weight: float,
    clip_c: float,
    device: torch.device,
    num_workers: int,
    show_tqdm: bool,
) -> None:
    dataset = TensorDataset(theta, delta, reward, log_prop_b)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    policy.train()
    for _ in range(epochs):
        iterator = loader if not show_tqdm else tqdm(loader, leave=False)
        for b_theta, b_delta, b_reward, b_logb in iterator:
            b_theta = b_theta.to(device)
            b_delta = b_delta.to(device)
            b_reward = b_reward.to(device)
            b_logb = b_logb.to(device)

            opt.zero_grad(set_to_none=True)
            log_p_new = policy.log_prob(delta=b_delta, theta=b_theta)
            log_ratio = torch.clamp(log_p_new - b_logb, min=-20.0, max=float(np.log(max_weight)))
            w = torch.exp(log_ratio)
            w_clip = torch.clamp(w, max=float(clip_c))

            ips_obj = torch.mean(w * b_reward)
            snips_obj = torch.sum(w * b_reward) / torch.sum(w).clamp_min(1e-12)
            cips_obj = torch.mean(w_clip * b_reward)
            csnips_obj = torch.sum(w_clip * b_reward) / torch.sum(w_clip).clamp_min(1e-12)

            if objective == "ips":
                obj = ips_obj
            elif objective == "snips":
                obj = snips_obj
            elif objective == "cips":
                obj = cips_obj
            else:
                obj = csnips_obj

            reg = l2_coef * (torch.mean(policy.beta_mu ** 2) + torch.mean(policy.beta_sigma ** 2))
            loss = -obj + reg
            loss.backward()
            opt.step()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential training of one global policy with order-sequence batches."
    )
    parser.add_argument("--tuples-csv", type=str, default="pix_mapping/ktm_gaussian_propensity_order_u10k_mu1_sigma2.csv")
    parser.add_argument("--behavior-fit-csv", type=str, default="pix_mapping/ktm_gaussian_regression_order_u10k_mu1_sigma2.csv")
    parser.add_argument("--order-col", type=str, default="order_sequence")
    parser.add_argument("--theta-col", type=str, default="proficiency")
    parser.add_argument("--delta-col", type=str, default="difficulties")
    parser.add_argument("--reward-col", type=str, default="reward")
    parser.add_argument("--log-prop-col", type=str, default="log_propensity")
    parser.add_argument("--mu-degree", type=int, default=1)
    parser.add_argument("--sigma-degree", type=int, default=2)
    parser.add_argument("--sigma-floor", type=float, default=1e-4)
    parser.add_argument("--objective", type=str, default="snips", choices=["ips", "snips", "cips", "csnips"])
    parser.add_argument(
        "--denominator",
        type=str,
        default="logged",
        choices=["logged", "mixture"],
        help="logged: row-wise logged propensity; mixture: behavior-mixture propensity over the training set.",
    )
    parser.add_argument(
        "--train-scope",
        type=str,
        default="cumulative",
        choices=["cumulative", "current"],
        help="cumulative: train each round on orders 1..t; current: train each round only on current order batch.",
    )
    parser.add_argument("--clip-c", type=float, default=10.0)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--epochs-per-round", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--l2-coef", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scatter", type=int, default=3000)
    parser.add_argument("--band-scale", type=float, default=1.0)
    parser.add_argument("--y-min", type=float, default=-4.0)
    parser.add_argument("--y-max", type=float, default=4.0)
    parser.add_argument("--show-tqdm", action="store_true")
    parser.add_argument("--irt-mc-samples", type=int, default=64)
    parser.add_argument("--make-animation", action="store_true")
    parser.add_argument("--animation-fps", type=int, default=2)
    parser.add_argument("--animation-name", type=str, default="sequential_policy_animation.gif")
    parser.add_argument("--out-dir", type=str, default="pix_mapping/sequential_single_policy_mu1_sigma2")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fit = pd.read_csv(args.behavior_fit_csv)
    if "order_sequence" not in fit.columns:
        if "order_min" in fit.columns and "order_max" in fit.columns:
            fit = fit[fit["order_min"] == fit["order_max"]].copy()
            fit["order_sequence"] = fit["order_min"].astype(int)
        else:
            raise ValueError("Behavior fit CSV must contain order_sequence or order_min/order_max.")
    fit["order_sequence"] = fit["order_sequence"].astype(int)
    fit = fit.sort_values("order_sequence").reset_index(drop=True)
    behavior_lookup = build_behavior_lookup(
        fit,
        mu_degree=args.mu_degree,
        sigma_degree=args.sigma_degree,
    )

    usecols = [args.order_col, args.theta_col, args.delta_col, args.reward_col, args.log_prop_col]
    data = pd.read_csv(args.tuples_csv, usecols=usecols).dropna().copy()
    data[args.order_col] = data[args.order_col].astype(int)
    keep_orders = set(fit["order_sequence"].tolist())
    data = data[data[args.order_col].isin(keep_orders)].copy()

    order_list = sorted(set(data[args.order_col].tolist()) & set(fit["order_sequence"].tolist()))
    if not order_list:
        raise ValueError("No overlapping orders between tuples and behavior fit.")

    theta_min = float(data[args.theta_col].min())
    theta_max = float(data[args.theta_col].max())
    theta_grid = np.linspace(theta_min, theta_max, 500)
    delta_min = float(data[args.delta_col].min())
    delta_max = float(data[args.delta_col].max())
    opt_curve = optimal_curve_irt(theta_grid, delta_min=delta_min, delta_max=delta_max)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = None
    if args.train_scope == "cumulative":
        policy = GlobalGaussianPolicy(
            mu_degree=args.mu_degree,
            sigma_degree=args.sigma_degree,
            sigma_floor=args.sigma_floor,
        )
        # Cumulative mode starts from the first behavior row and keeps updating.
        first_row = fit[fit["order_sequence"] == order_list[0]].iloc[0]
        policy.init_from_behavior_row(first_row)
        policy = policy.to(device)

    summary_rows = []
    coef_rows = []
    rng = np.random.default_rng(args.seed)

    for t, order_val in enumerate(order_list, start=1):
        seen_orders = order_list[:t]
        seen = data[data[args.order_col].isin(seen_orders)].copy()
        batch = data[data[args.order_col] == order_val].copy()
        behavior_row = fit[fit["order_sequence"] == order_val].iloc[0]

        if args.train_scope == "current":
            # Strict per-batch training: reset policy each round and train only on current batch.
            policy = GlobalGaussianPolicy(
                mu_degree=args.mu_degree,
                sigma_degree=args.sigma_degree,
                sigma_floor=args.sigma_floor,
            ).to(device)
            policy.init_from_behavior_row(behavior_row)
            train_df = batch
        else:
            train_df = seen

        th = torch.tensor(train_df[args.theta_col].to_numpy(dtype=np.float32))
        de = torch.tensor(train_df[args.delta_col].to_numpy(dtype=np.float32))
        rw = torch.tensor(train_df[args.reward_col].to_numpy(dtype=np.float32))
        if args.denominator == "mixture":
            comp_counts = train_df[args.order_col].value_counts().sort_index()
            component_orders = comp_counts.index.to_numpy(dtype=int)
            component_weights = comp_counts.to_numpy(dtype=np.float64)
            component_weights = component_weights / float(component_weights.sum())
            log_denom = compute_log_mixture_propensity(
                theta=train_df[args.theta_col].to_numpy(dtype=np.float64),
                delta=train_df[args.delta_col].to_numpy(dtype=np.float64),
                component_orders=component_orders,
                component_weights=component_weights,
                behavior_lookup=behavior_lookup,
                sigma_floor=args.sigma_floor,
            )
        else:
            component_orders = np.array(sorted(train_df[args.order_col].unique()), dtype=int)
            log_denom = train_df[args.log_prop_col].to_numpy(dtype=np.float32)
        lb = torch.tensor(log_denom, dtype=torch.float32)

        train_one_round(
            policy,
            th,
            de,
            rw,
            lb,
            objective=args.objective,
            epochs=args.epochs_per_round,
            batch_size=args.batch_size,
            lr=args.lr,
            l2_coef=args.l2_coef,
            max_weight=args.max_weight,
            clip_c=args.clip_c,
            device=device,
            num_workers=args.num_workers,
            show_tqdm=args.show_tqdm,
        )

        metrics = compute_metrics(
            policy,
            theta=th,
            delta=de,
            reward=rw,
            log_prop_b=lb,
            max_weight=args.max_weight,
            clip_c=args.clip_c,
            device=device,
        )

        with torch.no_grad():
            bmu = policy.beta_mu.detach().cpu().numpy()
            bsig = policy.beta_sigma.detach().cpu().numpy()
        coef_rec = {
            "round": int(t),
            "order_sequence": int(order_val),
            "n_seen": int(len(seen)),
            "objective": args.objective,
            "denominator": args.denominator,
            "n_components": int(len(component_orders)),
        }
        for d, v in enumerate(bmu):
            coef_rec[f"beta_mu_{d}"] = float(v)
        for d, v in enumerate(bsig):
            coef_rec[f"beta_sigma_{d}"] = float(v)
        coef_rows.append(coef_rec)

        b_mu = np.array([float(behavior_row.get(f"beta_mu_{d}", 0.0)) for d in range(args.mu_degree + 1)], dtype=float)
        b_sig = np.array([float(behavior_row.get(f"beta_sigma_{d}", 0.0)) for d in range(args.sigma_degree + 1)], dtype=float)
        beh_mu = poly_eval(theta_grid, b_mu)
        beh_sigma = np.maximum(np.exp(poly_eval(theta_grid, b_sig)), args.sigma_floor)

        pol_mu = poly_eval(theta_grid, bmu)
        pol_sigma = np.maximum(np.exp(poly_eval(theta_grid, bsig)), args.sigma_floor)

        # Value comparisons on current batch.
        batch_theta = batch[args.theta_col].to_numpy(dtype=float)
        behavior_empirical = float(batch[args.reward_col].mean())
        behavior_irt = policy_value_irt_mc(
            batch_theta,
            b_mu,
            b_sig,
            delta_min=delta_min,
            sigma_floor=args.sigma_floor,
            n_mc=args.irt_mc_samples,
            seed=args.seed + 17 * t + 1,
        )
        learned_irt = policy_value_irt_mc(
            batch_theta,
            bmu,
            bsig,
            delta_min=delta_min,
            sigma_floor=args.sigma_floor,
            n_mc=args.irt_mc_samples,
            seed=args.seed + 17 * t + 2,
        )
        optimal_irt = optimal_value_irt(
            batch_theta,
            delta_min=delta_min,
            delta_max=delta_max,
        )

        if len(batch) > args.max_scatter:
            idx = rng.choice(len(batch), size=args.max_scatter, replace=False)
            xs = batch.iloc[idx][args.theta_col].to_numpy(dtype=float)
            ys = batch.iloc[idx][args.delta_col].to_numpy(dtype=float)
        else:
            xs = batch[args.theta_col].to_numpy(dtype=float)
            ys = batch[args.delta_col].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(8.8, 5.6))
        ax.scatter(xs, ys, s=8, alpha=0.12, color="#888888", label=f"batch order={order_val}")

        ax.plot(
            theta_grid,
            beh_mu,
            color="#ff7f0e",
            lw=2.0,
            label=f"behavior mean | V_emp={behavior_empirical:.3f} V_irt={behavior_irt:.3f}",
        )
        ax.fill_between(
            theta_grid,
            beh_mu - args.band_scale * beh_sigma,
            beh_mu + args.band_scale * beh_sigma,
            color="#ff7f0e",
            alpha=0.17,
            label=f"behavior ±{args.band_scale:g} sigma",
        )

        ax.plot(
            theta_grid,
            pol_mu,
            color="#1f77b4",
            lw=2.2,
            label=f"learned mean | V_irt={learned_irt:.3f}",
        )
        ax.fill_between(
            theta_grid,
            pol_mu - args.band_scale * pol_sigma,
            pol_mu + args.band_scale * pol_sigma,
            color="#1f77b4",
            alpha=0.17,
            label=f"learned ±{args.band_scale:g} sigma",
        )

        ax.plot(
            theta_grid,
            opt_curve,
            color="#2ca02c",
            lw=2.2,
            linestyle="--",
            label=f"optimal (IRT) | V_irt={optimal_irt:.3f}",
        )

        ax.set_xlabel("theta")
        ax.set_ylabel("delta")
        ax.set_ylim(args.y_min, args.y_max)
        ax.grid(alpha=0.25)
        ax.set_title(
            f"Round {t}/{len(order_list)} | order={order_val} | train={len(train_df)} | seen={len(seen)} | "
            f"scope={args.train_scope} | denom={args.denominator}\n"
            f"ips={metrics['ips']:.3f} snips={metrics['snips']:.3f} cips={metrics['cips']:.3f} csnips={metrics['csnips']:.3f}"
        )
        ax.legend(loc="best", fontsize=8, ncol=2)
        fig.tight_layout()
        out_plot = os.path.join(plots_dir, f"round_{t:03d}_order_{int(order_val):03d}.png")
        fig.savefig(out_plot, dpi=170)
        plt.close(fig)

        summary_rows.append(
            {
                "round": int(t),
                "order_sequence": int(order_val),
                "n_batch": int(len(batch)),
                "n_train": int(len(train_df)),
                "n_seen": int(len(seen)),
                "objective": args.objective,
                "train_scope": args.train_scope,
                "denominator": args.denominator,
                "n_components": int(len(component_orders)),
                "behavior_empirical_batch_mean": behavior_empirical,
                "behavior_irt_value": behavior_irt,
                "learned_irt_value": learned_irt,
                "optimal_irt_value": optimal_irt,
                **metrics,
                "plot_path": out_plot,
            }
        )
        print(
            f"round={t}/{len(order_list)} order={order_val} n_train={len(train_df)} n_seen={len(seen)} "
            f"scope={args.train_scope} denom={args.denominator} k={len(component_orders)} "
            f"ips={metrics['ips']:.4f} snips={metrics['snips']:.4f} "
            f"beh_emp={behavior_empirical:.4f} beh_irt={behavior_irt:.4f} "
            f"learn_irt={learned_irt:.4f} opt_irt={optimal_irt:.4f}"
        )

    summary = pd.DataFrame(summary_rows)
    coefs = pd.DataFrame(coef_rows)
    summary_csv = os.path.join(args.out_dir, "sequential_round_summary.csv")
    coef_csv = os.path.join(args.out_dir, "sequential_policy_coefficients.csv")
    summary.to_csv(summary_csv, index=False)
    coefs.to_csv(coef_csv, index=False)

    # Value trajectory plot over rounds.
    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    x = summary["round"].to_numpy(dtype=int)
    ax.plot(x, summary["behavior_empirical_batch_mean"], color="#444444", lw=2.0, label="behavior empirical (batch mean)")
    ax.plot(x, summary["behavior_irt_value"], color="#ff7f0e", lw=2.0, label="behavior IRT expected value")
    ax.plot(x, summary["learned_irt_value"], color="#1f77b4", lw=2.2, label="learned policy IRT expected value")
    ax.plot(x, summary["optimal_irt_value"], color="#2ca02c", lw=2.2, linestyle="--", label="optimal policy IRT value")
    ax.set_xlabel("round (order batches)")
    ax.set_ylabel("value")
    ax.set_title(
        f"Policy Value Trajectory by Round | scope={args.train_scope} | obj={args.objective} | denom={args.denominator}"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    value_plot = os.path.join(args.out_dir, "sequential_value_trajectory.png")
    fig.savefig(value_plot, dpi=170)
    plt.close(fig)

    animation_path = ""
    if args.make_animation:
        frame_paths = sorted(
            [
                os.path.join(plots_dir, f)
                for f in os.listdir(plots_dir)
                if f.endswith(".png")
            ]
        )
        if frame_paths:
            images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in frame_paths]
            duration_ms = int(1000 / max(args.animation_fps, 1))
            animation_path = os.path.join(args.out_dir, args.animation_name)
            images[0].save(
                animation_path,
                save_all=True,
                append_images=images[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
            )
            for im in images:
                im.close()

    print(f"saved_summary={os.path.abspath(summary_csv)}")
    print(f"saved_coefficients={os.path.abspath(coef_csv)}")
    print(f"saved_plots_dir={os.path.abspath(plots_dir)}")
    print(f"saved_value_plot={os.path.abspath(value_plot)}")
    if animation_path:
        print(f"saved_animation={os.path.abspath(animation_path)}")


if __name__ == "__main__":
    main()
