from __future__ import annotations

import argparse
import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ktm import processing_data


def _load_dataset(name: str, path: str) -> pd.DataFrame:
    lname = name.lower()
    if lname in {"assistment8000", "assistments8000", "assistment100", "assistments100"}:
        df = pd.read_csv(path, usecols=["user", "item", "correct"]).copy()
        df["answer_number"] = df.groupby("user").cumcount() + 1
        df["skill"] = "NA"
        return df
    if lname in {"skillbuilder", "skill_builder_2015", "skillbuilder2015"}:
        df = pd.read_csv(path, usecols=["user_id", "sequence_id", "correct"]).copy()
        df = df.rename(columns={"user_id": "user", "sequence_id": "item"})
        df["answer_number"] = df.groupby("user").cumcount() + 1
        df["skill"] = "NA"
        return df
    if lname == "attempts":
        df = pd.read_csv(path, usecols=["student", "problem", "solved"]).copy()
        df = df.rename(columns={"student": "user", "problem": "item", "solved": "correct"})
        df["correct"] = df["correct"].astype(int)
        df["answer_number"] = df.groupby("user").cumcount() + 1
        df["skill"] = "NA"
        return df
    if lname in {"pix_data", "pix"}:
        df = pd.read_csv(path, usecols=["user_id", "challenge_id", "answer_result", "answer_number"]).copy()
        amap = {"ok": 1.0, "ko": 0.0, "aband": 0.0}
        df["correct"] = df["answer_result"].map(amap).fillna(0.0)
        df = df.rename(columns={"user_id": "user", "challenge_id": "item"})
        df["skill"] = "NA"
        return df[["user", "item", "correct", "answer_number", "skill"]]
    raise ValueError(f"Unknown dataset name: {name}")


def _poly_design_torch(x: torch.Tensor, degree: int) -> torch.Tensor:
    cols = [torch.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return torch.stack(cols, dim=1)


def _poly_eval_np(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    for d, c in enumerate(coeffs):
        y += c * (x ** d)
    return y


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def _fit_trunc_gaussian_torch(
    theta: np.ndarray,
    delta: np.ndarray,
    *,
    mu_degree: int,
    sigma_degree: int,
    sigma_floor: float,
    lr: float,
    max_epochs: int,
    tol: float,
    patience: int,
    l2_coef: float,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    th = torch.from_numpy(theta.astype(np.float32)).to(device)
    de = torch.from_numpy(delta.astype(np.float32)).to(device)
    x_mu = _poly_design_torch(th, mu_degree)
    x_sig = _poly_design_torch(th, sigma_degree)
    p = x_mu.shape[1]
    q = x_sig.shape[1]

    beta_mu = torch.nn.Parameter(torch.zeros(p, device=device))
    beta_sig = torch.nn.Parameter(torch.zeros(q, device=device))

    # Robust init.
    x_mu_np = np.column_stack([np.ones_like(theta)] + [theta ** d for d in range(1, mu_degree + 1)])
    beta0 = np.linalg.lstsq(x_mu_np, delta, rcond=None)[0]
    resid = delta - x_mu_np @ beta0
    sigma0 = max(float(np.std(resid)), sigma_floor)
    with torch.no_grad():
        beta_mu.copy_(torch.from_numpy(beta0.astype(np.float32)).to(device))
        beta_sig[0] = float(np.log(sigma0))

    a = float(delta.min())
    b = float(delta.max())
    log2pi = float(np.log(2.0 * np.pi))
    optimizer = torch.optim.Adam([beta_mu, beta_sig], lr=lr)

    best = float("inf")
    best_state = None
    bad = 0

    for _ in range(max_epochs):
        optimizer.zero_grad(set_to_none=True)
        mu = x_mu @ beta_mu
        sigma = torch.exp(x_sig @ beta_sig).clamp_min(float(sigma_floor))
        z = (de - mu) / sigma
        alpha = (float(a) - mu) / sigma
        beta = (float(b) - mu) / sigma
        log_phi = -0.5 * log2pi - torch.log(sigma) - 0.5 * (z ** 2)
        z_norm = (_normal_cdf(beta) - _normal_cdf(alpha)).clamp_min(1e-12)
        log_pdf = log_phi - torch.log(z_norm)
        nll = -torch.mean(log_pdf)
        reg = float(l2_coef) * (torch.mean(beta_mu ** 2) + torch.mean(beta_sig ** 2))
        loss = nll + reg
        loss.backward()
        optimizer.step()

        cur = float(nll.detach().cpu().item())
        if cur + tol < best:
            best = cur
            bad = 0
            best_state = (
                beta_mu.detach().cpu().numpy().copy(),
                beta_sig.detach().cpu().numpy().copy(),
            )
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        best_state = (
            beta_mu.detach().cpu().numpy().copy(),
            beta_sig.detach().cpu().numpy().copy(),
        )
    return best_state[0], best_state[1], float(best)


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate behavior policy evolution with truncated Gaussian.")
    parser.add_argument("--dataset-name", type=str, default="skill_builder_2015")
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--mu-degree", type=int, default=1)
    parser.add_argument("--sigma-degree", type=int, default=2)
    parser.add_argument("--sigma-floor", type=float, default=0.2)
    parser.add_argument("--min-obs-per-order", type=int, default=200)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--l2-coef", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out-gif",
        type=str,
        default="pix_mapping/behavior_trunc_gaussian_skill_builder_2015.gif",
    )
    parser.add_argument(
        "--out-fit-csv",
        type=str,
        default="pix_mapping/behavior_trunc_gaussian_skill_builder_2015.csv",
    )
    args = parser.parse_args()

    if not args.data_path:
        default_map = {
            "assistment8000": os.path.join(PROJECT_ROOT, "data", "data_top8000_items.csv"),
            "assistment100": os.path.join(PROJECT_ROOT, "data", "data_top8000_items.csv"),
            "skill_builder_2015": os.path.join(PROJECT_ROOT, "data", "2015_100_skill_builders_main_problems.csv"),
            "attempts": os.path.join(PROJECT_ROOT, "data", "attempts.csv"),
            "pix_data": os.path.join(PROJECT_ROOT, "data", "pix_data.csv"),
        }
        if args.dataset_name not in default_map:
            raise ValueError("Unknown dataset_name and no --data-path provided")
        data_path = default_map[args.dataset_name]
    else:
        data_path = args.data_path

    raw = _load_dataset(args.dataset_name, data_path)
    df_ktm = processing_data(
        raw.copy(),
        user_col="user",
        item_col="item",
        correct_col="correct",
        skill_col="skill",
        order_cols=["answer_number"],
        reduce=False,
        rank_start=0,
        c=0.1,
        random_state=args.seed,
        fit_intercept=True,
        center_latents=True,
        target_std=None,
    )

    groups = []
    for order_val, g in df_ktm.groupby("order_sequence", sort=True):
        if len(g) < args.min_obs_per_order:
            continue
        groups.append((int(order_val), g.copy()))
    if args.max_rounds > 0:
        groups = groups[: int(args.max_rounds)]
    if not groups:
        raise ValueError("No order groups satisfy min-obs filter.")

    frames = []
    fit_rows = []
    for order_val, g in groups:
        theta = g["proficiency"].to_numpy(dtype=float)
        delta = g["difficulties"].to_numpy(dtype=float)
        beta_mu, beta_sigma, nll = _fit_trunc_gaussian_torch(
            theta=theta,
            delta=delta,
            mu_degree=args.mu_degree,
            sigma_degree=args.sigma_degree,
            sigma_floor=args.sigma_floor,
            lr=args.lr,
            max_epochs=args.max_epochs,
            tol=args.tol,
            patience=args.patience,
            l2_coef=args.l2_coef,
            seed=args.seed + order_val,
            device=args.device,
        )
        frames.append(
            {
                "order_sequence": order_val,
                "theta": theta,
                "delta": delta,
                "beta_mu": beta_mu.copy(),
                "beta_sigma": beta_sigma.copy(),
                "nll": float(nll),
                "n_obs": int(len(g)),
            }
        )
        rec = {"order_sequence": order_val, "n_obs": len(g), "nll": float(nll)}
        for d, v in enumerate(beta_mu):
            rec[f"beta_mu_{d}"] = float(v)
        for d, v in enumerate(beta_sigma):
            rec[f"beta_sigma_{d}"] = float(v)
        fit_rows.append(rec)
        print(f"fit order={order_val} n={len(g)} nll={nll:.6f}")

    fit_df = pd.DataFrame(fit_rows).sort_values("order_sequence")
    os.makedirs(os.path.dirname(args.out_fit_csv), exist_ok=True)
    fit_df.to_csv(args.out_fit_csv, index=False)

    theta_min = min(float(np.min(fr["theta"])) for fr in frames)
    theta_max = max(float(np.max(fr["theta"])) for fr in frames)
    delta_min = min(float(np.min(fr["delta"])) for fr in frames)
    delta_max = max(float(np.max(fr["delta"])) for fr in frames)
    theta_grid = np.linspace(theta_min, theta_max, 360, dtype=float)
    rng = np.random.default_rng(args.seed)

    fig, ax = plt.subplots(figsize=(9.2, 6.2))

    def draw(i: int) -> None:
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
        ax.scatter(xs, ys, s=8, alpha=0.15, color="#777777", label="data")

        mu = _poly_eval_np(theta_grid, fr["beta_mu"])
        sigma = np.maximum(np.exp(_poly_eval_np(theta_grid, fr["beta_sigma"])), args.sigma_floor)
        ax.plot(theta_grid, mu, color="#1f77b4", lw=2.3, label="truncated Gaussian mean")
        ax.fill_between(theta_grid, mu - 2.0 * sigma, mu + 2.0 * sigma, color="#1f77b4", alpha=0.20, label="±2σ")

        ax.set_xlim(theta_min, theta_max)
        ax.set_ylim(delta_min, delta_max)
        ax.set_xlabel("theta (proficiency)")
        ax.set_ylabel("delta (difficulty)")
        ax.set_title(
            f"Truncated Gaussian behavior | order={fr['order_sequence']} | "
            f"n={fr['n_obs']} | nll={fr['nll']:.4f}"
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / max(args.fps, 1))
    os.makedirs(os.path.dirname(args.out_gif), exist_ok=True)
    ani.save(args.out_gif, writer=animation.PillowWriter(fps=max(args.fps, 1)))
    plt.close(fig)

    print(f"saved_gif={os.path.abspath(args.out_gif)}")
    print(f"saved_fit_csv={os.path.abspath(args.out_fit_csv)}")
    print(f"n_frames={len(frames)}")


if __name__ == "__main__":
    main()
