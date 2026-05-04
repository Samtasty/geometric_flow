from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ktm import processing_data


def _poly_design_np(x: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return np.column_stack(cols).astype(np.float32)


def _poly_design_torch(x: torch.Tensor, degree: int) -> torch.Tensor:
    cols = [torch.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return torch.stack(cols, dim=1)


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _component_mu_sigma_np(
    theta: np.ndarray,
    beta_mu: np.ndarray,
    beta_sig: np.ndarray,
    sigma_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_mu = _poly_design_np(theta, degree=len(beta_mu) - 1)
    x_sig = _poly_design_np(theta, degree=len(beta_sig) - 1)
    mu = x_mu @ beta_mu
    sigma = np.maximum(np.exp(x_sig @ beta_sig), sigma_floor)
    return mu, sigma


def _init_params(
    theta: np.ndarray,
    delta: np.ndarray,
    *,
    k: int,
    mu_degree: int,
    sigma_degree: int,
    sigma_floor: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_mu = _poly_design_np(theta, mu_degree)
    p = x_mu.shape[1]
    q = sigma_degree + 1

    beta_mu = np.zeros((k, p), dtype=np.float32)
    beta_sig = np.zeros((k, q), dtype=np.float32)
    logits_pi = np.zeros(k, dtype=np.float32)

    qs = np.quantile(delta, np.linspace(0.0, 1.0, k + 1))
    comp = np.digitize(delta, qs[1:-1], right=True)
    if len(np.unique(comp)) < k:
        comp = rng.integers(0, k, size=len(delta))

    for j in range(k):
        mask = comp == j
        if int(mask.sum()) < p:
            beta = np.linalg.lstsq(x_mu, delta, rcond=None)[0]
            resid = delta - x_mu @ beta
            nj = max(int(mask.sum()), 1)
        else:
            beta = np.linalg.lstsq(x_mu[mask], delta[mask], rcond=None)[0]
            resid = delta[mask] - x_mu[mask] @ beta
            nj = int(mask.sum())

        beta_mu[j] = beta.astype(np.float32)
        s0 = max(float(np.std(resid)), sigma_floor)
        beta_sig[j, 0] = float(np.log(s0))
        logits_pi[j] = float(np.log(nj))

    return beta_mu, beta_sig, logits_pi


def fit_trunc_gmm_torch(
    theta: np.ndarray,
    delta: np.ndarray,
    *,
    k: int,
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
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    th = torch.from_numpy(theta.astype(np.float32)).to(device)
    de = torch.from_numpy(delta.astype(np.float32)).to(device)
    x_mu = _poly_design_torch(th, mu_degree)
    x_sig = _poly_design_torch(th, sigma_degree)
    a = float(delta.min())
    b = float(delta.max())

    bmu0, bsig0, lpi0 = _init_params(
        theta,
        delta,
        k=k,
        mu_degree=mu_degree,
        sigma_degree=sigma_degree,
        sigma_floor=sigma_floor,
        seed=seed,
    )

    beta_mu = torch.nn.Parameter(torch.from_numpy(bmu0).to(device))
    beta_sig = torch.nn.Parameter(torch.from_numpy(bsig0).to(device))
    logits_pi = torch.nn.Parameter(torch.from_numpy(lpi0).to(device))

    opt = torch.optim.Adam([beta_mu, beta_sig, logits_pi], lr=lr)
    best = float("inf")
    best_state = None
    bad = 0
    log2pi = math.log(2.0 * math.pi)

    for epoch in range(1, max_epochs + 1):
        opt.zero_grad(set_to_none=True)
        mu = x_mu @ beta_mu.T
        sigma = torch.exp(x_sig @ beta_sig.T).clamp_min(float(sigma_floor))

        z = (de[:, None] - mu) / sigma
        alpha = (float(a) - mu) / sigma
        beta = (float(b) - mu) / sigma
        log_phi = -0.5 * log2pi - torch.log(sigma) - 0.5 * (z ** 2)
        z_norm = (_normal_cdf(beta) - _normal_cdf(alpha)).clamp_min(1e-12)
        log_pdf = log_phi - torch.log(z_norm)
        log_pi = F.log_softmax(logits_pi, dim=0)[None, :]
        log_mix = torch.logsumexp(log_pi + log_pdf, dim=1)
        nll = -torch.mean(log_mix)
        reg = float(l2_coef) * (torch.mean(beta_mu ** 2) + torch.mean(beta_sig ** 2))
        loss = nll + reg
        loss.backward()
        opt.step()

        cur = float(nll.detach().cpu().item())
        if cur + tol < best:
            best = cur
            bad = 0
            best_state = (
                beta_mu.detach().cpu().numpy().copy(),
                beta_sig.detach().cpu().numpy().copy(),
                logits_pi.detach().cpu().numpy().copy(),
                epoch,
            )
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        best_state = (
            beta_mu.detach().cpu().numpy().copy(),
            beta_sig.detach().cpu().numpy().copy(),
            logits_pi.detach().cpu().numpy().copy(),
            max_epochs,
        )
    bmu, bsig, lpi, best_epoch = best_state
    pi = np.exp(lpi - np.max(lpi))
    pi = pi / np.clip(pi.sum(), 1e-12, None)
    return {
        "pi": pi,
        "beta_mu": bmu,
        "beta_sig": bsig,
        "nll": float(best),
        "best_epoch": int(best_epoch),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate behavior policy evolution with truncated GMM-3.")
    parser.add_argument("--dataset-name", type=str, default="assistment8000")
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--mu-degree", type=int, default=1)
    parser.add_argument("--sigma-degree", type=int, default=2)
    parser.add_argument("--sigma-floor", type=float, default=0.2)
    parser.add_argument("--min-obs-per-order", type=int, default=200)
    parser.add_argument("--max-rounds", type=int, default=0, help="0 means all available orders.")
    parser.add_argument("--max-points", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--l2-coef", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-gif", type=str, default="pix_mapping/behavior_trunc_gmm3_assistment8000.gif")
    parser.add_argument("--out-fit-csv", type=str, default="pix_mapping/behavior_trunc_gmm3_assistment8000.csv")
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
    for idx, (order_val, g) in enumerate(groups, start=1):
        theta = g["proficiency"].to_numpy(dtype=np.float32)
        delta = g["difficulties"].to_numpy(dtype=np.float32)
        fit = fit_trunc_gmm_torch(
            theta,
            delta,
            k=args.k,
            mu_degree=args.mu_degree,
            sigma_degree=args.sigma_degree,
            sigma_floor=args.sigma_floor,
            lr=args.lr,
            max_epochs=args.max_epochs,
            tol=args.tol,
            patience=args.patience,
            l2_coef=args.l2_coef,
            seed=args.seed + idx,
            device=args.device,
        )
        frames.append(
            {
                "order_sequence": order_val,
                "theta": theta,
                "delta": delta,
                "pi": fit["pi"],
                "beta_mu": fit["beta_mu"],
                "beta_sig": fit["beta_sig"],
                "nll": fit["nll"],
                "n_obs": len(g),
            }
        )
        rec = {
            "order_sequence": order_val,
            "n_obs": len(g),
            "nll": float(fit["nll"]),
        }
        for j in range(args.k):
            rec[f"pi_{j}"] = float(fit["pi"][j])
            for m in range(args.mu_degree + 1):
                rec[f"beta_mu_{j}_{m}"] = float(fit["beta_mu"][j, m])
            for m in range(args.sigma_degree + 1):
                rec[f"beta_sig_{j}_{m}"] = float(fit["beta_sig"][j, m])
        fit_rows.append(rec)
        print(f"fit order={order_val} n={len(g)} nll={fit['nll']:.6f}")

    fit_df = pd.DataFrame(fit_rows).sort_values("order_sequence")
    os.makedirs(os.path.dirname(args.out_fit_csv), exist_ok=True)
    fit_df.to_csv(args.out_fit_csv, index=False)

    theta_min = min(float(np.min(fr["theta"])) for fr in frames)
    theta_max = max(float(np.max(fr["theta"])) for fr in frames)
    delta_min = min(float(np.min(fr["delta"])) for fr in frames)
    delta_max = max(float(np.max(fr["delta"])) for fr in frames)
    theta_grid = np.linspace(theta_min, theta_max, 360, dtype=np.float32)
    rng = np.random.default_rng(args.seed)

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    colors = plt.cm.tab10(np.linspace(0, 1, max(args.k, 3)))

    def draw(i: int) -> None:
        ax.clear()
        fr = frames[i]
        theta = fr["theta"]
        delta = fr["delta"]
        if len(theta) > args.max_points:
            ix = rng.choice(len(theta), size=args.max_points, replace=False)
            xs = theta[ix]
            ys = delta[ix]
        else:
            xs = theta
            ys = delta
        ax.scatter(xs, ys, s=8, alpha=0.15, color="#777777", label="data")

        mix_mean = np.zeros_like(theta_grid, dtype=np.float32)
        for j in range(args.k):
            mu_j, sig_j = _component_mu_sigma_np(
                theta_grid,
                fr["beta_mu"][j],
                fr["beta_sig"][j],
                sigma_floor=args.sigma_floor,
            )
            mix_mean += fr["pi"][j] * mu_j
            c = colors[j]
            ax.plot(theta_grid, mu_j, color=c, lw=1.8, label=f"comp {j+1} (pi={fr['pi'][j]:.2f})")
            ax.fill_between(theta_grid, mu_j - 2.0 * sig_j, mu_j + 2.0 * sig_j, color=c, alpha=0.12)

        ax.plot(theta_grid, mix_mean, color="black", lw=2.4, label="mixture mean")
        ax.set_xlim(theta_min, theta_max)
        ax.set_ylim(delta_min, delta_max)
        ax.set_xlabel("theta (proficiency)")
        ax.set_ylabel("delta (difficulty)")
        ax.set_title(
            f"Truncated GMM-3 behavior | order={fr['order_sequence']} | "
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
