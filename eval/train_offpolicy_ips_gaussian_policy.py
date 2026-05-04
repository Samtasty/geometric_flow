import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.offpolicy_gaussian_policy import GaussianOrderPolicy


def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)


@torch.no_grad()
def evaluate_ips(
    policy: GaussianOrderPolicy,
    theta: torch.Tensor,
    delta: torch.Tensor,
    reward: torch.Tensor,
    order_idx: torch.Tensor,
    log_propensity_b: torch.Tensor,
    *,
    max_weight: float,
    device: torch.device,
) -> dict[str, float]:
    policy.eval()
    theta_d = _to_device(theta, device)
    delta_d = _to_device(delta, device)
    reward_d = _to_device(reward, device)
    order_idx_d = _to_device(order_idx, device)
    log_prop_b_d = _to_device(log_propensity_b, device)

    log_p_new = policy.log_prob(delta=delta_d, theta=theta_d, order_idx=order_idx_d)
    log_ratio = torch.clamp(log_p_new - log_prop_b_d, min=-20.0, max=float(np.log(max_weight)))
    w = torch.exp(log_ratio)

    ips = torch.mean(w * reward_d)
    w_sum = torch.sum(w).clamp_min(1e-12)
    snips = torch.sum(w * reward_d) / w_sum
    ess = (w_sum ** 2) / torch.sum(w ** 2).clamp_min(1e-12)
    mean_w = torch.mean(w)
    std_w = torch.std(w)

    return {
        "ips": float(ips.item()),
        "snips": float(snips.item()),
        "ess": float(ess.item()),
        "mean_w": float(mean_w.item()),
        "std_w": float(std_w.item()),
    }


@torch.no_grad()
def evaluate_clipped(
    policy: GaussianOrderPolicy,
    theta: torch.Tensor,
    delta: torch.Tensor,
    reward: torch.Tensor,
    order_idx: torch.Tensor,
    log_propensity_b: torch.Tensor,
    *,
    clip_c: float,
    device: torch.device,
) -> dict[str, float]:
    policy.eval()
    theta_d = _to_device(theta, device)
    delta_d = _to_device(delta, device)
    reward_d = _to_device(reward, device)
    order_idx_d = _to_device(order_idx, device)
    log_prop_b_d = _to_device(log_propensity_b, device)

    log_p_new = policy.log_prob(delta=delta_d, theta=theta_d, order_idx=order_idx_d)
    log_ratio = torch.clamp(log_p_new - log_prop_b_d, min=-20.0, max=20.0)
    w = torch.exp(log_ratio)
    w_clip = torch.clamp(w, max=float(clip_c))

    cips = torch.mean(w_clip * reward_d)
    w_sum = torch.sum(w_clip).clamp_min(1e-12)
    csnips = torch.sum(w_clip * reward_d) / w_sum
    ess_clip = (w_sum ** 2) / torch.sum(w_clip ** 2).clamp_min(1e-12)
    mean_w_clip = torch.mean(w_clip)
    std_w_clip = torch.std(w_clip)
    return {
        "cips": float(cips.item()),
        "csnips": float(csnips.item()),
        "ess_clip": float(ess_clip.item()),
        "mean_w_clip": float(mean_w_clip.item()),
        "std_w_clip": float(std_w_clip.item()),
    }


def _extract_order_values(df: pd.DataFrame, order_col: str) -> tuple[np.ndarray, dict[int, int]]:
    vals = np.array(sorted(df[order_col].dropna().astype(int).unique().tolist()), dtype=int)
    mapping = {int(v): i for i, v in enumerate(vals)}
    return vals, mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Gaussian policy with IPS objective on logged tuples."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="pix_mapping/ktm_offpolicy_tuples_notebook.csv",
    )
    parser.add_argument(
        "--init-fit-csv",
        type=str,
        default="pix_mapping/ktm_gaussian_regression_order_u10k_mu2_sigma2.csv",
    )
    parser.add_argument("--order-col", type=str, default="order_model")
    parser.add_argument("--theta-col", type=str, default="proficiency")
    parser.add_argument("--delta-col", type=str, default="difficulties")
    parser.add_argument("--reward-col", type=str, default="reward")
    parser.add_argument("--log-prop-col", type=str, default="log_propensity")
    parser.add_argument("--mu-degree", type=int, default=2)
    parser.add_argument("--sigma-degree", type=int, default=2)
    parser.add_argument("--sigma-floor", type=float, default=1e-4)
    parser.add_argument(
        "--objective",
        type=str,
        default="ips",
        choices=["ips", "snips", "cips", "csnips"],
    )
    parser.add_argument("--clip-c", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--l2-coef", type=float, default=1e-6)
    parser.add_argument("--max-weight", type=float, default=50.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-history-csv",
        type=str,
        default="pix_mapping/offpolicy_ips_history.csv",
    )
    parser.add_argument(
        "--out-policy-csv",
        type=str,
        default="pix_mapping/offpolicy_ips_learned_policy.csv",
    )
    args = parser.parse_args()

    if args.mu_degree < 1:
        raise ValueError("--mu-degree must be >= 1")
    if args.sigma_degree < 1:
        raise ValueError("--sigma-degree must be >= 1")
    if args.max_weight <= 0:
        raise ValueError("--max-weight must be > 0")
    if args.clip_c <= 0:
        raise ValueError("--clip-c must be > 0")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    usecols = [args.theta_col, args.delta_col, args.reward_col, args.order_col, args.log_prop_col]
    df = pd.read_csv(args.data, usecols=usecols).dropna()

    order_vals, order_to_idx = _extract_order_values(df, args.order_col)
    order_idx_np = df[args.order_col].astype(int).map(order_to_idx).to_numpy(dtype=np.int64)

    theta_np = df[args.theta_col].to_numpy(dtype=np.float32)
    delta_np = df[args.delta_col].to_numpy(dtype=np.float32)
    reward_np = df[args.reward_col].to_numpy(dtype=np.float32)
    log_prop_b_np = df[args.log_prop_col].to_numpy(dtype=np.float32)

    theta = torch.from_numpy(theta_np)
    delta = torch.from_numpy(delta_np)
    reward = torch.from_numpy(reward_np)
    order_idx = torch.from_numpy(order_idx_np)
    log_prop_b = torch.from_numpy(log_prop_b_np)

    dataset = TensorDataset(theta, delta, reward, order_idx, log_prop_b)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = GaussianOrderPolicy(
        n_orders=len(order_vals),
        mu_degree=args.mu_degree,
        sigma_degree=args.sigma_degree,
        sigma_floor=args.sigma_floor,
    ).to(device)

    if args.init_fit_csv and os.path.exists(args.init_fit_csv):
        fit_df = pd.read_csv(args.init_fit_csv)
        policy.init_from_fit_csv(fit_df, order_to_idx=order_to_idx)

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    init_metrics = evaluate_ips(
        policy,
        theta=theta,
        delta=delta,
        reward=reward,
        order_idx=order_idx,
        log_propensity_b=log_prop_b,
        max_weight=args.max_weight,
        device=device,
    )
    history.append(
        {
            "epoch": 0,
            "train_loss": np.nan,
            "objective": args.objective,
            **init_metrics,
        }
    )
    history[-1].update(
        evaluate_clipped(
            policy,
            theta=theta,
            delta=delta,
            reward=reward,
            order_idx=order_idx,
            log_propensity_b=log_prop_b,
            clip_c=args.clip_c,
            device=device,
        )
    )

    for epoch in range(1, args.epochs + 1):
        policy.train()
        running_loss = 0.0
        n_seen = 0
        iterator = loader if args.disable_tqdm else tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in iterator:
            b_theta, b_delta, b_reward, b_order, b_logp_b = batch
            b_theta = _to_device(b_theta, device)
            b_delta = _to_device(b_delta, device)
            b_reward = _to_device(b_reward, device)
            b_order = _to_device(b_order, device)
            b_logp_b = _to_device(b_logp_b, device)

            optimizer.zero_grad(set_to_none=True)
            log_p_new = policy.log_prob(delta=b_delta, theta=b_theta, order_idx=b_order)
            log_ratio = torch.clamp(
                log_p_new - b_logp_b,
                min=-20.0,
                max=float(np.log(args.max_weight)),
            )
            w = torch.exp(log_ratio)

            ips_obj = torch.mean(w * b_reward)
            snips_obj = torch.sum(w * b_reward) / torch.sum(w).clamp_min(1e-12)
            w_clip = torch.clamp(w, max=float(args.clip_c))
            cips_obj = torch.mean(w_clip * b_reward)
            csnips_obj = torch.sum(w_clip * b_reward) / torch.sum(w_clip).clamp_min(1e-12)
            if args.objective == "ips":
                target_obj = ips_obj
            elif args.objective == "snips":
                target_obj = snips_obj
            elif args.objective == "cips":
                target_obj = cips_obj
            else:
                target_obj = csnips_obj
            reg = args.l2_coef * (
                torch.mean(policy.beta_mu ** 2) + torch.mean(policy.beta_sigma ** 2)
            )
            loss = -target_obj + reg
            loss.backward()
            optimizer.step()

            bs = b_theta.shape[0]
            running_loss += float(loss.item()) * bs
            n_seen += bs
            if not args.disable_tqdm:
                iterator.set_postfix(
                    {
                        "loss": f"{loss.item():.5f}",
                        "ips": f"{ips_obj.item():.5f}",
                        "snips": f"{snips_obj.item():.5f}",
                        "cips": f"{cips_obj.item():.5f}",
                        "csnips": f"{csnips_obj.item():.5f}",
                    }
                )

        train_loss = running_loss / max(n_seen, 1)
        metrics = evaluate_ips(
            policy,
            theta=theta,
            delta=delta,
            reward=reward,
            order_idx=order_idx,
            log_propensity_b=log_prop_b,
            max_weight=args.max_weight,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "objective": args.objective,
                **metrics,
            }
        )
        history[-1].update(
            evaluate_clipped(
                policy,
                theta=theta,
                delta=delta,
                reward=reward,
                order_idx=order_idx,
                log_propensity_b=log_prop_b,
                clip_c=args.clip_c,
                device=device,
            )
        )
        print(
            f"epoch={epoch} loss={train_loss:.6f} ips={metrics['ips']:.6f} "
            f"snips={metrics['snips']:.6f} cips={history[-1]['cips']:.6f} "
            f"csnips={history[-1]['csnips']:.6f} ess={metrics['ess']:.1f}"
        )

    hist_df = pd.DataFrame(history).sort_values("epoch").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_history_csv), exist_ok=True)
    hist_df.to_csv(args.out_history_csv, index=False)

    rows = []
    with torch.no_grad():
        bmu = policy.beta_mu.detach().cpu().numpy()
        bsig = policy.beta_sigma.detach().cpu().numpy()
    for i, order_val in enumerate(order_vals):
        rec = {
            "order_value": int(order_val),
            "objective": args.objective,
            "clip_c": float(args.clip_c),
        }
        for d in range(args.mu_degree + 1):
            rec[f"beta_mu_{d}"] = float(bmu[i, d])
        for d in range(args.sigma_degree + 1):
            rec[f"beta_sigma_{d}"] = float(bsig[i, d])
        rows.append(rec)
    pol_df = pd.DataFrame(rows).sort_values("order_value").reset_index(drop=True)
    pol_df.to_csv(args.out_policy_csv, index=False)

    print(f"rows={len(df)} n_orders={len(order_vals)}")
    print(f"objective={args.objective}")
    print(f"clip_c={args.clip_c}")
    print(f"saved_history={os.path.abspath(args.out_history_csv)}")
    print(f"saved_policy={os.path.abspath(args.out_policy_csv)}")
    print("\nFinal:")
    print(hist_df.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
