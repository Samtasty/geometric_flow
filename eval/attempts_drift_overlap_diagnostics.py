from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.offpolicy_gaussian_policy import GlobalGaussianPolicy
from offpolicy_ktm_pipeline.run_last_round_simple import _compute_context_ratio_hist, _evaluate_estimators


def _build_behavior_policy(row: pd.Series, *, delta_min: float, delta_max: float, sigma_floor: float) -> GlobalGaussianPolicy:
    policy = GlobalGaussianPolicy(
        mu_degree=1,
        sigma_degree=2,
        sigma_floor=float(sigma_floor),
        delta_min=float(delta_min),
        delta_max=float(delta_max),
        bound_mean_to_action_range=False,
    )
    policy.init_from_behavior_row(row)
    policy.eval()
    return policy


def _build_learned_policy(row: pd.Series, *, delta_min: float, delta_max: float, sigma_floor: float) -> GlobalGaussianPolicy:
    policy = GlobalGaussianPolicy(
        mu_degree=1,
        sigma_degree=2,
        sigma_floor=float(sigma_floor),
        delta_min=float(delta_min),
        delta_max=float(delta_max),
        bound_mean_to_action_range=True,
    )
    with torch.no_grad():
        policy.beta_mu[0] = float(row["beta_mu_0"])
        policy.beta_mu[1] = float(row["beta_mu_1"])
        policy.beta_sigma[0] = float(row["beta_sigma_0"])
        policy.beta_sigma[1] = float(row["beta_sigma_1"])
        policy.beta_sigma[2] = float(row["beta_sigma_2"])
    policy.eval()
    return policy


def _policy_label(name: str) -> str:
    return {
        "behavior_last_round": "Behavior (round last)",
        "learned_ips": "Learned IPS",
        "learned_snips": "Learned SNIPS",
        "learned_dr": "Learned DR",
    }.get(name, name)


def _build_eval_frame(
    *,
    ktm_df: pd.DataFrame,
    split_df: pd.DataFrame,
    split_name: str,
    last_round: int,
    delta_min: float,
) -> pd.DataFrame:
    users = set(split_df.loc[split_df["split"] == split_name, "user"])
    out = ktm_df[(ktm_df["user"].isin(users)) & (ktm_df["order_sequence"].astype(int) <= int(last_round))].copy()
    out["reward"] = out["correct"].astype(float) * (out["difficulties"].astype(float) - float(delta_min))
    return out


def _round_drift_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    diff_floor = float(df["difficulties"].min())
    for order, g in df.groupby("order_sequence", sort=True):
        diff = g["difficulties"].to_numpy(dtype=float)
        rows.append(
            {
                "order_sequence": int(order),
                "n_rows": int(len(g)),
                "difficulty_mean": float(np.mean(diff)),
                "difficulty_p10": float(np.quantile(diff, 0.10)),
                "difficulty_p25": float(np.quantile(diff, 0.25)),
                "difficulty_p50": float(np.quantile(diff, 0.50)),
                "difficulty_p75": float(np.quantile(diff, 0.75)),
                "difficulty_p90": float(np.quantile(diff, 0.90)),
                "difficulty_floor_frac": float(np.mean(np.isclose(diff, diff_floor))),
                "proficiency_mean": float(g["proficiency"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows)


def _policy_overlap_by_round(
    *,
    eval_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    target_theta: np.ndarray,
    policies: dict[str, GlobalGaussianPolicy],
    delta_min: float,
    delta_max: float,
    sigma_floor: float,
    max_weight: float,
    dm_delta_grid: int,
    context_bins: int,
    context_ratio_clip: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for order, g in eval_df.groupby("order_sequence", sort=True):
        order = int(order)
        logged_row = fit_df.loc[fit_df["order_sequence"].astype(int) == order].iloc[0]
        logged_policy = _build_behavior_policy(
            logged_row,
            delta_min=delta_min,
            delta_max=delta_max,
            sigma_floor=sigma_floor,
        )

        theta_np = g["proficiency"].to_numpy(dtype=float)
        context_w_np = _compute_context_ratio_hist(
            theta_ref=target_theta,
            theta_cur=theta_np,
            theta_eval=theta_np,
            n_bins=int(context_bins),
            ratio_clip=float(context_ratio_clip),
        ).astype(np.float32)

        theta_t = torch.from_numpy(theta_np.astype(np.float32))
        delta_t = torch.from_numpy(g["difficulties"].to_numpy(dtype=np.float32))
        reward_t = torch.from_numpy(g["reward"].to_numpy(dtype=np.float32))
        context_t = torch.from_numpy(context_w_np)
        with torch.no_grad():
            log_prop_t = logged_policy.log_prob(delta=delta_t, theta=theta_t)

        for name, policy in policies.items():
            with torch.no_grad():
                log_num = policy.log_prob(delta=delta_t, theta=theta_t)
                log_ratio = torch.clamp(log_num - log_prop_t, min=-20.0, max=float(np.log(max_weight)))
                weights = torch.exp(log_ratio)
            est = _evaluate_estimators(
                policy,
                theta=theta_t,
                delta=delta_t,
                reward=reward_t,
                context_w=context_t,
                log_prop_logged=log_prop_t,
                log_prop_mix=None,
                delta_min=float(delta_min),
                delta_max=float(delta_max),
                dm_delta_grid=int(dm_delta_grid),
                max_weight=float(max_weight),
                device=torch.device("cpu"),
            )
            w_np = weights.cpu().numpy()
            rows.append(
                {
                    "order_sequence": order,
                    "policy": name,
                    "n_rows": int(len(g)),
                    "ips": float(est["ips"]),
                    "snips": float(est["snips"]),
                    "dr": float(est["dr"]),
                    "dm": float(est["dm"]),
                    "ess_logged": float(est["ess_logged"]),
                    "ess_frac": float(est["ess_logged"]) / max(float(len(g)), 1.0),
                    "w_mean": float(np.mean(w_np)),
                    "w_std": float(np.std(w_np)),
                    "w_p95": float(np.quantile(w_np, 0.95)),
                    "w_p99": float(np.quantile(w_np, 0.99)),
                    "w_max": float(np.max(w_np)),
                    "clip_hi_frac": float(np.mean(w_np >= float(max_weight) - 1e-12)),
                }
            )
    return pd.DataFrame(rows)


def _plot_diagnostics(
    *,
    drift_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    out_path: Path,
) -> None:
    colors = {
        "behavior_last_round": "#2C7BB6",
        "learned_ips": "#D7191C",
        "learned_snips": "#FDAE61",
        "learned_dr": "#1A9641",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    x = drift_df["order_sequence"].to_numpy()
    ax.plot(x, drift_df["difficulty_mean"], color="#2C7BB6", linewidth=2.5, label="mean difficulty")
    ax.fill_between(
        x,
        drift_df["difficulty_p10"],
        drift_df["difficulty_p90"],
        color="#2C7BB6",
        alpha=0.18,
        label="p10-p90",
    )
    ax.set_title("Behavior Drift by Round")
    ax.set_xlabel("source round")
    ax.set_ylabel("difficulty")
    ax2 = ax.twinx()
    ax2.plot(x, 100.0 * drift_df["difficulty_floor_frac"], color="#CC3311", linestyle="--", linewidth=2, label="floor mass (%)")
    ax2.set_ylabel("floor mass (%)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="lower right")

    ax = axes[1]
    for name, g in overlap_df.groupby("policy", sort=False):
        ax.plot(g["order_sequence"], g["ess_frac"], marker="o", markersize=3.5, linewidth=2, color=colors[name], label=_policy_label(name))
    ax.set_title("Overlap With Round-Dependent Logger")
    ax.set_xlabel("source round")
    ax.set_ylabel("ESS / n")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    for name, g in overlap_df.groupby("policy", sort=False):
        ax.plot(g["order_sequence"], g["w_p99"], marker="o", markersize=3.5, linewidth=2, color=colors[name], label=_policy_label(name))
    ax.set_title("Upper-Tail Importance Weights")
    ax.set_xlabel("source round")
    ax.set_ylabel("weight p99")
    ax.set_ylim(bottom=0.0)

    fig.suptitle("Attempts: Adaptive Drift and IPS Overlap Diagnostics", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose adaptive drift and IPS overlap on attempts.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/attempts_user_split_trunc30_testonly_ep10_bs16000"),
    )
    parser.add_argument(
        "--ktm-csv",
        type=Path,
        default=Path("results/attempts_trunc30/ktm_dataframe.csv"),
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="attempts_ips_overlap",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_prefix = run_dir / args.output_prefix

    ktm_df = pd.read_csv(args.ktm_csv.resolve())
    split_df = pd.read_csv(run_dir / "user_split_assignments.csv")
    fit_df = pd.read_csv(run_dir / "gaussian_fit_train_by_order.csv")
    learned_df = pd.read_csv(run_dir / "learned_policy_coefficients.csv")
    summary_df = pd.read_csv(run_dir / "run_summary.csv")

    last_round = int(summary_df["common_last_round"].iloc[0])
    target_round = int(summary_df["target_round_train_first_half"].iloc[0])
    delta_min = float(fit_df["delta_min"].iloc[0])
    delta_max = float(fit_df["delta_max"].iloc[0])
    sigma_floor = 0.2
    max_weight = 20.0
    dm_delta_grid = 121
    context_bins = 120
    context_ratio_clip = 20.0

    train_eval = _build_eval_frame(
        ktm_df=ktm_df,
        split_df=split_df,
        split_name="train",
        last_round=last_round,
        delta_min=delta_min,
    )
    test_eval = _build_eval_frame(
        ktm_df=ktm_df,
        split_df=split_df,
        split_name="test",
        last_round=last_round,
        delta_min=delta_min,
    )
    target_theta = train_eval.loc[train_eval["order_sequence"].astype(int) == target_round, "proficiency"].to_numpy(dtype=float)

    behavior_last = _build_behavior_policy(
        fit_df.loc[fit_df["order_sequence"].astype(int) == last_round].iloc[0],
        delta_min=delta_min,
        delta_max=delta_max,
        sigma_floor=sigma_floor,
    )
    policies = {"behavior_last_round": behavior_last}
    for _, row in learned_df.iterrows():
        policies[f"learned_{row['objective']}"] = _build_learned_policy(
            row,
            delta_min=delta_min,
            delta_max=delta_max,
            sigma_floor=sigma_floor,
        )

    drift_df = _round_drift_summary(test_eval)
    overlap_df = _policy_overlap_by_round(
        eval_df=test_eval,
        fit_df=fit_df,
        target_theta=target_theta,
        policies=policies,
        delta_min=delta_min,
        delta_max=delta_max,
        sigma_floor=sigma_floor,
        max_weight=max_weight,
        dm_delta_grid=dm_delta_grid,
        context_bins=context_bins,
        context_ratio_clip=context_ratio_clip,
    )

    drift_df.to_csv(out_prefix.with_name(f"{args.output_prefix}_drift_by_round.csv"), index=False)
    overlap_df.to_csv(out_prefix.with_name(f"{args.output_prefix}_overlap_by_round.csv"), index=False)
    _plot_diagnostics(
        drift_df=drift_df,
        overlap_df=overlap_df,
        out_path=out_prefix.with_name(f"{args.output_prefix}_diagnostic.png"),
    )

    selected = overlap_df[overlap_df["order_sequence"].isin([0, 5, 10, 15, 20, 25, last_round])].copy()
    selected.to_csv(out_prefix.with_name(f"{args.output_prefix}_selected_rounds.csv"), index=False)

    print(f"saved_drift={out_prefix.with_name(f'{args.output_prefix}_drift_by_round.csv')}")
    print(f"saved_overlap={out_prefix.with_name(f'{args.output_prefix}_overlap_by_round.csv')}")
    print(f"saved_selected={out_prefix.with_name(f'{args.output_prefix}_selected_rounds.csv')}")
    print(f"saved_plot={out_prefix.with_name(f'{args.output_prefix}_diagnostic.png')}")


if __name__ == "__main__":
    main()
