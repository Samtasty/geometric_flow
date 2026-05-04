"""
Train/test split off-policy evaluation.

Students are split 70/30 by user ID (no leakage).
Behavior policy and learned policies are fitted on TRAIN only.
The target context distribution is fixed from the TRAIN set.
All policies are then evaluated on both TRAIN and TEST sets,
allowing a direct out-of-sample comparison.

Usage:
    python run_train_test.py \
        --base-ktm results/attempts_trunc30/ktm_dataframe.csv \
        --output-dir results/attempts_train_test \
        --train-fraction 0.7 \
        --epochs 20

Repeat for each dataset by changing --base-ktm and --output-dir.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# --------------------------------------------------------------------------- #
# Import internal helpers from run_last_round_simple without calling main()    #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
_rls_path = PROJECT_ROOT / "offpolicy_ktm_pipeline" / "run_last_round_simple.py"
_spec = importlib.util.spec_from_file_location("_rls", _rls_path)
_rls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rls)

_evaluate_estimators           = _rls._evaluate_estimators
_train_last_round_policy       = _rls._train_last_round_policy
_build_behavior_policy_from_row = _rls._build_behavior_policy_from_row
_select_target_round            = _rls._select_target_round_closest_to_last
_compute_context_ratio_hist     = _rls._compute_context_ratio_hist
IRTOptimalGaussianPolicy        = _rls.IRTOptimalGaussianPolicy

from offpolicy_ktm_pipeline.src.gaussian_regression import fit_gaussian_by_order
from offpolicy_ktm_pipeline.src.propensity import build_offpolicy_dataset
from offpolicy_ktm_pipeline.src.estimators_and_training import compute_log_mixture_propensity
from offpolicy_ktm_pipeline.src.utils import get_default_device, set_global_seed
from offpolicy_ktm_pipeline.src.visualization import (
    plot_objective_comparison,
    plot_behavior_evolution,
    _anchor_indices,
)


# --------------------------------------------------------------------------- #
# Helper                                                                        #
# --------------------------------------------------------------------------- #

def _split_users(
    df: pd.DataFrame,
    train_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by user ID so no student appears in both train and test."""
    rng = np.random.default_rng(seed)
    users = np.array(sorted(df["user"].unique()))
    rng.shuffle(users)
    n_train = max(1, int(math.floor(len(users) * train_fraction)))
    train_users = set(users[:n_train].tolist())
    test_users  = set(users[n_train:].tolist())
    return (
        df[df["user"].isin(train_users)].copy(),
        df[df["user"].isin(test_users)].copy(),
    )


def _eval_split(
    *,
    split_name: str,
    offpolicy_df: pd.DataFrame,
    last_order: int,
    target_theta: np.ndarray,   # fixed target distribution from train
    fit_df: pd.DataFrame,
    named_policies: list,
    delta_min: float,
    delta_max: float,
    dm_delta_grid: int,
    max_weight: float,
    context_bins: int,
    context_ratio_clip: float,
    fit_sigma_floor: float,
    device: torch.device,
) -> list[dict]:
    """Evaluate all named policies on a given offpolicy dataframe slice."""
    eval_df = offpolicy_df[offpolicy_df["order_sequence"].astype(int) <= int(last_order)].copy()
    if eval_df.empty:
        print(f"  [{split_name}] empty — skipping")
        return []

    theta_np = eval_df["proficiency"].to_numpy(dtype=float)
    theta_t  = torch.from_numpy(theta_np.astype(np.float32))
    delta_t  = torch.from_numpy(eval_df["difficulties"].to_numpy(np.float32))
    reward_t = torch.from_numpy(eval_df["reward"].to_numpy(np.float32))

    # Context weights: reweight this split's theta to match the fixed train target
    cw_np = _compute_context_ratio_hist(
        theta_ref=target_theta,
        theta_cur=theta_np,
        theta_eval=theta_np,
        n_bins=context_bins,
        ratio_clip=context_ratio_clip,
    ).astype(np.float32)
    cw_t = torch.from_numpy(cw_np)

    # Behavior log-propensity (train-fitted model applied to this split's data)
    log_prop_b = torch.from_numpy(
        eval_df["log_propensity"].to_numpy(np.float32)
    )

    # Mixture log-propensity
    log_mix_np = compute_log_mixture_propensity(
        theta=theta_np,
        delta=eval_df["difficulties"].to_numpy(dtype=float),
        orders=eval_df["order_sequence"].to_numpy(dtype=int),
        fit_df=fit_df,
        mu_degree=1,
        sigma_degree=2,
        sigma_floor=float(fit_sigma_floor),
    )
    log_mix_t = torch.from_numpy(log_mix_np.astype(np.float32))

    rows = []
    for name, trained_obj, pol in named_policies:
        m = _evaluate_estimators(
            pol,
            theta=theta_t,
            delta=delta_t,
            reward=reward_t,
            context_w=cw_t,
            log_prop_logged=log_prop_b,
            log_prop_mix=log_mix_t,
            delta_min=delta_min,
            delta_max=delta_max,
            dm_delta_grid=dm_delta_grid,
            max_weight=max_weight,
            device=device,
        )
        rows.append({
            "split": split_name,
            "policy": name,
            "trained_objective": trained_obj,
            "n_rows": len(eval_df),
            "ips":  float(m["ips"]),
            "snips": float(m["snips"]),
            "dr":   float(m["dr"]),
            "mis":  float(m["mis"]),
            "dm":   float(m["dm"]),
            "ess_logged": float(m["ess_logged"]),
        })
    return rows


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-ktm",       type=str, required=True,
                        help="Path to existing ktm_dataframe.csv (already preprocessed).")
    parser.add_argument("--output-dir",     type=str, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch-size",     type=int,   default=8192)
    parser.add_argument("--lr",             type=float, default=0.005)
    parser.add_argument("--l2-coef",        type=float, default=1e-6)
    parser.add_argument("--max-weight",     type=float, default=20.0)
    parser.add_argument("--dm-delta-grid",  type=int,   default=121)
    parser.add_argument("--context-bins",   type=int,   default=120)
    parser.add_argument("--context-ratio-clip", type=float, default=20.0)
    parser.add_argument("--fit-sigma-floor",    type=float, default=1e-4)
    parser.add_argument("--policy-sigma-floor", type=float, default=0.2)
    parser.add_argument("--optimal-sigma",      type=float, default=0.2)
    parser.add_argument("--min-obs-per-order",  type=int,   default=100)
    parser.add_argument("--objectives",     type=str, nargs="+",
                        default=["ips", "snips", "dr", "mis"],
                        choices=["ips", "snips", "dr", "mis"])
    parser.add_argument("--device",         type=str, default=None)
    parser.add_argument("--dataset-name",   type=str, default="",
                        help="Human-readable dataset label for the report.")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device   = get_default_device(args.device)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # ------------------------------------------------------------------ #
    # 1. Load KTM dataframe and split students                            #
    # ------------------------------------------------------------------ #
    print(f"Loading ktm: {args.base_ktm}")
    df_ktm = pd.read_csv(args.base_ktm)
    print(f"  total rows={len(df_ktm)}, users={df_ktm['user'].nunique()}")

    delta_min = float(df_ktm["difficulties"].min())
    delta_max = float(df_ktm["difficulties"].max())

    train_df, test_df = _split_users(df_ktm, args.train_fraction, args.seed)
    print(f"  train: {train_df['user'].nunique()} users, {len(train_df)} rows")
    print(f"  test:  {test_df['user'].nunique()} users,  {len(test_df)} rows")

    pd.DataFrame({"split": ["train", "test"],
                  "n_users": [train_df["user"].nunique(), test_df["user"].nunique()],
                  "n_rows":  [len(train_df), len(test_df)]}).to_csv(
        out_dir / "split_summary.csv", index=False)

    # ------------------------------------------------------------------ #
    # 2. Fit behavior policy on TRAIN only                                #
    # ------------------------------------------------------------------ #
    print("Fitting behavior policy on train...")
    fit_df = fit_gaussian_by_order(
        train_df,
        order_col="order_sequence",
        theta_col="proficiency",
        delta_col="difficulties",
        mu_degree=1,
        sigma_degree=2,
        sigma_floor=float(args.fit_sigma_floor),
        min_obs_per_order=int(args.min_obs_per_order),
        distribution="truncated_gaussian",
        delta_min=delta_min,
        delta_max=delta_max,
    )
    if fit_df.empty:
        raise ValueError("No behavior fit rows — decrease --min-obs-per-order")
    fit_df.to_csv(out_dir / "gaussian_fit_by_order.csv", index=False)

    # ------------------------------------------------------------------ #
    # 3. Build offpolicy dataset from TRAIN (propensities from train fit) #
    # ------------------------------------------------------------------ #
    print("Building offpolicy dataset from train...")
    offpolicy_train = build_offpolicy_dataset(
        train_df, fit_df, sigma_floor=float(args.fit_sigma_floor)
    )

    # Build offpolicy dataset for TEST using the SAME train-fitted behavior
    print("Building offpolicy dataset from test (using train-fitted behavior)...")
    offpolicy_test = build_offpolicy_dataset(
        test_df, fit_df, sigma_floor=float(args.fit_sigma_floor)
    )

    # ------------------------------------------------------------------ #
    # 4. Select target round and fix context distribution from TRAIN      #
    # ------------------------------------------------------------------ #
    target_order, last_order, first_half, best_js = _select_target_round(
        offpolicy_train,
        order_col="order_sequence",
        theta_col="proficiency",
        n_bins=int(args.context_bins),
    )
    print(f"  target_order={target_order}, last_order={last_order}")

    # Fixed target context theta from TRAIN — used for both train and test evaluation
    target_theta = offpolicy_train[
        offpolicy_train["order_sequence"].astype(int) == int(target_order)
    ]["proficiency"].to_numpy(dtype=float)
    print(f"  target context: n={len(target_theta)}, mean={target_theta.mean():.3f}")

    # ------------------------------------------------------------------ #
    # 5. Build training tensors (cumulative, context-reweighted)          #
    # ------------------------------------------------------------------ #
    train_eval_df = offpolicy_train[
        offpolicy_train["order_sequence"].astype(int) <= int(last_order)
    ].copy()

    theta_tr = train_eval_df["proficiency"].to_numpy(dtype=float)
    cw_train = _compute_context_ratio_hist(
        theta_ref=target_theta,
        theta_cur=theta_tr,
        theta_eval=theta_tr,
        n_bins=int(args.context_bins),
        ratio_clip=float(args.context_ratio_clip),
    ).astype(np.float32)
    train_eval_df["context_weight"] = cw_train

    log_mix_train = compute_log_mixture_propensity(
        theta=theta_tr,
        delta=train_eval_df["difficulties"].to_numpy(dtype=float),
        orders=train_eval_df["order_sequence"].to_numpy(dtype=int),
        fit_df=fit_df,
        mu_degree=1,
        sigma_degree=2,
        sigma_floor=float(args.fit_sigma_floor),
    )

    theta_t  = torch.from_numpy(theta_tr.astype(np.float32))
    delta_t  = torch.from_numpy(train_eval_df["difficulties"].to_numpy(np.float32))
    reward_t = torch.from_numpy(train_eval_df["reward"].to_numpy(np.float32))
    cw_t     = torch.from_numpy(cw_train)
    lb_t     = torch.from_numpy(train_eval_df["log_propensity"].to_numpy(np.float32))
    lm_t     = torch.from_numpy(log_mix_train.astype(np.float32))

    # ------------------------------------------------------------------ #
    # 6. Train policies on TRAIN                                           #
    # ------------------------------------------------------------------ #
    fit_last = fit_df[fit_df["order_sequence"].astype(int) == int(last_order)]
    init_row = fit_last.iloc[0] if not fit_last.empty else fit_df.sort_values("order_sequence").iloc[-1]

    learned_policies: dict[str, object] = {}
    all_histories   = []
    all_coef_rows   = []

    for obj in args.objectives:
        print(f"Training {obj} ...")
        policy, hist_df = _train_last_round_policy(
            objective=obj,
            theta=theta_t,
            delta=delta_t,
            reward=reward_t,
            context_w=cw_t,
            log_prop_logged=lb_t,
            log_prop_mix=lm_t,
            delta_min=delta_min,
            delta_max=delta_max,
            policy_sigma_floor=float(args.policy_sigma_floor),
            init_row=init_row,
            dm_delta_grid=int(args.dm_delta_grid),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            l2_coef=float(args.l2_coef),
            max_weight=float(args.max_weight),
            device=device,
        )
        learned_policies[obj] = policy
        all_histories.append(hist_df)
        with torch.no_grad():
            all_coef_rows.append({
                "objective":    obj,
                "beta_mu_0":    float(policy.beta_mu[0].item()),
                "beta_mu_1":    float(policy.beta_mu[1].item()),
                "beta_sigma_0": float(policy.beta_sigma[0].item()),
                "beta_sigma_1": float(policy.beta_sigma[1].item()),
                "beta_sigma_2": float(policy.beta_sigma[2].item()),
            })

    pd.concat(all_histories, ignore_index=True).to_csv(
        out_dir / "training_history.csv", index=False)
    coeff_df = pd.DataFrame(all_coef_rows)
    coeff_df.to_csv(out_dir / "learned_policy_coefficients.csv", index=False)

    # ------------------------------------------------------------------ #
    # 7. Evaluate on TRAIN and TEST                                        #
    # ------------------------------------------------------------------ #
    behavior_policy = _build_behavior_policy_from_row(
        init_row, delta_min=delta_min, delta_max=delta_max,
        sigma_floor=float(args.fit_sigma_floor), device=device,
    )
    optimal_policy = IRTOptimalGaussianPolicy(
        delta_min=delta_min, delta_max=delta_max, sigma=float(args.optimal_sigma),
    )

    named_policies = (
        [("behavior", "n/a", behavior_policy), ("optimal_irt", "n/a", optimal_policy)]
        + [(f"learned_{obj}", obj, pol.eval()) for obj, pol in learned_policies.items()]
    )

    eval_kwargs = dict(
        last_order=last_order,
        target_theta=target_theta,
        fit_df=fit_df,
        named_policies=named_policies,
        delta_min=delta_min,
        delta_max=delta_max,
        dm_delta_grid=int(args.dm_delta_grid),
        max_weight=float(args.max_weight),
        context_bins=int(args.context_bins),
        context_ratio_clip=float(args.context_ratio_clip),
        fit_sigma_floor=float(args.fit_sigma_floor),
        device=device,
    )

    print("Evaluating on train split...")
    train_rows = _eval_split(split_name="train", offpolicy_df=offpolicy_train, **eval_kwargs)
    print("Evaluating on test split...")
    test_rows  = _eval_split(split_name="test",  offpolicy_df=offpolicy_test,  **eval_kwargs)

    all_rows = train_rows + test_rows
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(out_dir / "estimator_results_train_test.csv", index=False)

    # ------------------------------------------------------------------ #
    # 8. Plots                                                             #
    # ------------------------------------------------------------------ #
    theta_grid = np.linspace(
        float(df_ktm["proficiency"].min()),
        float(df_ktm["proficiency"].max()),
        400,
    )
    _irt = IRTOptimalGaussianPolicy(delta_min=delta_min, delta_max=delta_max, sigma=float(args.optimal_sigma))
    with torch.no_grad():
        optimal_delta = _irt._optimal_delta(
            torch.from_numpy(theta_grid.astype(np.float32))
        ).numpy().astype(float)

    fig1, _ = plot_objective_comparison(
        theta_grid=theta_grid,
        fit_df=fit_df,
        coeff_df=coeff_df,
        delta_min=delta_min,
        delta_max=delta_max,
        sigma_floor=float(args.policy_sigma_floor),
        optimal_delta=optimal_delta,
        title=f"Learned Policies vs Behavior — {args.dataset_name or Path(args.base_ktm).parent.name}",
    )
    fig1.savefig(str(out_dir / "policy_comparison_by_objective.png"), dpi=170)
    plt.close(fig1)

    fig2, _ = plot_behavior_evolution(
        theta_grid=theta_grid,
        fit_df=fit_df,
        delta_min=delta_min,
        delta_max=delta_max,
        sigma_floor=float(args.fit_sigma_floor),
        title="Behavior Policy Evolution (train data)",
    )
    fig2.savefig(str(out_dir / "behavior_evolution_by_round.png"), dpi=170)
    plt.close(fig2)

    # ------------------------------------------------------------------ #
    # 9. HTML report                                                       #
    # ------------------------------------------------------------------ #
    _write_report(
        out_dir=out_dir,
        results_df=results_df,
        dataset_name=args.dataset_name or Path(args.base_ktm).parent.name,
        n_train_users=train_df["user"].nunique(),
        n_test_users=test_df["user"].nunique(),
        n_train_rows=len(train_df),
        n_test_rows=len(test_df),
        train_fraction=args.train_fraction,
        target_order=target_order,
        last_order=last_order,
        delta_min=delta_min,
        delta_max=delta_max,
    )

    total = time.perf_counter() - t0
    print(f"\nDone in {total:.1f}s  ->  {out_dir}/")


# --------------------------------------------------------------------------- #
# Report generation                                                             #
# --------------------------------------------------------------------------- #

def _fmt(v: float) -> str:
    return f"{v:.3f}"

def _row_html(row: pd.Series, best_per_metric: dict[str, float], is_baseline: bool) -> str:
    policy_label = {
        "behavior": "Behavior", "optimal_irt": "Optimal (IRT)",
        "learned_ips": "Learned (IPS)", "learned_snips": "Learned (SNIPS)",
        "learned_dr": "Learned (DR)", "learned_mis": "Learned (MIS)",
    }.get(row["policy"], row["policy"])
    obj_label = row.get("trained_objective", "—")
    if obj_label in (None, "n/a", float("nan"), "nan"):
        obj_label = "—"

    sep = ' class="sep"' if is_baseline else ""
    cells = f'<td>{policy_label}</td><td>{obj_label}</td>'
    for m in ["ips", "snips", "dr", "mis", "dm"]:
        v = float(row[m])
        bold = (not is_baseline) and abs(v - best_per_metric.get(m, -999)) < 1e-6
        cls = ' class="best"' if bold else ""
        cells += f'<td{cls}>{_fmt(v)}</td>'
    cells += f'<td>{int(row["ess_logged"]):,}</td>'
    return f'  <tr{sep}>{cells}</tr>'


def _write_report(
    *,
    out_dir: Path,
    results_df: pd.DataFrame,
    dataset_name: str,
    n_train_users: int,
    n_test_users: int,
    n_train_rows: int,
    n_test_rows: int,
    train_fraction: float,
    target_order: int,
    last_order: int,
    delta_min: float,
    delta_max: float,
) -> None:
    METRICS = ["ips", "snips", "dr", "mis", "dm"]

    def _table_html(split: str) -> str:
        df_s = results_df[results_df["split"] == split].copy()
        learned_mask = df_s["policy"].str.startswith("learned")
        best: dict[str, float] = {}
        for m in METRICS:
            learned_vals = df_s.loc[learned_mask, m].dropna()
            if not learned_vals.empty:
                best[m] = float(learned_vals.max())
        rows_html = []
        for _, row in df_s.iterrows():
            is_sep = row["policy"] == "learned_ips"
            rows_html.append(_row_html(row, best, is_baseline=not learned_mask[row.name]))
        return "\n".join(rows_html)

    train_table = _table_html("train")
    test_table  = _table_html("test")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Train/Test Split — {dataset_name}</title>
<style>
  body {{ font-family: sans-serif; max-width: 1200px; margin: 40px auto; color: #222; }}
  h1, h2, h3 {{ border-bottom: 1px solid #ccc; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; margin: 16px 0; }}
  th, td {{ border: 1px solid #bbb; padding: 7px 14px; text-align: right; }}
  th {{ background: #f0f0f0; }}
  td:first-child, td:nth-child(2), th:first-child, th:nth-child(2) {{ text-align: left; }}
  .best {{ font-weight: bold; }}
  .sep {{ border-top: 2px solid #888; }}
  img {{ max-width: 100%; margin: 12px 0; border: 1px solid #ddd; border-radius: 4px; }}
  .meta {{ color: #555; font-size: 14px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .note {{ background: #fffbe6; border: 1px solid #f0c040; border-radius: 4px; padding: 12px 16px; margin: 12px 0; font-size: 14px; }}
</style>
</head>
<body>

<h1>Train/Test Split Evaluation — <code>{dataset_name}</code></h1>
<p class="meta">
  Train fraction: {train_fraction:.0%} &nbsp;|&nbsp;
  Train: {n_train_users:,} students / {n_train_rows:,} interactions &nbsp;|&nbsp;
  Test: {n_test_users:,} students / {n_test_rows:,} interactions &nbsp;|&nbsp;
  Last round: {last_order} &nbsp;|&nbsp;
  Target context round: {target_order} &nbsp;|&nbsp;
  Difficulty range: [{delta_min:.3f}, {delta_max:.3f}]
</p>

<div class="note">
  <strong>Design:</strong> Behavior policy and learned policies are fitted on the <strong>train split only</strong>.
  The target context distribution &theta;<sub>target</sub> is fixed from the train set at round {target_order}.
  Both splits are then evaluated using the same trained policies and the same fixed target context,
  enabling a clean out-of-sample test.
</div>

<h2>Results — Train Split</h2>
<p>Bold = best among learned policies per estimator.</p>
<table>
  <thead>
    <tr><th>Policy</th><th>Trained Obj.</th><th>IPS</th><th>SNIPS</th><th>DR</th><th>MIS</th><th>DM</th><th>ESS (logged)</th></tr>
  </thead>
  <tbody>
{train_table}
  </tbody>
</table>

<h2>Results — Test Split (out-of-sample)</h2>
<p>
  Policies are <strong>not re-trained</strong> on test data.
  Behavior log-propensities on the test set are computed using the train-fitted behavior model.
  Context weights use the same fixed target distribution from train.
  Bold = best among learned policies per estimator.
</p>
<table>
  <thead>
    <tr><th>Policy</th><th>Trained Obj.</th><th>IPS</th><th>SNIPS</th><th>DR</th><th>MIS</th><th>DM</th><th>ESS (logged)</th></tr>
  </thead>
  <tbody>
{test_table}
  </tbody>
</table>

<h2>Discussion</h2>
<p>
  If learned policies genuinely improve over the behavior policy, this should hold on both
  the train and test splits. A policy that only wins on train but not test may be overfitting
  to the importance weights or the training distribution.
  The DM estimator is particularly informative here: since it does not rely on importance weights,
  it provides a direct model-based estimate that is less susceptible to train/test distribution shift
  in the behavior policy estimate.
</p>

<h2>Policy Comparison by Objective</h2>
<img src="policy_comparison_by_objective.png" alt="Policy comparison">

<h2>Behavior Policy Evolution (Train Data)</h2>
<img src="behavior_evolution_by_round.png" alt="Behavior evolution">

</body>
</html>
"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")
    print(f"Report saved: {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
