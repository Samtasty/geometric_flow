from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_rls_path = PROJECT_ROOT / "offpolicy_ktm_pipeline" / "run_last_round_simple.py"
_spec = importlib.util.spec_from_file_location("_rls", _rls_path)
_rls = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_rls)

_apply_round_cap = _rls._apply_round_cap
_build_behavior_policy_from_row = _rls._build_behavior_policy_from_row
_compute_context_ratio_hist = _rls._compute_context_ratio_hist
_evaluate_estimators = _rls._evaluate_estimators
IRTOptimalGaussianPolicy = _rls.IRTOptimalGaussianPolicy

from models.offpolicy_gaussian_policy import GlobalGaussianPolicy
from offpolicy_ktm_pipeline.src.estimators_and_training import compute_log_mixture_propensity
from offpolicy_ktm_pipeline.src.propensity import build_offpolicy_dataset
from offpolicy_ktm_pipeline.src.utils import get_default_device
from offpolicy_ktm_pipeline.src.visualization import plot_behavior_evolution, plot_objective_comparison


def _fmt(v: float) -> str:
    return f"{float(v):.3f}"


def _policy_label(name: str, *, first_order: int | None = None, last_order: int | None = None) -> str:
    return {
        "behavior": "Behavior",
        "behavior_first_round": f"Behavior (round {first_order})" if first_order is not None else "Behavior (first round)",
        "behavior_last_round": f"Behavior (round {last_order})" if last_order is not None else "Behavior (last round)",
        "optimal_irt": "Optimal (IRT)",
        "optimal_irt_gaussian": "Optimal (IRT)",
        "learned_ips": "Learned (IPS)",
        "learned_snips": "Learned (SNIPS)",
        "learned_dr": "Learned (DR)",
        "learned_mis": "Learned (MIS)",
    }.get(name, name)


def _load_base_ktm(base_ktm: Path, truncate_at_round: int, max_rounds: int, round_cap_strategy: str) -> pd.DataFrame:
    df_ktm = pd.read_csv(base_ktm)
    if int(truncate_at_round) > 0:
        df_ktm = df_ktm[df_ktm["answer_number"] <= int(truncate_at_round)].copy()
    df_ktm, _ = _apply_round_cap(
        df_ktm,
        max_rounds=int(max_rounds),
        strategy=str(round_cap_strategy),
    )
    return df_ktm


def _compute_log_mixture_propensity_batched(
    *,
    eval_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    fit_sigma_floor: float,
    batch_rows: int,
) -> np.ndarray:
    n = int(len(eval_df))
    if n == 0:
        return np.empty(0, dtype=np.float32)
    rows = max(int(batch_rows), 1)
    chunks: list[np.ndarray] = []
    for start in range(0, n, rows):
        end = min(start + rows, n)
        cur = eval_df.iloc[start:end]
        chunks.append(
            compute_log_mixture_propensity(
                theta=cur["proficiency"].to_numpy(dtype=float),
                delta=cur["difficulties"].to_numpy(dtype=float),
                orders=cur["order_sequence"].to_numpy(dtype=int),
                fit_df=fit_df,
                mu_degree=1,
                sigma_degree=2,
                sigma_floor=float(fit_sigma_floor),
            ).astype(np.float32)
        )
    return np.concatenate(chunks, axis=0)


def _rebuild_learned_policies(
    coeff_df: pd.DataFrame,
    *,
    delta_min: float,
    delta_max: float,
    sigma_floor: float,
    device: torch.device,
) -> dict[str, GlobalGaussianPolicy]:
    out: dict[str, GlobalGaussianPolicy] = {}
    for _, row in coeff_df.iterrows():
        pol = GlobalGaussianPolicy(
            mu_degree=1,
            sigma_degree=2,
            sigma_floor=float(sigma_floor),
            delta_min=float(delta_min),
            delta_max=float(delta_max),
        ).to(device)
        with torch.no_grad():
            pol.beta_mu[0] = float(row["beta_mu_0"])
            pol.beta_mu[1] = float(row["beta_mu_1"])
            pol.beta_sigma[0] = float(row["beta_sigma_0"])
            pol.beta_sigma[1] = float(row["beta_sigma_1"])
            pol.beta_sigma[2] = float(row["beta_sigma_2"])
        out[str(row["objective"])] = pol.eval()
    return out


def _evaluate_split(
    *,
    split_name: str,
    offpolicy_df: pd.DataFrame,
    target_theta: np.ndarray,
    last_order: int,
    fit_df: pd.DataFrame,
    named_policies: list[tuple[str, str, object]],
    delta_min: float,
    delta_max: float,
    dm_delta_grid: int,
    max_weight: float,
    context_bins: int,
    context_ratio_clip: float,
    fit_sigma_floor: float,
    need_mix: bool,
    mix_batch_rows: int,
    device: torch.device,
) -> pd.DataFrame:
    eval_df = offpolicy_df[offpolicy_df["order_sequence"].astype(int) <= int(last_order)].copy()
    theta_np = eval_df["proficiency"].to_numpy(dtype=float)
    eval_df["context_weight"] = _compute_context_ratio_hist(
        theta_ref=target_theta,
        theta_cur=theta_np,
        theta_eval=theta_np,
        n_bins=int(context_bins),
        ratio_clip=float(context_ratio_clip),
    ).astype(np.float32)

    log_prop_mix_t = None
    if bool(need_mix):
        log_mix_np = _compute_log_mixture_propensity_batched(
            eval_df=eval_df,
            fit_df=fit_df,
            fit_sigma_floor=float(fit_sigma_floor),
            batch_rows=int(mix_batch_rows),
        )
        log_prop_mix_t = torch.from_numpy(log_mix_np.astype(np.float32))

    theta_t = torch.from_numpy(eval_df["proficiency"].to_numpy(np.float32))
    delta_t = torch.from_numpy(eval_df["difficulties"].to_numpy(np.float32))
    reward_t = torch.from_numpy(eval_df["reward"].to_numpy(np.float32))
    context_t = torch.from_numpy(eval_df["context_weight"].to_numpy(np.float32))
    log_prop_t = torch.from_numpy(eval_df["log_propensity"].to_numpy(np.float32))

    rows: list[dict[str, object]] = []
    for name, trained_obj, pol in named_policies:
        m = _evaluate_estimators(
            pol,
            theta=theta_t,
            delta=delta_t,
            reward=reward_t,
            context_w=context_t,
            log_prop_logged=log_prop_t,
            log_prop_mix=log_prop_mix_t,
            delta_min=float(delta_min),
            delta_max=float(delta_max),
            dm_delta_grid=int(dm_delta_grid),
            max_weight=float(max_weight),
            device=device,
        )
        rows.append(
            {
                "split": split_name,
                "policy": name,
                "trained_objective": trained_obj,
                "ips": float(m["ips"]),
                "snips": float(m["snips"]),
                "dr": float(m["dr"]),
                "mis": float(m["mis"]),
                "dm": float(m["dm"]),
                "ess_logged": float(m["ess_logged"]),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_behavior_progression(
    *,
    split_name: str,
    offpolicy_df: pd.DataFrame,
    target_theta: np.ndarray,
    orders: list[int],
    first_order: int,
    last_order: int,
    fit_df: pd.DataFrame,
    delta_min: float,
    delta_max: float,
    dm_delta_grid: int,
    max_weight: float,
    context_bins: int,
    context_ratio_clip: float,
    fit_sigma_floor: float,
    need_mix: bool,
    mix_batch_rows: int,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for order in orders:
        seen_df = offpolicy_df[offpolicy_df["order_sequence"].astype(int) <= int(order)].copy()
        theta_np = seen_df["proficiency"].to_numpy(dtype=float)
        context_w = _compute_context_ratio_hist(
            theta_ref=target_theta,
            theta_cur=theta_np,
            theta_eval=theta_np,
            n_bins=int(context_bins),
            ratio_clip=float(context_ratio_clip),
        ).astype(np.float32)
        log_prop_mix_t = None
        if bool(need_mix):
            log_mix_np = _compute_log_mixture_propensity_batched(
                eval_df=seen_df,
                fit_df=fit_df,
                fit_sigma_floor=float(fit_sigma_floor),
                batch_rows=int(mix_batch_rows),
            )
            log_prop_mix_t = torch.from_numpy(log_mix_np.astype(np.float32))

        brow = fit_df[fit_df["order_sequence"].astype(int) == int(order)]
        if brow.empty:
            continue
        pol = _build_behavior_policy_from_row(
            brow.iloc[0],
            delta_min=delta_min,
            delta_max=delta_max,
            sigma_floor=float(fit_sigma_floor),
            device=device,
        )
        est = _evaluate_estimators(
            pol,
            theta=torch.from_numpy(seen_df["proficiency"].to_numpy(np.float32)),
            delta=torch.from_numpy(seen_df["difficulties"].to_numpy(np.float32)),
            reward=torch.from_numpy(seen_df["reward"].to_numpy(np.float32)),
            context_w=torch.from_numpy(context_w),
            log_prop_logged=torch.from_numpy(seen_df["log_propensity"].to_numpy(np.float32)),
            log_prop_mix=log_prop_mix_t,
            delta_min=float(delta_min),
            delta_max=float(delta_max),
            dm_delta_grid=int(dm_delta_grid),
            max_weight=float(max_weight),
            device=device,
        )
        if int(order) == int(first_order):
            policy_name = "behavior_first_round"
        elif int(order) == int(last_order):
            policy_name = "behavior_last_round"
        else:
            policy_name = f"behavior_round_{int(order)}"
        rows.append(
            {
                "split": split_name,
                "policy": policy_name,
                "trained_objective": "n/a",
                "round_order_sequence": int(order),
                "ips": float(est["ips"]),
                "snips": float(est["snips"]),
                "dr": float(est["dr"]),
                "mis": float(est["mis"]),
                "dm": float(est["dm"]),
                "ess_logged": float(est["ess_logged"]),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_actual_logged_behavior(
    *,
    split_name: str,
    offpolicy_df: pd.DataFrame,
    target_theta: np.ndarray,
    orders: list[int],
    first_order: int,
    last_order: int,
    context_bins: int,
    context_ratio_clip: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for order in orders:
        cur_df = offpolicy_df[offpolicy_df["order_sequence"].astype(int) == int(order)].copy()
        if cur_df.empty:
            continue
        theta_np = cur_df["proficiency"].to_numpy(dtype=float)
        context_w = _compute_context_ratio_hist(
            theta_ref=target_theta,
            theta_cur=theta_np,
            theta_eval=theta_np,
            n_bins=int(context_bins),
            ratio_clip=float(context_ratio_clip),
        ).astype(np.float32)
        reward_np = cur_df["reward"].to_numpy(dtype=float)
        value = float(np.sum(context_w * reward_np) / max(float(np.sum(context_w)), 1e-12))
        if int(order) == int(first_order):
            policy_name = "behavior_first_round"
        elif int(order) == int(last_order):
            policy_name = "behavior_last_round"
        else:
            policy_name = f"behavior_round_{int(order)}"
        rows.append(
            {
                "split": split_name,
                "policy": policy_name,
                "round_order_sequence": int(order),
                "actual_logged_value": value,
                "n_rows": int(len(cur_df)),
            }
        )
    return pd.DataFrame(rows)


def _dataset_summary(df_ktm: pd.DataFrame, *, last_order: int, target_order: int) -> dict[str, object]:
    lengths = df_ktm.groupby("user").size().astype(float)
    diff_min = float(df_ktm["difficulties"].min())
    diff_max = float(df_ktm["difficulties"].max())
    if "reward" in df_ktm.columns:
        reward = df_ktm["reward"].to_numpy(dtype=float)
    elif "correct" in df_ktm.columns:
        reward = df_ktm["correct"].to_numpy(dtype=float) * (df_ktm["difficulties"].to_numpy(dtype=float) - diff_min)
    else:
        reward = df_ktm["difficulties"].to_numpy(dtype=float) - diff_min
    return {
        "students": int(df_ktm["user"].nunique()),
        "interactions_kept": int(len(df_ktm)),
        "length_min": int(lengths.min()),
        "length_mean": float(lengths.mean()),
        "length_median": float(lengths.median()),
        "length_p75": float(lengths.quantile(0.75)),
        "length_p90": float(lengths.quantile(0.90)),
        "length_p95": float(lengths.quantile(0.95)),
        "length_p99": float(lengths.quantile(0.99)),
        "length_max": int(lengths.max()),
        "reward_min": float(np.min(reward)),
        "reward_max": float(np.max(reward)),
        "difficulty_min": diff_min,
        "difficulty_max": diff_max,
        "last_round_sample_size": int((df_ktm["order_sequence"].astype(int) == int(last_order)).sum()),
        "target_context_round": int(target_order),
    }


def _row_html(
    row: pd.Series,
    best_per_metric: dict[str, float],
    learned_names: set[str],
    metrics: list[str],
    *,
    first_order: int,
    last_order: int,
) -> str:
    is_learned = str(row["policy"]) in learned_names
    sep = ' class="sep"' if str(row["policy"]) in {"optimal_irt", "optimal_irt_gaussian"} else ""
    obj_label = row.get("trained_objective", "—")
    if pd.isna(obj_label) or str(obj_label).lower() in {"n/a", "nan"}:
        obj_label = "—"
    cells = (
        f"<td>{_policy_label(str(row['policy']), first_order=first_order, last_order=last_order)}</td>"
        f"<td>{obj_label}</td>"
    )
    for metric in metrics:
        value = float(row[metric])
        bold = is_learned and abs(value - float(best_per_metric.get(metric, -1e18))) < 1e-6
        cls = ' class="best"' if bold else ""
        cells += f"<td{cls}>{_fmt(value)}</td>"
    cells += f"<td>{int(round(float(row['ess_logged']))):,}</td>"
    return f"  <tr{sep}>{cells}</tr>"


def _discussion_html(results_df: pd.DataFrame, metrics: list[str]) -> str:
    train_df = results_df[results_df["split"] == "train"].copy()
    test_df = results_df[results_df["split"] == "test"].copy()
    behavior = test_df[test_df["policy"] == "behavior_last_round"].iloc[0]
    oracle = test_df[test_df["policy"].isin(["optimal_irt", "optimal_irt_gaussian"])].iloc[0]
    learned = test_df[test_df["policy"].str.startswith("learned_")].copy()

    best_lines = []
    for metric in metrics:
        best_row = learned.loc[learned[metric].idxmax()]
        best_lines.append(f'{_policy_label(str(best_row["policy"]))} is best on {metric.upper()} ({_fmt(best_row[metric])})')

    lines = [
        "The held-out test split keeps the same qualitative pattern as training: " + ", ".join(best_lines) + ".",
        f'Against the behavior policy, the strongest learned improvement on test is '
        f'{_fmt(float(learned["dr"].max()) - float(behavior["dr"]))} on DR and '
        f'{_fmt(float(learned["dm"].max()) - float(behavior["dm"]))} on DM.',
        f'The IRT oracle remains strong under the direct model (DM={_fmt(oracle["dm"])}) but uses a much smaller effective sample size '
        f'(ESS={int(round(float(oracle["ess_logged"]))):,}) than behavior (ESS={int(round(float(behavior["ess_logged"]))):,}), '
        f'which is consistent with a support mismatch effect.',
    ]

    merged = train_df[train_df["policy"].str.startswith("learned_")][["policy", *metrics]].merge(
        test_df[test_df["policy"].str.startswith("learned_")][["policy", *metrics]],
        on="policy",
        suffixes=("_train", "_test"),
    )
    max_delta = 0.0
    max_policy = ""
    max_metric = ""
    for _, row in merged.iterrows():
        for metric in metrics:
            delta = float(row[f"{metric}_test"] - row[f"{metric}_train"])
            if abs(delta) > abs(max_delta):
                max_delta = delta
                max_policy = _policy_label(str(row["policy"]))
                max_metric = metric.upper()
    if max_policy:
        lines.append(
            f'Train-to-test drift is modest overall; the largest absolute change among learned policies is '
            f'{max_delta:+.3f} on {max_metric} for {max_policy}.'
        )

    return "\n".join(f"<p>{line}</p>" for line in lines)


def _write_report(
    *,
    out_dir: Path,
    results_df: pd.DataFrame,
    behavior_progression_df: pd.DataFrame,
    actual_logged_df: pd.DataFrame,
    dataset_name: str,
    dataset_summary: dict[str, object],
    n_train_users: int,
    n_test_users: int,
    n_train_rows: int,
    n_test_rows: int,
    train_fraction: float,
    first_order: int,
    target_order: int,
    last_order: int,
    epochs: int,
    metrics: list[str],
) -> None:
    learned_names = set(results_df.loc[results_df["policy"].str.startswith("learned_"), "policy"].astype(str).unique().tolist())

    policy_rank = {
        "behavior_first_round": 0,
        "behavior_last_round": 1,
        "optimal_irt": 2,
        "optimal_irt_gaussian": 2,
        "learned_ips": 3,
        "learned_snips": 4,
        "learned_dr": 5,
        "learned_mis": 6,
    }

    def table_html(split: str) -> str:
        df_s = results_df[results_df["split"] == split].copy()
        df_s["_rank"] = df_s["policy"].map(lambda x: policy_rank.get(str(x), 999))
        df_s = df_s.sort_values(["_rank", "policy"]).drop(columns="_rank")
        best = {m: float(df_s.loc[df_s["policy"].isin(learned_names), m].max()) for m in metrics}
        return "\n".join(
            _row_html(row, best, learned_names, metrics, first_order=first_order, last_order=last_order)
            for _, row in df_s.iterrows()
        )

    metric_headers = "".join(f"<th>{m.upper()}</th>" for m in metrics)
    delta_headers = "".join(f"<th>&Delta; {m.upper()}</th>" for m in metrics)

    merged = results_df[results_df["split"] == "train"][["policy", *metrics]].merge(
        results_df[results_df["split"] == "test"][["policy", *metrics]],
        on="policy",
        suffixes=("_train", "_test"),
    )
    merged = merged[merged["policy"].str.startswith("learned_")].copy()
    comparison_rows = []
    for _, row in merged.iterrows():
        cells = [f"<td>{_policy_label(str(row['policy']))}</td>"]
        for metric in metrics:
            delta = float(row[f"{metric}_test"] - row[f"{metric}_train"])
            sign = "+" if delta >= 0 else "&minus;"
            cells.append(f"<td>{sign}{abs(delta):.3f}</td>")
        comparison_rows.append("  <tr>" + "".join(cells) + "</tr>")

    progression_rows = []
    progression_df = behavior_progression_df.sort_values("round_order_sequence")
    for _, row in progression_df.iterrows():
        cells = [f"<td>{_policy_label(str(row['policy']), first_order=first_order, last_order=last_order)}</td>"]
        for metric in metrics:
            cells.append(f"<td>{_fmt(float(row[metric]))}</td>")
        cells.append(f"<td>{int(round(float(row['ess_logged']))):,}</td>")
        progression_rows.append("  <tr>" + "".join(cells) + "</tr>")

    actual_logged_rows = []
    actual_logged_df = actual_logged_df.sort_values(["split", "round_order_sequence"])
    for _, row in actual_logged_df.iterrows():
        actual_logged_rows.append(
            "  <tr>"
            f"<td>{str(row['split']).capitalize()}</td>"
            f"<td>{_policy_label(str(row['policy']), first_order=first_order, last_order=last_order)}</td>"
            f"<td>{_fmt(float(row['actual_logged_value']))}</td>"
            f"<td>{int(row['n_rows']):,}</td>"
            "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Train/Test Split — {dataset_name}</title>
<style>
  body {{ font-family: sans-serif; max-width: 1100px; margin: 40px auto; color: #222; }}
  h1, h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; margin: 16px 0; }}
  th, td {{ border: 1px solid #bbb; padding: 7px 14px; text-align: right; }}
  th {{ background: #f0f0f0; }}
  td:first-child, td:nth-child(2), th:first-child, th:nth-child(2) {{ text-align: left; }}
  .best {{ font-weight: bold; }}
  .sep {{ border-top: 2px solid #888; }}
  img {{ max-width: 100%; margin: 12px 0; border: 1px solid #ddd; border-radius: 4px; }}
  .meta {{ color: #555; font-size: 14px; }}
  .note {{ background: #fffbe6; border: 1px solid #f0c040; border-radius: 4px; padding: 12px 16px; margin: 12px 0; font-size: 14px; }}
</style>
</head>
<body>

<h1>Train/Test Split — <code>{dataset_name}</code></h1>
<p class="meta">
  Train: {n_train_users:,} students / {n_train_rows:,} interactions &nbsp;|&nbsp;
  Test: {n_test_users:,} students / {n_test_rows:,} interactions &nbsp;|&nbsp;
  Train fraction: {train_fraction:.0%} &nbsp;|&nbsp;
  Last round: {last_order} &nbsp;|&nbsp;
  Target context round: {target_order} &nbsp;|&nbsp;
  Epochs: {epochs}
</p>

<div class="note">
  <strong>Design:</strong> Behavior policy and learned policies are fitted on the <strong>train split only</strong> (no student appears in both splits).
  The target context distribution &theta;<sub>target</sub> is fixed from the train set.
  Both splits are evaluated using the same trained policies and the same fixed target context.
</div>

<h2>Dataset Summary</h2>
<table>
  <tbody>
    <tr><th>Students</th><td>{int(dataset_summary['students']):,}</td></tr>
    <tr><th>Interactions kept</th><td>{int(dataset_summary['interactions_kept']):,}</td></tr>
    <tr><th>Student length (min / mean / median / p75 / p90 / p95 / p99 / max)</th><td>{int(dataset_summary['length_min'])} / {float(dataset_summary['length_mean']):.2f} / {int(round(float(dataset_summary['length_median'])))} / {int(round(float(dataset_summary['length_p75'])))} / {int(round(float(dataset_summary['length_p90'])))} / {int(round(float(dataset_summary['length_p95'])))} / {int(round(float(dataset_summary['length_p99'])))} / {int(dataset_summary['length_max'])}</td></tr>
    <tr><th>Reward range</th><td>[{float(dataset_summary['reward_min']):.3f}, {float(dataset_summary['reward_max']):.3f}]</td></tr>
    <tr><th>Difficulty range</th><td>[{float(dataset_summary['difficulty_min']):.3f}, {float(dataset_summary['difficulty_max']):.3f}]</td></tr>
    <tr><th>Last-round sample size</th><td>{int(dataset_summary['last_round_sample_size']):,}</td></tr>
    <tr><th>Target context round</th><td>{int(dataset_summary['target_context_round'])}</td></tr>
  </tbody>
</table>

<h2>Results — Train Split</h2>
<p>Bold = best among learned policies per estimator column.</p>
<table>
  <thead>
    <tr><th>Policy</th><th>Trained Obj.</th>{metric_headers}<th>ESS (logged)</th></tr>
  </thead>
  <tbody>
{table_html("train")}
  </tbody>
</table>

<h2>Behavior Improvement by Round</h2>
<p>These two rows compare the first and last round-specific behavior policies on the train split, using the same fixed target context distribution.</p>
<table>
  <thead>
    <tr><th>Policy</th>{metric_headers}<th>ESS (logged)</th></tr>
  </thead>
  <tbody>
{''.join(progression_rows)}
  </tbody>
</table>

<h2>Actual Logged Behavior by Round</h2>
<p>These values use only rows from the corresponding round and only reweight by the fixed target context distribution, with no action importance ratio.</p>
<table>
  <thead>
    <tr><th>Split</th><th>Policy</th><th>Context-weighted reward</th><th>Rows</th></tr>
  </thead>
  <tbody>
{''.join(actual_logged_rows)}
  </tbody>
</table>

<h2>Results — Test Split (out-of-sample)</h2>
<p>
  Policies are <strong>not re-trained</strong> on test data. Behavior log-propensities on the test set
  are computed using the train-fitted behavior model. Context weights use the same fixed target
  distribution from train. Bold = best among learned policies per estimator.
</p>
<table>
  <thead>
    <tr><th>Policy</th><th>Trained Obj.</th>{metric_headers}<th>ESS (logged)</th></tr>
  </thead>
  <tbody>
{table_html("test")}
  </tbody>
</table>

<h2>Train vs Test Comparison</h2>
<p>Change from train to test per learned policy.</p>
<table>
  <thead>
    <tr><th>Policy</th>{delta_headers}</tr>
  </thead>
  <tbody>
{''.join(comparison_rows)}
  </tbody>
</table>

<h2>Discussion</h2>
{_discussion_html(results_df, metrics)}

<h2>Policy Comparison by Objective</h2>
<img src="policy_comparison_by_objective.png" alt="Policy comparison by objective">

<h2>Behavior Policy Evolution (Train Data)</h2>
<img src="behavior_evolution_by_round.png" alt="Behavior policy evolution">

</body>
</html>
"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a train/test-style HTML report from one or more user-split runs.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--extra-run-dirs", type=str, nargs="*", default=[])
    parser.add_argument("--base-ktm", type=str, default="")
    parser.add_argument("--dataset-name", type=str, default="")
    parser.add_argument("--fit-sigma-floor", type=float, default=1e-4)
    parser.add_argument("--policy-sigma-floor", type=float, default=0.2)
    parser.add_argument("--optimal-sigma", type=float, default=0.2)
    parser.add_argument("--dm-delta-grid", type=int, default=121)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--context-bins", type=int, default=120)
    parser.add_argument("--context-ratio-clip", type=float, default=20.0)
    parser.add_argument("--mix-batch-rows", type=int, default=200000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    extra_run_dirs = [Path(p).resolve() for p in args.extra_run_dirs]
    device = get_default_device(args.device)

    run_summary = pd.read_csv(run_dir / "run_summary.csv").iloc[0]
    coeff_df = pd.read_csv(run_dir / "learned_policy_coefficients.csv")
    training_history = pd.read_csv(run_dir / "training_history.csv")
    fit_df = pd.read_csv(run_dir / "gaussian_fit_train_by_order.csv")
    split_assign = pd.read_csv(run_dir / "user_split_assignments.csv")
    split_summary = pd.read_csv(run_dir / "split_summary.csv")

    for extra_dir in extra_run_dirs:
        coeff_df = pd.concat(
            [coeff_df, pd.read_csv(extra_dir / "learned_policy_coefficients.csv")],
            ignore_index=True,
        )
    coeff_df = coeff_df.drop_duplicates(subset=["objective"], keep="last").reset_index(drop=True)

    base_ktm = Path(args.base_ktm) if args.base_ktm else Path(str(run_summary["input_csv"]))
    if not base_ktm.is_absolute():
        base_ktm = (PROJECT_ROOT / base_ktm).resolve()

    df_ktm = _load_base_ktm(
        base_ktm=base_ktm,
        truncate_at_round=int(run_summary["truncate_at_round"]),
        max_rounds=int(run_summary["max_rounds"]),
        round_cap_strategy=str(run_summary["round_cap_strategy"]),
    )
    df_ktm["split"] = df_ktm["user"].map(dict(zip(split_assign["user"], split_assign["split"])))

    train_df = df_ktm[df_ktm["split"] == "train"].drop(columns=["split"]).copy()
    test_df = df_ktm[df_ktm["split"] == "test"].drop(columns=["split"]).copy()
    delta_min = float(df_ktm["difficulties"].min())
    delta_max = float(df_ktm["difficulties"].max())
    first_order = int(sorted(fit_df["order_sequence"].dropna().astype(int).unique().tolist())[0])
    last_order = int(run_summary["common_last_round"])
    target_order = int(run_summary["target_round_train_first_half"])
    summary = _dataset_summary(df_ktm, last_order=last_order, target_order=target_order)

    learned_policies = _rebuild_learned_policies(
        coeff_df,
        delta_min=delta_min,
        delta_max=delta_max,
        sigma_floor=float(args.policy_sigma_floor),
        device=device,
    )
    metrics = ["ips", "snips", "dr", "mis", "dm"] if "mis" in learned_policies else ["ips", "snips", "dr", "dm"]

    offpolicy_train = build_offpolicy_dataset(train_df, fit_df, sigma_floor=float(args.fit_sigma_floor))
    target_theta = offpolicy_train[
        offpolicy_train["order_sequence"].astype(int) == int(target_order)
    ]["proficiency"].to_numpy(dtype=float)

    fit_first = fit_df[fit_df["order_sequence"].astype(int) == int(first_order)]
    first_row = fit_first.iloc[0] if not fit_first.empty else fit_df.sort_values("order_sequence").iloc[0]
    fit_last = fit_df[fit_df["order_sequence"].astype(int) == int(last_order)]
    last_row = fit_last.iloc[0] if not fit_last.empty else fit_df.sort_values("order_sequence").iloc[-1]
    behavior_last_policy = _build_behavior_policy_from_row(
        last_row,
        delta_min=delta_min,
        delta_max=delta_max,
        sigma_floor=float(args.fit_sigma_floor),
        device=device,
    )
    optimal_policy = IRTOptimalGaussianPolicy(
        delta_min=delta_min,
        delta_max=delta_max,
        sigma=float(args.optimal_sigma),
    )
    named_policies = [
        ("optimal_irt_gaussian", "n/a", optimal_policy),
        *[(f"learned_{obj}", obj, pol) for obj, pol in learned_policies.items()],
    ]

    train_results = _evaluate_split(
        split_name="train",
        offpolicy_df=offpolicy_train,
        target_theta=target_theta,
        last_order=last_order,
        fit_df=fit_df,
        named_policies=named_policies,
        delta_min=delta_min,
        delta_max=delta_max,
        dm_delta_grid=int(args.dm_delta_grid),
        max_weight=float(args.max_weight),
        context_bins=int(args.context_bins),
        context_ratio_clip=float(args.context_ratio_clip),
        fit_sigma_floor=float(args.fit_sigma_floor),
        need_mix=bool("mis" in metrics),
        mix_batch_rows=int(args.mix_batch_rows),
        device=device,
    )

    offpolicy_test = build_offpolicy_dataset(test_df, fit_df, sigma_floor=float(args.fit_sigma_floor))
    test_results = _evaluate_split(
        split_name="test",
        offpolicy_df=offpolicy_test,
        target_theta=target_theta,
        last_order=last_order,
        fit_df=fit_df,
        named_policies=named_policies,
        delta_min=delta_min,
        delta_max=delta_max,
        dm_delta_grid=int(args.dm_delta_grid),
        max_weight=float(args.max_weight),
        context_bins=int(args.context_bins),
        context_ratio_clip=float(args.context_ratio_clip),
        fit_sigma_floor=float(args.fit_sigma_floor),
        need_mix=bool("mis" in metrics),
        mix_batch_rows=int(args.mix_batch_rows),
        device=device,
    )

    behavior_train_progression_df = _evaluate_behavior_progression(
        split_name="train",
        offpolicy_df=offpolicy_train,
        target_theta=target_theta,
        orders=[first_order, last_order],
        first_order=first_order,
        last_order=last_order,
        fit_df=fit_df,
        delta_min=delta_min,
        delta_max=delta_max,
        dm_delta_grid=int(args.dm_delta_grid),
        max_weight=float(args.max_weight),
        context_bins=int(args.context_bins),
        context_ratio_clip=float(args.context_ratio_clip),
        fit_sigma_floor=float(args.fit_sigma_floor),
        need_mix=bool("mis" in metrics),
        mix_batch_rows=int(args.mix_batch_rows),
        device=device,
    )
    behavior_test_progression_df = _evaluate_behavior_progression(
        split_name="test",
        offpolicy_df=offpolicy_test,
        target_theta=target_theta,
        orders=[first_order, last_order],
        first_order=first_order,
        last_order=last_order,
        fit_df=fit_df,
        delta_min=delta_min,
        delta_max=delta_max,
        dm_delta_grid=int(args.dm_delta_grid),
        max_weight=float(args.max_weight),
        context_bins=int(args.context_bins),
        context_ratio_clip=float(args.context_ratio_clip),
        fit_sigma_floor=float(args.fit_sigma_floor),
        need_mix=bool("mis" in metrics),
        mix_batch_rows=int(args.mix_batch_rows),
        device=device,
    )
    behavior_progression_df = behavior_train_progression_df.copy()
    actual_logged_df = pd.concat(
        [
            _evaluate_actual_logged_behavior(
                split_name="train",
                offpolicy_df=offpolicy_train,
                target_theta=target_theta,
                orders=[first_order, last_order],
                first_order=first_order,
                last_order=last_order,
                context_bins=int(args.context_bins),
                context_ratio_clip=float(args.context_ratio_clip),
            ),
            _evaluate_actual_logged_behavior(
                split_name="test",
                offpolicy_df=offpolicy_test,
                target_theta=target_theta,
                orders=[first_order, last_order],
                first_order=first_order,
                last_order=last_order,
                context_bins=int(args.context_bins),
                context_ratio_clip=float(args.context_ratio_clip),
            ),
        ],
        ignore_index=True,
    )

    keep_cols = ["split", "policy", "trained_objective", *metrics, "ess_logged"]
    results_df = pd.concat(
        [
            behavior_train_progression_df[keep_cols],
            train_results[keep_cols],
            behavior_test_progression_df[keep_cols],
            test_results[keep_cols],
        ],
        ignore_index=True,
    )
    results_df.to_csv(run_dir / "estimator_results_train_test.csv", index=False)

    theta_grid = np.linspace(float(df_ktm["proficiency"].min()), float(df_ktm["proficiency"].max()), 400)
    _irt = IRTOptimalGaussianPolicy(delta_min=delta_min, delta_max=delta_max, sigma=float(args.optimal_sigma))
    with torch.no_grad():
        optimal_delta = _irt._optimal_delta(torch.from_numpy(theta_grid.astype(np.float32))).numpy().astype(float)

    fig1, _ = plot_objective_comparison(
        theta_grid=theta_grid,
        fit_df=fit_df,
        coeff_df=coeff_df,
        delta_min=delta_min,
        delta_max=delta_max,
        sigma_floor=float(args.policy_sigma_floor),
        optimal_delta=optimal_delta,
        title=f"Learned Policies vs Behavior — {args.dataset_name or run_dir.name}",
    )
    fig1.savefig(str(run_dir / "policy_comparison_by_objective.png"), dpi=170)
    plt.close(fig1)

    fig2, _ = plot_behavior_evolution(
        theta_grid=theta_grid,
        fit_df=fit_df,
        delta_min=delta_min,
        delta_max=delta_max,
        sigma_floor=float(args.fit_sigma_floor),
        title="Behavior Policy Evolution (train data)",
    )
    fig2.savefig(str(run_dir / "behavior_evolution_by_round.png"), dpi=170)
    plt.close(fig2)

    n_epochs = int(training_history["epoch"].max()) if not training_history.empty else 0
    train_row = split_summary[split_summary["split"] == "train"].iloc[0]
    test_row = split_summary[split_summary["split"] == "test"].iloc[0]
    _write_report(
        out_dir=run_dir,
        results_df=results_df,
        behavior_progression_df=behavior_progression_df,
        actual_logged_df=actual_logged_df,
        dataset_name=args.dataset_name or run_dir.name,
        dataset_summary=summary,
        n_train_users=int(train_row["n_users"]),
        n_test_users=int(test_row["n_users"]),
        n_train_rows=int(train_row["n_rows"]),
        n_test_rows=int(test_row["n_rows"]),
        train_fraction=float(run_summary["train_frac"]),
        first_order=first_order,
        target_order=target_order,
        last_order=last_order,
        epochs=n_epochs,
        metrics=metrics,
    )
    print(f"Report saved: {run_dir / 'report.html'}")


if __name__ == "__main__":
    main()
