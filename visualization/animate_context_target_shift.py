from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _hist_prob(theta: np.ndarray, edges: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    counts, _ = np.histogram(theta, bins=edges)
    p = counts.astype(float) + eps
    p = p / np.clip(float(p.sum()), 1e-12, None)
    return p


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    m = 0.5 * (p + q)
    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)
    m_safe = np.clip(m, eps, None)
    kl_pm = np.sum(p_safe * np.log(p_safe / m_safe))
    kl_qm = np.sum(q_safe * np.log(q_safe / m_safe))
    return float(0.5 * (kl_pm + kl_qm))


def _select_target_theta(
    df: pd.DataFrame,
    orders: list[int],
    *,
    context_target: str,
    k_rounds: int,
    bins: int,
    order_col: str,
    theta_col: str,
) -> tuple[np.ndarray, str]:
    if context_target == "global":
        return df[theta_col].to_numpy(dtype=float), "global"

    if context_target == "round1":
        o = int(orders[0])
        th = df[df[order_col] == o][theta_col].to_numpy(dtype=float)
        return th, f"round1(order={o})"

    k = max(1, min(int(k_rounds), len(orders)))
    first_k = [int(x) for x in orders[:k]]
    df_k = df[df[order_col].isin(first_k)].copy()

    if context_target == "pooled_first_k":
        th = df_k[theta_col].to_numpy(dtype=float)
        return th, f"pooled_first_{k}"

    if context_target == "best_round_in_k":
        lo = float(df_k[theta_col].min())
        hi = float(df_k[theta_col].max())
        if hi <= lo:
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, max(4, int(bins)) + 1)
        round_hist: dict[int, np.ndarray] = {}
        for o in first_k:
            th = df[df[order_col] == o][theta_col].to_numpy(dtype=float)
            round_hist[o] = _hist_prob(th, edges)

        best_o = first_k[0]
        best_score = float("inf")
        for cand in first_k:
            p = round_hist[cand]
            score = 0.0
            for o in first_k:
                score += _js_divergence(p, round_hist[o])
            score /= float(len(first_k))
            if score < best_score:
                best_score = score
                best_o = cand
        th = df[df[order_col] == best_o][theta_col].to_numpy(dtype=float)
        return th, f"best_round_in_first_{k}(order={best_o},mean_js={best_score:.6f})"

    raise ValueError(f"Unsupported context_target: {context_target}")


def build_animation(
    *,
    input_csv: str,
    out_gif: str,
    order_col: str,
    theta_col: str,
    context_target: str,
    k_rounds: int,
    bins: int,
    fps: int,
    max_orders: int,
    dataset_name: str,
) -> None:
    df = pd.read_csv(input_csv, usecols=[order_col, theta_col]).dropna().copy()
    df[order_col] = df[order_col].astype(int)
    orders_all = sorted(df[order_col].unique().tolist())
    if not orders_all:
        raise ValueError("No order values found.")

    target_theta, target_label = _select_target_theta(
        df,
        orders_all,
        context_target=context_target,
        k_rounds=k_rounds,
        bins=bins,
        order_col=order_col,
        theta_col=theta_col,
    )

    orders = orders_all[: max_orders if max_orders > 0 else len(orders_all)]
    grouped = {int(o): df[df[order_col] == o][theta_col].to_numpy(dtype=float) for o in orders}

    all_theta = df[theta_col].to_numpy(dtype=float)
    lo = float(min(all_theta.min(), target_theta.min()))
    hi = float(max(all_theta.max(), target_theta.max()))
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, max(4, int(bins)) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(edges[1] - edges[0])

    target_hist = _hist_prob(target_theta, edges)
    y_max = float(max(1e-6, target_hist.max()))
    for o in orders:
        y_max = max(y_max, float(_hist_prob(grouped[o], edges).max()))

    fig, ax = plt.subplots(figsize=(9.6, 5.6))

    def draw(i: int) -> None:
        o = int(orders[i])
        th = grouped[o]
        p_o = _hist_prob(th, edges)
        js = _js_divergence(p_o, target_hist)

        ax.clear()
        ax.bar(centers, p_o, width=width * 0.92, color="#1f77b4", alpha=0.70, label=f"round {o} distribution")
        ax.plot(centers, target_hist, color="#d62728", lw=2.2, label=f"target: {target_label}")
        ax.fill_between(centers, target_hist, color="#d62728", alpha=0.12)

        ax.set_xlim(lo, hi)
        ax.set_ylim(0.0, y_max * 1.15)
        ax.set_xlabel("theta")
        ax.set_ylabel("probability per bin")
        ax.set_title(f"{dataset_name} | Context Shift vs Target | frame {i + 1}/{len(orders)}")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        ax.text(
            0.01,
            0.98,
            f"order={o}\nN_round={len(th)}\nN_target={len(target_theta)}\nJS(round,target)={js:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "#cccccc"},
        )

    ani = animation.FuncAnimation(fig, draw, frames=len(orders), interval=1000 / max(1, fps))
    Path(out_gif).parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_gif, writer=animation.PillowWriter(fps=max(1, fps)))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate context distribution shift with a fixed target context.")
    parser.add_argument("--input-csv", type=str, default="pix_mapping/data_full_ktm.csv")
    parser.add_argument("--out-gif", type=str, default="pix_mapping/context_shift_target.gif")
    parser.add_argument("--order-col", type=str, default="order_sequence")
    parser.add_argument("--theta-col", type=str, default="proficiency")
    parser.add_argument(
        "--context-target",
        type=str,
        default="round1",
        choices=["round1", "global", "pooled_first_k", "best_round_in_k"],
    )
    parser.add_argument("--k-rounds", type=int, default=5)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-orders", type=int, default=50)
    parser.add_argument("--dataset-name", type=str, default="Dataset")
    args = parser.parse_args()

    build_animation(
        input_csv=args.input_csv,
        out_gif=args.out_gif,
        order_col=args.order_col,
        theta_col=args.theta_col,
        context_target=args.context_target,
        k_rounds=args.k_rounds,
        bins=args.bins,
        fps=args.fps,
        max_orders=args.max_orders,
        dataset_name=args.dataset_name,
    )
    print(f"saved_animation={Path(args.out_gif).resolve()}")


if __name__ == "__main__":
    main()
