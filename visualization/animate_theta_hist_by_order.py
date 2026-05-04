from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_animation(
    *,
    input_csv: str,
    out_gif: str,
    order_col: str,
    theta_col: str,
    bins: int,
    fps: int,
    max_orders: int,
    dataset_name: str,
    normalize_prob: bool,
    overlay_user_reference: bool,
    user_col: str,
) -> None:
    usecols = [order_col, theta_col]
    if overlay_user_reference:
        usecols.append(user_col)
    df = pd.read_csv(input_csv, usecols=usecols).dropna().copy()
    df[order_col] = df[order_col].astype(int)

    grouped = {int(k): v[theta_col].to_numpy(dtype=float) for k, v in df.groupby(order_col)}
    orders = sorted(grouped.keys())
    if max_orders > 0:
        orders = orders[:max_orders]

    if not orders:
        raise ValueError("No orders found after filtering.")

    user_theta_ref = None
    if overlay_user_reference:
        # One theta per user for a global context reference curve.
        user_theta_ref = (
            df[[user_col, theta_col]]
            .groupby(user_col, as_index=False)[theta_col]
            .mean()[theta_col]
            .to_numpy(dtype=float)
        )

    theta_all = df[theta_col].to_numpy(dtype=float)
    x_min = float(np.quantile(theta_all, 0.01))
    x_max = float(np.quantile(theta_all, 0.99))
    if x_max <= x_min:
        x_min, x_max = float(theta_all.min()), float(theta_all.max())
    if x_max <= x_min:
        x_min, x_max = x_min - 1.0, x_max + 1.0

    y_max = 0.0
    for o in orders:
        vals = grouped[o]
        if normalize_prob and len(vals) > 0:
            w = np.ones(len(vals), dtype=float) / float(len(vals))
            counts, _ = np.histogram(vals, bins=bins, range=(x_min, x_max), weights=w)
        else:
            counts, _ = np.histogram(vals, bins=bins, range=(x_min, x_max))
        y_max = max(y_max, float(counts.max()) if len(counts) > 0 else 0.0)
    y_max = max(1e-6, y_max)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    def draw(i: int) -> None:
        order_val = orders[i]
        vals = grouped[order_val]
        ax.clear()
        hist_kwargs = {}
        if normalize_prob and len(vals) > 0:
            hist_kwargs["weights"] = np.ones(len(vals), dtype=float) / float(len(vals))
        ax.hist(
            vals,
            bins=bins,
            range=(x_min, x_max),
            color="#1f77b4",
            alpha=0.85,
            edgecolor="white",
            **hist_kwargs,
        )
        if user_theta_ref is not None and len(user_theta_ref) > 0:
            ref_kwargs = {}
            if normalize_prob:
                ref_kwargs["weights"] = np.ones(len(user_theta_ref), dtype=float) / float(len(user_theta_ref))
            ax.hist(
                user_theta_ref,
                bins=bins,
                range=(x_min, x_max),
                histtype="step",
                linewidth=2.0,
                color="#ff7f0e",
                label="all users (1 theta/user)",
                **ref_kwargs,
            )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max * 1.1)
        ax.set_xlabel("theta (proficiency)")
        ax.set_ylabel("frequency in [0,1]" if normalize_prob else "frequency")
        ax.set_title(
            f"{dataset_name} | Theta Histogram by order_sequence | "
            f"order={order_val} ({i + 1}/{len(orders)})"
        )
        ax.grid(alpha=0.25)
        if user_theta_ref is not None:
            ax.legend(loc="upper left")
        ax.text(
            0.98,
            0.95,
            (
                f"n(theta) = {len(vals)}\n"
                + (f"n(users) = {len(user_theta_ref)}" if user_theta_ref is not None else "")
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    ani = animation.FuncAnimation(fig, draw, frames=len(orders), interval=1000 / max(1, fps))
    Path(out_gif).parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_gif, writer=animation.PillowWriter(fps=max(1, fps)))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate theta histogram per order_sequence.")
    parser.add_argument("--input-csv", type=str, default="pix_mapping/data_full_ktm.csv")
    parser.add_argument("--out-gif", type=str, default="pix_mapping/theta_hist_by_order.gif")
    parser.add_argument("--order-col", type=str, default="order_sequence")
    parser.add_argument("--theta-col", type=str, default="proficiency")
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-orders", type=int, default=0, help="0 = all orders")
    parser.add_argument("--dataset-name", type=str, default="Dataset")
    parser.add_argument(
        "--normalize-prob",
        action="store_true",
        help="Plot normalized bin frequencies so histogram values are in [0,1].",
    )
    parser.add_argument(
        "--overlay-user-reference",
        action="store_true",
        help="Overlay global histogram of theta with one theta per user.",
    )
    parser.add_argument("--user-col", type=str, default="user")
    args = parser.parse_args()

    build_animation(
        input_csv=args.input_csv,
        out_gif=args.out_gif,
        order_col=args.order_col,
        theta_col=args.theta_col,
        bins=args.bins,
        fps=args.fps,
        max_orders=args.max_orders,
        dataset_name=args.dataset_name,
        normalize_prob=args.normalize_prob,
        overlay_user_reference=args.overlay_user_reference,
        user_col=args.user_col,
    )
    print(f"saved_animation={Path(args.out_gif).resolve()}")


if __name__ == "__main__":
    main()
