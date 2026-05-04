"""
Post-training analysis of item parameters.

Focus: do item discrimination vectors a_j cluster by competence?
Metrics: silhouette score (cosine), PCA visualisation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def _item_vectors(model) -> np.ndarray:
    """Return normalised discrimination vectors (N_challenges, K)."""
    return model.item_discrimination_normalised().cpu().numpy()


def silhouette_by_k(results: dict, item_labels: np.ndarray) -> pd.Series:
    """
    Compute cosine silhouette score of item discrimination vectors
    grouped by competence for each K in results.
    """
    scores = {}
    for K, res in results.items():
        a_norm = _item_vectors(res["model"])
        if K == 1:
            scores[K] = float("nan")
        else:
            scores[K] = silhouette_score(a_norm, item_labels, metric="cosine")
    return pd.Series(scores, name="silhouette")


def summary_table(results: dict) -> pd.DataFrame:
    rows = []
    for K, res in results.items():
        h = res["history"]
        m = res["model"]
        n_item_params = (
            m.a_raw.weight.numel() + m.d.weight.numel()
        )
        rows.append({
            "K": K,
            "n_item_params": n_item_params,
            "best_test_loss": min(h["test_loss"]),
            "final_test_loss": h["test_loss"][-1],
            "best_test_auc": max(h["test_auc"]),
            "final_test_auc": h["test_auc"][-1],
            "final_train_bce": h["train_bce"][-1],
        })
    df = pd.DataFrame(rows).set_index("K")
    df["delta_loss_vs_K1"] = df["final_test_loss"] - df.loc[1, "final_test_loss"]
    df["delta_auc_vs_K1"]  = df["final_test_auc"]  - df.loc[1, "final_test_auc"]
    return df


def plot_elbow(results: dict, out_path=None, lambda_theta: float = 1.0,
               n_students: int = 0):
    dims = sorted(results)
    summary = summary_table(results)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    colors = cm.plasma(np.linspace(0, 0.85, len(dims)))

    # Learning curves
    for K, color in zip(dims, colors):
        axes[0].plot(results[K]["history"]["test_loss"], label=f"K={K}", color=color)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Test BCE loss")
    axes[0].set_title("Test BCE loss per epoch")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Elbow — test loss
    axes[1].plot(dims, summary.loc[dims, "final_test_loss"].values,
                 "o-", color="steelblue", lw=2, ms=8)
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Final test BCE loss")
    axes[1].set_title("Elbow: test loss vs K")
    axes[1].set_xticks(dims); axes[1].grid(True, alpha=0.3)

    # Elbow — AUC
    axes[2].plot(dims, summary.loc[dims, "final_test_auc"].values,
                 "o-", color="tomato", lw=2, ms=8)
    axes[2].set_xlabel("K"); axes[2].set_ylabel("Final test AUC-ROC")
    axes[2].set_title("Elbow: AUC-ROC vs K")
    axes[2].set_xticks(dims); axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"M-IRT higher-K sweep  |  {n_students} students  |  λ_θ={lambda_theta}",
        fontsize=13,
    )
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.show()


def plot_item_pca(
    results: dict,
    item_labels: np.ndarray,
    comp_codes: list,
    out_path=None,
    lambda_theta: float = 1.0,
    n_students: int = 0,
):
    """
    2×4 grid of PCA scatter plots of normalised item discrimination vectors,
    coloured by competence code.  One subplot per K value.
    """
    dims = sorted(results)
    comp_to_int = {c: i for i, c in enumerate(comp_codes)}
    colors_arr = np.array([comp_to_int[c] for c in item_labels])

    n_cols = 4
    n_rows = int(np.ceil(len(dims) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    sil_scores = {}
    for ax_idx, K in enumerate(dims):
        a_norm = _item_vectors(results[K]["model"])

        sil = (
            silhouette_score(a_norm, colors_arr, metric="cosine")
            if K > 1 else float("nan")
        )
        sil_scores[K] = sil

        n_pc = min(2, K)
        pca  = PCA(n_components=n_pc, random_state=42)
        pts  = pca.fit_transform(a_norm)
        if n_pc == 1:
            pts = np.column_stack([pts, np.zeros(len(pts))])

        ax = axes[ax_idx]
        ax.scatter(pts[:, 0], pts[:, 1], c=colors_arr, cmap="tab20",
                   s=18, alpha=0.75)
        var_str = f"PC1={pca.explained_variance_ratio_[0]:.0%}"
        if n_pc > 1:
            var_str += f", PC2={pca.explained_variance_ratio_[1]:.0%}"
        ax.set_title(f"K={K}  |  sil={sil:.3f}\n{var_str}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    for ax_idx in range(len(dims), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(
        f"Item discrimination vectors (normalised) — PCA by competence\n"
        f"{n_students} students, λ_θ={lambda_theta}",
        fontsize=13,
    )
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.show()

    return sil_scores
