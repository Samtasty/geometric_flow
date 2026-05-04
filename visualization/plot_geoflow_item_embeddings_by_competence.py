import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train.geoflow_simulator_learner import dataframe_to_seqs, fit_item_embeddings_batched


def _build_item_to_competence_map(df):
    m = df[["challenge_id", "competence_code"]].drop_duplicates().copy()
    counts = m.groupby("challenge_id")["competence_code"].nunique()
    bad = counts[counts > 1]
    if len(bad) > 0:
        raise ValueError(f"Found {len(bad)} challenge_id with multiple competence_code values.")
    return m.set_index("challenge_id")["competence_code"].astype(str).to_dict()


def _pca_project_2d(x):
    if x.ndim != 2:
        raise ValueError("Expected a 2D array for PCA projection.")
    if x.shape[1] < 2:
        raise ValueError("Need at least 2 dimensions to project to 2D.")
    x0 = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x0, full_matrices=False)
    coords = x0 @ vt[:2].T
    var = (s ** 2) / max(x0.shape[0] - 1, 1)
    total = float(var.sum())
    if total <= 0:
        return coords, 0.0, 0.0
    evr = var / total
    pc1 = float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = float(evr[1]) if len(evr) > 1 else 0.0
    return coords, pc1, pc2


def _plot(df_out, out_plot, title, draw_unit_circle=False):
    codes = sorted(df_out["competence_code"].unique().tolist())
    cmap = plt.get_cmap("tab20")

    plt.figure(figsize=(10, 8))
    for i, code in enumerate(codes):
        sub = df_out[df_out["competence_code"] == code]
        plt.scatter(
            sub["dim1"].to_numpy(),
            sub["dim2"].to_numpy(),
            s=26,
            alpha=0.82,
            color=cmap(i % 20),
            label=code,
        )

    if draw_unit_circle:
        # GeoFlow embeddings are row-normalized; unit circle helps interpretation.
        t = np.linspace(0, 2 * np.pi, 400)
        plt.plot(np.cos(t), np.sin(t), linestyle="--", linewidth=1.0, alpha=0.6, color="gray")

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    if draw_unit_circle:
        plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.legend(
        title="competence_code",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    plt.savefig(out_plot, dpi=170)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train GeoFlow item embeddings on PIX and plot in 2D by competence_code."
    )
    parser.add_argument("--processed-csv", type=str, default="pix_mapping/pix_processed.csv")
    parser.add_argument("--emb-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--tbptt-k", type=int, default=16)
    parser.add_argument("--max-t", type=int, default=64)
    parser.add_argument("--min-user-len", type=int, default=5)
    parser.add_argument("--kappa", type=float, default=5.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--eta-pos", type=float, default=0.5)
    parser.add_argument("--eta-neg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out-csv",
        type=str,
        default="pix_mapping/geoflow_item_embedding_by_competence.csv",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="pix_mapping/geoflow_item_embedding_by_competence.png",
    )
    args = parser.parse_args()
    if args.emb_dim < 2:
        raise ValueError("--emb-dim must be >= 2")

    usecols = ["user_id", "challenge_id", "outcome", "answer_number", "competence_code"]
    df = pd.read_csv(args.processed_csv, usecols=usecols)
    df["user_id"] = df["user_id"].astype(int)
    df["challenge_id"] = df["challenge_id"].astype(int)
    df["outcome"] = df["outcome"].astype(float)
    df["answer_number"] = df["answer_number"].astype(int)
    df["competence_code"] = df["competence_code"].astype(str)

    item_to_comp = _build_item_to_competence_map(df)

    prep = dataframe_to_seqs(
        df,
        user_col="user_id",
        item_col="challenge_id",
        resp_col="outcome",
        order_cols=["answer_number"],
        threshold=0.5,
        min_user_len=args.min_user_len,
    )
    seqs = prep["seqs"]
    n_items = int(prep["n_items"])
    item_values = np.asarray(prep["item_values"], dtype=np.int64)

    print(
        f"users_kept={len(seqs)} n_items={n_items} "
        f"emb_dim={args.emb_dim} epochs={args.epochs} batch_size={args.batch_size} max_t={args.max_t}"
    )

    w_hat, _ = fit_item_embeddings_batched(
        seqs,
        n_items=n_items,
        d=args.emb_dim,
        kappa=args.kappa,
        lam=args.lam,
        eta_pos=args.eta_pos,
        eta_neg=args.eta_neg,
        use_prox=True,
        epochs=args.epochs,
        lr=args.lr,
        tbptt_k=(None if args.tbptt_k <= 0 else args.tbptt_k),
        batch_size=args.batch_size,
        max_t=(None if args.max_t <= 0 else args.max_t),
        random_window=True,
        use_geoopt=False,
        device=args.device,
        seed=args.seed,
        verbose=True,
    )

    if args.emb_dim == 2:
        coords = w_hat
        pc1, pc2 = None, None
        title = "GeoFlow Item Embeddings (native 2D), colored by competence_code"
        draw_unit_circle = True
    else:
        coords, pc1, pc2 = _pca_project_2d(w_hat)
        title = (
            "GeoFlow Item Embeddings "
            f"(trained in {args.emb_dim}D, PCA to 2D), colored by competence_code"
        )
        draw_unit_circle = False

    out_df = pd.DataFrame(
        {
            "item_idx_geoflow": np.arange(n_items, dtype=np.int64),
            "challenge_id": item_values,
            "competence_code": [item_to_comp.get(int(cid), "UNKNOWN") for cid in item_values],
            "dim1": coords[:, 0],
            "dim2": coords[:, 1],
        }
    ).sort_values(["competence_code", "challenge_id"])
    out_df["emb_dim"] = int(args.emb_dim)
    if pc1 is not None:
        out_df["pca_explained_var_pc1"] = pc1
        out_df["pca_explained_var_pc2"] = pc2

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    _plot(out_df, args.out_plot, title=title, draw_unit_circle=draw_unit_circle)

    print(f"saved_csv={os.path.abspath(args.out_csv)}")
    print(f"saved_plot={os.path.abspath(args.out_plot)}")
    print(
        f"competences={out_df['competence_code'].nunique()} "
        f"items={len(out_df)}"
    )
    if pc1 is not None:
        print(f"pca_explained_var_pc1={pc1:.4f} pca_explained_var_pc2={pc2:.4f}")


if __name__ == "__main__":
    main()
