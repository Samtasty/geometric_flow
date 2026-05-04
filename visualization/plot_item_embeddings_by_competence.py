import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.assistments_dataset import AssistmentsDataset
from train.train_mirt import train_mirt


def _build_item_to_competence_map(processed_csv):
    df = pd.read_csv(processed_csv, usecols=["challenge_id", "competence_code"])
    df["challenge_id"] = df["challenge_id"].astype(int)
    df["competence_code"] = df["competence_code"].astype(str)

    counts = df.groupby("challenge_id")["competence_code"].nunique()
    bad = counts[counts > 1]
    if len(bad) > 0:
        raise ValueError(
            f"Found {len(bad)} challenge_id values with multiple competence_code labels."
        )

    mapping = (
        df.drop_duplicates(subset=["challenge_id"])
        .set_index("challenge_id")["competence_code"]
        .to_dict()
    )
    return mapping


def _plot_embeddings(df, out_plot):
    competences = sorted(df["competence_code"].unique().tolist())
    cmap = plt.get_cmap("tab20")

    plt.figure(figsize=(10, 8))
    for idx, comp in enumerate(competences):
        sub = df[df["competence_code"] == comp]
        color = cmap(idx % 20)
        plt.scatter(
            sub["dim1"].to_numpy(),
            sub["dim2"].to_numpy(),
            s=25,
            alpha=0.8,
            label=comp,
            color=color,
        )

    plt.title("Item Embeddings in 2D, Colored by competence_code")
    plt.xlabel("Item Embedding Dim 1")
    plt.ylabel("Item Embedding Dim 2")
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
    plt.savefig(out_plot, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train dim=2 MIRT and plot item embeddings colored by competence_code."
    )
    parser.add_argument("--data", type=str, default="pix_mapping/pix_irt_outcome.csv")
    parser.add_argument("--processed-csv", type=str, default="pix_mapping/pix_processed.csv")
    parser.add_argument("--emb-dim", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--student-l2", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--use-softplus",
        action="store_true",
        help="Plot transformed item embedding softplus(item_emb) instead of raw item_emb.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="pix_mapping/item_embedding_dim2_by_competence.csv",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="pix_mapping/item_embedding_dim2_by_competence.png",
    )
    args = parser.parse_args()

    if args.emb_dim != 2:
        raise ValueError("This plotting script expects emb_dim=2.")

    item_to_comp = _build_item_to_competence_map(args.processed_csv)
    dataset = AssistmentsDataset(args.data)

    model, _ = train_mirt(
        dataset,
        emb_dim=args.emb_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        student_l2=args.student_l2,
        num_workers=args.num_workers,
        device=args.device,
    )

    with torch.no_grad():
        item_emb = model.item_emb.weight.detach().cpu()
        if args.use_softplus:
            item_emb = F.softplus(item_emb)
        item_emb = item_emb.numpy()

    item_raw = np.asarray(dataset.items, dtype=np.int64)
    item_idx = np.arange(len(item_raw), dtype=np.int64)
    competence = [item_to_comp.get(int(ch), "UNKNOWN") for ch in item_raw]

    out_df = pd.DataFrame(
        {
            "item_idx": item_idx,
            "challenge_id": item_raw,
            "competence_code": competence,
            "dim1": item_emb[:, 0],
            "dim2": item_emb[:, 1],
        }
    )
    out_df = out_df.sort_values(["competence_code", "challenge_id"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    _plot_embeddings(out_df, args.out_plot)

    print(
        f"saved_csv={os.path.abspath(args.out_csv)}\n"
        f"saved_plot={os.path.abspath(args.out_plot)}\n"
        f"items={len(out_df)} competences={out_df['competence_code'].nunique()}"
    )


if __name__ == "__main__":
    main()
