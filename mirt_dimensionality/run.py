"""
M-IRT dimensionality sweep — entry point.

Usage (from project root):
    # Full data, cosine LR schedule
    python -m mirt_dimensionality.run --full-data

    # Small student batch (for item-param focus)
    python -m mirt_dimensionality.run --n-students 5000 --lambda-theta 1.0

    # Custom sweep
    python -m mirt_dimensionality.run --dims 1 2 4 8 16 32 64 --lr 1e-2 --epochs 50
"""
import argparse
import os

from .data import load_pix, challenge_metadata, sample_student_batch, response_split
from .train import sweep
from .analysis import summary_table, plot_elbow, plot_item_pca


def parse_args():
    p = argparse.ArgumentParser(description="M-IRT higher-K sweep with student Gaussian reg")
    p.add_argument("--data",          default="data/pix_data.csv")
    p.add_argument("--dims",          nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    # Data
    p.add_argument("--full-data",     action="store_true",
                   help="Use all students (ignores --n-students)")
    p.add_argument("--n-students",    type=int, default=5_000)
    # Regularisation
    p.add_argument("--lambda-theta",  type=float, default=1.0,
                   help="Gaussian prior on student embeddings (0 = no reg)")
    # Optimisation
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch-size",    type=int,   default=4096)
    p.add_argument("--lr",            type=float, default=5e-3)
    p.add_argument("--lr-schedule",   default="cosine", choices=["cosine", "none"])
    p.add_argument("--warmup-epochs", type=int,   default=2)
    p.add_argument("--min-lr-ratio",  type=float, default=0.05,
                   help="Cosine floor = lr * min_lr_ratio")
    p.add_argument("--patience",      type=int,   default=0,
                   help="Early stopping patience on test_loss (0 = disabled)")
    # Misc
    p.add_argument("--out-dir",       default="results")
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--device",        default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    df = load_pix(args.data)
    meta = challenge_metadata(df)

    if args.full_data:
        print(f"Using full dataset ({df['uid'].nunique():,} students)")
        df_work = df
    else:
        df_work = sample_student_batch(df, n_students=args.n_students, seed=args.seed)

    df_train, df_test = response_split(df_work, seed=args.seed + 1)
    print(f"Train: {len(df_train):,}  |  Test: {len(df_test):,}")

    # ------------------------------------------------------------------
    # 2. Train
    # ------------------------------------------------------------------
    results = sweep(
        df_train, df_test,
        dims=args.dims,
        lambda_theta=args.lambda_theta,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        patience=args.patience,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    table = summary_table(results)
    print("\nSummary:")
    print(table.to_string(float_format="%.5f"))
    table.to_csv(os.path.join(args.out_dir, "mirt_high_k_summary.csv"))

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    comp_codes_sorted = sorted(df["competence_code"].unique())
    item_labels = meta.sort_index()["competence_code"].values

    n_students = df_work["uid"].nunique()

    plot_elbow(
        results,
        out_path=os.path.join(args.out_dir, "mirt_high_k_elbow.png"),
        lambda_theta=args.lambda_theta,
        n_students=n_students,
    )

    sil = plot_item_pca(
        results,
        item_labels=item_labels,
        comp_codes=comp_codes_sorted,
        out_path=os.path.join(args.out_dir, "mirt_high_k_item_pca.png"),
        lambda_theta=args.lambda_theta,
        n_students=n_students,
    )

    print("\nSilhouette scores (cosine, by competence):")
    for K, s in sil.items():
        print(f"  K={K:3d}  sil={s:.4f}")


if __name__ == "__main__":
    main()
