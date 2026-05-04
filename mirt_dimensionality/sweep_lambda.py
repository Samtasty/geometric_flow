"""
Compare K sweep across different lambda_theta values.
Run from project root: python -m mirt_dimensionality.sweep_lambda
"""
import sys
from .data import load_pix, response_split
from .train import sweep


def main():
    df = load_pix("data/pix_data.csv")
    df_train, df_test = response_split(df, seed=1)
    print(f"Train: {len(df_train):,}  Test: {len(df_test):,}\n")

    dims = [1, 2, 4, 8, 16, 32, 64]
    lambdas = [0.0, 0.1, 1.0, 5.0]

    all_results = {}
    for lam in lambdas:
        print(f"\n{'='*60}")
        print(f"lambda_theta = {lam}")
        print(f"{'='*60}")
        sys.stdout.flush()
        all_results[lam] = sweep(
            df_train, df_test,
            dims=dims,
            lambda_theta=lam,
            n_epochs=50,
            batch_size=8192,
            lr=5e-3,
            lr_schedule="cosine",
            warmup_epochs=2,
            min_lr_ratio=0.05,
            patience=5,
        )

    # Print combined summary table
    print("\n\n" + "=" * 90)
    print("SUMMARY: best test_loss / best test_auc  (by K and lambda_theta)")
    print("=" * 90)

    header = f"{'K':>4}"
    for lam in lambdas:
        header += f"  |  λ={lam:<4}  loss       auc"
    print(header)
    print("-" * len(header))

    for K in dims:
        row = f"{K:>4}"
        for lam in lambdas:
            h = all_results[lam][K]["history"]
            row += f"  |  {min(h['test_loss']):>10.5f}  {max(h['test_auc']):.5f}"
        print(row)


if __name__ == "__main__":
    main()
