"""
Overfit check: train M-IRT with NO regularisation on a tiny batch.
Expect training BCE → 0 (or near 0) to confirm the model has enough capacity.

Run from project root:
    python -m mirt_dimensionality.overfit_check
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .data import load_pix, sample_student_batch
from .model import MIRT


def overfit(
    n_students: int = 200,
    K: int = 16,
    epochs: int = 200,
    batch_size: int = 2048,
    lr: float = 1e-2,
    seed: int = 0,
    data_path: str = "data/pix_data.csv",
):
    df = load_pix(data_path)
    df_batch = sample_student_batch(df, n_students=n_students, seed=seed)

    n_users      = int(df_batch["uid"].max()) + 1
    n_challenges = int(df_batch["cid"].max()) + 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = MIRT(n_users, n_challenges, K).to(device)
    opt    = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    tr_u = torch.tensor(df_batch["uid"].values,     dtype=torch.long)
    tr_c = torch.tensor(df_batch["cid"].values,     dtype=torch.long)
    tr_y = torch.tensor(df_batch["correct"].values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tr_u, tr_c, tr_y),
                        batch_size=batch_size, shuffle=True)

    print(f"\nOverfit check | K={K} | {n_students} students | {len(df_batch):,} responses | NO reg")
    print(f"{'epoch':>6}  {'train_bce':>10}")

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for u_b, c_b, y_b in loader:
            u_b, c_b, y_b = u_b.to(device), c_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = loss_fn(model(u_b, c_b), y_b)
            loss.backward()
            opt.step()
            total += loss.item(); n += 1
        bce = total / n
        if epoch % 20 == 0 or epoch == 1 or bce < 1e-3:
            print(f"{epoch:>6}  {bce:>10.6f}")
        if bce < 1e-4:
            print(f"  → converged at epoch {epoch}")
            break


if __name__ == "__main__":
    overfit()
