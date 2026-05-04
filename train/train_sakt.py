import torch
from torch.utils.data import DataLoader
import numpy as np

from models.sakt import SAKTModel
from data.kt_dataset import KTSequenceDataset


def train_sakt(
    csv_path,
    max_seq_len=50,
    emb_dim=64,
    num_heads=4,
    dropout=0.2,
    batch_size=128,
    lr=1e-3,
    epochs=10,
    train_ratio=0.8,
    device=None,
):
    train_set = KTSequenceDataset(
        csv_path, max_seq_len=max_seq_len, train_ratio=train_ratio, split="train"
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SAKTModel(train_set.num_items, emb_dim=emb_dim, num_heads=num_heads, dropout=dropout)
    model.to(device)

    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            q_seq = batch["q_seq"].to(device)
            r_seq = batch["r_seq"].to(device)
            q_next = batch["q_next"].to(device)
            labels = batch["label"].to(device)

            preds = model(q_seq, r_seq, q_next)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"Epoch {epoch+1}: Loss = {mean_loss:.4f}")

    return model, train_set
