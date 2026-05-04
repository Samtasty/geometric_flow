import os
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

from models.mgirt import MGIRTModel
from data.assistments_dataset import AssistmentsDataset


def _base_dataset(dataset):
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "students"):
        return dataset.dataset
    return dataset


def _ordered_indices(dataset, chronological, seed):
    n = len(dataset)
    if chronological:
        if hasattr(dataset, "idxs"):
            base_idxs = np.array(dataset.idxs)
            return np.argsort(base_idxs).tolist()
        if hasattr(dataset, "indices"):
            base_idxs = np.array(dataset.indices)
            return np.argsort(base_idxs).tolist()
        return list(range(n))
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    return order.tolist()


def _split_train_val(dataset, val_split=0.1, seed=42, chronological=True):
    if val_split <= 0.0 or len(dataset) < 2:
        return dataset, None
    n = len(dataset)
    val_size = int(n * val_split)
    if val_size <= 0:
        return dataset, None
    if val_size >= n:
        val_size = n - 1
    order = _ordered_indices(dataset, chronological, seed)
    train_idxs = order[: n - val_size]
    val_idxs = order[n - val_size :]
    return Subset(dataset, train_idxs), Subset(dataset, val_idxs)


def _mean_loss(model, loader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            student_idx = batch["student_idx"].long().to(device)
            item_idx = batch["item_idx"].long().to(device)
            correct = batch["correct"].float().to(device)
            logits = model.logits(student_idx, item_idx)
            loss = criterion(logits, correct)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def train_mgirt_mle(
    data_or_path,
    emb_dim=2,
    alpha=1.0,
    beta=1.0,
    batch_size=128,
    lr=1e-2,
    epochs=10,
    weight_decay=0.0,
    val_split=0.1,
    seed=42,
    chronological_val=True,
    early_stopping=True,
    patience=1,
    min_delta=0.0,
    plot_path=None,
    device=None,
):
    if isinstance(data_or_path, str):
        dataset = AssistmentsDataset(data_or_path)
    else:
        dataset = data_or_path

    base = _base_dataset(dataset)
    num_students = len(base.students)
    num_items = len(base.items)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MGIRTModel(num_students, num_items, emb_dim, alpha=alpha, beta=beta).to(device)
    train_ds, val_ds = _split_train_val(
        dataset,
        val_split=val_split,
        seed=seed,
        chronological=chronological_val,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            student_idx = batch["student_idx"].long().to(device)
            item_idx = batch["item_idx"].long().to(device)
            correct = batch["correct"].float().to(device)

            logits = model.logits(student_idx, item_idx)
            loss = criterion(logits, correct)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        train_losses.append(mean_loss)

        if val_loader is not None:
            val_loss = _mean_loss(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            print(f"[MG-IRT MLE] Epoch {epoch+1}: Train = {mean_loss:.4f}, Val = {val_loss:.4f}")
            if early_stopping:
                if val_loss < best_val - min_delta:
                    best_val = val_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print(f"[MG-IRT MLE] Early stop at epoch {epoch+1}")
                        break
        else:
            print(f"[MG-IRT MLE] Epoch {epoch+1}: Loss = {mean_loss:.4f}")

    if plot_path:
        try:
            import matplotlib.pyplot as plt
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.figure()
            plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
            if val_losses:
                plt.plot(range(1, len(val_losses) + 1), val_losses, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"[MG-IRT MLE] Saved loss plot to {plot_path}")
        except Exception as exc:
            print(f"[MG-IRT MLE] Skipped plotting: {exc}")

    return model, dataset
