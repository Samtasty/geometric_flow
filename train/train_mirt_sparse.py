import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader, Subset

from data.irt_tensor_dataset import IRTTensorDataset
from models.mirt_sparse import SparseMIRTModel


def _base_dataset(dataset):
    if isinstance(dataset, Subset):
        return dataset.dataset
    return dataset


def temporal_split_by_student(dataset, train_ratio=0.75):
    """
    Temporal split per student:
    first train_ratio interactions -> train, rest -> test.
    Uses answer_number when available, else row order.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")

    if isinstance(dataset, Subset):
        base = dataset.dataset
        global_idx = np.asarray(dataset.indices, dtype=np.int64)
    else:
        base = dataset
        global_idx = np.arange(len(base), dtype=np.int64)

    users = base.users[global_idx].cpu().numpy()
    row_order = np.arange(len(global_idx), dtype=np.int64)
    if getattr(base, "answer_number", None) is not None:
        answer = base.answer_number[global_idx].cpu().numpy()
        order = np.lexsort((row_order, answer, users))
    else:
        order = np.lexsort((row_order, users))

    sorted_idx = global_idx[order]
    sorted_users = users[order]
    boundaries = np.concatenate(
        ([0], np.flatnonzero(np.diff(sorted_users)) + 1, [len(sorted_users)])
    )

    train_idx = []
    test_idx = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        rows = sorted_idx[a:b]
        n = len(rows)
        if n < 2:
            train_idx.extend(rows.tolist())
            continue
        cut = int(n * train_ratio)
        if cut <= 0:
            cut = 1
        if cut >= n:
            cut = n - 1
        train_idx.extend(rows[:cut].tolist())
        test_idx.extend(rows[cut:].tolist())

    return Subset(base, train_idx), Subset(base, test_idx)


def evaluate_sparse_mirt(model, dataset, batch_size=16384, device="cpu", num_workers=0):
    pin_memory = str(device).startswith("cuda")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for u, i, y in loader:
            u = u.long().to(device, non_blocking=pin_memory)
            i = i.long().to(device, non_blocking=pin_memory)
            p = model(u, i).detach().cpu().numpy()
            y_true.extend(y.numpy().tolist())
            y_pred.extend(p.tolist())

    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)
    acc = accuracy_score(y_true, y_pred >= 0.5)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    nll = log_loss(y_true, y_pred, labels=[0, 1]) if len(np.unique(y_true)) > 1 else float("nan")
    return {"accuracy": float(acc), "auc": float(auc), "nll": float(nll)}


def train_sparse_mirt(
    data_or_dataset,
    emb_dim=2,
    batch_size=2048,
    lr=0.01,
    epochs=10,
    student_l2=1e-3,
    item_l2=0.0,
    device=None,
    num_workers=0,
    eval_train_set=None,
    eval_test_set=None,
    eval_every=1,
    eval_batch_size=16384,
    show_progress=False,
):
    if isinstance(data_or_dataset, str):
        dataset = IRTTensorDataset(data_or_dataset)
    else:
        dataset = data_or_dataset

    base = _base_dataset(dataset)
    num_users = base.num_users
    num_items = base.num_items

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = str(device).startswith("cuda")

    model = SparseMIRTModel(num_users, num_items, emb_dim=emb_dim).to(device)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    history = {"epoch": [], "loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        iterable = loader
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterable = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            except Exception:
                iterable = loader
        for u, i, y in iterable:
            u = u.long().to(device, non_blocking=pin_memory)
            i = i.long().to(device, non_blocking=pin_memory)
            y = y.float().to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            probs = model(u, i)
            loss = criterion(probs, y)

            if student_l2 > 0.0:
                theta = model.student_emb(u)
                loss = loss + student_l2 * theta.pow(2).sum(dim=1).mean()
            if item_l2 > 0.0:
                alpha_raw = model.item_emb(i)
                loss = loss + item_l2 * alpha_raw.pow(2).sum(dim=1).mean()

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if show_progress and hasattr(iterable, "set_postfix"):
                iterable.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history["epoch"].append(epoch)
        history["loss"].append(mean_loss)

        train_acc = float("nan")
        test_acc = float("nan")
        if eval_every > 0 and (epoch % eval_every == 0):
            if eval_train_set is not None:
                train_acc = evaluate_sparse_mirt(
                    model,
                    eval_train_set,
                    batch_size=eval_batch_size,
                    device=device,
                    num_workers=num_workers,
                )["accuracy"]
            if eval_test_set is not None:
                test_acc = evaluate_sparse_mirt(
                    model,
                    eval_test_set,
                    batch_size=eval_batch_size,
                    device=device,
                    num_workers=num_workers,
                )["accuracy"]
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(
            f"epoch={epoch} loss={mean_loss:.4f} "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
        )

    return model, history
