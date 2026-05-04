import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


def evaluate_mirt(model, dataset, batch_size=4096, device=None, num_workers=0):
    if device is None:
        device = next(model.parameters()).device
    pin_memory = str(device).startswith("cuda")
    model.eval()
    y_true, y_pred = [], []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    with torch.no_grad():
        for batch in loader:
            student_idx = batch["student_idx"].long().to(device, non_blocking=pin_memory)
            item_idx = batch["item_idx"].long().to(device, non_blocking=pin_memory)
            correct = batch["correct"].float().cpu().numpy()
            prob = model(student_idx, item_idx).detach().cpu().numpy()
            y_true.extend(correct.tolist())
            y_pred.extend(prob.tolist())
    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)
    acc = accuracy_score(y_true, y_pred >= 0.5)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    nll = log_loss(y_true, y_pred, labels=[0, 1]) if len(np.unique(y_true)) > 1 else float("nan")
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}, NLL: {nll:.4f}")
    return {'accuracy': acc, 'auc': auc, 'nll': nll}


def evaluate_mirt_item_only(model, dataset, batch_size=4096, device=None, num_workers=0):
    """
    Evaluate with item-only signal for unseen-student settings.

    Uses sigmoid(item_bias[item]) and ignores student embedding.
    """
    if device is None:
        device = next(model.parameters()).device
    pin_memory = str(device).startswith("cuda")
    model.eval()
    y_true, y_pred = [], []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    with torch.no_grad():
        for batch in loader:
            item_idx = batch["item_idx"].long().to(device, non_blocking=pin_memory)
            correct = batch["correct"].float().cpu().numpy()
            bias = model.item_bias(item_idx).squeeze(-1)
            prob = torch.sigmoid(bias).detach().cpu().numpy()
            y_true.extend(correct.tolist())
            y_pred.extend(prob.tolist())
    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)
    acc = accuracy_score(y_true, y_pred >= 0.5)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    nll = log_loss(y_true, y_pred, labels=[0, 1]) if len(np.unique(y_true)) > 1 else float("nan")
    print(f"[Item-only] Accuracy: {acc:.4f}, AUC: {auc:.4f}, NLL: {nll:.4f}")
    return {"accuracy": acc, "auc": auc, "nll": nll}


def _resolve_eval_frame(dataset):
    if hasattr(dataset, "dataset") and hasattr(dataset, "idxs") and hasattr(dataset.dataset, "data"):
        idxs = np.asarray(dataset.idxs, dtype=np.int64)
        df = dataset.dataset.data.iloc[idxs].copy()
        df["__row_idx"] = idxs
        return df
    if hasattr(dataset, "data"):
        df = dataset.data.copy()
        df["__row_idx"] = np.arange(len(df), dtype=np.int64)
        return df
    raise ValueError("Expected dataset with .data or subset exposing .dataset/.idxs")


def _extract_item_tables(model):
    # Dense MIRTModel path (item_module exists).
    if hasattr(model, "item_module"):
        item_vec = F.softplus(model.item_module.item_emb.weight.detach().cpu())
        item_bias = model.item_module.item_bias.weight.detach().cpu().squeeze(-1)
        return item_vec, item_bias

    # Fallback for other models with item_emb/item_bias.
    raw = model.item_emb.weight.detach().cpu()
    if getattr(model.item_emb, "sparse", False):
        item_vec = torch.exp(raw)
    else:
        item_vec = F.softplus(raw)
    item_bias = model.item_bias.weight.detach().cpu().squeeze(-1)
    return item_vec, item_bias


def evaluate_mirt_student_holdout_online(
    model,
    dataset,
    theta_lr=0.1,
    theta_prior_mean=0.0,
    theta_prior_std=1.0,
    seed=42,
):
    """
    Sequential cold-start evaluation for student-holdout.

    For each unseen student in the test set:
    1) initialize theta_0 ~ N(theta_prior_mean, theta_prior_std^2 I)
    2) predict each interaction in time order
    3) update theta using observed outcome and fixed item parameters
    """
    if theta_prior_std <= 0:
        raise ValueError("theta_prior_std must be > 0")
    if theta_lr <= 0:
        raise ValueError("theta_lr must be > 0")

    model.eval()
    frame = _resolve_eval_frame(dataset)
    required = {"student_idx", "item_idx", "correct"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")

    if "answer_number" in frame.columns:
        frame = frame.sort_values(["student_idx", "answer_number", "__row_idx"])
    else:
        frame = frame.sort_values(["student_idx", "__row_idx"])

    item_vec, item_bias = _extract_item_tables(model)
    emb_dim = int(item_vec.shape[1])
    prior_var = float(theta_prior_std) ** 2
    prior_mean = float(theta_prior_mean)
    rng = np.random.default_rng(seed)

    y_true = []
    y_pred = []

    for _, group in frame.groupby("student_idx", sort=False):
        theta = torch.tensor(
            rng.normal(loc=prior_mean, scale=theta_prior_std, size=(emb_dim,)),
            dtype=torch.float32,
        )
        items = group["item_idx"].to_numpy(dtype=np.int64)
        labels = group["correct"].to_numpy(dtype=np.float32)

        for item_idx, y in zip(items, labels):
            beta = item_vec[item_idx]
            bias = item_bias[item_idx]
            logit = (theta * beta).sum() + bias
            p = torch.sigmoid(logit).item()
            y_true.append(float(y))
            y_pred.append(float(p))

            grad = (p - float(y)) * beta + (theta - prior_mean) / prior_var
            theta = theta - float(theta_lr) * grad

    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)
    acc = accuracy_score(y_true, y_pred >= 0.5)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    nll = log_loss(y_true, y_pred, labels=[0, 1]) if len(np.unique(y_true)) > 1 else float("nan")
    print(
        "[Holdout-online-theta] "
        f"Accuracy: {acc:.4f}, AUC: {auc:.4f}, NLL: {nll:.4f} "
        f"(theta_lr={theta_lr}, prior_mean={theta_prior_mean}, prior_std={theta_prior_std}, seed={seed})"
    )
    return {"accuracy": acc, "auc": auc, "nll": nll}
