import math
import torch
import numpy as np

from models.mgirt import MGIRTModel
from data.assistments_dataset import AssistmentsDataset


def _base_dataset(dataset):
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "students"):
        return dataset.dataset
    return dataset


def _iter_samples(dataset):
    if hasattr(dataset, "dataset") and hasattr(dataset, "idxs"):
        base = dataset.dataset
        for idx in dataset.idxs:
            yield base[idx]
        return
    for i in range(len(dataset)):
        yield dataset[i]


def train_mgirt_elo(
    data_or_path,
    emb_dim=4,
    alpha=1.0,
    beta=1.0,
    lr_item=0.01,
    lr_pos=0.1,
    lr_neg=0.1,
    epochs=20,
    lr_decay=0.0,
    decay_mode="sqrt",
    l2=0.0,
    init_std=0.01,
    normalize_items=True,
    eps=1e-7,
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

    model = MGIRTModel(num_students, num_items, emb_dim, alpha=alpha, beta=beta, eps=eps).to(device)

    with torch.no_grad():
        model.student_emb.weight.normal_(0.0, init_std)
        model.item_emb.weight.normal_(0.0, init_std)
        # Normalize student vectors to the sphere
        z = model.student_emb.weight
        z /= (z.norm(dim=1, keepdim=True) + eps)
        if normalize_items:
            w = model.item_emb.weight
            w /= (w.norm(dim=1, keepdim=True) + eps)

    student_counts = torch.zeros(num_students, dtype=torch.long)
    item_counts = torch.zeros(num_items, dtype=torch.long)

    for epoch in range(epochs):
        losses = []
        for sample in _iter_samples(dataset):
            s_idx = int(sample["student_idx"])
            i_idx = int(sample["item_idx"])
            y = float(sample["correct"])

            with torch.no_grad():
                z = model.student_emb.weight[s_idx]
                w = model.item_emb.weight[i_idx]

                z = z / (z.norm() + eps)
                z_old = z.clone()
                w_old = w.clone()

                s = (z_old * w_old).sum()
                w_perp = w_old - s * z_old
                g = torch.sqrt((w_perp * w_perp).sum() + eps)
                logit = model.alpha * s - model.beta * g
                p = torch.sigmoid(logit)

                step_item = lr_item
                step_pos = lr_pos
                step_neg = lr_neg
  

                # Update item vector using gradient of log-likelihood
                err = y - p
                if g.item() > 0:
                    grad_w = err * (model.alpha * z_old - model.beta * (w_perp / g))
                else:
                    grad_w = err * (model.alpha * z_old)
                w_new = w_old + step_item * grad_w
                if l2 > 0.0:
                    w_new = w_new * (1.0 - l2 * step_item)
                if normalize_items:
                    w_new = w_new / (w_new.norm() + eps)

                # Update student state with soft pruning + retraction
                if y >= 0.5:
                    z_new = z_old + step_pos * (1.0 - p) * w_perp
                else:
                    denom = (w_old * w_old).sum() + eps
                    proj_w = (s / denom) * w_old
                    z_new = z_old - step_neg * p * proj_w
                z_new = z_new / (z_new.norm() + eps)

                model.item_emb.weight[i_idx] = w_new
                model.student_emb.weight[s_idx] = z_new

                loss = torch.nn.functional.binary_cross_entropy(p, torch.tensor(y, device=p.device))
                losses.append(loss.item())

            student_counts[s_idx] += 1
            item_counts[i_idx] += 1

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[MG-IRT ELO] Epoch {epoch+1}: Loss = {mean_loss:.4f}")

    return model, dataset
