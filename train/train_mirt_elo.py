import math
import torch
import numpy as np

from models.mirt import MIRTModel
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


def train_mirt_elo(
    data_or_path,
    emb_dim=2,
    lr=0.05,
    epochs=1,
    lr_decay=0.0,
    decay_mode="sqrt",
    l2=0.0,
    init_std=0.01,
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

    model = MIRTModel(num_students, num_items, emb_dim).to(device)

    # Elo-style training is sensitive to initialization; small values work well.
    with torch.no_grad():
        model.student_emb.weight.normal_(0.0, init_std)
        model.item_emb.weight.normal_(0.0, init_std)
        model.item_bias.weight.zero_()

    student_counts = torch.zeros(num_students, dtype=torch.long)
    item_counts = torch.zeros(num_items, dtype=torch.long)
    for epoch in range(epochs):
        losses = []
        for sample in _iter_samples(dataset):
            s = int(sample["student_idx"])
            i = int(sample["item_idx"])
            y = float(sample["correct"])

            with torch.no_grad():
                theta = model.student_emb.weight[s]
                beta = model.item_emb.weight[i]
                bias = model.item_bias.weight[i, 0]

                logit = (theta * beta).sum() + bias
                p = torch.sigmoid(logit)
                err = y - p

                step = lr
                if lr_decay > 0.0:
                    count = student_counts[s].item() + item_counts[i].item()
                    if decay_mode == "linear":
                        step = lr / (1.0 + lr_decay * count)
                    else:
                        step = lr / (1.0 + lr_decay * math.sqrt(count))

                theta_old = theta.clone()
                theta += step * err * beta
                beta += step * err * theta_old
                bias += step * err

                if l2 > 0.0:
                    theta *= (1.0 - l2 * step)
                    beta *= (1.0 - l2 * step)

                model.item_bias.weight[i, 0] = bias

                # Track loss for monitoring
                loss = torch.nn.functional.binary_cross_entropy(p, torch.tensor(y, device=p.device))
                losses.append(loss.item())

            student_counts[s] += 1
            item_counts[i] += 1

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[ELO] Epoch {epoch+1}: Loss = {mean_loss:.4f}")

    return model, dataset
