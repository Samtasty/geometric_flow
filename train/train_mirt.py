import torch
from torch.utils.data import DataLoader
from models.mirt import MIRTModel
from data.assistments_dataset import AssistmentsDataset
from data.student_batch_sampler import StudentBatchSampler, MixedStudentBatchSampler
import torch.optim as optim
import numpy as np

def _base_dataset(dataset):
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "students"):
        return dataset.dataset
    return dataset


def _mean_bce(model, dataset, batch_size=4096, device="cpu", num_workers=0):
    pin_memory = device.startswith("cuda")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    criterion = torch.nn.BCELoss()
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            student_idx = batch["student_idx"].long().to(device, non_blocking=pin_memory)
            item_idx = batch["item_idx"].long().to(device, non_blocking=pin_memory)
            correct = batch["correct"].float().to(device, non_blocking=pin_memory)
            pred = model(student_idx, item_idx)
            loss = criterion(pred, correct)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def _renorm_embeddings(model, eps=1e-8):
    with torch.no_grad():
        for emb in (model.student_emb, model.item_emb):
            w = emb.weight.data
            norms = w.norm(dim=1, keepdim=True).clamp_min(eps)
            emb.weight.data = w / norms


def train_mirt(
    data_or_path,
    emb_dim=2,
    batch_size=128,
    lr=1e-2,
    epochs=10,
    val_set=None,
    early_stopping=False,
    patience=3,
    min_delta=0.0,
    renorm=False,
    device=None,
    num_workers=0,
    batch_mode="random",
    seed=42,
    students_per_batch=64,
    interactions_per_student=None,
    student_l2=0.0,
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
    pin_memory = device.startswith("cuda")
    model = MIRTModel(num_students, num_items, emb_dim).to(device)
    if batch_mode not in {"random", "student", "student_mix"}:
        raise ValueError("batch_mode must be 'random', 'student', or 'student_mix'")

    if batch_mode == "student":
        batch_sampler = StudentBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            shuffle_within_student=True,
            drop_last=False,
            seed=seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    elif batch_mode == "student_mix":
        batch_sampler = MixedStudentBatchSampler(
            dataset,
            batch_size=batch_size,
            students_per_batch=students_per_batch,
            interactions_per_student=interactions_per_student,
            shuffle=True,
            drop_last=False,
            seed=seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    model.train()
    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            student_idx = batch['student_idx'].long().to(device, non_blocking=pin_memory)
            item_idx = batch['item_idx'].long().to(device, non_blocking=pin_memory)
            correct = batch['correct'].float().to(device, non_blocking=pin_memory)
            pred = model(student_idx, item_idx)
            loss = criterion(pred, correct)
            if student_l2 > 0.0:
                theta = model.student_emb(student_idx)
                # MAP-style stabilization on student abilities.
                loss = loss + student_l2 * theta.pow(2).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if renorm:
                _renorm_embeddings(model)
            losses.append(loss.item())
        mean_loss = float(np.mean(losses)) if losses else float("nan")

        if val_set is not None:
            val_loss = _mean_bce(model, val_set, device=device, num_workers=num_workers)
            print(f"Epoch {epoch+1}: Train Loss = {mean_loss:.4f}, Val Loss = {val_loss:.4f}")
            if early_stopping:
                if val_loss < best_val - min_delta:
                    best_val = val_loss
                    bad_epochs = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        else:
            print(f"Epoch {epoch+1}: Loss = {mean_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, dataset
