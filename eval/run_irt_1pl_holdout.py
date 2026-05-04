import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.assistments_dataset import AssistmentsDataset
from data.splits import split_by_student_holdout


class IRT1PLModel(torch.nn.Module):
    """
    1PL / Rasch model:
    p(correct|u,i) = sigmoid(theta_u - diff_i)
    """

    def __init__(self, num_students, num_items):
        super().__init__()
        self.student_theta = torch.nn.Embedding(num_students, 1)
        self.item_diff = torch.nn.Embedding(num_items, 1)

    def forward(self, student_idx, item_idx):
        theta = self.student_theta(student_idx).squeeze(-1)
        diff = self.item_diff(item_idx).squeeze(-1)
        return torch.sigmoid(theta - diff)


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


def _evaluate_batch(model, dataset, device="cpu", batch_size=16384, num_workers=0):
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
        for batch in loader:
            u = batch["student_idx"].long().to(device, non_blocking=pin_memory)
            i = batch["item_idx"].long().to(device, non_blocking=pin_memory)
            y = batch["correct"].float().cpu().numpy()
            p = model(u, i).detach().cpu().numpy()
            y_true.extend(y.tolist())
            y_pred.extend(p.tolist())
    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)
    acc = accuracy_score(y_true, y_pred >= 0.5)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    nll = log_loss(y_true, y_pred, labels=[0, 1]) if len(np.unique(y_true)) > 1 else float("nan")
    return {"accuracy": float(acc), "auc": float(auc), "nll": float(nll)}


def _evaluate_holdout_online_1pl(
    model,
    test_set,
    theta_lr=0.3,
    theta_prior_mean=0.0,
    theta_prior_std=1.0,
    seed=42,
):
    if theta_prior_std <= 0:
        raise ValueError("theta_prior_std must be > 0")
    if theta_lr <= 0:
        raise ValueError("theta_lr must be > 0")

    frame = _resolve_eval_frame(test_set)
    required = {"student_idx", "item_idx", "correct"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    if "answer_number" in frame.columns:
        frame = frame.sort_values(["student_idx", "answer_number", "__row_idx"])
    else:
        frame = frame.sort_values(["student_idx", "__row_idx"])

    diff_table = model.item_diff.weight.detach().cpu().squeeze(-1).numpy()
    rng = np.random.default_rng(seed)
    prior_var = float(theta_prior_std) ** 2
    prior_mean = float(theta_prior_mean)

    y_true = []
    y_pred = []
    for _, group in frame.groupby("student_idx", sort=False):
        theta = float(rng.normal(loc=prior_mean, scale=theta_prior_std))
        items = group["item_idx"].to_numpy(dtype=np.int64)
        labels = group["correct"].to_numpy(dtype=np.float32)
        for item_idx, y in zip(items, labels):
            diff = float(diff_table[item_idx])
            p = 1.0 / (1.0 + np.exp(-(theta - diff)))
            y_true.append(float(y))
            y_pred.append(float(p))
            grad = (p - float(y)) + (theta - prior_mean) / prior_var
            theta = theta - float(theta_lr) * grad

    y_true = np.array(y_true)
    y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)
    acc = accuracy_score(y_true, y_pred >= 0.5)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    nll = log_loss(y_true, y_pred, labels=[0, 1]) if len(np.unique(y_true)) > 1 else float("nan")
    return {"accuracy": float(acc), "auc": float(auc), "nll": float(nll)}


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate strict 1PL IRT on student holdout.")
    parser.add_argument("--data", type=str, default="pix_mapping/pix_irt_outcome.csv")
    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--student-l2", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--theta-lr", type=float, default=0.3)
    parser.add_argument("--theta-prior-mean", type=float, default=0.0)
    parser.add_argument("--theta-prior-std", type=float, default=1.0)
    parser.add_argument("--theta-seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = str(device).startswith("cuda")

    dataset = AssistmentsDataset(args.data, cache_path=(args.cache_path or None))
    train_set, test_set = split_by_student_holdout(dataset, test_size=args.test_size, seed=args.seed)
    base = train_set.dataset if hasattr(train_set, "dataset") else train_set
    num_students = len(base.students)
    num_items = len(base.items)

    print(
        f"split=student_holdout rows_total={len(dataset)} train_rows={len(train_set)} "
        f"test_rows={len(test_set)} num_students={num_students} num_items={num_items}"
    )
    print(
        f"model=1PL p=sigmoid(theta-diff) epochs={args.epochs} lr={args.lr} "
        f"student_l2={args.student_l2} theta_lr={args.theta_lr}"
    )

    model = IRT1PLModel(num_students, num_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            u = batch["student_idx"].long().to(device, non_blocking=pin_memory)
            i = batch["item_idx"].long().to(device, non_blocking=pin_memory)
            y = batch["correct"].float().to(device, non_blocking=pin_memory)
            optimizer.zero_grad()
            p = model(u, i)
            loss = criterion(p, y)
            if args.student_l2 > 0.0:
                theta = model.student_theta(u).squeeze(-1)
                loss = loss + args.student_l2 * theta.pow(2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"epoch={epoch} loss={float(np.mean(losses)):.4f}")

    train_metrics = _evaluate_batch(
        model,
        train_set,
        device=device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    test_metrics = _evaluate_holdout_online_1pl(
        model,
        test_set,
        theta_lr=args.theta_lr,
        theta_prior_mean=args.theta_prior_mean,
        theta_prior_std=args.theta_prior_std,
        seed=args.theta_seed,
    )
    print(
        f"train accuracy={train_metrics['accuracy']:.4f} auc={train_metrics['auc']:.4f} "
        f"nll={train_metrics['nll']:.4f}"
    )
    print(
        f"test_online_1pl accuracy={test_metrics['accuracy']:.4f} auc={test_metrics['auc']:.4f} "
        f"nll={test_metrics['nll']:.4f}"
    )


if __name__ == "__main__":
    main()
