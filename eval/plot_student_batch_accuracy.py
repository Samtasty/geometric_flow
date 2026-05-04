import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.assistments_dataset import AssistmentsDataset
from data.splits import IndexedSubset, split_temporal_by_student
from data.student_batch_sampler import StudentBatchSampler, MixedStudentBatchSampler
from models.mirt import MIRTModel


def evaluate_accuracy(model, dataset, batch_size=8192, device="cpu", num_workers=0):
    pin_memory = str(device).startswith("cuda")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            s = batch["student_idx"].long().to(device, non_blocking=pin_memory)
            i = batch["item_idx"].long().to(device, non_blocking=pin_memory)
            y = batch["correct"].float().cpu().numpy()
            p = model(s, i).detach().cpu().numpy()
            y_true.extend(y.tolist())
            y_pred.extend((p >= 0.5).astype(float).tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def main():
    parser = argparse.ArgumentParser(
        description="Plot train/test accuracy vs epochs for MIRT with student-batch mode."
    )
    parser.add_argument("--data", type=str, default="pix_mapping/pix_irt_outcome.csv")
    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-students", type=int, default=5000)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--emb-dim", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--batch-mode",
        type=str,
        default="student",
        choices=["random", "student", "student_mix"],
    )
    parser.add_argument("--students-per-batch", type=int, default=64)
    parser.add_argument("--interactions-per-student", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="", help="cpu|cuda, empty=auto")
    parser.add_argument(
        "--out-plot",
        type=str,
        default="pix_mapping/mirt_student_batch_lr001_accuracy_vs_epoch.png",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="pix_mapping/mirt_student_batch_lr001_accuracy_vs_epoch.csv",
    )
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    dataset = AssistmentsDataset(args.data, cache_path=(args.cache_path or None))

    # Student-level sampling keeps student-batch training tractable.
    if args.max_students > 0:
        all_students = np.array(dataset.data["student_idx"].unique())
        n_take = min(args.max_students, len(all_students))
        selected_students = set(rng.choice(all_students, size=n_take, replace=False).tolist())
        selected_rows = dataset.data.index[dataset.data["student_idx"].isin(selected_students)].to_numpy()
        dataset = IndexedSubset(dataset, selected_rows)

    train_set, test_set = split_temporal_by_student(dataset, train_ratio=args.train_ratio)
    base = train_set.dataset if hasattr(train_set, "dataset") else train_set

    print(f"device={device}")
    print(f"train_rows={len(train_set)} test_rows={len(test_set)}")
    print(
        f"batch_mode={args.batch_mode} batch_size={args.batch_size} "
        f"students_per_batch={args.students_per_batch} "
        f"interactions_per_student={args.interactions_per_student}"
    )

    model = MIRTModel(len(base.students), len(base.items), emb_dim=args.emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    train_accs = []
    test_accs = []
    epochs = []

    for epoch in range(1, args.epochs + 1):
        pin_memory = device.startswith("cuda")
        if args.batch_mode == "student":
            sampler = StudentBatchSampler(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                shuffle_within_student=True,
                seed=args.seed + epoch,
            )
            loader = DataLoader(
                train_set,
                batch_sampler=sampler,
                pin_memory=pin_memory,
                num_workers=args.num_workers,
            )
        elif args.batch_mode == "student_mix":
            sampler = MixedStudentBatchSampler(
                train_set,
                batch_size=args.batch_size,
                students_per_batch=args.students_per_batch,
                interactions_per_student=(
                    None if args.interactions_per_student <= 0 else args.interactions_per_student
                ),
                shuffle=True,
                drop_last=False,
                seed=args.seed + epoch,
            )
            loader = DataLoader(
                train_set,
                batch_sampler=sampler,
                pin_memory=pin_memory,
                num_workers=args.num_workers,
            )
        else:
            loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=pin_memory,
                num_workers=args.num_workers,
            )

        model.train()
        losses = []
        for batch in loader:
            s = batch["student_idx"].long().to(device, non_blocking=pin_memory)
            i = batch["item_idx"].long().to(device, non_blocking=pin_memory)
            y = batch["correct"].float().to(device, non_blocking=pin_memory)
            p = model(s, i)
            loss = criterion(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_acc = evaluate_accuracy(
            model,
            train_set,
            batch_size=args.eval_batch_size,
            device=device,
            num_workers=args.num_workers,
        )
        test_acc = evaluate_accuracy(
            model,
            test_set,
            batch_size=args.eval_batch_size,
            device=device,
            num_workers=args.num_workers,
        )
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epochs.append(epoch)
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(
            f"epoch={epoch} loss={mean_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
        )

    # Save CSV
    import pandas as pd

    out_df = pd.DataFrame({"epoch": epochs, "train_acc": train_accs, "test_acc": test_accs})
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    # Save plot
    os.makedirs(os.path.dirname(args.out_plot), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, marker="o", label="Train Accuracy")
    plt.plot(epochs, test_accs, marker="o", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MIRT Accuracy vs Epoch (student batch, lr=0.01)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=140)
    plt.close()

    print(f"saved_plot={os.path.abspath(args.out_plot)}")
    print(f"saved_csv={os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
