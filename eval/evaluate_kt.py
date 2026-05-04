import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


def evaluate_kt(model, dataset, batch_size=256, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    y_true, y_pred = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            q_seq = batch["q_seq"].to(device)
            r_seq = batch["r_seq"].to(device)
            q_next = batch["q_next"].to(device)
            labels = batch["label"].to(device)
            preds = model(q_seq, r_seq, q_next)
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    if not y_true:
        print("No samples to evaluate.")
        return {"accuracy": float("nan"), "auc": float("nan"), "nll": float("nan")}

    y_pred = np.clip(np.array(y_pred), 1e-7, 1 - 1e-7)
    y_true_arr = np.array(y_true)

    acc = accuracy_score(y_true_arr, y_pred >= 0.5)
    auc = roc_auc_score(y_true_arr, y_pred) if len(np.unique(y_true_arr)) > 1 else float("nan")
    nll = log_loss(y_true_arr, y_pred, labels=[0, 1]) if len(np.unique(y_true_arr)) > 1 else float("nan")

    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}, NLL: {nll:.4f}")
    return {"accuracy": acc, "auc": auc, "nll": nll}
