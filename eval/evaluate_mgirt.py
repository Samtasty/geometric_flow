import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


def evaluate_mgirt(model, dataset):
    model.eval()
    y_true, y_pred = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            student_idx = torch.tensor([sample['student_idx']], device=device)
            item_idx = torch.tensor([sample['item_idx']], device=device)
            correct = sample['correct']
            prob = model(student_idx, item_idx).item()
            y_true.append(correct)
            y_pred.append(prob)

    acc = accuracy_score(y_true, [p >= 0.5 for p in y_pred])
    auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else float("nan")
    nll = log_loss(y_true, y_pred) if len(set(y_true)) > 1 else float("nan")
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}, NLL: {nll:.4f}")
    return {'accuracy': acc, 'auc': auc, 'nll': nll}
