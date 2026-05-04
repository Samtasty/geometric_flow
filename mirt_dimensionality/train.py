"""
Training loop for M-IRT with student-only Gaussian regularisation.

Loss:
    L = BCE(logits, y) + lambda_theta * mean_batch(||theta_u||^2)

Item parameters (a_raw, d) are left unpenalised so all representational
capacity flows into learning item structure.  Student embeddings are pulled
toward 0 (N(0, 1/lambda_theta) prior), preventing them from memorising
individual response patterns on a small data batch.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from .model import MIRT


def _make_tensors(df):
    return (
        torch.tensor(df["uid"].values,     dtype=torch.long),
        torch.tensor(df["cid"].values,     dtype=torch.long),
        torch.tensor(df["correct"].values, dtype=torch.float32),
    )


def _eval(model, te_u, te_c, te_y, criterion, device, chunk=16_384):
    model.eval()
    probs_all, y_all = [], []
    with torch.no_grad():
        for i in range(0, len(te_u), chunk):
            p = torch.sigmoid(model(te_u[i:i+chunk].to(device),
                                    te_c[i:i+chunk].to(device))).cpu()
            probs_all.append(p)
            y_all.append(te_y[i:i+chunk])
    probs = torch.cat(probs_all)
    y     = torch.cat(y_all)
    loss  = criterion(torch.logit(probs.clamp(1e-6, 1 - 1e-6)), y).item()
    auc   = roc_auc_score(y.numpy(), probs.numpy())
    return loss, auc


def train(
    df_train,
    df_test,
    K: int,
    lambda_theta: float = 1.0,
    n_epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 5e-3,
    lr_schedule: str = "cosine",   # "cosine" | "none"
    warmup_epochs: int = 2,
    min_lr_ratio: float = 0.05,    # floor = lr * min_lr_ratio for cosine
    patience: int = 0,             # early stopping on test_loss; 0 = disabled
    device=None,
    verbose: bool = True,
):
    """
    Train one M-IRT model with student Gaussian regularisation.

    LR schedule options
    -------------------
    cosine : linear warmup for `warmup_epochs`, then cosine decay to
             lr * min_lr_ratio over the remaining epochs.
    none   : constant lr throughout.

    Returns (model, history) where history has keys:
        train_bce, train_reg, train_loss, test_loss, test_auc, lr_items, lr_students
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_users      = int(df_train["uid"].max()) + 1
    n_challenges = int(
        max(df_train["cid"].max(), df_test["cid"].max())
    ) + 1

    model = MIRT(n_users, n_challenges, K).to(device)

    # Separate optimisers — items get no penalty from the optimiser itself
    opt_items    = optim.Adam(
        list(model.a_raw.parameters()) + list(model.d.parameters()), lr=lr
    )
    opt_students = optim.Adam(model.theta.parameters(), lr=lr)

    # LR schedulers (applied to both optimisers together)
    def _make_scheduler(opt):
        if lr_schedule == "cosine":
            # Warmup: linear ramp from 0 → lr over warmup_epochs,
            # then cosine from lr → lr*min_lr_ratio over remaining epochs.
            def _lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / max(warmup_epochs, 1)
                progress = (epoch - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
                cosine   = 0.5 * (1 + np.cos(np.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
            return optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
        return None  # constant lr

    sched_items    = _make_scheduler(opt_items)
    sched_students = _make_scheduler(opt_students)

    criterion = nn.BCEWithLogitsLoss()

    tr_u, tr_c, tr_y = _make_tensors(df_train)
    te_u, te_c, te_y = _make_tensors(df_test)
    loader = DataLoader(TensorDataset(tr_u, tr_c, tr_y),
                        batch_size=batch_size, shuffle=True)

    history = {k: [] for k in ("train_bce", "train_reg", "train_loss",
                                "test_loss", "test_auc", "lr_items", "lr_students")}

    best_test_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_bce = total_reg = total_loss = 0.0
        n_batches = 0

        for u_b, c_b, y_b in loader:
            u_b, c_b, y_b = u_b.to(device), c_b.to(device), y_b.to(device)

            opt_items.zero_grad()
            opt_students.zero_grad()

            bce = criterion(model(u_b, c_b), y_b)
            reg = lambda_theta * model.theta(u_b).pow(2).mean()
            loss = bce + reg
            loss.backward()

            opt_items.step()
            opt_students.step()

            total_bce  += bce.item()
            total_reg  += reg.item()
            total_loss += loss.item()
            n_batches  += 1

        if sched_items:
            sched_items.step()
        if sched_students:
            sched_students.step()

        train_bce  = total_bce  / n_batches
        train_reg  = total_reg  / n_batches
        train_loss = total_loss / n_batches
        cur_lr_items    = opt_items.param_groups[0]["lr"]
        cur_lr_students = opt_students.param_groups[0]["lr"]

        test_loss, test_auc = _eval(model, te_u, te_c, te_y, criterion, device)

        history["train_bce"].append(train_bce)
        history["train_reg"].append(train_reg)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_auc"].append(test_auc)
        history["lr_items"].append(cur_lr_items)
        history["lr_students"].append(cur_lr_students)

        # Early stopping
        if patience > 0:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    if verbose:
                        print(f"  Early stop at epoch {epoch} (best test_loss={best_test_loss:.4f})")
                    model.load_state_dict(best_state)
                    break

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == n_epochs):
            marker = " *" if (patience > 0 and test_loss == best_test_loss) else ""
            print(
                f"  K={K:3d} | ep {epoch:3d}/{n_epochs} | "
                f"bce={train_bce:.4f}  reg={train_reg:.4f}  "
                f"test_loss={test_loss:.4f}  test_auc={test_auc:.4f}  "
                f"lr={cur_lr_items:.2e}{marker}"
            )

    if patience > 0 and best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def sweep(
    df_train,
    df_test,
    dims,
    lambda_theta: float = 1.0,
    n_epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 5e-3,
    lr_schedule: str = "cosine",
    warmup_epochs: int = 2,
    min_lr_ratio: float = 0.05,
    patience: int = 0,
    device=None,
):
    """Train one model per K value; return {K: {'model': ..., 'history': ...}}."""
    results = {}
    for K in dims:
        print(f"\n=== K={K} | λ_θ={lambda_theta} | schedule={lr_schedule} ===")
        model, history = train(
            df_train, df_test, K=K,
            lambda_theta=lambda_theta,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            lr_schedule=lr_schedule,
            warmup_epochs=warmup_epochs,
            min_lr_ratio=min_lr_ratio,
            patience=patience,
            device=device,
        )
        results[K] = {"model": model, "history": history}
    return results
