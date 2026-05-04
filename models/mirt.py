import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class MIRTItemModule(nn.Module):
    """
    Item-side module for MIRT.
    Useful for DDP where item parameters are synchronized but student embeddings remain local.
    """

    def __init__(self, num_items, emb_dim=2):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.item_bias = nn.Embedding(num_items, 1)

    def logits(self, theta, item_idx):
        beta = F.softplus(self.item_emb(item_idx))  # non-negative coordinates
        bias = self.item_bias(item_idx).squeeze(-1)
        return (theta * beta).sum(dim=1) + bias

    def forward(self, theta, item_idx):
        return torch.sigmoid(self.logits(theta, item_idx))


class MIRTModel(nn.Module):
    def __init__(self, num_students, num_items, emb_dim=2):
        super().__init__()
        self.student_emb = nn.Embedding(num_students, emb_dim)
        self.item_module = MIRTItemModule(num_items, emb_dim)
        # Compatibility aliases used elsewhere in the repo.
        self.item_emb = self.item_module.item_emb
        self.item_bias = self.item_module.item_bias

    def forward(self, student_idx, item_idx):
        theta = self.student_emb(student_idx)  # (batch, emb_dim)
        return self.item_module(theta, item_idx)


class MIRTModelEM(nn.Module):
    def __init__(self, num_students, num_items, emb_dim=2, prior_mean=0.0, prior_std=1.0, num_em_iter=5):
        super().__init__()
        self.num_students = num_students
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.student_emb = nn.Embedding(num_students, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.item_bias = nn.Embedding(num_items, 1)
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.num_em_iter = num_em_iter

    def forward(self, student_idx, item_idx):
        theta = self.student_emb(student_idx)
        beta = F.softplus(self.item_emb(item_idx))
        bias = self.item_bias(item_idx).squeeze(-1)
        logits = (theta * beta).sum(dim=1) + bias
        prob = torch.sigmoid(logits)
        return prob

    def _extract_frame(self, dataset):
        # Support both full datasets and index-based subsets.
        if hasattr(dataset, "dataset") and hasattr(dataset, "idxs") and hasattr(dataset.dataset, "data"):
            idxs = np.asarray(dataset.idxs, dtype=np.int64)
            return dataset.dataset.data.iloc[idxs]
        if hasattr(dataset, "data"):
            return dataset.data
        raise ValueError("Expected dataset with .data or subset exposing .dataset/.idxs")

    def _build_student_buffers(self, dataset, device):
        full_data = self._extract_frame(dataset)
        grouped = full_data.groupby("student_idx", sort=False)
        student_items = {}
        student_correct = {}
        for s_idx, group in grouped:
            s = int(s_idx)
            student_items[s] = torch.tensor(
                group["item_idx"].to_numpy(), dtype=torch.long, device=device
            )
            student_correct[s] = torch.tensor(
                group["correct"].to_numpy(), dtype=torch.float, device=device
            )
        return student_items, student_correct

    def em_train(self, dataset, batch_size=128, lr=1e-2, epochs=10, device=None, num_workers=0):
        """
        EM training loop for MIRT with prior on student knowledge state.
        E-step: Estimate posterior of student ability given current item params and responses.
        M-step: Update item params and bias by maximizing expected log-likelihood.
        """
        import torch.optim as optim
        from torch.utils.data import DataLoader
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        pin_memory = str(device).startswith("cuda")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        optimizer = optim.Adam(list(self.item_emb.parameters()) + list(self.item_bias.parameters()), lr=lr)
        student_items, student_correct = self._build_student_buffers(dataset, device=device)
        student_ids = sorted(student_items.keys())
        for epoch in range(epochs):
            # E-step: update student embeddings (MAP estimate for each student)
            for s in student_ids:
                item_idx = student_items[s]
                correct = student_correct[s]
                # MAP estimate: maximize log posterior
                theta = self.student_emb.weight[s].detach().clone().requires_grad_(True)
                for _ in range(self.num_em_iter):
                    beta = F.softplus(self.item_emb(item_idx)).detach()
                    bias = self.item_bias(item_idx).squeeze(-1).detach()
                    logits = (theta * beta).sum(dim=1) + bias
                    log_lik = F.binary_cross_entropy_with_logits(logits, correct, reduction='sum')
                    # Gaussian prior
                    log_prior = 0.5 * ((theta - self.prior_mean) ** 2).sum() / (self.prior_std ** 2)
                    loss = log_lik + log_prior
                    grad = torch.autograd.grad(loss, theta)[0]
                    theta = theta - lr * grad
                self.student_emb.weight.data[s] = theta.detach()
            # M-step: update item params (item_emb, item_bias)
            self.item_emb.weight.requires_grad = True
            self.item_bias.weight.requires_grad = True
            self.student_emb.weight.requires_grad = False
            losses = []
            for batch in loader:
                student_idx = batch['student_idx'].long().to(device, non_blocking=pin_memory)
                item_idx = batch['item_idx'].long().to(device, non_blocking=pin_memory)
                correct = batch['correct'].float().to(device, non_blocking=pin_memory)
                theta = self.student_emb(student_idx)
                beta = F.softplus(self.item_emb(item_idx))
                bias = self.item_bias(item_idx).squeeze(-1)
                logits = (theta * beta).sum(dim=1) + bias
                loss = F.binary_cross_entropy_with_logits(logits, correct)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print(f"[EM] Epoch {epoch+1}: M-step Loss = {sum(losses)/len(losses):.4f}")
        # Re-enable gradients for all
        self.student_emb.weight.requires_grad = True
        self.item_emb.weight.requires_grad = True
        self.item_bias.weight.requires_grad = True

    def _split_chunks(self, values, chunk_size):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]

    def _solve_student_chunk(
        self,
        chunk_students,
        theta_snapshot,
        student_items,
        student_correct,
        beta_table,
        bias_table,
        lr_step,
    ):
        updates = []
        prior_scale = float(self.prior_std) ** 2
        for s in chunk_students:
            theta = theta_snapshot[s].detach().clone().requires_grad_(True)
            item_idx = student_items[s]
            correct = student_correct[s]
            beta = beta_table[item_idx]
            bias = bias_table[item_idx]
            for _ in range(self.num_em_iter):
                logits = (theta * beta).sum(dim=1) + bias
                log_lik = F.binary_cross_entropy_with_logits(logits, correct, reduction="sum")
                log_prior = 0.5 * ((theta - self.prior_mean) ** 2).sum() / prior_scale
                loss = log_lik + log_prior
                grad = torch.autograd.grad(loss, theta)[0]
                theta = (theta - lr_step * grad).detach().requires_grad_(True)
            updates.append((s, theta.detach()))
        return updates

    def em_train_async_online(
        self,
        dataset,
        batch_size=1024,
        lr=1e-2,
        epochs=10,
        device=None,
        num_workers=0,
        e_workers=12,
        student_chunk_size=256,
        participation_rate=1.0,
        m_step_batches_per_round=1,
        seed=42,
    ):
        """
        Asynchronous/online EM variant:
        - E-step: parallel MAP updates for student chunks
        - M-step: interleave mini-batch item updates after each E-step chunk completion
        """
        import torch.optim as optim
        from torch.utils.data import DataLoader

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if not 0.0 < participation_rate <= 1.0:
            raise ValueError("participation_rate must be in (0, 1]")
        if m_step_batches_per_round <= 0:
            raise ValueError("m_step_batches_per_round must be > 0")

        self.to(device)
        pin_memory = str(device).startswith("cuda")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        optimizer = optim.Adam(
            list(self.item_emb.parameters()) + list(self.item_bias.parameters()),
            lr=lr,
        )
        student_items, student_correct = self._build_student_buffers(dataset, device="cpu")
        student_ids = sorted(student_items.keys())
        if not student_ids:
            raise ValueError("No students found in dataset for EM training.")

        self.student_emb.weight.requires_grad = False
        self.item_emb.weight.requires_grad = True
        self.item_bias.weight.requires_grad = True

        for epoch in range(epochs):
            rng = np.random.default_rng(seed + epoch)
            if participation_rate < 1.0:
                n_active = max(1, int(len(student_ids) * participation_rate))
                active_students = rng.choice(np.array(student_ids), size=n_active, replace=False).tolist()
            else:
                active_students = student_ids.copy()
            rng.shuffle(active_students)
            chunks = self._split_chunks(active_students, student_chunk_size)

            # Snapshot item params once per epoch for E-step consistency.
            with torch.no_grad():
                beta_table = F.softplus(self.item_emb.weight).detach().cpu()
                bias_table = self.item_bias.weight.squeeze(-1).detach().cpu()
                theta_snapshot = self.student_emb.weight.detach().cpu()

            m_iter = iter(loader)
            m_losses = []

            if e_workers <= 1:
                chunk_results = []
                for chunk in chunks:
                    chunk_results.append(
                        self._solve_student_chunk(
                            chunk,
                            theta_snapshot,
                            student_items,
                            student_correct,
                            beta_table,
                            bias_table,
                            lr,
                        )
                    )
            else:
                chunk_results = []
                with ThreadPoolExecutor(max_workers=e_workers) as pool:
                    futures = [
                        pool.submit(
                            self._solve_student_chunk,
                            chunk,
                            theta_snapshot,
                            student_items,
                            student_correct,
                            beta_table,
                            bias_table,
                            lr,
                        )
                        for chunk in chunks
                    ]
                    for fut in as_completed(futures):
                        chunk_results.append(fut.result())

            for updates in chunk_results:
                with torch.no_grad():
                    for s, theta_new in updates:
                        self.student_emb.weight.data[s] = theta_new.to(device)

                for _ in range(m_step_batches_per_round):
                    try:
                        batch = next(m_iter)
                    except StopIteration:
                        m_iter = iter(loader)
                        batch = next(m_iter)
                    student_idx = batch["student_idx"].long().to(device, non_blocking=pin_memory)
                    item_idx = batch["item_idx"].long().to(device, non_blocking=pin_memory)
                    correct = batch["correct"].float().to(device, non_blocking=pin_memory)

                    theta = self.student_emb(student_idx)
                    beta = F.softplus(self.item_emb(item_idx))
                    bias = self.item_bias(item_idx).squeeze(-1)
                    logits = (theta * beta).sum(dim=1) + bias
                    loss = F.binary_cross_entropy_with_logits(logits, correct)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    m_losses.append(loss.item())

            mean_loss = float(np.mean(m_losses)) if m_losses else float("nan")
            print(
                f"[EM-ASYNC] Epoch {epoch+1}: "
                f"active_students={len(active_students)}, chunks={len(chunks)}, "
                f"M-step Loss = {mean_loss:.4f}"
            )

        self.student_emb.weight.requires_grad = True
        self.item_emb.weight.requires_grad = True
        self.item_bias.weight.requires_grad = True
