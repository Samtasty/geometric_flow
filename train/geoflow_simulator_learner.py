"""GeoFlow sequence simulator and embedding learner.

This module extracts the essential modeling code from `data_analysis.ipynb`
into a reusable Python file.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import geoopt

    _HAS_GEOOPT = True
except Exception:
    geoopt = None
    _HAS_GEOOPT = False


def np_row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def np_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    return v / max(n, eps)


def sigmoid_np(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def stable_softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits)
    p = np.exp(x)
    return p / p.sum()


def dataframe_to_seqs(
    df: pd.DataFrame,
    user_col: str = "user",
    item_col: str = "item",
    resp_col: str = "correct",
    order_cols: list[str] | None = None,
    threshold: float = 0.5,
    min_user_len: int = 5,
) -> dict[str, Any]:
    cols = [user_col, item_col, resp_col] + (order_cols or [])
    x = df[cols].dropna(subset=[user_col, item_col, resp_col]).copy()

    if order_cols:
        x = x.sort_values([user_col] + order_cols, kind="mergesort")
    else:
        x = x.reset_index(drop=False).sort_values([user_col, "index"], kind="mergesort")

    u_idx, u_vals = pd.factorize(x[user_col], sort=False)
    i_idx, i_vals = pd.factorize(x[item_col], sort=False)

    r = x[resp_col].to_numpy()
    if np.issubdtype(r.dtype, np.floating):
        r = (r >= threshold).astype(np.int64)
    else:
        r = r.astype(np.int64)

    x["_u"] = u_idx.astype(np.int64)
    x["_i"] = i_idx.astype(np.int64)
    x["_r"] = r

    seqs: list[list[tuple[int, int]]] = []
    for _, g in x.groupby("_u", sort=False):
        seq = list(zip(g["_i"].tolist(), g["_r"].tolist()))
        if len(seq) >= min_user_len:
            seqs.append(seq)

    return {
        "seqs": seqs,
        "n_users": len(seqs),
        "n_items": len(i_vals),
        "user_values": u_vals,
        "item_values": i_vals,
    }


def split_user_sequences(
    seqs: list[list[tuple[int, int]]], train_frac: float = 0.8, min_test_len: int = 1
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    train_seqs, test_seqs = [], []
    for seq in seqs:
        cut = int(len(seq) * train_frac)
        if len(seq) - cut >= min_test_len and cut > 0:
            train_seqs.append(seq[:cut])
            test_seqs.append(seq[cut:])
    return train_seqs, test_seqs


def simulate_sequences_from_true(
    w_true: np.ndarray,
    z0_true: np.ndarray,
    horizon: int,
    *,
    kappa: float = 5.0,
    lam: float = 1.0,
    eta_pos: float = 0.5,
    eta_neg: float = 1.0,
    use_prox: bool = True,
    seed: int = 0,
) -> list[list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    w = np_row_normalize(w_true)
    z0 = np_row_normalize(z0_true)

    n_users = z0.shape[0]
    n_items = w.shape[0]

    seqs: list[list[tuple[int, int]]] = []
    for u in range(n_users):
        z = z0[u].copy()
        seq: list[tuple[int, int]] = []
        for _ in range(horizon):
            logits = kappa * (w @ z)
            pi = stable_softmax_np(logits)
            i = int(rng.choice(n_items, p=pi))
            wi = w[i]

            s = float(kappa * (z @ wi))
            p = float(sigmoid_np(s))
            r = int(rng.random() < p)

            if r == 1:
                tangent = wi - float(z @ wi) * z
                z = z + eta_pos * kappa * (1.0 - p) * tangent
            else:
                proj = float(z @ wi) * wi
                gamma = eta_neg * p
                shrink = (gamma * lam) / (1.0 + gamma * lam) if use_prox else (gamma * lam)
                z = z - shrink * proj

            z = np_normalize(z)
            seq.append((i, r))
        seqs.append(seq)
    return seqs


def torch_row_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def torch_vec_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class GeoFlowEmbeddingLearner(nn.Module):
    def __init__(
        self,
        n_items: int,
        d: int,
        n_users: int,
        device: str = "cpu",
        use_geoopt: bool = False,
    ):
        super().__init__()
        self.use_geoopt = bool(use_geoopt and _HAS_GEOOPT)

        if self.use_geoopt:
            sphere = geoopt.manifolds.Sphere()
            w_init = sphere.projx(torch.randn(n_items, d, device=device))
            z_init = sphere.projx(torch.randn(n_users, d, device=device))
            self.W_raw = geoopt.ManifoldParameter(w_init, manifold=sphere)
            self.z0_raw = geoopt.ManifoldParameter(z_init, manifold=sphere)
        else:
            self.W_raw = nn.Parameter(torch.randn(n_items, d, device=device))
            self.z0_raw = nn.Parameter(torch.randn(n_users, d, device=device))

        self.device = device

    def current_w(self, eps: float = 1e-12) -> torch.Tensor:
        if self.use_geoopt:
            return self.W_raw
        return torch_row_normalize(self.W_raw, eps)

    def current_z0(self, user_id: int, eps: float = 1e-12) -> torch.Tensor:
        if self.use_geoopt:
            return self.z0_raw[user_id]
        return torch_vec_normalize(self.z0_raw[user_id], eps)

    def forward_sequence(
        self,
        user_id: int,
        item_ids: torch.Tensor,
        responses: torch.Tensor,
        *,
        kappa: float = 5.0,
        lam: float = 1.0,
        eta_pos: float = 0.5,
        eta_neg: float = 1.0,
        use_prox: bool = True,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        w = self.current_w(eps)
        z = self.current_z0(user_id, eps)

        total_loss = torch.tensor(0.0, device=self.device)
        for t in range(item_ids.shape[0]):
            i = item_ids[t]
            r = responses[t]

            wi = w[i]
            dot = torch.dot(z, wi)
            logit = kappa * dot
            p = torch.sigmoid(logit)

            total_loss = total_loss + F.binary_cross_entropy_with_logits(logit, r)

            if r.item() == 1.0:
                tangent = wi - dot * z
                z = z + eta_pos * kappa * (1.0 - p) * tangent
            else:
                proj = dot * wi
                gamma = eta_neg * p
                shrink = (gamma * lam) / (1.0 + gamma * lam) if use_prox else (gamma * lam)
                z = z - shrink * proj

            z = torch_vec_normalize(z, eps)

        return total_loss

    @torch.no_grad()
    def get_w(self) -> np.ndarray:
        return torch_row_normalize(self.W_raw).cpu().numpy()

    @torch.no_grad()
    def get_z0(self) -> np.ndarray:
        return torch_row_normalize(self.z0_raw).cpu().numpy()


def fit_item_embeddings(
    seqs: list[list[tuple[int, int]]],
    n_items: int,
    d: int,
    *,
    kappa: float = 5.0,
    lam: float = 1.0,
    eta_pos: float = 0.5,
    eta_neg: float = 1.0,
    use_prox: bool = True,
    learn_user_init: bool = True,
    tbptt_k: int | None = 32,
    use_geoopt: bool = False,
    epochs: int = 30,
    lr: float = 1e-2,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_users = len(seqs)
    geoopt_enabled = bool(use_geoopt and _HAS_GEOOPT)
    if use_geoopt and not _HAS_GEOOPT and verbose:
        print("geoopt not available; falling back to Euclidean Adam + manual projection.")

    model = GeoFlowEmbeddingLearner(
        n_items, d, n_users, device=device, use_geoopt=geoopt_enabled
    ).to(device)

    if not learn_user_init:
        model.z0_raw.requires_grad_(False)

    if geoopt_enabled:
        opt = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    seq_tensors: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for u, seq in enumerate(seqs):
        item_ids = torch.tensor([it for (it, _) in seq], dtype=torch.long, device=device)
        responses = torch.tensor([float(r) for (_, r) in seq], dtype=torch.float32, device=device)
        seq_tensors.append((u, item_ids, responses))

    for ep in range(1, epochs + 1):
        total = 0.0
        perm = np.random.permutation(len(seq_tensors))
        for idx in perm:
            u, item_ids, responses = seq_tensors[idx]

            # Standard full-BPTT update
            if not tbptt_k or tbptt_k <= 0:
                opt.zero_grad()
                loss = model.forward_sequence(
                    u,
                    item_ids,
                    responses,
                    kappa=kappa,
                    lam=lam,
                    eta_pos=eta_pos,
                    eta_neg=eta_neg,
                    use_prox=use_prox,
                )
                loss.backward()
                opt.step()

                if not geoopt_enabled:
                    with torch.no_grad():
                        model.W_raw.copy_(torch_row_normalize(model.W_raw))
                        if learn_user_init:
                            model.z0_raw.copy_(torch_row_normalize(model.z0_raw))

                total += float(loss.item())
                continue

            # Truncated BPTT: backprop and detach hidden state every K steps.
            opt.zero_grad()
            z = model.current_z0(u)
            w = model.current_w()
            chunk_loss = torch.tensor(0.0, device=device)
            seq_loss_total = 0.0

            for t in range(item_ids.shape[0]):
                i = item_ids[t]
                r = responses[t]
                wi = w[i]

                dot = torch.dot(z, wi)
                logit = kappa * dot
                p = torch.sigmoid(logit)
                step_loss = F.binary_cross_entropy_with_logits(logit, r)
                chunk_loss = chunk_loss + step_loss
                seq_loss_total += float(step_loss.detach().item())

                if r.item() == 1.0:
                    tangent = wi - dot * z
                    z = z + eta_pos * kappa * (1.0 - p) * tangent
                else:
                    proj = dot * wi
                    gamma = eta_neg * p
                    shrink = (gamma * lam) / (1.0 + gamma * lam) if use_prox else (gamma * lam)
                    z = z - shrink * proj

                z = torch_vec_normalize(z)

                if (t + 1) % tbptt_k == 0:
                    chunk_loss.backward()
                    opt.step()
                    if not geoopt_enabled:
                        with torch.no_grad():
                            model.W_raw.copy_(torch_row_normalize(model.W_raw))
                            if learn_user_init:
                                model.z0_raw.copy_(torch_row_normalize(model.z0_raw))
                    opt.zero_grad()

                    z = z.detach()
                    w = model.current_w()
                    chunk_loss = torch.tensor(0.0, device=device)

            if chunk_loss.requires_grad:
                chunk_loss.backward()
                opt.step()
                if not geoopt_enabled:
                    with torch.no_grad():
                        model.W_raw.copy_(torch_row_normalize(model.W_raw))
                        if learn_user_init:
                            model.z0_raw.copy_(torch_row_normalize(model.z0_raw))

            total += seq_loss_total

        if verbose:
            print(f"epoch {ep:03d} | mean loss/user = {total/len(seq_tensors):.4f}")

    return model.get_w(), model.get_z0()


def _maybe_crop_sequence(
    seq: list[tuple[int, int]],
    max_t: int | None,
    rng: np.random.Generator,
    random_window: bool = True,
) -> list[tuple[int, int]]:
    if not max_t or max_t <= 0 or len(seq) <= max_t:
        return seq
    if random_window:
        start = int(rng.integers(0, len(seq) - max_t + 1))
    else:
        start = 0
    return seq[start : start + max_t]


def fit_item_embeddings_batched(
    seqs: list[list[tuple[int, int]]],
    n_items: int,
    d: int,
    *,
    kappa: float = 5.0,
    lam: float = 1.0,
    eta_pos: float = 0.5,
    eta_neg: float = 1.0,
    use_prox: bool = True,
    learn_user_init: bool = True,
    tbptt_k: int | None = 32,
    use_geoopt: bool = False,
    epochs: int = 30,
    lr: float = 1e-2,
    batch_size: int = 128,
    max_t: int | None = None,
    random_window: bool = True,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    n_users = len(seqs)
    geoopt_enabled = bool(use_geoopt and _HAS_GEOOPT)
    if use_geoopt and not _HAS_GEOOPT and verbose:
        print("geoopt not available; falling back to Euclidean Adam + manual projection.")

    model = GeoFlowEmbeddingLearner(
        n_items, d, n_users, device=device, use_geoopt=geoopt_enabled
    ).to(device)

    if not learn_user_init:
        model.z0_raw.requires_grad_(False)

    if geoopt_enabled:
        opt = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    user_ids = np.arange(n_users, dtype=np.int64)
    for ep in range(1, epochs + 1):
        perm = rng.permutation(user_ids)
        total = 0.0

        for b_start in range(0, n_users, batch_size):
            batch_users = perm[b_start : b_start + batch_size]
            if len(batch_users) == 0:
                continue

            user_seq_pairs = []
            for u in batch_users:
                seq_u = _maybe_crop_sequence(
                    seqs[int(u)],
                    max_t=max_t,
                    rng=rng,
                    random_window=random_window,
                )
                if len(seq_u) > 0:
                    user_seq_pairs.append((int(u), seq_u))

            if not user_seq_pairs:
                continue
            batch_users_kept = [u for (u, _) in user_seq_pairs]
            batch_seqs = [s for (_, s) in user_seq_pairs]

            b = len(batch_seqs)
            t_max = max(len(s) for s in batch_seqs)

            item_ids = torch.full((b, t_max), -1, dtype=torch.long, device=device)
            responses = torch.zeros((b, t_max), dtype=torch.float32, device=device)
            mask = torch.zeros((b, t_max), dtype=torch.bool, device=device)
            for bi, seq in enumerate(batch_seqs):
                tt = len(seq)
                item_ids[bi, :tt] = torch.tensor([it for (it, _) in seq], dtype=torch.long, device=device)
                responses[bi, :tt] = torch.tensor(
                    [float(r) for (_, r) in seq], dtype=torch.float32, device=device
                )
                mask[bi, :tt] = True

            batch_users_t = torch.tensor(batch_users_kept, dtype=torch.long, device=device)
            if model.use_geoopt:
                z = model.z0_raw[batch_users_t]
            else:
                z = torch_vec_normalize(model.z0_raw[batch_users_t])

            w = model.current_w()
            opt.zero_grad()
            chunk_loss = torch.tensor(0.0, device=device)
            chunk_obs = 0
            batch_loss_sum = 0.0

            for t in range(t_max):
                active = mask[:, t]
                if not torch.any(active):
                    continue

                idx = item_ids[:, t][active]
                r = responses[:, t][active]
                z_active = z[active]
                wi = w[idx]

                dot = torch.sum(z_active * wi, dim=1)
                logit = kappa * dot
                p = torch.sigmoid(logit)
                step_loss_sum = F.binary_cross_entropy_with_logits(logit, r, reduction="sum")
                chunk_loss = chunk_loss + step_loss_sum
                obs_here = int(active.sum().item())
                chunk_obs += obs_here
                batch_loss_sum += float(step_loss_sum.detach().item())

                z_new = z_active.clone()

                pos = r == 1.0
                if torch.any(pos):
                    z_pos = z_active[pos]
                    wi_pos = wi[pos]
                    dot_pos = dot[pos]
                    p_pos = p[pos]
                    tangent = wi_pos - dot_pos.unsqueeze(1) * z_pos
                    z_new[pos] = z_pos + eta_pos * kappa * (1.0 - p_pos).unsqueeze(1) * tangent

                neg = ~pos
                if torch.any(neg):
                    z_neg = z_active[neg]
                    wi_neg = wi[neg]
                    dot_neg = dot[neg]
                    p_neg = p[neg]
                    proj = dot_neg.unsqueeze(1) * wi_neg
                    gamma = eta_neg * p_neg
                    if use_prox:
                        shrink = (gamma * lam) / (1.0 + gamma * lam)
                    else:
                        shrink = gamma * lam
                    z_new[neg] = z_neg - shrink.unsqueeze(1) * proj

                z_new = torch_vec_normalize(z_new)
                z_next = z.clone()
                z_next[active] = z_new
                z = z_next

                if tbptt_k and tbptt_k > 0 and (t + 1) % tbptt_k == 0:
                    if chunk_obs > 0:
                        (chunk_loss / chunk_obs).backward()
                        opt.step()
                        if not geoopt_enabled:
                            with torch.no_grad():
                                model.W_raw.copy_(torch_row_normalize(model.W_raw))
                                if learn_user_init:
                                    model.z0_raw.copy_(torch_row_normalize(model.z0_raw))
                        opt.zero_grad()
                    z = z.detach()
                    w = model.current_w()
                    chunk_loss = torch.tensor(0.0, device=device)
                    chunk_obs = 0

            if chunk_obs > 0:
                (chunk_loss / chunk_obs).backward()
                opt.step()
                if not geoopt_enabled:
                    with torch.no_grad():
                        model.W_raw.copy_(torch_row_normalize(model.W_raw))
                        if learn_user_init:
                            model.z0_raw.copy_(torch_row_normalize(model.z0_raw))

            total += batch_loss_sum

        if verbose:
            print(f"epoch {ep:03d} | mean loss/user = {total/max(n_users, 1):.4f}")

    return model.get_w(), model.get_z0()


def compare_trainers_on_dataframe(
    df: pd.DataFrame,
    *,
    user_col: str = "user",
    item_col: str = "item",
    resp_col: str = "correct",
    train_frac: float = 0.8,
    d: int = 3,
    kappa: float = 5.0,
    lam: float = 1.0,
    eta_pos: float = 0.5,
    eta_neg: float = 0.5,
    use_prox: bool = True,
    epochs: int = 3,
    lr: float = 1e-1,
    tbptt_k: int | None = 32,
    batch_size: int = 128,
    max_t: int | None = None,
    random_window: bool = True,
    use_geoopt: bool = False,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    prep = dataframe_to_seqs(df, user_col=user_col, item_col=item_col, resp_col=resp_col)
    seqs = prep["seqs"]
    n_items = prep["n_items"]
    train_seqs, test_seqs = split_user_sequences(seqs, train_frac=train_frac)

    w_base, z0_base = fit_item_embeddings(
        train_seqs,
        n_items=n_items,
        d=d,
        kappa=kappa,
        lam=lam,
        eta_pos=eta_pos,
        eta_neg=eta_neg,
        use_prox=use_prox,
        epochs=epochs,
        lr=lr,
        tbptt_k=tbptt_k,
        use_geoopt=use_geoopt,
        device=device,
        seed=seed,
    )
    m_base = evaluate_sequences_metrics(
        test_seqs,
        w_base,
        z0_base,
        kappa=kappa,
        lam=lam,
        eta_pos=eta_pos,
        eta_neg=eta_neg,
        use_prox=use_prox,
    )

    w_batch, z0_batch = fit_item_embeddings_batched(
        train_seqs,
        n_items=n_items,
        d=d,
        kappa=kappa,
        lam=lam,
        eta_pos=eta_pos,
        eta_neg=eta_neg,
        use_prox=use_prox,
        epochs=epochs,
        lr=lr,
        tbptt_k=tbptt_k,
        batch_size=batch_size,
        max_t=max_t,
        random_window=random_window,
        use_geoopt=use_geoopt,
        device=device,
        seed=seed,
    )
    m_batch = evaluate_sequences_metrics(
        test_seqs,
        w_batch,
        z0_batch,
        kappa=kappa,
        lam=lam,
        eta_pos=eta_pos,
        eta_neg=eta_neg,
        use_prox=use_prox,
    )

    return {
        "n_users": len(seqs),
        "n_items": n_items,
        "baseline_metrics": m_base,
        "batched_metrics": m_batch,
        "delta_accuracy_batched_minus_baseline": m_batch["accuracy"] - m_base["accuracy"],
        "delta_auc_batched_minus_baseline": m_batch["auc"] - m_base["auc"],
        "delta_bce_batched_minus_baseline": m_batch["bce"] - m_base["bce"],
    }


def orthogonal_procrustes_align(w_hat: np.ndarray, w_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = w_hat.T @ w_true
    u, _, vt = np.linalg.svd(a)
    r = u @ vt
    return w_hat @ r, r


def embedding_metrics(w_true: np.ndarray, w_hat: np.ndarray, topk: int = 5) -> dict[str, float]:
    wt = np_row_normalize(w_true)
    wh = np_row_normalize(w_hat)

    wh_aligned, _ = orthogonal_procrustes_align(wh, wt)
    cos_i = np.sum(wh_aligned * wt, axis=1)
    cos_i = np.clip(cos_i, -1.0, 1.0)
    ang_deg = np.degrees(np.arccos(cos_i))
    rel_frob = np.linalg.norm(wh_aligned - wt, "fro") / np.linalg.norm(wt, "fro")

    c_true = wt @ wt.T
    c_hat = wh @ wh.T
    iu = np.triu_indices(c_true.shape[0], k=1)
    ct, ch = c_true[iu], c_hat[iu]
    pair_corr = float(np.corrcoef(ct, ch)[0, 1])
    pair_mse = float(np.mean((ct - ch) ** 2))

    def topk_neighbors(c: np.ndarray, k: int) -> np.ndarray:
        c2 = c.copy()
        np.fill_diagonal(c2, -np.inf)
        return np.argsort(-c2, axis=1)[:, :k]

    nn_true = topk_neighbors(c_true, topk)
    nn_hat = topk_neighbors(c_hat, topk)
    overlap = []
    for i in range(wt.shape[0]):
        overlap.append(len(set(nn_true[i]) & set(nn_hat[i])) / topk)

    return {
        "mean_cos_aligned": float(np.mean(cos_i)),
        "median_cos_aligned": float(np.median(cos_i)),
        "mean_angle_deg_aligned": float(np.mean(ang_deg)),
        "median_angle_deg_aligned": float(np.median(ang_deg)),
        "rel_frobenius_after_align": float(rel_frob),
        "pairwise_cos_corr": pair_corr,
        "pairwise_cos_mse": pair_mse,
        f"nn_overlap@{topk}": float(np.mean(overlap)),
    }


def heldout_bce(
    seqs: list[list[tuple[int, int]]],
    w: np.ndarray,
    z0: np.ndarray,
    *,
    kappa: float = 5.0,
    lam: float = 1.0,
    eta_pos: float = 0.5,
    eta_neg: float = 1.0,
    use_prox: bool = True,
) -> float:
    w = np_row_normalize(w)
    z0 = np_row_normalize(z0)

    total_loss = 0.0
    total_t = 0
    for u, seq in enumerate(seqs):
        z = z0[u].copy()
        for i, r in seq:
            wi = w[i]
            s = float(kappa * (z @ wi))
            p = float(sigmoid_np(s))

            total_loss += -(r * np.log(max(p, 1e-12)) + (1 - r) * np.log(max(1 - p, 1e-12)))
            total_t += 1

            if r == 1:
                tangent = wi - float(z @ wi) * z
                z = z + eta_pos * kappa * (1.0 - p) * tangent
            else:
                proj = float(z @ wi) * wi
                gamma = eta_neg * p
                shrink = (gamma * lam) / (1.0 + gamma * lam) if use_prox else (gamma * lam)
                z = z - shrink * proj
            z = np_normalize(z)

    return float(total_loss / max(total_t, 1))


def binary_auc_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)

    i = 0
    rank = 1.0
    while i < len(order):
        j = i + 1
        while j < len(order) and y_score[order[j]] == y_score[order[i]]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1.0)) / 2.0
        ranks[order[i:j]] = avg_rank
        rank += j - i
        i = j

    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def evaluate_sequences_metrics(
    seqs: list[list[tuple[int, int]]],
    w: np.ndarray,
    z0: np.ndarray,
    *,
    kappa: float = 5.0,
    lam: float = 1.0,
    eta_pos: float = 0.5,
    eta_neg: float = 1.0,
    use_prox: bool = True,
    decision_threshold: float = 0.5,
) -> dict[str, float]:
    w = np_row_normalize(w)
    z0 = np_row_normalize(z0)

    total_nll = 0.0
    total_t = 0
    y_true: list[int] = []
    y_prob: list[float] = []

    for u, seq in enumerate(seqs):
        z = z0[u].copy()
        for i, r in seq:
            wi = w[i]
            s = float(kappa * (z @ wi))
            p = float(sigmoid_np(s))

            y = int(r)
            y_true.append(y)
            y_prob.append(p)

            total_nll += -(y * np.log(max(p, 1e-12)) + (1 - y) * np.log(max(1 - p, 1e-12)))
            total_t += 1

            if y == 1:
                tangent = wi - float(z @ wi) * z
                z = z + eta_pos * kappa * (1.0 - p) * tangent
            else:
                proj = float(z @ wi) * wi
                gamma = eta_neg * p
                shrink = (gamma * lam) / (1.0 + gamma * lam) if use_prox else (gamma * lam)
                z = z - shrink * proj

            z = np_normalize(z)

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob_arr >= decision_threshold).astype(np.int64)

    bce = float(total_nll / max(total_t, 1))
    acc = float((y_pred == y_true_arr).mean()) if total_t > 0 else float("nan")
    auc = binary_auc_from_scores(y_true_arr, y_prob_arr)
    return {"bce": bce, "auc": auc, "accuracy": acc, "n_obs": int(total_t)}


def run_full_pipeline(
    *,
    n_items: int = 60,
    n_users: int = 40,
    d: int = 8,
    horizon: int = 200,
    kappa: float = 5.0,
    lam: float = 1.0,
    eta_pos: float = 0.5,
    eta_neg: float = 1.0,
    use_prox: bool = True,
    tbptt_k: int | None = 32,
    use_geoopt: bool = False,
    epochs: int = 30,
    lr: float = 1e-2,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    w_true = np_row_normalize(rng.normal(size=(n_items, d)))
    z0_true = np_row_normalize(rng.normal(size=(n_users, d)))

    seqs = simulate_sequences_from_true(
        w_true,
        z0_true,
        horizon,
        kappa=kappa,
        lam=lam,
        eta_pos=eta_pos,
        eta_neg=eta_neg,
        use_prox=use_prox,
        seed=seed + 123,
    )

    split = int(0.8 * horizon)
    train_seqs = [seq[:split] for seq in seqs]
    test_seqs = [seq[split:] for seq in seqs]

    w_hat, z0_hat = fit_item_embeddings(
        train_seqs,
        n_items,
        d,
        kappa=kappa,
        lam=lam,
        eta_pos=eta_pos,
        eta_neg=eta_neg,
        use_prox=use_prox,
        learn_user_init=True,
        tbptt_k=tbptt_k,
        use_geoopt=use_geoopt,
        epochs=epochs,
        lr=lr,
        device=device,
        seed=seed,
    )

    metrics = embedding_metrics(w_true, w_hat, topk=5)
    bce_true = heldout_bce(
        test_seqs,
        w_true,
        z0_true,
        kappa=kappa,
        lam=lam,
        eta_pos=eta_pos,
        eta_neg=eta_neg,
        use_prox=use_prox,
    )
    bce_hat = heldout_bce(
        test_seqs,
        w_hat,
        z0_hat,
        kappa=kappa,
        lam=lam,
        eta_pos=eta_pos,
        eta_neg=eta_neg,
        use_prox=use_prox,
    )

    return {
        "W_true": w_true,
        "z0_true": z0_true,
        "W_hat": w_hat,
        "z0_hat": z0_hat,
        "metrics": metrics,
        "heldout_bce_true": bce_true,
        "heldout_bce_hat": bce_hat,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GeoFlow simulator + embedding learner.")
    parser.add_argument("--n-items", type=int, default=60)
    parser.add_argument("--n-users", type=int, default=40)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--tbptt-k", type=int, default=32, help="Set <=0 to disable TBPTT.")
    parser.add_argument("--use-geoopt", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = run_full_pipeline(
        n_items=args.n_items,
        n_users=args.n_users,
        d=args.dim,
        horizon=args.horizon,
        tbptt_k=args.tbptt_k if args.tbptt_k > 0 else None,
        use_geoopt=args.use_geoopt,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )

    print("\nEmbedding recovery metrics:")
    for k, v in out["metrics"].items():
        print(f"{k:28s}: {v}")

    print("\nHeld-out response BCE (lower is better):")
    print("true W,true z0:", out["heldout_bce_true"])
    print("learned W,learned z0:", out["heldout_bce_hat"])
