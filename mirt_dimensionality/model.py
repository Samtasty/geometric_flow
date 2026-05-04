"""
Multidimensional 2PL IRT model.

    logit(correct) = softplus(a_j) · theta_u - d_j

- theta_u ∈ R^K  : student proficiency (not constrained)
- a_j    ∈ R^K_+ : item discrimination (softplus keeps it positive)
- d_j    ∈ R     : item difficulty scalar
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MIRT(nn.Module):
    def __init__(self, n_users: int, n_challenges: int, K: int):
        super().__init__()
        self.K = K
        self.n_users = n_users
        self.n_challenges = n_challenges

        self.theta = nn.Embedding(n_users, K)
        nn.init.normal_(self.theta.weight, 0.0, 0.1)

        self.a_raw = nn.Embedding(n_challenges, K)
        nn.init.constant_(self.a_raw.weight, 0.5)  # softplus(0.5) ≈ 0.97

        self.d = nn.Embedding(n_challenges, 1)
        nn.init.zeros_(self.d.weight)

    def forward(self, u_idx: torch.Tensor, c_idx: torch.Tensor) -> torch.Tensor:
        """Returns logits (before sigmoid), shape (B,)."""
        theta = self.theta(u_idx)                      # (B, K)
        a     = F.softplus(self.a_raw(c_idx))          # (B, K)  positive
        d     = self.d(c_idx).squeeze(-1)              # (B,)
        return (a * theta).sum(dim=-1) - d

    def item_discrimination(self) -> torch.Tensor:
        """Returns softplus(a_raw) for all items, shape (N_challenges, K)."""
        with torch.no_grad():
            return F.softplus(self.a_raw.weight)

    def item_discrimination_normalised(self) -> torch.Tensor:
        """Unit-norm discrimination vectors, shape (N_challenges, K)."""
        a = self.item_discrimination()
        return a / (a.norm(dim=1, keepdim=True) + 1e-8)
