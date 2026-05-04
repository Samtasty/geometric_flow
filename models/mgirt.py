import torch
import torch.nn as nn


class MGIRTModel(nn.Module):
    def __init__(self, num_students, num_items, emb_dim=2, alpha=1.0, beta=1.0, eps=1e-7):
        super().__init__()
        self.student_emb = nn.Embedding(num_students, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def logits(self, student_idx, item_idx):
        z = self.student_emb(student_idx)  # (B, D)
        w = self.item_emb(item_idx)        # (B, D)

        z = z / (z.norm(dim=1, keepdim=True) + self.eps)
        s = (z * w).sum(dim=1)
        w_perp = w - s.unsqueeze(1) * z
        g = torch.sqrt((w_perp * w_perp).sum(dim=1) + self.eps)
        return self.alpha * s - self.beta * g

    def forward(self, student_idx, item_idx):
        return torch.sigmoid(self.logits(student_idx, item_idx))
