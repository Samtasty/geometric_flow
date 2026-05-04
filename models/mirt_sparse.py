import torch
import torch.nn as nn


class SparseMIRTModel(nn.Module):
    """
    Standard sparse-embedding MIRT model.
    """

    def __init__(self, num_users, num_items, emb_dim=2):
        super().__init__()
        self.student_emb = nn.Embedding(num_users, emb_dim, sparse=True)
        self.item_emb = nn.Embedding(num_items, emb_dim, sparse=True)
        self.item_bias = nn.Embedding(num_items, 1, sparse=True)

    def forward(self, user_idx, item_idx):
        theta = self.student_emb(user_idx)
        # Positive discrimination parameters.
        alpha = torch.exp(self.item_emb(item_idx))
        beta = self.item_bias(item_idx).squeeze(-1)
        logits = (theta * alpha).sum(dim=1) + beta
        return torch.sigmoid(logits)
