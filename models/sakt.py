import torch
import torch.nn as nn


class SAKTModel(nn.Module):
    def __init__(self, num_items, emb_dim=64, num_heads=4, dropout=0.2):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")
        self.num_items = num_items
        self.q_embed = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * num_items + 1, emb_dim, padding_idx=0)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.ffn_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(emb_dim, 1)

    def forward(self, q_seq, r_seq, q_next):
        # q_seq: (B, L) question ids, 0 padded
        # r_seq: (B, L) correctness (0/1)
        # q_next: (B,) next question id
        q_seq = q_seq.long()
        r_seq = r_seq.long()
        q_next = q_next.long()

        k_embed = self.q_embed(q_seq)  # (B, L, D)
        qa_ids = q_seq + r_seq * self.num_items
        v_embed = self.qa_embed(qa_ids)  # (B, L, D)
        q_embed = self.q_embed(q_next).unsqueeze(1)  # (B, 1, D)

        key_padding_mask = q_seq.eq(0)
        attn_out, _ = self.attn(q_embed, k_embed, v_embed, key_padding_mask=key_padding_mask)
        x = self.attn_norm(attn_out + q_embed)
        ffn_out = self.ffn(x)
        x = self.ffn_norm(ffn_out + x)
        x = self.dropout(x)
        logits = self.out(x).squeeze(-1).squeeze(-1)
        return torch.sigmoid(logits)
