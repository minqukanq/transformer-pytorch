import torch.nn as nn
from models.self_attention import SelfAttention
from models.encoder_block import EncoderBlock


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm              = nn.LayerNorm(embed_size)
        self.attention         = SelfAttention(embed_size, heads=heads)
        self.transformer_block = EncoderBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout           = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query     = self.dropout(self.norm(attention + x))
        out       = self.transformer_block(value, key, query, src_mask)
        return out