import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange


# ── MultiHeadAttention ──────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores /= math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(self.dropout(weights), v)
        return output, weights

    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.shape
        _, T_k, _ = key.shape
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        Q = rearrange(
            Q,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.num_heads,
        )
        K = rearrange(
            K,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.num_heads,
        )
        V = rearrange(
            V,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.num_heads,
        )
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = rearrange(
            attn_output, "batch heads seq head_dim -> batch seq (heads head_dim)"
        )
        output = self.W_o(attn_output)
        return output, attn_weights


# ── 位置编码 ────────────────────────────────────────────────────────────────
def pos_sinusoid_embedding(d_model: int, seq_len: int):
    embeddings = torch.zeros((seq_len, d_model))
    for i in range(d_model):
        f = torch.sin if i % 2 == 0 else torch.cos
        embeddings[:, i] = f(
            torch.arange(0, seq_len) / np.power(1e4, 2 * (i // 2) / d_model)
        )
    return embeddings.float()


# ── 掩码 ────────────────────────────────────────────────────────────────────
def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )
    return causal_mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)


def make_padding_mask(token_ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    padding_mask = token_ids == pad_id
    return padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)


# ── PositionwiseFFN ──────────────────────────────────────────────────────────
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()  # ← Bug1修复：nn.ReLU() 而非 nn.ReLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.W1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.W2(x)
        return x


# ── EncoderLayer / Encoder ───────────────────────────────────────────────────
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ── DecoderLayer / Decoder ───────────────────────────────────────────────────
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        self_attn_out, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_out))
        cross_attn_out, _ = self.cross_attn(tgt, enc_out, enc_out, src_mask)
        tgt = self.norm2(tgt + self.dropout(cross_attn_out))
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_out))
        return tgt


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, enc_out, tgt_mask, src_mask)
        return tgt


# ── Embedding ────────────────────────────────────────────────────────────────
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        pe = pos_sinusoid_embedding(d_model, max_seq_len)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.token_embedding(x)
        x = x + self.pe[: x.size(1), :]
        return self.dropout(x)


# ── Transformer ──────────────────────────────────────────────────────────────
class Transformer(nn.Module):
    def __init__(
        self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, max_seq_len, dropout)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.out_proj = nn.Linear(d_model, vocab_size)

    # ← Bug2修复：forward 在类级别，不能嵌套在 __init__ 里
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(self.embedding(src), src_mask)
        dec_out = self.decoder(self.embedding(tgt), enc_out, tgt_mask, src_mask)
        logits = self.out_proj(dec_out)
        return logits


# ── 前向传播测试 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    VOCAB_SIZE = 1000
    D_MODEL = 64
    NUM_HEADS = 8
    D_FF = 256  # 4 × d_model
    NUM_LAYERS = 2
    MAX_SEQ_LEN = 50
    BATCH_SIZE = 2
    SRC_LEN = 10
    TGT_LEN = 8

    model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, MAX_SEQ_LEN)
    model.eval()

    # 模拟含 padding（0 = PAD）的输入
    src = torch.tensor(
        [[5, 12, 8, 3, 7, 0, 0, 0, 0, 0], [1, 4, 9, 2, 6, 11, 8, 0, 0, 0]]
    )  # (2, 10)
    tgt = torch.tensor([[1, 6, 9, 3, 0, 0, 0, 0], [1, 4, 7, 0, 0, 0, 0, 0]])  # (2, 8)

    src_mask = make_padding_mask(src, pad_id=0)  # (2,1,1,10)
    tgt_mask = make_causal_mask(TGT_LEN, device=src.device)  # (1,1,8,8)

    with torch.no_grad():
        logits = model(src, tgt, src_mask, tgt_mask)

    print(f"✅ 前向传播成功！")
    print(f"   src shape    : {src.shape}")
    print(f"   tgt shape    : {tgt.shape}")
    print(f"   logits shape : {logits.shape}")  # 期望 (2, 8, 1000)
    print(f"   nan in output: {logits.isnan().any().item()}")
