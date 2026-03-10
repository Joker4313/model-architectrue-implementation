import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Vaswani et al., "Attention Is All You Need", 2017)

    核心思路：
      将 Q/K/V 投影到 h 个低维子空间，各自计算 Scaled Dot-Product Attention，
      再把结果拼接后做一次线性投影输出。

    维度约定（全文统一）：
      B  = batch size
      T  = sequence length (token 数)
      d  = model dimension (d_model)
      h  = number of heads
      d_k = d // h  (每个 head 的维度)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # head dimension

        # ------------------------------------------------------------------ #
        # [BLOCK 1] 线性投影层                                                 #
        # 需要 4 个 Linear：Q / K / V 各一个，输出拼接后再一个                     #
        # 注意：这里不加 bias 是 Transformer 原始实现的惯例（可加可不加，说明即可）   #
        # ------------------------------------------------------------------ #
        # TODO: 定义 self.W_q, self.W_k, self.W_v (d_model -> d_model, no bias)
        # TODO: 定义 self.W_o (d_model -> d_model, no bias)
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # YOUR CODE
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # YOUR CODE
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # YOUR CODE
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # YOUR CODE

        self.dropout = nn.Dropout(dropout)

    # ---------------------------------------------------------------------- #
    # [BLOCK 2] Scaled Dot-Product Attention                                  #
    # 公式：Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) · V                 #
    # 参数：                                                                   #
    #   q, k, v : shape (B, h, T, d_k)                                        #
    #   mask    : optional bool tensor (B, 1, T_q, T_k)，True 位置被 mask 掉   #
    # 返回：                                                                   #
    #   output  : shape (B, h, T, d_k)                                        #
    #   weights : shape (B, h, T_q, T_k)  ← 用于可视化/调试                    #
    # ---------------------------------------------------------------------- #
    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        # Step 1: 计算 attention score = Q @ K^T
        #         Q: (B, h, T_q, d_k)  K^T: (B, h, d_k, T_k)
        #         结果: (B, h, T_q, T_k)
        # TODO:
        scores = torch.matmul(q, k.transpose(-2, -1))  # YOUR CODE

        # Step 2: 除以 sqrt(d_k) 做缩放，防止点积过大导致 softmax 梯度消失
        # TODO:
        scores = scores / math.sqrt(self.d_k)  # YOUR CODE

        # Step 3: 如果有 mask，将 mask=True 位置的 score 设为 -inf
        #         这样 softmax 后这些位置权重≈0，实现因果掩码或 padding 掩码
        if mask is not None:
            # TODO: 用 masked_fill_ 或 masked_fill 完成
            scores = scores.masked_fill(mask, float("-inf"))  # YOUR CODE

        # Step 4: softmax 归一化（在最后一维，即 Key 维度上）
        # TODO:
        weights = F.softmax(scores, dim=-1)  # YOUR CODE

        # Step 5: dropout（作用在 attention weights 上，原论文有此操作）
        weights = self.dropout(weights)

        # Step 6: 加权求和 V
        #         weights: (B, h, T_q, T_k)  V: (B, h, T_k, d_k)
        #         结果: (B, h, T_q, d_k)
        # TODO:
        output = torch.matmul(weights, v)  # YOUR CODE

        return output, weights

    # ---------------------------------------------------------------------- #
    # [BLOCK 3] Forward 主流程                                                 #
    # 步骤：                                                                   #
    #   1. 线性投影 → 得到 Q, K, V (B, T, d_model)                             #
    #   2. 拆分多头 → reshape 成 (B, h, T, d_k)                                #
    #   3. 调用 scaled_dot_product_attention                                   #
    #   4. 拼接多头 → reshape 回 (B, T, d_model)                               #
    #   5. 输出线性投影 W_o                                                     #
    # ---------------------------------------------------------------------- #
    def forward(
        self,
        query: torch.Tensor,  # (B, T_q, d_model)
        key: torch.Tensor,  # (B, T_k, d_model)
        value: torch.Tensor,  # (B, T_k, d_model)
        mask: torch.Tensor = None,
    ):
        B, T_q, _ = query.shape
        _, T_k, _ = key.shape

        # Step 1: 线性投影
        # TODO: 分别对 query/key/value 做投影，结果 shape 均为 (B, T, d_model)
        Q = self.W_q(query)  # YOUR CODE
        K = self.W_k(key)  # YOUR CODE
        V = self.W_v(value)  # YOUR CODE

        # Step 2: 拆分多头
        # 目标 shape: (B, num_heads, T, d_k)
        # 技巧：先 reshape 成 (B, T, num_heads, d_k)，再 transpose(1, 2)
        # TODO:
        Q = Q.reshape(B, T_q, self.num_heads, self.d_k).transpose(
            1, 2
        )  # YOUR CODE  → (B, h, T_q, d_k)
        K = K.reshape(B, T_k, self.num_heads, self.d_k).transpose(
            1, 2
        )  # YOUR CODE  → (B, h, T_k, d_k)
        V = V.reshape(B, T_k, self.num_heads, self.d_k).transpose(
            1, 2
        )  # YOUR CODE  → (B, h, T_k, d_k)

        # Step 3: Scaled Dot-Product Attention
        # TODO:
        attn_output, attn_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )  # YOUR CODE
        # attn_output: (B, h, T_q, d_k)

        # Step 4: 拼接多头
        # 先 transpose 回 (B, T_q, h, d_k)，再 contiguous().reshape → (B, T_q, d_model)
        # NOTE: 必须先调 contiguous()，因为 transpose 后内存不连续，reshape 会报错
        # TODO:
        attn_output = (
            attn_output.transpose(1, 2).contiguous().reshape(B, T_q, self.d_model)
        )  # YOUR CODE  → (B, T_q, d_model)

        # Step 5: 输出投影
        # TODO:
        output = self.W_o(attn_output)  # YOUR CODE  → (B, T_q, d_model)

        return output, attn_weights


# ============================================================ #
#  参考答案（面试结束后对照检查，先别看！）                           #
# ============================================================ #
class MultiHeadAttention_Reference(nn.Module):
    """Reference implementation — DO NOT peek during the interview!"""

    def __init__(self, d_model, num_heads, dropout=0.0):
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
        scores = torch.matmul(q, k.transpose(-2, -1))  # (B,h,T_q,T_k)
        scores = scores / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = torch.matmul(weights, v)  # (B,h,T_q,d_k)
        return output, weights

    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.shape
        _, T_k, _ = key.shape
        Q = self.W_q(query)  # (B,T_q,d_model)
        K = self.W_k(key)
        V = self.W_v(value)
        Q = Q.reshape(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.reshape(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.reshape(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().reshape(B, T_q, self.d_model)
        )
        output = self.W_o(attn_output)
        return output, attn_weights


# ============================================================ #
#  Quick Smoke Test                                             #
# ============================================================ #
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, d_model, h = 2, 10, 64, 8

    mha = MultiHeadAttention(d_model, h, dropout=0.0)
    x = torch.randn(B, T, d_model)

    # Self-attention (Q=K=V=x)
    out, weights = mha(x, x, x)
    print(f"Output shape : {out.shape}")  # expect (2, 10, 64)
    print(f"Weights shape: {weights.shape}")  # expect (2, 8, 10, 10)

    # Cross-entropy sanity: output should match reference
    ref = MultiHeadAttention_Reference(d_model, h)
    ref.load_state_dict(mha.state_dict())
    ref_out, _ = ref(x, x, x)
    print(
        f"Max diff vs reference: {(out - ref_out).abs().max().item():.6f}"
    )  # expect ~0
