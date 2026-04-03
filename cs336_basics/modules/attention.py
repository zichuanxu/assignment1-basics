import torch
import torch.nn as nn
from cs336_basics.modules.linear import Linear
from cs336_basics.modules.rope import RoPEEmbedding

"""

仔细观察我们可以发现softmax 对所有输入同时加同一个常数不变。也就是说，对任意常数 c
softmax(v) = softmax(v + c)。
其实就相当于分子分母多乘一个 exp(c)，工程中为了数值稳定，通常会取 c = -max(v)，这样就能避免 exp(v_i) 过大导致的溢出问题。
  """


def stable_softmax(
    logits: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    max_logits = torch.max(logits, dim=dim, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)
    sum_exp_logits = torch.sum(exp_logits, dim=dim, keepdim=True)
    softmax = exp_logits / sum_exp_logits
    return softmax


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k**0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = stable_softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output


class MHA(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.q_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPEEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device,
            )

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        tril 是 triangle lower（下三角）的缩写。
        它会保留矩阵的主对角线及其下方的元素，把对角线右上方的元素全部变成 0。
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        """
        .unsqueeze(0) 两次：
        刚才生成的 mask 形状是 (seq_len, seq_len)。
        但在 Transformer 的 Multi-Head Attention 中，
        注意力分数矩阵（Attention Scores）的形状通常是 (batch_size, num_heads, seq_len, seq_len)。
        为了能让 mask 和注意力分数矩阵进行广播相加（Broadcast Add），
        我们必须给 mask 在最前面增加两个维度。
        两次 unsqueeze(0) 后，它的形状就变成了 (1, 1, seq_len, seq_len)。这样它就能完美适配各种批次大小和注意力头数了。
        """
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        batch_size, seq_len, _ = x.size()
        causal_mask = self._create_causal_mask(seq_len, x.device)
        """
        那么，计算 Q,K,V
        的线性投影后，我们需要把它们 reshape 成 (batch_size, num_heads, seq_len, d_k)，
        以便每个 head 独立计算注意力。实现上通常用以下两步：

        先用 view() 把最后一维拆成 (num_heads, d_k)，变成 (batch_size, seq_len, num_heads, d_k)
        再用 transpose() 把 num_heads 维度移到第二维，变成 (batch_size, num_heads, seq_len, d_k)

        """
        # 线性变换并分头
        query = (
            self.q_linear(x)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        key = (
            self.k_linear(x)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        value = (
            self.v_linear(x)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        """
        在使用 RoPE 的版本中，需要对 Q 和 K 做同样的位置旋转：
        对每个 head 的 Q
        应用 RoPE
        对每个 head 的 K
        应用 RoPE
        不要对 V
        应用 RoPE
        原因是：RoPE 影响的是“相似度打分”（Q@K^T）的相对位置信息；而 V
        是被加权汇聚的内容本身，通常不需要做旋转。
        """
        if self.use_rope:
            query, key = self.rope(query, token_positions), self.rope(
                key, token_positions
            )

        # 计算注意力
        attn_output = scaled_dot_product_attention(query, key, value, causal_mask)

        # 合并头并线性变换输出
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        output = self.out_linear(attn_output)
        return output
