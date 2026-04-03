import einops
import torch
import torch.nn as nn


class RoPEEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.theta = theta  # 基础频率，通常设为 10000.0 或更大
        self.d_k = d_k  # 注意力头的维度 (例如 128)
        self.max_seq_len = max_seq_len

        # torch.arange(0, d_k, 2) 生成 [0, 2, 4, ... d_k-2]。
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    """
    它把相邻的两个元素 $(x_1, x_2)$ 变成了 $(-x_2, x_1)$，为后面的高效计算做准备。
    """

    def _rotate_half(self, x):
        x = einops.rearrange(x, "... (d j) -> ... d j", j=2)
        x1, x2 = x.unbind(dim=-1)
        return einops.rearrange(torch.stack((-x2, x1), dim=-1), "... d j-> ... (d j)")

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. 如果没有给位置，默认就是 0 到 seq_len-1 (比如 [0, 1, 2, ..., seq_len-1])
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)
            token_positions = token_positions.unsqueeze(0)
        # 2. 计算每个 token 在每个平面的旋转角度 (m * theta_i)
        # token_positions 形状: (1, seq_len)
        # inv_freq 形状: (d_k / 2)
        # theta 形状: (1, seq_len, d_k / 2)
        theta = torch.einsum("...i , j -> ... i j", token_positions, self.inv_freq)
        # 3. 算 cos 和 sin，并把每个值复制两次 (因为一个平面有2个维度共用一个角度)
        # 形状变回: (1, seq_len, d_k)
        cos = torch.cos(theta).repeat_interleave(2, dim=-1)
        sin = torch.sin(theta).repeat_interleave(2, dim=-1)

        x_rotated = (x * cos) + (self._rotate_half(x) * sin)
        return x_rotated
