import torch
import torch.nn as nn

from cs336_basics.config import ModelConfig
from cs336_basics.modules import FFN, MHA, Linear, RMSNorm, Embedding

"""
总的来说，Part 02 就是在 Part 01 的 Tokenization 之后，
把“能训练的语言模型”真正搭起来：我们从最基础的 Linear / Embedding 出发，
逐步实现 RMSNorm（Pre-Norm）、现代 LLM 常用的 SwiGLU-FFN、
再到最核心也最容易写错的 (RoPE + Causal) Multi-Head Self-Attention，
最终像搭积木一样组装出完整的 TransformerBlock，并串联成 TransformerLM，
通过 Output Layer 输出 vocabulary logits 用于 next-token prediction。

这一部分最值得记住的工程要点有三类：

稳定性（stability）： Softmax 的数值稳定（减 max）、
Pre-Norm（RMSNorm 放在子层前）、以及 causal mask 防止未来信息泄露，都是“训练能不能跑起来”的关键。

效率（efficiency）： Q/K/V 投影应当是 3 次矩阵乘法（更进一步可以合成 1 次），
mask 用 “softmax 前加 -” 而不是切子序列，RoPE 用预计算的 sin/cos buffer 复用跨 batch/跨层，避免显式构造 d*d
 旋转矩阵。

结构（architecture）： 现代 LLM 的 Block 基本都遵循 “RMSNorm → MHA/FFN → Residual” 的 Pre-Norm 模式；
FFN 常用 SwiGLU（激活 + gating）；RoPE 只作用在 Q/K（不作用在 V）；
最后再接一个输出头（可选 final norm / weight tying）把 hidden states 映射到词表分布。

"""


class TransformerBlock(nn.Module):
    """
    (1) 归一化：先把输入x做 RMSNorm，得到更稳定的输入分布
    (2) 主操作：把归一化后的向量送入 MHA，计算注意力输出
    (3) 残差：把注意力输出加回原输入x，形成y
    Pre-Norm
    这里我们采用的是 Pre-Norm 结构，也就是在每个子层（MHA 或 FFN）前做归一化。
    Pre-Norm相对于 Post-Norm（先做子层再归一化）有几个优点：
    1. 训练更稳定：Pre-Norm 可以缓解深层 Transformer 的梯度消失问题，使得训练更稳定。
    2. 更深的模型：Pre-Norm 允许我们训练更深的 Transformer，因为每个子层的输入都经过归一化，减少了内部协变量偏移。
    3. 对Learning Rate更不敏感：Pre-Norm 结构对学习率的选择不那么敏感，允许使用更大的学习率进行训练。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.mha = MHA(
            d_model=config.d_model,
            num_heads=config.num_heads,
            use_rope=config.use_rope,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
        )
        self.use_moe = config.use_moe
        if self.use_moe:
            from cs336_basics.modules import MoE

            self.ffn = MoE(
                d_model=config.d_model,
                d_ff=config.d_ff,
                num_experts=config.num_experts,
                top_k=config.top_k,
                router_jitter=config.router_jitter,
                z_loss_coef=config.z_loss_coef,
                lb_loss_coef=config.lb_loss_coef,
            )
        else:
            self.ffn = FFN(
                d_model=config.d_model,
                d_ff=config.d_ff,
            )
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor | tuple:
        aux = {
            "z_loss": x.new_zeros(()),
            "z_loss_scaled": x.new_zeros(()),
            "tokens_per_expert": None,  # 可选：debug/monitor
            "lb_loss": x.new_zeros(()),  # 可选：debug/monitor
            "lb_loss_scaled": x.new_zeros(()),  # 可选：debug/monitor
        }

        x = x + self.mha(self.norm1(x), token_positions=token_positions)

        if self.use_moe:
            out = self.ffn(self.norm2(x))
            x = x + out["output"]

            aux["tokens_per_expert"] = out.get("tokens_per_expert", None)
            aux["z_loss"] = out.get("z_loss", x.new_zeros(()))
            aux["z_loss_scaled"] = out.get("z_loss_scaled", x.new_zeros(()))
            aux["lb_loss"] = out.get("lb_loss", x.new_zeros(()))
            aux["lb_loss_scaled"] = out.get("lb_loss_scaled", x.new_zeros(()))
        else:
            x = x + self.ffn(self.norm2(x))

        return x, aux


class OutputLayer(nn.Module):
    def __init__(self, d_model, vocab_size, use_norm: bool = False):
        super().__init__()
        self.linear = Linear(d_model, vocab_size)
        self.norm = RMSNorm(d_model) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        logits = self.linear(x)
        return logits


class TransformerLM(nn.Module):
    """
    当我们实现完 embedding、Transformer block（MHA + FFN）、
    以及输出层之后，就可以按照高层结构把整个语言模型串起来了。整体流程可以概括为三步：

    1. Token Embedding：把 token id 映射到向量表示
    2. 堆叠 num_layers 个 Transformer Blocks
    3. Output Layers：映射到词表分布
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.output_layer = OutputLayer(
            config.d_model, config.vocab_size, use_norm=config.use_final_norm
        )

        if config.tie_weights:
            self._tie_weights()

    """
    token ids → embedding 得到 X→ 经过L个 Transformer blocks 得到 H
    → 输出头（norm + linear + softmax）得到词表分布，用于 next-token prediction。
    """

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> tuple:
        x = self.token_embedding(x)
        total_z_scaled = x.new_zeros(())
        tokens_per_expert_all = []  # list[Tensor] or []
        total_lb_loss_scaled = x.new_zeros(())
        moe_layers = 0

        for layer in self.layers:
            x, aux = layer(x, token_positions=token_positions)
            if self.config.use_moe:
                total_z_scaled = total_z_scaled + aux["z_loss_scaled"]
                total_lb_loss_scaled = total_lb_loss_scaled + aux["lb_loss_scaled"]
                tokens_per_expert_all.append(aux["tokens_per_expert"])
                moe_layers += 1
        x = self.final_norm(x)
        logits = self.output_layer(x)

        aux_out = {
            "z_loss_scaled": total_z_scaled,
            "moe_layers": moe_layers,
            "tokens_per_expert": tokens_per_expert_all,  # list[Tensor] or []
            "lb_loss_scaled": total_lb_loss_scaled,
        }
        return logits, aux_out

    def _tie_weights(self):
        self.output_layer.linear.weight = self.token_embedding.weight

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def _generate_core(self):
        self.eval()
        pass

    def generate(self, x: torch.Tensor, max_length: int) -> torch.Tensor:
        pass

    def generate_streaming(self, x: torch.Tensor, max_length: int) -> torch.Tensor:
        pass
