import torch
import torch.nn.functional as F

from cs336_basics.tokenizer import BPETokenizer

"""

为了解决 Top-K 太死板的问题，Top-P 提出了一个极其聪明的做法：
按概率从大到小收集，只要收集到的总概率达到了 P（比如 0.9 或 90%），就不再收集了！
如果模型这次很确定，可能只收集了 2 个词，
总概率就达到 90% 了。剩下的词全部丢弃。（动态缩小候选集）
如果模型这次很犹豫，可能要收集 40 个词，总概率才能凑够 90%。（动态放大候选集）

从累计概率达到 P 的 token 集合中采样，P越大多样性越高
动态调整候选 token 数量	需要选择合适的 P 值，通常 P 值在 0.8 到 0.9 之间


"""


def top_p_sampling(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    logits: (B, V)
    returns: (B,) sampled token ids
    """
    assert 0.0 < top_p <= 1.0

    # 把得分从大到小排序，并算出它们当前的概率百分比
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    # 计算累积概率 (cumsum)。
    # 假设排好序的概率是: [0.5, 0.3, 0.1, 0.05, 0.05]
    # 算完 cumsum 变成:  [0.5, 0.8, 0.9, 0.95, 1.0 ]
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # 找出累积概率大于 top_p (假设设为 0.85) 的那些词。
    # 此时掩码是: [False, False, True, True, True] (True 表示要剔除)
    sorted_indices_to_remove = cumulative_probs > top_p
    # 【向右平移】
    # 为什么要平移？cumsum > p 会在刚好跨过门槛的那个词上亮起红灯（设为 True，准备剔除）。
    # 但这不符合 Top-P “包含该门槛词”的定义。
    # 所以，我们平移 1 位，把亮红灯的时机故意延迟 1 步。这样就能精准地把“跨门槛功臣”囊括在内，然后从它的下一个词开始真正执行剔除。
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    # 强行保底：确保第 1 名永远不被剔除（防止 P 设得太小，导致转盘空了）
    sorted_indices_to_remove[..., 0] = False

    # 后面的步骤就和 Top-K 一样了：
    # 把要剔除的位置映射回原始排序，填充 -inf，重新算 Softmax，最后掷骰子！
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    filtered_logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = F.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


"""
从概率最高的 K 个 token 中采样, K越大多样性越高	
增加多样性	需要选择合适的 K 值，通常 K 值在 10 到 50 之间
"""


def top_k_sampling(
    logits: torch.Tensor,
    top_k: int,
):
    if top_k <= 0:
        # sample from full distribution
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # 1. keep only top-k logits
    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

    # 创建一个全都是负无穷（-inf）的张量。在 Softmax 里，-inf 就代表 0% 概率。
    filtered_logits = torch.full_like(logits, float("-inf"))
    # 把刚才选出来的前 K 个高分，填回它们原本的位置。其他位置依然是 -inf
    filtered_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)

    # 2. softmax over filtered logits
    # 重新计算概率。那些 -inf 的词概率变成了 0，前 K 个词的概率被放大，总和变为 100%
    probs = F.softmax(filtered_logits, dim=-1)

    # 3. sample
    # 掷骰子！根据新的概率分布抽选 1 个词
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt: torch.Tensor | str,
    tokenizer: BPETokenizer,
    max_new_tokens: int = 256,
    top_k: int = 0,
    top_p: float = 0.0,
    temperature: float = 1.0,
) -> dict:
    model.eval()
    if isinstance(prompt, str):
        out = tokenizer.encode(prompt)
        input_ids = out.ids if hasattr(out, "ids") else out  # List[int]
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    else:
        input_ids = prompt.unsqueeze(0)  # Add batch dimension

    input_ids = input_ids.to(model.device)
    input_len = input_ids.shape[1]

    with torch.amp.autocast("cuda", enabled=False):
        for _ in range(max_new_tokens):
            logits, _ = model(input_ids)
            next_token_logits = logits[
                :, -1, :
            ].float()  # Get logits for the last token

            # Sample from the distribution
            """
            当 temperature < 1 时，概率分布会变得更陡峭，
            模型更倾向于选择高概率的 token，生成的文本更确定性。
            当 temperature > 1 时，概率分布会变得更平坦，模型更倾向于选择低概率的 token，
            生成的文本更具多样性和创造性。
            """
            assert temperature > 0.0, "Temperature must be positive."
            assert (
                top_p == 0.0 or top_k == 0
            ), "Only one of top_p or top_k should be set."
            next_token_logits = next_token_logits / temperature

            if top_k > 0:
                next_token_id = top_k_sampling(next_token_logits, top_k)
            elif top_p > 0.0:
                next_token_id = top_p_sampling(next_token_logits, top_p)
            else:
                # Greedy Sampling	选择概率最高的 token
                # 简单高效	可能缺乏多样性，容易陷入局部最优
                next_token_id = next_token_logits.argmax(
                    dim=-1, keepdim=True
                )  # Greedy if no sampling

            if next_token_id.item() == tokenizer.eos_token_id:
                break  # Stop if EOS token is generated
            input_ids = torch.cat(
                [input_ids, next_token_id], dim=-1
            )  # Append to input_ids

    input_ids = input_ids.squeeze(0)  # Remove batch dimension
    all_text = tokenizer.decode(input_ids.tolist())
    generated_ids = input_ids[input_len:]
    generated_text = tokenizer.decode(generated_ids.tolist())

    model.train()
    return {
        "all_text": all_text,
        "generated_text": generated_text,
        "generated_ids": generated_ids,
    }
