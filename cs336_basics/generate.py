import torch
import torch.nn.functional as F

from cs336_basics.tokenizer import BPETokenizer

"""
从累计概率达到 P 的 token 集合中采样，P越大多样性越高
动态调整候选 token 数量	需要选择合适的 P 值，通常 P 值在 0.8 到 0.9 之间
"""
def top_p_sampling(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    logits: (B, V)
    returns: (B,) sampled token ids
    """
    assert 0.0 < top_p <= 1.0

    # sort
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # mask tokens with cumulative prob > top_p (but keep at least 1 token)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # scatter mask back to original vocab positions
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

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

    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)

    # 2. softmax over filtered logits
    probs = F.softmax(filtered_logits, dim=-1)

    # 3. sample
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
            next_token_logits = logits[:, -1, :].float()  # Get logits for the last token

            # Sample from the distribution
            """
            当 temperature < 1 时，概率分布会变得更陡峭，
            模型更倾向于选择高概率的 token，生成的文本更确定性。
            当 temperature > 1 时，概率分布会变得更平坦，模型更倾向于选择低概率的 token，
            生成的文本更具多样性和创造性。
            """
            assert temperature > 0.0, "Temperature must be positive."
            assert top_p == 0.0 or top_k == 0, "Only one of top_p or top_k should be set."
            next_token_logits = next_token_logits / temperature

            if top_k > 0:
                next_token_id = top_k_sampling(next_token_logits, top_k)
            elif top_p > 0.0:
                next_token_id = top_p_sampling(next_token_logits, top_p)
            else:
                # Greedy Sampling	选择概率最高的 token	
                # 简单高效	可能缺乏多样性，容易陷入局部最优
                next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)  # Greedy if no sampling

            if next_token_id.item() == tokenizer.eos_token_id:
                break  # Stop if EOS token is generated
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)  # Append to input_ids

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