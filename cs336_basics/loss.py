import torch

"""
其中 log_probs.gather(1, labels) 这一行代码的作用是
从 log_probs 张量中提取出每个样本对应的真实标签的对数概率值。具体来说：

log_probs 的 shape 是 (N, C)，表示 N个样本在C个类别上的对数概率分布。
labels 的 shape 是 (N, 1)，表示每个样本的真实类别索引。
gather(1, labels) 会根据 labels 中的索引，从 log_probs 的第二维（类别维度）中提取对应的对数概率值，结果的 shape 是 (N, 1)。
在使用这个交叉熵损失函数时，我们通常会将模型的输出 logits 和对应的真实标签展开成一维向量。
这样做的好处是，可以简化计算过程，使得每个时间步的预测都被视为一个独立的样本，从而方便地计算整体的损失。
"""


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor):
    # Subtract the largest element for numerical stability.
    # 如果 logits 中的数值过大，计算 exp(logits) 可能会导致数值溢出（overflow）。
    # 通过减去 logits 中的最大值，可以确保所有的数值都变得较小，从而避免溢出问题。
    logits = logits - torch.max(logits, dim=1, keepdim=True).values

    # 计算 log_probs。log_probs 的 shape 仍然是 (N, C)，表示每个样本在每个类别上的对数概率。
    # logP(x) = log(exp(logits) / sum(exp(logits))) = logits - log(sum(exp(logits)))
    log_probs = logits - torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True))

    # 现在 log_probs 矩阵里装满了模型对每一个选项的“对数自信度”。
    # 但是，我们在计算损失时，只关心模型对“正确答案”的自信度有多高。
    # 其他选错的项，我们根本不看。
    labels = labels.unsqueeze(1)
    loss = log_probs.gather(1, labels).squeeze(1)
    loss = -loss.mean()
    return loss


"""
困惑度衡量的是模型对测试数据的预测能力，数值越低表示模型越好。
PPL 等价于“模型给真实 token 的概率 p”的倒数 1/p 的几何平均，所以越小表示模型平均给真值的概率越大。
为什么用 exp(loss) 来计算 perplexity 呢？因为交叉熵损失 loss 是 -log(p)，所以 perplexity 就是 exp(-log(p)) = 1/p。
"""


def perplexity(loss: torch.Tensor) -> torch.Tensor:
    return torch.exp(loss)
