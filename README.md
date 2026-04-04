# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

---

```
本仓库参考了教程如下
https://yyzhang2025.github.io/posts/CS336/

原作者仓库
https://github.com/YYZhang2025/Stanford-CS336
```

## BPE 分词器实现

### 整体架构

分词器分两个阶段：**训练**（`train_bpe.py`）和**编码**（`tokenizer.py` 中的 `BPETokenizer.encode()`）。

```
训练:  原始文本 → Multi-Process 预分词 → 词频统计 → Heap + 倒排索引迭代合并 → vocab.json + merges.txt
编码:  输入文本 → regex 预分词 → bytes → 双向链表 + 最小堆贪心合并 → token IDs
```

### BPE 基础

#### 什么是 BPE

BPE（Byte-Pair Encoding）是一种**子词分词算法**。核心思想很朴素：从最细粒度的单位（单字节或字符）出发，反复合并最频繁出现的相邻 token 对，逐步构建出更大的词汇单元。

用一个极简例子说明整个过程。假设语料库里只有一句话 `"aaabdaa"`：

```
初始状态（每个字符就是一个 token）：
  a  a  a  b  d  a  a

第 1 轮 — 最高频的 pair 是 (a, a)，出现 3 次 → 合并它
  合并后: aa  a  b  d  aa
  新增词表: aa

第 2 轮 — 最高频的 pair 是 (aa, a)，出现 2 次 → 合并它
  合并后: aaa  b  d  aaa
  新增词表: aaa
```

每次合并把 2 个 token 变成 1 个，词表大小 +1。重复这个过程直到词表达到目标大小，我们就学会了「`aa → aa`」「`aaa → aaa`」这些合并规则。编码新文本时，按规则的创建顺序依次应用合并即可。

#### 为什么用 BPE 而不是纯字符级分词

| 维度 | 字符级 | BPE 子词 |
|---|---|---|
| 词表大小 | ~256 (ASCII) | 几万（可控） |
| 序列长度 | 很长 | 显著缩短，注意力开销更低 |
| 未登录词 | 不存在 | 不存在（任何未知词都可拆解到字符级） |
| 语义单元 | 一个字符通常无语义 | `" playing"`、`"ing"` 等是有意义的语言单元 |

GPT 系列、Llama 等主流模型都采用 BPE 或其变种。

#### 关键概念速查

| 概念 | 含义 |
|---|---|
| **pre-token（预分词单元）** | regex 把原始文本拆成的基本单元，一个 pre-token 是一个单词或空格片段 |
| **pair** | 两个相邻 token 的有序对，如 `(a, b)` |
| **merge（合并）** | 把 pair 替换为一个新 token 的操作 |
| **rank** | merge 的创建顺序编号，rank 越小优先级越高 |
| **vocab（词表）** | 从 token ID 到字节表示的映射 |
| **merges（合并规则表）** | 按创建顺序排列的 pair 列表，编码时依次应用 |

#### 编码过程（朴素版）

理解了合并规则表（merges），编码就变得直观：

```
输入: "playing"
↓
1. regex 预分词: [" playing"]  →  转 bytes: [b"p", b"l", b"a", b"y", b"i", b"n", b"g"]
↓
2. 按 merges 顺序从上往下扫:
   - 第一条 merge (e, s)? "playing" 里没有 → 跳过
   - ...
   - 某条 merge (i, n)? 有！→ 合并: [b"p", b"l", b"a", b"y", b"新_token(in)", b"g"]
   - 继续往下扫...
   - 某条 merge (ay,ing)? 有！→ 合并: [b"p", b"l", b"新_token(aying)"]
↓
3. 输出: [token_id]
```

关键规则：**merges 表的顺序就是应用顺序**，先定义的规则优先级更高。这保证了编码结果的确定性。

---

### 训练阶段 (`train_bpe.py`)

**Step 1 — 文件切分 + 多进程预分词**

不一次性把整个语料读进内存。`find_chunk_boundaries()` 按换行符对齐把大文件切成 N 块，每块交给独立进程做预分词：

```python
chunks = find_chunk_boundaries(f, desired_num_chunks=NUM_PROCESSES, split_special_token=b"\n")
# 每个进程处理 [start, end) 区间, 通过 Queue 回传 Counter 结果
```



使用 Special Token-aware Splitting 和 Regex-based Pre-Tokenizatio，把文本拆成单词级 token 元组，统计每个 token 元组出现多少次。

- Special Token-aware Splitting: 我们已经了解过了，在初始化vocab 时，我们也需要初始化special tokens，其中一个常见的special tokens就是  . 这个token意味着一段文本的结束。给出一段很长的文本，我们要做的第一件事情就是把这个文本分成许多段。
- Regex-based Pre-Tokenization: Pre-Tokenization（预分词） 就是在真正训练 BPE 合并规则之前，先对整份语料做一次粗粒度的切分，把文本切成一段段“更大的片段”（pre-token），然后在这些片段内部去统计相邻字节（byte pair）的出现频率。具体来说，使用GPT-2 使用的那条正则表达式PAT。

**Step 2 — 构建 pair 频率堆 + 倒排索引**


一个很明显的优化点是：每一轮都要找当前频率最高的 pair。在 Version 0 (Section 2.2) 里，我们每轮都通过遍历 pairs_counter 来取最大值，这一步是$O(n)$
（n是 pair 的数量）。而这个操作正好符合堆（heap）的使用场景：用堆维护“当前最大的元素”，就能把“取最大”降到$O(\log n)$
（严格来说是：取堆顶是$O(1)$，但如果包含 pop/push 更新则是 $O(\log n)$  ）。

具体做法是把每个 pair 作为堆元素，并把“排序依据”设计成：

- 频次越大优先级越高
- 频次相同则按 pair 的字典序更大者优先

在 Python 的 heapq 是最小堆，因此我们可以用负号把它变成“最大堆”，例如存成：

- key = (-freq, a, b)

这样每一轮我们都能快速拿到候选的“最常见 pair”。

不过要注意一点：频次在 merge 之后会发生变化，因此堆里旧的条目可能变“过期”。在 pop 堆顶时，我们需要检查该 pair 的当前频次是否和堆里存的频次一致；如果不一致，说明堆顶是过期的，就继续 pop 直到找到一个有效的 pair。

训练的核心数据结构不是简单的 `word_counter`，而是两个关键索引：

| 数据结构 | 作用 |
|---|---|
| `pairs_counter: Counter` | 每个相邻 pair 在全语料中的总出现次数 |
| `pair_to_words: dict[pair, set[word]]` | 每个 pair 出现在哪些 word 中（倒排索引） |
| `pair_heap: list[HeapItem]` | 最大堆维护当前最频繁的 pair |

构建方式：遍历 `word_counter` 中的每个 word，对其中所有相邻 pair 累加频率，同时把 word 加入 `pair_to_words[pair]` 的集合。然后用 Heap 按 `-freq, 字典序` 排序。

**Step 3 — 增量合并循环（核心优化）**


除了用 Heap 加速“选出频率最高的 pair”，另一个更关键的瓶颈在于 merge 更新阶段：我们每一轮都会遍历 word_counter 里的所有 word，检查这个 word 里是否出现了目标 pair；这一步的代价通常非常高，因为绝大多数 word 根本不包含 当前要 merge 的 pair，但我们还是把它们都扫了一遍。


因此我们可以用一个“倒排索引”来做 空间换时间：提前维护一个映射 pair -> {words…}，记录每个 pair 出现在哪些 word 中。这样当我们决定 merge 某个 pair 时，就只需要遍历 pair_to_words[pair] 里的那一小部分 word，而不必全量扫描所有 word。


这也正是我们搭建 pair_to_words 的原因：


- 没有索引：每轮 merge 都是 全量扫描所有 words（慢，$O(#Words)$级别）。
- 有索引：每轮只处理 包含该 pair 的 words 子集（快，复杂度取决于该 pair 的覆盖范围，通常远小于全量）。


接下来，我们还需要在 merge 之后，更新这个索引：当某个 pair 被 merge 成一个新 token 后，所有包含该 pair 的 word 都会发生变化，因此我们需要把这些 word 从旧 pair 的索引里移除，并把它们添加到新 pair 的索引里。


每轮迭代只做**局部精准更新**，而不是全量重新扫描：

```python
for _ in trange(num_merges):
    # 1. 从堆顶取最频繁的 pair（跳过过期条目）
    most_frequent_pair = pop_most_frequent_pair(pair_heap, pairs_counter)

    # 2. 更新 vocab, 分配新 token ID
    new_id = update_vocab(vocab, most_frequent_pair)

    # 3. 只对包含该 pair 的 word 做合并（利用 pair_to_words 索引）
    word_counter, pairs_counter, pair_heap, pair_to_words = \
        merge_pairs_with_heap_index(...)

    # 4. 记录合并规则
    merges.append((vocab[pair[0]], vocab[pair[1]]))
```

**`merge_pairs_with_heap_index()` 做了什么**：

1. 取出 `pair_to_words[target_pair]` 中所有受影响的 word
2. 对这些 word：先从 `word_counter` 和 `pairs_counter` 中**扣除**旧数据
3. 构造合并后的新 word，**加回** `word_counter` 和新产生的 pair 频率
4. 更新 `pair_to_words` 倒排索引
5. 把频率变化的 pair 重新 **heappush** 进堆（不删除旧条目，靠 lazy validation 处理）

这个策略把每轮复杂度从 O(全量 word 数) 降到 O(受影响的 word 数)，是训练从秒级降到毫秒级的关键。

### 编码阶段 (`tokenizer.py` — `BPETokenizer.encode()`)

#### 编码流程拆解

编码阶段面对的是一个**新问题**：给定训练好的 merges 表，如何把一段文本高效地编码为 token IDs？

先看最朴素的做法，再看我们的优化：

**朴素做法：** 对每个 pre-token 的 ID 列表，反复遍历整个 merges 表，找到能合并的 pair 就合并，直到所有 merge 规则都检查过一遍。这个做法的复杂度是 `O(merges数量 × 序列长度)` — merges 表有 50000 条规则时非常慢。

**我们的做法：双向链表 + 最小堆。** 核心观察是，**不是每条 merge 规则都需要检查**。只有那些「当前存在的 pair 恰好出现在 merges 表中」的位置才可能发生合并。于是：

```

堆里存 (rank, i)，表示当前位置 i 与其右邻居 nxt[i] 的 pair 在 merge 规则中的优先级（rank 越小越先合并）。
每次取出最小 rank 的候选，做一次合并，然后只需要重新检查局部的两个 pair：

(prev[i], i)
(i, nxt[i])

输入 pre-token: [72, 101, 108, 108, 111]  （"Hello" 的初始 byte IDs）

① 扫描一遍，检查每个相邻 pair 是否在 merges 表中
   (72,101) rank=5     → 推入堆: (5, 0)
   (101,108) rank=??   → 不在 merges 中，跳过
   (108,108) rank=100  → 推入堆: (100, 2)
   (108,111) rank=??   → 不在 merges 中，跳过

   堆: [(5, 0), (100, 2)]

② 弹出 rank 最小的 (5, 0) → 合并位置 0 和 1
   合并后: [新ID, 108, 108, 111]
   链表从: 0→1→2→3→4 变为: 0→2→3→4（位置 1 被跳过）

   只需检查位置 0 的新邻居:
   - 左邻居: 无
   - 右邻居: (新ID, 108) → 查 rank → 推入堆

③ 弹出下一个 (50, 0) → 合并 → 更新邻居

   堆空了 → 结束
```

```
1. 预处理: regex 拆分 → 每个 token 转 byte → 映射为初始 token ID 列表

2. 对每个 pre-token 执行 merge_one_pretoken():
   - 用双向链表维护当前存活的 token 位置
   - 初始化: 遍历所有相邻位置, 把有对应 merge rank 的 pair 推入最小堆
   - 循环弹出 rank 最小的 pair (最早定义的 merge 优先级最高):
     a) 校验: pair 是否仍然相邻? rank 是否匹配?
     b) 合并: 把两个位置的值替换为 merge_to_new_id 中的新 ID
     c) 从链表中"删除"被合并的位置
     d) 只重新检查被合并位置的前驱和后继邻居 → 局部更新堆

3. 遍历最终链表, 收集所有存活位置的 ID
```

关键设计：

- **双向链表**避免数组元素搬移，合并只是标记 `alive[j] = False` + 重连指针
- **堆 + stale check** 避免全局扫描，每次合并只影响 2 个相邻 pair
- **merge_to_new_id 映射** 提前预计算 `(a_id, b_id) → new_id`，运行时 O(1) 查找

### 与 tiktoken 对齐

编码结果与 OpenAI 的 `tiktoken` (GPT-2 encoding) **完全一致**。测试验证了：
- 空字符串、单字符、Unicode emoji（如 🙃）
- 包含特殊 token 的混合文本
- 多语言文本（德语）
- 长文本（TinyStories 样本）

### 序列化格式

训练输出与 tiktoken 兼容的格式：

```
vocab.json   → { "token_string": id, ... }  (GPT-2 bytes_to_unicode 编码)
merges.txt   → #version: 0.2 \n token1 token2 \n ...
special_tokens.txt → 每行一个特殊 token
```

通过 `BPETokenizer.from_files(vocab_path, merges_path, special_tokens_path)` 加载。

### 测试

```bash
uv run pytest tests/test_train_bpe.py
uv run pytest tests/test_tokenizer.py
```

### 训练

们通过运行以下代码来完成TinyStory的Tokenization与保存：
```bash
uv run python ./train_bpe.py
```
在训练完之后，我们可以到的一下的directory
```
datasets/
└── tiny_stories/
    ├── eval.bin
    ├── merges.txt
    ├── special_tokens.txt
    ├── train.bin
    └── vocab.json
```

---

## Language Model 实现

### 整体架构

在 BPE 分词器把文本转换为 token IDs 之后，我们需要搭建一个完整的语言模型来学习这些 token 的分布。整体流程可以概括为：

```
Token IDs → Embedding → [TransformerBlock × num_layers] → Final Norm → Output Layer → Vocabulary Logits
```

每个 TransformerBlock 内部采用 **Pre-Norm** 结构：

```
x → RMSNorm → MHA (RoPE + Causal) → Residual → RMSNorm → SwiGLU-FFN → Residual → 输出
```

### 核心组件速查

| 组件 | 文件 | 作用 |
|---|---|---|
| `Linear` | `modules/linear.py` | 自定义线性层（支持权重初始化策略） |
| `Embedding` | `modules/embedding.py` | Token ID → 向量映射 |
| `RMSNorm` | `modules/norm.py` | 归一化层，替代 LayerNorm |
| `RoPEEmbedding` | `modules/rope.py` | 旋转位置编码 |
| `MHA` | `modules/attention.py` | 多头自注意力（含 RoPE + Causal Mask） |
| `FFN` | `modules/ffn.py` | SwiGLU 前馈网络 |
| `TransformerBlock` | `model.py` | 完整的 Transformer 层 |
| `TransformerLM` | `model.py` | 完整语言模型 |

---

### 基础组件

#### Linear 层

自定义的 `Linear` 类封装了 `nn.Linear`，主要目的是统一权重初始化策略，方便后续实验调整。所有矩阵乘法（Q/K/V 投影、FFN、输出头）都通过这个类完成。

#### Embedding 层

```
Token IDs (batch, seq_len) → Embedding → (batch, seq_len, d_model)
```

把离散的 token ID 映射到连续的向量空间。这是模型学习语义表示的第一步。

---

### RMSNorm（Root Mean Square Layer Normalization）

#### 为什么不用 LayerNorm？

LayerNorm 同时计算均值和方差来做归一化：

```python
# LayerNorm: 减去均值，除以标准差
x_norm = (x - mean) / std
```

RMSNorm 去掉了均值项，只保留 RMS（均方根）：

```python
# RMSNorm: 只除以 RMS
rms = sqrt(mean(x²) + eps)
x_norm = x / rms
```

**优点：**
- 计算更简单，少一次减法操作
- Llama、Mistral 等现代 LLM 都采用 RMSNorm
- 实验表明去掉均值项对性能影响极小

#### 实现要点

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.weight = nn.Parameter(torch.ones(d_model))  # 可学习的缩放因子

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)  # 数值稳定性：在 fp32 下计算
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return (x_normed * self.weight).to(input_dtype)  # 恢复原始精度
```

**关键细节：**
- `eps = 1e-5` 防止除零
- 计算时升到 `float32`，输出时恢复原始 dtype（如 `bfloat16`）
- `weight` 初始化为全 1，让模型自己学习每个维度的缩放

---

### SwiGLU-FFN（前馈网络）

#### 从 ReLU-FFN 到 SwiGLU

传统的 Transformer FFN 结构：

```python
# 传统 FFN
FFN(x) = (x @ W1) · ReLU · W2 + b
```

现代 LLM（Llama、PaLM）广泛采用 **SwiGLU** 变体：

```python
# SwiGLU FFN
FFN(x) = (SiLU(x @ W_up) * (x @ W_gate)) @ W_down
```

其中 `SiLU(x) = x * sigmoid(x)` 是 Sigmoid-Linear-Unit 激活函数。

#### 为什么用 SwiGLU？

| 维度 | ReLU-FFN | SwiGLU-FFN |
|---|---|---|
| 参数量 | 2 个矩阵 (W1, W2) | 3 个矩阵 (W_up, W_gate, W_down) |
| 表达能力 | 单路变换 | 门控机制（gate 控制信息流） |
| 实际效果 | 基线 | 同等参数量下效果更好 |

SwiGLU 的核心思想是**门控**：`x @ W_gate` 学习"哪些信息应该通过"，`SiLU(x @ W_up)` 学习"要传递什么信息"，两者逐元素相乘实现动态信息路由。

#### 实现

```python
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        self.up = Linear(d_model, d_ff)      # 上投影
        self.gate = Linear(d_model, d_ff)    # 门控投影
        self.down = Linear(d_ff, d_model)    # 下投影

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(silu(self.up(x)) * self.gate(x))
```

注意 `d_ff` 通常设置为 `d_model` 的 2-4 倍（如 d_model=768, d_ff=3072），中间层更宽以增强表达能力。

---

### RoPE（Rotary Position Embedding）

#### 位置编码的演进

| 方法 | 原理 | 缺点 |
|---|---|---|
| 绝对位置编码 | 在 embedding 上加可学习的位置向量 | 外推性差（超出训练长度效果骤降） |
| 相对位置编码 | 在注意力分数中编码相对距离 | 实现复杂，计算开销大 |
| **RoPE** | 通过旋转矩阵注入位置信息 | 当前主流方案 |

#### RoPE 的核心思想

RoPE 不直接"加"位置信息，而是通过**旋转**向量的方式来编码位置。

对于二维向量 $(x_1, x_2)$ 和位置 $m$，旋转操作为：

$$
\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
$$

**关键性质：** 两个位置 $m$ 和 $n$ 的向量做内积时，结果只依赖于相对位置 $m-n$：

$$
\langle f_q(x, m), f_k(y, n) \rangle = \text{只与 } (m-n) \text{ 有关的函数}
$$

这正是注意力机制需要的——我们关心的是 token 之间的**相对距离**，而不是绝对位置。

#### 高维推广

对于 $d_k$ 维向量，把它拆成 $d_k/2$ 个二维平面，每个平面用不同的旋转频率：

$$
\theta_i = \theta^{-2i/d_k} \quad (i = 0, 1, ..., d_k/2-1)
$$

其中 $\theta$ 是基础频率（通常设为 10000）。频率随维度递减，低维用高频（捕捉短距离关系），高维用低频（捕捉长距离关系）。

#### 实现要点

```python
class RoPEEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        # 预计算逆频率: [d_k/2]
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, token_positions=None):
        # 计算旋转角度: (batch, seq_len, d_k/2)
        theta = torch.einsum("...i, j -> ...ij", token_positions, self.inv_freq)
        cos, sin = torch.cos(theta), torch.sin(theta)
        # repeat_interleave 让每个角度对应两个维度
        cos = cos.repeat_interleave(2, dim=-1)  # (batch, seq_len, d_k)
        sin = sin.repeat_interleave(2, dim=-1)
        # 旋转: x' = x*cos + rotate_half(x)*sin
        return x * cos + self._rotate_half(x) * sin
```

**关键细节：**
- `register_buffer(..., persistent=False)`：buffer 不保存到模型 state_dict
- `_rotate_half` 用 `einops.rearrange` 把相邻两元素配对，然后 $(-x_2, x_1)$ 实现 90° 旋转
- RoPE **只作用于 Q 和 K**，不作用于 V（因为 V 是被加权的内容，不需要位置信息）
- 预计算的 `inv_freq` 在所有 batch 和所有层之间共享

---

### Multi-Head Self-Attention（MHA）

#### Scaled Dot-Product Attention

注意力机制的本质是**加权平均**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**为什么要除以 $\sqrt{d_k}$？**

当 $d_k$ 较大时，$QK^T$ 的方差会变大，导致 softmax 的输入绝对值过大。softmax 对大输入非常敏感（梯度趋近于 0），除以 $\sqrt{d_k}$ 可以把方差归一化到 1 附近，保持梯度流动。

**数值稳定的 Softmax：**

```python
def stable_softmax(logits, dim=-1):
    max_logits = torch.max(logits, dim=dim, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)  # 减去最大值防止溢出
    return exp_logits / torch.sum(exp_logits, dim=dim, keepdim=True)
```

减去最大值不改变 softmax 的数学结果（分子分母同乘 $e^{-\max}$），但能避免 `exp()` 溢出。

#### Causal Mask（因果掩码）

语言模型是**自回归**的：预测第 $t$ 个 token 时只能看到前 $t-1$ 个 token。Causal Mask 确保注意力不会"偷看"未来信息：

```python
def _create_causal_mask(self, seq_len, device):
    # 下三角矩阵: 位置 (i, j) 为 True 当且仅当 j <= i
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
```

应用方式是在 softmax 之前，把 mask 为 0 的位置填充为 $-\infty$：

```python
scores = scores.masked_fill(mask == 0, float("-inf"))
```

这样 softmax 之后这些位置的权重就是 0，模型无法关注到未来的 token。

#### 多头机制

单个注意力头只能学习一种"关注模式"。多头注意力把 $d_{model}$ 拆成 $h$ 个头，每个头独立计算注意力，最后拼接：

```
Q, K, V (batch, seq_len, d_model)
  → view(batch, seq_len, num_heads, d_k)
  → transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
  → 每个头独立计算注意力
  → transpose(1, 2).contiguous().view(batch, seq_len, d_model)
```

**为什么需要 `contiguous()`？** `transpose` 返回的是非连续张量（只是改变了 stride），`view` 要求内存连续，所以需要先调用 `contiguous()` 重新排列内存。

#### MHA 完整流程

```python
class MHA(nn.Module):
    def forward(self, x, token_positions=None):
        # 1. 线性投影 + 分头
        Q = self.q_linear(x).view(...).transpose(1, 2)
        K = self.k_linear(x).view(...).transpose(1, 2)
        V = self.v_linear(x).view(...).transpose(1, 2)

        # 2. RoPE 位置编码（只作用于 Q, K）
        if self.use_rope:
            Q, K = self.rope(Q, token_positions), self.rope(K, token_positions)

        # 3. Causal Mask
        causal_mask = self._create_causal_mask(seq_len, x.device)

        # 4. Scaled Dot-Product Attention
        attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)

        # 5. 合并头 + 输出投影
        output = self.out_linear(attn_output.transpose(1, 2).contiguous().view(...))
        return output
```

---

### TransformerBlock

#### Pre-Norm 结构

```
x → RMSNorm → MHA → x + MHA_output → RMSNorm → FFN → x + FFN_output
```

**为什么用 Pre-Norm 而不是 Post-Norm？**

| 维度 | Post-Norm | Pre-Norm |
|---|---|---|
| 归一化位置 | 子层之后 | 子层之前 |
| 梯度流 | 深层容易梯度消失 | 残差路径畅通 |
| 训练稳定性 | 需要 warmup | 更稳定 |
| 可训练深度 | 通常 ≤ 24 层 | 可以更深 |

Pre-Norm 的核心优势是**残差路径更干净**：输入 $x$ 直接加到子层输出上，不经过归一化，梯度可以无损地回传。

#### 实现

```python
class TransformerBlock(nn.Module):
    def forward(self, x, token_positions=None):
        # Pre-Norm + MHA + Residual
        x = x + self.mha(self.norm1(x), token_positions=token_positions)
        # Pre-Norm + FFN + Residual
        x = x + self.ffn(self.norm2(x))
        return x
```

简洁但强大。每个 block 的输入输出形状保持一致 `(batch, seq_len, d_model)`，可以任意堆叠。

---

### TransformerLM（完整语言模型）

#### 整体流程

```
Token IDs (batch, seq_len)
    ↓
Embedding → (batch, seq_len, d_model)
    ↓
[TransformerBlock × num_layers]
    ↓
Final RMSNorm
    ↓
Output Layer (RMSNorm + Linear) → (batch, seq_len, vocab_size)
    ↓
Logits（用于 next-token prediction）
```

#### 实现

```python
class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        self.token_embedding = Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)
        self.output_layer = OutputLayer(
            config.d_model, config.vocab_size, use_norm=config.use_final_norm
        )

        if config.tie_weights:
            self._tie_weights()

    def forward(self, x, token_positions=None):
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits
```

#### Weight Tying（权重共享）

当 `tie_weights=True` 时，输出层的权重矩阵与 embedding 矩阵共享：

```python
def _tie_weights(self):
    self.output_layer.linear.weight = self.token_embedding.weight
```

**好处：**
- 减少参数量（省一个 $d_{model} \times vocab_{size}$ 的矩阵）
- 实验表明对语言模型性能有正面影响（GPT-2 采用此策略）
- 相当于让 embedding 和输出层学习同一套语义表示

#### Output Layer 设计

```python
class OutputLayer(nn.Module):
    def __init__(self, d_model, vocab_size, use_norm=False):
        self.linear = Linear(d_model, vocab_size)
        self.norm = RMSNorm(d_model) if use_norm else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        return self.linear(x)
```

可选的 final norm 在输出头之前再做一次归一化，进一步稳定 logits 分布。

---

### 测试

```bash
# 测试所有模型组件
uv run pytest tests/test_model.py

# 测试特定组件
uv run pytest tests/test_model.py::test_run_transformer_block
uv run pytest tests/test_model.py::test_run_rope
uv run pytest tests/test_model.py::test_run_multihead_self_attention_with_rope

# 测试 nn 工具函数
uv run pytest tests/test_nn_utils.py
```

测试覆盖了：
- `Linear` / `Embedding` 基础层
- `RMSNorm` / `SwiGLU` / `SiLU` 归一化和激活
- `scaled_dot_product_attention`（含 4D 张量版本）
- `RoPE` 位置编码
- `Multi-Head Self-Attention`（含/不含 RoPE）
- `TransformerBlock` 完整前向传播
- `TransformerLM` 完整模型（含截断输入测试）

### 关键设计决策总结

| 决策 | 选择 | 原因 |
|---|---|---|
| 归一化 | RMSNorm | 更简单，Llama 等现代模型标准 |
| 归一化位置 | Pre-Norm | 训练更稳定，支持更深模型 |
| 位置编码 | RoPE | 相对位置编码，外推性好 |
| FFN 激活 | SwiGLU | 门控机制，同等参数量效果更好 |
| 注意力掩码 | Causal Mask | 自回归生成，防止未来信息泄露 |
| 权重共享 | 可选 tie_weights | 减少参数，GPT-2 标准做法 |
| Softmax | 减 max 数值稳定 | 防止 exp 溢出 |

---

## 优化器与训练流程

### 整体架构

模型搭建完成后，我们需要优化器来更新参数，以及训练引擎来组织完整的前向-反向-优化循环。整体流程可以概括为：

```
数据加载 → 前向传播 → Cross-Entropy Loss → 反向传播 → 梯度裁剪 → AdamW 更新 → LR 调度
```

每个训练步骤的关键组件：

| 组件 | 文件 | 作用 |
|---|---|---|
| `cross_entropy` | `loss.py` | 计算交叉熵损失 |
| `AdamW` | `optim.py` | 带权重衰减的 Adam 优化器 |
| `cosine_annealing_lr` | `optim.py` | 线性 warmup + 余弦退火学习率调度 |
| `gradient_clip` | `optim.py` | 梯度裁剪（L2 范数） |
| `data_loading_sequential` | `data.py` | 顺序数据加载（无重叠 batch） |
| `train` / `eval_model` | `train_engine.py` | 训练循环与评估逻辑 |

---

### 交叉熵损失（Cross-Entropy Loss）

#### 从 Logits 到 Loss

语言模型的输出是 **logits**（未归一化的分数），形状为 `(batch * seq_len, vocab_size)`。我们需要将其转换为概率分布，然后计算真实 token 的负对数概率。

```
logits (N, C) → softmax → probs (N, C) → 取真实标签位置的概率 p → loss = -log(p)
```

#### 数值稳定性：减去最大值

直接计算 `exp(logits)` 会导致溢出——当 logits 中某个值很大时（如 1000），`exp(1000)` 会返回 `inf`。

**解决方案：** 对每个样本减去其 logits 的最大值：

```python
logits = logits - torch.max(logits, dim=1, keepdim=True).values
```

**数学等价性：** softmax 具有平移不变性：

$$
\text{softmax}(x - c) = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}} = \frac{e^{x_i}}{\sum_j e^{x_j}} = \text{softmax}(x)
$$

减去最大值不改变最终的概率分布，但能确保 `exp()` 的输入 ≤ 0，永远不会溢出。

#### 完整实现

```python
def cross_entropy(logits: torch.Tensor, labels: torch.Tensor):
    # 1. 数值稳定：减去最大值
    logits = logits - torch.max(logits, dim=1, keepdim=True).values

    # 2. 计算 log_softmax: logP(x) = logits - log(sum(exp(logits)))
    log_probs = logits - torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True))

    # 3. 只提取真实标签位置的对数概率
    labels = labels.unsqueeze(1)  # (N, 1)
    loss = log_probs.gather(1, labels).squeeze(1)  # (N,)

    # 4. 取平均
    return -loss.mean()
```

**`gather` 的作用：** `log_probs` 是 `(N, C)` 的矩阵，包含每个样本对所有词的对数概率。但计算 loss 时我们只关心**正确答案**对应的概率。`gather(1, labels)` 从每行中挑出真实标签位置的值，等价于：

```python
# 等价于：
loss = torch.tensor([log_probs[i, labels[i]] for i in range(N)])
```

但 `gather` 是向量化操作，比 Python 循环快几个数量级。

#### 困惑度（Perplexity）

困惑度是语言模型最常用的评估指标：

```python
def perplexity(loss: torch.Tensor) -> torch.Tensor:
    return torch.exp(loss)
```

**直觉理解：** 困惑度 ≈ "模型在预测时等效的随机选择范围"。PPL = 100 意味着模型在每个位置都像是从 100 个词中随机选一个。

| PPL | 含义 |
|---|---|
| 1.0 | 完美预测（概率 100%） |
| vocab_size | 均匀随机猜测 |
| ∞ | 给真实 token 的概率为 0 |

PPL 越小越好。GPT-2 在 WikiText-2 上的 PPL 约为 30，GPT-3 约为 20。

---

### AdamW 优化器

#### 从 SGD 到 Adam

| 优化器 | 更新规则 | 特点 |
|---|---|---|
| SGD | $θ = θ - η · g$ | 简单，但需要手动调 LR |
| Momentum SGD | 加入一阶动量 | 加速收敛，减少震荡 |
| Adam | 一阶 + 二阶动量自适应 | 几乎不需要调 LR |
| **AdamW** | Adam + 解耦权重衰减 | Llama、GPT 系列标准选择 |

#### Adam 的核心思想

Adam 维护两个状态：

1. **一阶动量（`exp_avg`）**：梯度的指数移动平均，相当于"方向"
   $$m_t = β_1 · m_{t-1} + (1 - β_1) · g_t$$

2. **二阶动量（`exp_avg_sq`）**：梯度平方的指数移动平均，相当于"步长自适应"
   $$v_t = β_2 · v_{t-1} + (1 - β_2) · g_t^2$$

更新时，用一阶动量决定方向，用二阶动量决定步长：

$$θ_t = θ_{t-1} - η · \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + ε}$$

其中 $\hat{m}_t$ 和 $\hat{v}_t$ 是**偏差校正**后的动量（见下文）。

#### 偏差校正（Bias Correction）

初始时 $m_0 = 0, v_0 = 0$，这会导致早期动量估计偏向 0。偏差校正公式：

$$\hat{m}_t = \frac{m_t}{1 - β_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - β_2^t}$$

当 $t$ 较小时，$1 - β_1^t$ 也较小（如 $t=1, β_1=0.9$ 时为 0.1），校正后动量被放大，补偿了初始偏差。随着 $t$ 增大，校正因子趋近于 1。

#### 为什么是 AdamW 而不是 Adam + L2？

Adam 的原始论文把权重衰减（L2 regularization）直接加到梯度里：

```python
# Adam + L2（不推荐）
grad = grad + weight_decay * param  # 权重衰减混入梯度
```

但这在 Adam 中**不等价于真正的 L2 正则化**，因为 Adam 的自适应学习率会改变权重衰减的实际效果。

**AdamW（Decoupled Weight Decay）** 把权重衰减从梯度更新中解耦出来：

```python
# AdamW：先做权重衰减，再做 Adam 更新
param = param * (1 - lr * weight_decay)  # 独立的权重衰减
param = param - lr * m_hat / (sqrt(v_hat) + eps)  # Adam 更新
```

Loshchilov & Hutter (2019) 的论文证明 AdamW 在大多数任务上都优于 Adam + L2。

#### 实现要点

```python
class AdamW(torch.optim.Optimizer):
    @torch.no_grad()
    def step(self):
        for p in group["params"]:
            grad = p.grad
            state = self.state[p]

            # 初始化状态
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1
            t = state["step"]

            # 更新动量
            exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))

            # 偏差校正
            bias_correction1 = 1.0 - beta1**t
            bias_correction2 = 1.0 - beta2**t

            # 计算步长
            step_size = lr / bias_correction1
            denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)

            # 解耦权重衰减
            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)

            # 参数更新
            p.addcdiv_(exp_avg, denom, value=-step_size)
```

**关键细节：**
- `@torch.no_grad()`：优化器更新不需要梯度，避免构建计算图
- `addcmul_`：原地计算 `exp_avg_sq += (1-β2) * grad * grad`，节省内存
- `addcdiv_`：原地计算 `param += (-step_size) * exp_avg / denom`
- 默认超参数：`lr=1e-3, β1=0.9, β2=0.999, ε=1e-8, weight_decay=1e-2`

---

### 学习率调度（Cosine Annealing with Warmup）

#### 为什么需要 Warmup？

训练初期，模型参数随机初始化，梯度方向不稳定。如果一开始就用大学习率，可能导致：
- 梯度爆炸
- 损失函数震荡
- 训练发散

**Warmup 策略：** 前 `Tw` 步线性增加学习率，从 0 缓慢升到 `alpha_max`：

```python
if t < Tw:
    lr = (t / Tw) * alpha_max  # 线性增长
```

#### 余弦退火（Cosine Annealing）

Warmup 结束后，学习率按余弦曲线从 `alpha_max` 衰减到 `alpha_min`：

```python
progress = (t - Tw) / (Tc - Tw)  # [0, 1]
lr = alpha_min + 0.5 * (1 + cos(π * progress)) * (alpha_max - alpha_min)
```

**为什么用余弦而不是线性衰减？**

| 维度 | 线性衰减 | 余弦衰减 |
|---|---|---|
| 前期衰减速度 | 恒定 | 缓慢（保留较大 LR） |
| 后期衰减速度 | 恒定 | 加速（快速收敛） |
| 实际效果 | 基线 | 通常更好 |

余弦曲线在前期下降缓慢，让模型有足够时间探索损失曲面；后期加速衰减，帮助模型稳定收敛到局部最优。

#### 完整调度曲线

```
LR
 ↑
 |     /\
 |    /  \
 |   /    \      ← 余弦衰减
 |  /      \
 | /        \____ ← alpha_min
 |/ warmup
 +----------------→ step
 0   Tw    Tc
```

#### 实现

```python
def cosine_annealing_lr(t, alpha_max, alpha_min, Tw, Tc):
    # Warm-up 阶段
    if Tw > 0 and t < Tw:
        return (t / Tw) * alpha_max

    # 余弦退火阶段
    if t <= Tc:
        if Tc == Tw:
            return alpha_max
        progress = (t - Tw) / (Tc - Tw)
        return alpha_min + 0.5 * (1 + cos(π * progress)) * (alpha_max - alpha_min)

    # 退火结束后保持最小学习率
    return alpha_min
```

**典型配置（GPT-2 风格）：**
- `alpha_max = 6e-4`（峰值学习率）
- `alpha_min = alpha_max * 0.1`（最小学习率）
- `Tw = 2000`（warmup 步数）
- `Tc = num_steps - Tw`（余弦退火总步数）

---

### 梯度裁剪（Gradient Clipping）

#### 为什么需要梯度裁剪？

深层网络中，反向传播的梯度可能累积到非常大的值（梯度爆炸），导致：
- 参数更新步长过大，跳过最优解
- 数值溢出（NaN）
- 训练不稳定

#### L2 范数裁剪

计算所有参数梯度的全局 L2 范数，如果超过阈值 `max_l2_norm`，则等比例缩放：

```python
def gradient_clip(parameters, max_l2_norm, eps=1e-6):
    # 1. 计算全局 L2 范数
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    # 2. 计算缩放系数
    clip_coef = max_l2_norm / (total_norm + eps)

    # 3. 只在需要时裁剪
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
```

**关键细节：**
- `eps = 1e-6` 防止除零（当 total_norm 接近 0 时）
- 只在 `clip_coef < 1.0` 时裁剪，避免不必要的计算
- 等比例缩放保持梯度方向不变，只缩小幅度
- 典型 `max_l2_norm = 1.0`（GPT-2 配置）

---

### 数据加载

#### 随机采样 vs 顺序采样

| 方式 | 特点 | 适用场景 |
|---|---|---|
| `get_batch`（随机） | 每次随机采样起始位置 | 训练（增加随机性） |
| `get_batch_sequential`（顺序） | 按顺序遍历数据，无重叠 | 评估、高效训练 |

#### 顺序数据加载（`data_loading_sequential`）

训练时使用顺序采样可以**确保每个 token 恰好被使用一次**，没有浪费：

```python
def get_batch_sequential(x_t, batch_size, context_length, device, state, stride=None):
    if stride is None:
        stride = context_length  # 默认无重叠

    n = x_t.numel()
    max_start = n - context_length - 1

    # 如果当前位置不够，重置到开头
    last_start = state.pos + (batch_size - 1) * stride
    end = last_start + context_length + 1
    if end > n:
        state.pos = 0

    # 使用 as_strided 创建视图，避免数据拷贝
    base = x_t[state.pos : end]
    inputs = base.as_strided(size=(batch_size, context_length), stride=(stride, 1))
    targets = base[1:].as_strided(size=(batch_size, context_length), stride=(stride, 1))

    state.pos += batch_size * stride

    # 传输到目标设备
    inputs = inputs.to(device).long()
    targets = targets.to(device).long()
    return inputs, targets
```

**`as_strided` 的妙用：** 不复制数据，只改变张量的"视角"。对于 1D 序列 `[0, 1, 2, 3, 4, 5, ...]`，可以高效地创建 2D batch：

```
原始: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]

inputs (B=2, T=3, stride=3):
  [[0, 1, 2],
   [3, 4, 5]]

targets (偏移 1):
  [[1, 2, 3],
   [4, 5, 6]]
```

**内存效率：** `as_strided` 创建的是视图（view），不分配新内存。对于 GB 级别的数据集，这比 `reshape` 或 `slice` 更高效。

---

### 训练引擎（`train_engine.py`）

#### 训练循环

```python
def train(model, optimizer, train_config):
    # 加载数据（memmap，不一次性读入内存）
    original_data = np.memmap(train_config.train_data_path, dtype=np.uint16, mode="r+")
    x = torch.from_numpy(original_data)

    ctx = get_ctx(train_config.use_mixed_precision, train_config.device)
    state = BatchState(pos=0)

    for step in range(train_config.num_steps):
        # 1. 获取 batch
        inputs, targets = data_loading_sequential(x, batch_size, context_length, device, state)

        # 2. 前向传播（可选混合精度）
        with ctx:
            logits, aux = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = cross_entropy(logits, targets)

        # 3. 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 4. 梯度裁剪
        gradient_clip(model.parameters(), max_l2_norm=train_config.max_grad_norm)

        # 5. 学习率调度
        lr = cosine_annealing_lr(step, alpha_max, alpha_min, Tw, Tc)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 6. 参数更新
        optimizer.step()
```

**关键设计：**
- `np.memmap`：内存映射文件，GB 级数据集不会 OOM
- `optimizer.zero_grad(set_to_none=True)`：把梯度设为 `None` 而不是 0，节省内存
- `get_ctx`：根据配置返回 `torch.autocast`（混合精度）或 `nullcontext`（普通精度）

#### 评估逻辑

```python
@torch.no_grad()
def eval_model(model, train_config):
    model.eval()
    eval_loss = 0.0
    eval_perplexity = 0.0

    # 加载评估数据
    original_data = np.memmap(train_config.eval_data_path, dtype=np.uint16, mode="r+")
    x = torch.from_numpy(original_data)

    total_tokens = len(original_data)
    num_eval_batches = total_tokens // (batch_size * context_length)

    state = BatchState(pos=0)
    for _ in trange(num_eval_batches):
        inputs, targets = data_loading_sequential(x, batch_size, context_length, device, state)
        logits, aux = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        eval_loss += loss.item()
        eval_perplexity += perplexity(loss).item()

    return eval_loss / num_eval_batches, eval_perplexity / num_eval_batches
```

**关键细节：**
- `@torch.no_grad()`：评估时不构建计算图，节省内存和计算
- `model.eval()`：关闭 dropout 等训练特有的行为
- 评估完成后调用 `model.train()` 恢复训练模式

#### 训练中的额外功能

| 功能 | 触发条件 | 作用 |
|---|---|---|
| 评估 | `eval_log_interval` | 定期计算验证集 loss 和 PPL |
| 文本生成 | `sampling_log_interval` | 定期生成文本，直观检查模型质量 |
| 检查点保存 | eval loss 创新低时 | 保存最佳模型权重 |
| W&B 日志 | `wandb_logging=True` | 可视化训练曲线 |
| MoE 监控 | `use_moe=True` | 记录每个 expert 的 token 分配 |

---

### 测试

```bash
# 测试优化器
uv run pytest tests/test_optimizer.py

# 测试数据加载
uv run pytest tests/test_data.py

# 测试序列化（检查点保存/加载）
uv run pytest tests/test_serialization.py
```

测试覆盖了：
- `AdamW` 优化器（与 PyTorch 原生实现对比）
- `cosine_annealing_lr` 学习率调度（验证 warmup + 余弦衰减曲线）
- `get_batch` 随机采样（验证分布均匀性）
- 检查点保存/加载（模型状态一致性）

### 关键设计决策总结

| 决策 | 选择 | 原因 |
|---|---|---|
| 优化器 | AdamW | 解耦权重衰减，Llama/GPT 标准 |
| 学习率调度 | Warmup + 余弦退火 | 训练稳定，收敛效果好 |
| 梯度裁剪 | L2 范数裁剪 | 防止梯度爆炸，训练更稳定 |
| 数据加载 | memmap + 顺序采样 | 内存高效，无 token 浪费 |
| 损失函数 | 手动 cross_entropy | 数值稳定，可定制 |
| 评估指标 | Perplexity | 语言模型标准指标，直观可解释 |
| 混合精度 | 可选 autocast | 加速训练，减少显存占用 |