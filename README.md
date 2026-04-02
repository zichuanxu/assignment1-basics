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

- Special Token-aware Splitting: 我们已经了解过了，在初始化vocab 时，我们也需要初始化special tokens，其中一个常见的special tokens就是 <|endoftext|>. 这个token意味着一段文本的结束。给出一段很长的文本，我们要做的第一件事情就是把这个文本分成许多段。
- Regex-based Pre-Tokenization: Pre-Tokenization（预分词） 就是在真正训练 BPE 合并规则之前，先对整份语料做一次粗粒度的切分，把文本切成一段段“更大的片段”（pre-token），然后在这些片段内部去统计相邻字节（byte pair）的出现频率。具体来说，使用GPT-2 使用的那条正则表达式PAT。

**Step 2 — 构建 pair 频率堆 + 倒排索引**


一个很明显的优化点是：每一轮都要找当前频率最高的 pair。在 Version 0 (Section 2.2) 里，我们每轮都通过遍历 pairs_counter 来取最大值，这一步是$O(n)$
（n是 pair 的数量）。而这个操作正好符合堆（heap）的使用场景：用堆维护“当前最大的元素”，就能把“取最大”降到$O(\log n)$
（严格来说是：取堆顶是$O(1)$，但如果包含 pop/push 更新则是 $O(\log n)$ ）。

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
