from collections import defaultdict
import regex as re
from collections import Counter
from cs336_basics.utils_bpe import string_to_bytes, HeapItem
from cs336_basics.utils import deprecated
import heapq
from collections.abc import Iterable, Iterator
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# 初始化我们的词汇表：
# vocab key: id, value: bytes
def init_vocab(special_tokens: list[str] | None = None) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {
        x: bytes([x]) for x in range(256)
    }  # idx -> byte representation
    current_index = 256

    if special_tokens is not None:
        for token in special_tokens:
            vocab[current_index] = token.encode("utf-8")
            current_index += 1

    return vocab


# 统计文本中所有相邻字节对的出现频率。
# word_counter是整个token序列出现多少次
# Key 是由 ID 组成的元组（表示单词目前的切分状态），Value 是该单词出现的次数。
@deprecated(
    "这个函数在 train_bpe.py 里已经被 merge_pairs_with_heap_index 替代了，因为它效率太低了。"
)
def pair_counts(word_counter: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    pairs: dict[tuple[int, int], int] = {}
    for word, count in word_counter.items():
        # example: word = (10, 20, 30, 40)
        # list(zip(word, word[1:])) [(10, 20), (20, 30), (30, 40)]
        for a, b in zip(word[:-1], word[1:]):
            pairs[(a, b)] = pairs.get((a, b), 0) + count
    return pairs


# 找到出现频率最高的字节对
# 1. 频率最高的pair
# 2. 若多个 pair 频率相同，我们按 pair 的字典序（先比左 token，再比右 token）选择更大的那个。
@deprecated(
    "这个函数在 train_bpe.py 里已经被 pop_most_frequent_pair 替代了,不再遍历字典，而是直接看堆顶。"
)
def get_most_frequent_pair(pair_counter: dict[tuple[int, int], int]) -> tuple[int, int]:
    most_frequent_pair = max(pair_counter.items(), key=lambda item: (item[1], item[0]))
    return most_frequent_pair[0]


# 将这个新的字节对加入词汇表
def update_vocab(vocab: dict[int, bytes], pair: tuple[int, int]) -> int:
    index1, index2 = pair
    vocab[len(vocab)] = vocab[index1] + vocab[index2]
    return len(vocab) - 1


"""
假设我们的语料库里只有一个单词 "abac"，它出现了 10次。
此时，单词被拆解为最小单位（假设 a=1, b=2, c=3）：
word_counter: {(1, 2, 1, 3): 10}
pair_counter: {(1, 2): 10, (2, 1): 10, (1, 3): 10}
函数执行完毕，返回一个元组：
(
    # 第一个字典：告诉主循环，现在的单词长这样了
    {(99, 1, 3): 10},

    # 第二个字典：告诉主循环，下一轮你可以从这两对里选最高频的
    {(99, 1): 10, (1, 3): 10}
)
"""


@deprecated(
    "这个函数在 train_bpe.py 里已经被 build_pair_heap 替代了，因为它效率太低了。"
    "只在开头用一次build_pair_heap：它被改名或合并到了训练开始前的“初始统计”阶段，用来建立第一份 pair 频率堆。"
    "在 merge_pairs_with_heap_index 函数里，它从一个“全量统计函数”变成了“初始统计 + 增量更新”"
)
# 更新文本中所有出现该字节对的地方，以及重新统计文本中所有相邻字节对的出现频率
def merge_pair_ids(
    word_counter: dict[tuple[bytes, ...] | tuple[int, ...], int],
    pair: tuple[int, int],
    new_id: int,
) -> tuple[dict[tuple[int, ...], int], dict[tuple[int, int], int]]:
    new_word_counter: defaultdict[tuple[int, ...], int] = defaultdict(int)
    updated_pair_counter: defaultdict[tuple[int, int], int] = defaultdict(int)
    for token, freq in word_counter.items():
        new_token = []
        i = 0
        L = len(token)
        while i < L:
            if i < L - 1 and (token[i], token[i + 1]) == pair:
                new_token.append(new_id)
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        new_word_counter[tuple(new_token)] += freq

        for index1, index2 in zip(new_token[:-1], new_token[1:]):
            updated_pair_counter[(index1, index2)] += freq

    return dict(new_word_counter), dict(updated_pair_counter)


def get_new_word(
    word: tuple[int, ...],
    target_pair: tuple[int, int],
    new_id: int,
) -> tuple[int, ...]:
    a, b = target_pair
    new_word = []
    i = 0

    while i < len(word):
        if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1

    return tuple(new_word)


def pre_tokenize(
    string: str,
    special_tokens: list[str] | None = None,
    including_special: bool = False,
) -> Counter:
    word_counter = Counter()
    chunks = split_by_special_tokens(
        string, special_tokens, include_special=including_special
    )
    for chunk in chunks:
        if not chunk:
            continue
        if including_special and chunk in special_tokens:
            word_counter[tuple(string_to_bytes(chunk))] += 1
        else:
            for match in re.finditer(PAT, chunk):
                token = match.group(0)
                token_ids = tuple(string_to_bytes(token, return_int=True))
                word_counter[token_ids] += 1
    return word_counter


"""
如果没有这段代码，BPE 的 pre_tokenize 阶段会成为瓶颈：

单进程：读取 1GB 文件 -> 分词 -> 统计（耗时 10 分钟）。

多进程 (8核)：文件切 8 份 -> 8 个核心同时分词 -> 汇总（耗时约 1.5 分钟）。

这段代码通过文件指针定位 (seek) 和 进程间通信 (Queue)，
实现了对海量文本的高效预处理，为后续的 BPE 迭代打下了坚实的性能基础。
"""


def pre_tokenize_string_worker(*args):
    input_path, special_tokens, queue, start, end, include_special = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    word_counter = pre_tokenize(
        chunk, special_tokens, including_special=include_special
    )
    queue.put(word_counter)


"""
通过把所有 special tokens 先按长度降序排序，
并用正则构造匹配 pattern，我们可以把原始长文本拆成一系列 普通文本片段（以及可选的 special token 片段）。
当 include_special=True 时，
re.split(f"({pattern})", text) 会把匹配到的 special token 也保留下来，从而在后续编码时我们可以把它们当作“原子 token”直接映射到对应的 id；
当 include_special=False 时，
special token 会作为分隔符被丢弃，仅返回普通文本片段，适合训练阶段不想让 special tokens 参与 pair 统计 / merges 的场景。
"""


def split_by_special_tokens(
    text: str, special_tokens: list[str], include_special: bool = False
) -> list[str]:
    if not special_tokens:
        return [text]
    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in special_tokens_sorted)

    if include_special:
        parts = re.split(f"({pattern})", text)
    else:
        parts = re.split(pattern, text)
    return parts


"""
我们每一轮都会遍历 word_counter 里的所有 word，检查这个 word 里是否出现了目标 pair；这一步的代价通常非常高，因为绝大多数 word 根本不包含 当前要 merge 的 pair，但我们还是把它们都扫了一遍。

因此我们可以用一个“倒排索引”来做 空间换时间：提前维护一个映射 pair -> {words…}，记录每个 pair 出现在哪些 word 中。这样当我们决定 merge 某个 pair 时，就只需要遍历 pair_to_words[pair] 里的那一小部分 word，而不必全量扫描所有 word。

这也正是我们搭建 pair_to_words 的原因：有索引：每轮只处理 包含该 pair 的 words 子集（快，复杂度取决于该 pair 的覆盖范围，通常远小于全量）。
"""


def merge_pairs_with_heap_index(
    word_counter: dict[tuple[int, ...], int],
    pair_counter: Counter,
    target_pair: tuple[int, int],
    new_id: int,
    vocab: dict[int, bytes],
    pair_heap,
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]],
) -> tuple[
    dict[tuple[int, ...], int],
    Counter,
    list,
    dict[tuple[int, int], set[tuple[int, ...]]],
]:
    # 保留未受影响单词计数；仅对受影响单词做“替换”
    new_word_counter: Counter = Counter(word_counter)
    updated_pair_counter: Counter = pair_counter.copy()
    changed_pairs: set[tuple[int, int]] = set()

    affected_words = list(pair_to_words.get(target_pair, set()))

    # 更新 pair_to_words 索引：新单词会产生新的 pair，旧单词会失效一些旧 pair
    for w in affected_words:
        freq = word_counter.get(w, 0)
        if freq <= 0 or len(w) < 2:
            continue
        # 1. 从词典计数中扣除旧单词的频率
        new_word_counter[w] -= freq
        if new_word_counter.get(w, 0) <= 0:
            del new_word_counter[w]
        # 2. 关键：清理旧邻居的频率
        """
        为什么要扣除所有相邻对？ 因为只要单词 w 发生了合并，
        它内部所有的相邻关系都会断开或重组。为了保证计数准确，必须先“归零”旧的贡献。
        """
        for i in range(len(w) - 1):
            pair = (w[i], w[i + 1])
            # 这些相邻对即将消失或改变
            updated_pair_counter[pair] -= freq
            # 标记这些 pair 需要重新入堆
            changed_pairs.add(pair)
            # 在索引里删掉旧单词
            s = pair_to_words.get(pair)
            if s is not None:
                s.discard(w)
                if not s:
                    del pair_to_words[pair]
        # 3. 构造新单词，并加入词典计数
        new_word = get_new_word(w, target_pair, new_id)
        new_word_counter[new_word] += freq

        # 4. 更新新单词的相邻对频率，并更新索引
        if len(new_word) < 2:
            continue
        for i in range(len(new_word) - 1):
            pair = (new_word[i], new_word[i + 1])
            updated_pair_counter[pair] += freq
            changed_pairs.add(pair)
            pair_to_words.setdefault(pair, set()).add(new_word)

    # 5. 把受影响的 pair 重新入堆
    if pair_heap is not None:
        for pair in changed_pairs:
            f = updated_pair_counter.get(pair, 0)
            if f > 0:
                """
                这里并没有去堆里寻找并删除旧数据（因为那太慢了），
                而是直接把最新的频率作为新任务 heappush 进去。
                这会导致堆里存在多个相同的 pair，但频率不同。
                但是堆自己会在pop_most_frequent_pair函数校验
                """
                heapq.heappush(
                    pair_heap, HeapItem(-f, (vocab[pair[0]], vocab[pair[1]]), pair)
                )
    return dict(new_word_counter), updated_pair_counter, pair_heap, pair_to_words


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [t.encode("utf-8") for t in self.special_tokens]
        self.special_set = set(self.special_tokens_bytes)

        self.vocab_inv = {v: k for k, v in self.vocab.items()}

        # 记录合并的优先级。在 merges 列表里越靠前（r 越小），优先级越高。
        # 比如遇到 h e l l o，如果 h e 排第 1，l l 排第 5，那就必须先合并 h e。
        rank: dict[tuple[int, int], int] = {}
        # 速查表。直接记录 (ID_h, ID_e) -> ID_he。有了它，合并时瞬间就能拿到新 ID。
        merge_to_new_id: dict[tuple[int, int], int] = {}

        for r, (a_bytes, b_bytes) in enumerate(self.merges):
            a_id = self.vocab_inv.get(a_bytes)
            b_id = self.vocab_inv.get(b_bytes)
            # The merged token should be present in vocab; if not, skip this merge rule.
            new_id = self.vocab_inv.get(a_bytes + b_bytes)
            if a_id is None or b_id is None or new_id is None:
                continue
            pair = (a_id, b_id)
            rank[pair] = r
            merge_to_new_id[pair] = new_id

        self.rank = rank
        self.merge_to_new_id = merge_to_new_id

        self.eos_token_id = self.vocab_inv.get(b"<|endoftext|>", None)

    """
    当拿到一句长长的文本，比如 "Hello world! <|endoftext|>"，不能直接把它全拆成单字母。

    保护特殊符号：把 <|endoftext|> 这种单独拎出来，它不参与合并。

    正则切分（PAT）：把 "Hello world!" 切成 "Hello", " ", "world", "!"。

    转成 Bytes：最后把每个碎片转成 utf-8 字节。
    """

    def _pre_tokenize(self, text: str) -> list[bytes]:
        parts = split_by_special_tokens(text, self.special_tokens, include_special=True)
        token_list: list[bytes] = []

        for part in parts:
            if part == "":
                continue
            if part in self.special_tokens:
                token_list.append(part.encode("utf-8"))
            else:
                for tok in re.findall(PAT, part):
                    # Each regex token becomes a single bytestring.
                    token_list.append(tok.encode("utf-8"))

        return token_list

    """
    如果用普通的数组来合并，每次把两个元素变成一个元素，数组后面的所有元素都要往前挪一位。如果句子很长，这种操作会慢得让人崩溃。
    这段代码使用了一个高级数据结构组合：双向链表 (Doubly-Linked List) + 优先队列/堆 (Heap)。
    """

    def encode(self, text: str) -> list[int]:
        def merge_one_pretoken(ids: list[int]) -> list[int]:
            n = len(ids)
            if n <= 1:
                return ids

            """
            合并时并不真的 del 掉元素，而是：
            标记被吞掉的节点 alive[j] = False
            调整指针 nxt[i] = nxt[j]、prev[nxt[j]] = i
            这样就能在 O(1) 时间内完成合并，而不需要移动后续元素。
            """
            alive = [True] * n

            # Doubly-linked list over positions 0..n-1 (positions are stable; nodes get "deleted")
            prev = [-1] * n
            nxt = [-1] * n
            for i in range(n):
                prev[i] = i - 1
                nxt[i] = i + 1 if i + 1 < n else -1
            """
            堆里存 (rank, i)，
            表示当前位置 i 与其右邻居 nxt[i] 的 pair 在 merge 规则中的优先级（rank 越小越先合并）。
            每次取出最小 rank 的候选，做一次合并，然后只需要重新检查局部的两个 pair：

            (prev[i], i)
            (i, nxt[i])

            """
            # best pair per left-position i: (rank, i)
            heap: list[tuple[int, int]] = []

            def push_if_valid(i: int):
                cur_r = None
                j = nxt[i]
                if j == -1 or not alive[i] or not alive[j]:
                    cur_r = None
                else:
                    cur_r = self.rank.get((ids[i], ids[j]))

                if cur_r is not None:
                    heapq.heappush(heap, (cur_r, i))

            for i in range(n):
                push_if_valid(i)
            """

            与之前的heap一样，heap里面的内容会 “过期”：
            因为合并会改变邻接关系，堆中旧条目会过期，所以每次 pop 出来都要验证,

            接下来就是遍历这个heap，
            如果这个heap不是空的，我们就弹出，并且验证：

            这段 while heap: 是整个 merge_one_pretoken 的核心：
            堆里维护“当前可合并的相邻 pair”，
            每次取出 rank 最小（最优先） 的候选进行合并，并只更新合并点附近的候选。
            """
            while heap:  # 只要还有候选 pair，就继续尝试合并
                r, i = heapq.heappop(
                    heap
                )  # 取出当前 rank 最小的候选：(rank, 左端点位置 i)
                j = nxt[i]  # 右端点位置 j 是 i 在链表中的后继
                if (
                    j == -1 or not alive[i] or not alive[j]
                ):  # i/j 无效或 i 已到尾部：这是过期候选
                    continue
                # stale check：堆里的记录可能已过期（邻居关系/ids 已改变），需要重新验证
                # 注意后面合并后nxt[i]会变，所以每次都要检查当前 i 和 j 的 pair 是否仍然匹配当前 rank
                pair = (ids[i], ids[j])
                cur_r = self.rank.get(
                    pair
                )  # 查询这个 pair 在 merge 规则中的 rank（不可合并则为 None）
                if (
                    cur_r is None or cur_r != r
                ):  # 现在不可合并，或 rank 已不匹配：说明堆元素过期
                    continue

                # 执行合并：把 (ids[i], ids[j]) 合成一个新 token，并写回到位置 i
                new_id = self.merge_to_new_id.get(pair)
                if new_id is None:
                    continue
                ids[i] = new_id

                # 从链表中删除 j：j 被 i 吞掉了
                alive[j] = False
                nj = nxt[j]
                nxt[i] = nj
                if nj != -1:
                    prev[nj] = i

                # 局部更新：合并只会影响 i 附近的两个相邻 pair
                pi = prev[i]
                if pi != -1:
                    push_if_valid(pi)  # (pi, i) 这个 pair 可能变得可合并或 rank 改变
                push_if_valid(i)  # (i, nxt[i]) 这个 pair 也可能变得可合并或 rank 改变

            # 最终合并后的 token id 序列
            out: list[int] = []
            k = 0
            while k != -1:
                if alive[k]:
                    out.append(ids[k])
                k = nxt[k]
            return out

        byte_tokens = self._pre_tokenize(text)

        """
        Pre-tokenization：先粗粒度切分文本
        对每个切分出来的 pre-token 做 BPE merge (merge_one_pretoken)，得到最终的 token ID 序列。
        """
        token_ids: list[int] = []
        for btok in byte_tokens:
            if btok in self.special_set:
                token_ids.append(self.vocab_inv[btok])
            else:
                ids = [self.vocab_inv[bytes([b])] for b in btok]
                token_ids.extend(merge_one_pretoken(ids))

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Placeholder for iterable encoding logic
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        # https://en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character

        tokens = b"".join(self.vocab.get(i, b"\xef\xbf\xbd") for i in ids)
        return tokens.decode("utf-8", errors="replace")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | str | None = None,
    ) -> "BPETokenizer":
        with open(vocab_filepath) as vf:
            vocab_data = json.load(vf)
            vocab = {int(i): bytes(v, "latin1") for v, i in vocab_data.items()}

        merges = []
        with open(merges_filepath) as mf:
            # Skip the first line (header)
            next(mf)
            for line in mf:
                if line.strip() and not line.startswith("#"):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        merges.append(
                            (bytes(parts[0], "latin1"), bytes(parts[1], "latin1"))
                        )

        if isinstance(special_tokens, str):
            with open(special_tokens, encoding="utf-8") as stf:
                special_tokens_list = [line.strip() for line in stf if line.strip()]
        elif isinstance(special_tokens, list):
            special_tokens_list = special_tokens
        else:
            special_tokens_list = []

        return cls(vocab, merges, special_tokens_list)
