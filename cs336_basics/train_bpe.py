from cs336_basics.tokenizer import (
    init_vocab,
    pre_tokenize_string_worker,
    merge_pairs_with_heap_index,
    update_vocab,
    split_by_special_tokens,
    PAT,
)
from collections import Counter, defaultdict
import os
from multiprocessing import Process, Manager
from cs336_basics.utils import (
    build_pair_heap,
    pop_most_frequent_pair,
    find_chunk_boundaries,
    print_color,
    save_vocab_and_merges,
    timeit,
)
from tqdm import tqdm, trange
from queue import Empty
import numpy as np
from collections.abc import Iterable, Iterator
import json
import regex as re
import heapq


NUM_PROCESSES = min(4, os.cpu_count() or 1)


@timeit
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    verbose: bool = False,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    计算合并次数：BPE 每次合并产生一个新 Token。
    目标词表大小减去初始的 256 个字节和特殊字符，就是我们需要执行循环的次数。
    初始状态：vocab 此时只包含最基础的单位（0-255）和特殊符号。
    """
    num_merges = vocab_size - 256 - (len(special_tokens) if special_tokens else 0)
    vocab: dict[int, bytes] = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # 1. Pre-tokenization
    # 1.1 Find chunk boundaries
    # 切分文件
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=kwargs.get("desired_num_chunks", NUM_PROCESSES),
            split_special_token=b"\n",
        )

    if verbose:
        print_color(
            f"Identified {len(chunk_boundaries) - 1} chunks for pre-tokenization."
        )
    # 多进程并行
    # 1.2 Count word frequencies across chunks using multiprocessing
    manager = Manager()
    queue = manager.Queue()
    processes: list[Process] = []

    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        p = Process(
            target=pre_tokenize_string_worker,
            args=(input_path, special_tokens, queue, start, end, False),
        )
        processes.append(p)
        p.start()

    if verbose:
        print_color("Pre-tokenization processes completed. Aggregating results...")

    # 等待所有子进程结束，然后再收集词频，避免在长语料下超时导致结果丢失
    for p in processes:
        p.join()
        if p.exitcode is not None and p.exitcode != 0:
            raise RuntimeError(
                f"Pre-tokenization process failed with exit code {p.exitcode}."
            )

    word_counter = Counter()
    while True:
        try:
            partial_counter = queue.get(timeout=1)
            # 主进程使用 word_counter.update 将所有人的结果加在一起。
            # 结果：此时我们得到了语料库中所有词（以字节元组形式）出现的频率
            word_counter.update(partial_counter)
        except Empty:
            break

    if verbose:
        print_color(
            f"Completed pre-tokenization. Vocabulary size: {len(word_counter)} unique tokens."
        )

    """
    pairs_counter[pair]：记录该相邻 pair 在全语料中的总出现次数。 因为每个 word 在语料中出现了 word_counter[word] 次，所以 word 内部每出现一次 pair，就为全局频次贡献 word_counter[word]。
    pair_to_words[pair]：记录该 pair 出现在哪些 word（token 序列）里, 这个映射非常关键：当我们选择某个 pair 进行 merge 时，只有包含该 pair 的 word 会发生变化。
    借助 pair_to_words，我们可以只遍历这些“受影响的 words”，并对 pairs_counter 做局部增量更新，而不是每轮都重新扫描全部 word_counter。
     """
    pairs_counter = Counter()
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    for word in word_counter:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_to_words[pair].add(word)
            pairs_counter[pair] += word_counter[word]

    # 2. BPE Core Loop
    # 建立最大堆
    pair_heap = build_pair_heap(pairs_counter, vocab)

    for i in trange(num_merges):
        # 获取当前最强组合
        most_frequent_pair = pop_most_frequent_pair(pair_heap, pairs_counter)
        # 更新词表并获取新 ID
        new_id = update_vocab(vocab, most_frequent_pair)
        # 局部精准合并（这是最核心的性能优化点）
        word_counter, pairs_counter, pair_heap, pair_to_words = (
            merge_pairs_with_heap_index(
                word_counter,
                pairs_counter,
                most_frequent_pair,
                new_id,
                vocab,
                pair_heap,
                pair_to_words,
            )
        )
        # 记录合并规则
        merges.append((vocab[most_frequent_pair[0]], vocab[most_frequent_pair[1]]))
    # 将训练好的 vocab 和 merges 存入磁盘。
    # 这样在之后的分词阶段（Inference），你可以直接加载它们，而不需要重新训练。
    if kwargs.get("save_path"):
        save_vocab_and_merges(vocab, merges, kwargs["save_path"])
        with open(
            os.path.join(kwargs["save_path"], "special_tokens.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            if special_tokens:
                for token in special_tokens:
                    f.write(f"{token}\n")

    return vocab, merges


def encode_file_to_bin(tokenizer, text_path, out_bin_path, dtype=np.uint16):
    total_bytes = os.path.getsize(text_path)

    with open(text_path, encoding="utf-8") as f_in, open(out_bin_path, "wb") as f_out:
        p_bar = tqdm(
            total=total_bytes, desc="Encoding to binary", unit="B", unit_scale=True
        )

        for line in f_in:
            token_ids = tokenizer.encode(line)
            arr = np.array(token_ids, dtype=dtype)
            arr.tofile(f_out)

            p_bar.update(len(line.encode("utf-8")))


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

        rank: dict[tuple[int, int], int] = {}
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

    def encode(self, text: str) -> list[int]:
        def merge_one_pretoken(ids: list[int]) -> list[int]:
            n = len(ids)
            if n <= 1:
                return ids

            alive = [True] * n

            # Doubly-linked list over positions 0..n-1 (positions are stable; nodes get "deleted")
            prev = [-1] * n
            nxt = [-1] * n
            for i in range(n):
                prev[i] = i - 1
                nxt[i] = i + 1 if i + 1 < n else -1

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

            while heap:
                r, i = heapq.heappop(heap)
                j = nxt[i]
                if j == -1 or not alive[i] or not alive[j]:
                    continue
                # stale check: rank might no longer match current neighbor
                pair = (ids[i], ids[j])
                cur_r = self.rank.get(pair)
                if cur_r is None or cur_r != r:
                    continue

                # merge i and j into i (use precomputed mapping to avoid KeyError)
                new_id = self.merge_to_new_id.get(pair)
                if new_id is None:
                    continue
                ids[i] = new_id

                # delete j from the linked list
                alive[j] = False
                nj = nxt[j]
                nxt[i] = nj
                if nj != -1:
                    prev[nj] = i

                # Only pairs that can change are around i (prev[i], i) and (i, nxt[i])
                pi = prev[i]
                if pi != -1:
                    push_if_valid(pi)
                push_if_valid(i)

            # materialize result by walking the linked list
            out: list[int] = []
            k = 0
            while k != -1:
                if alive[k]:
                    out.append(ids[k])
                k = nxt[k]
            return out

        byte_tokens = self._pre_tokenize(text)

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


def load_tokenizer_from_dir(dir_path: str) -> BPETokenizer:
    vocab_path = os.path.join(dir_path, "vocab.json")
    merges_path = os.path.join(dir_path, "merges.txt")
    special_tokens_path = os.path.join(dir_path, "special_tokens.txt")
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens_path)
    return tokenizer
