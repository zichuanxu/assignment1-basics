from collections import Counter
import heapq
import warnings
import os
from typing import BinaryIO
import time
from functools import wraps

# ================================ 通用工具函数 ==================================


def deprecated(reason):
    """这是一个用于标记函数过时的装饰器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# 使用方法：
@deprecated("Use heap-based optimization for better performance")
def pair_counts(word_counter):
    # ... 原有代码 ...
    pass


# ================================ 通用工具函数 ==================================

# ================================== BPE 训练相关工具函数 ==================================

"""
一个很明显的优化点是：每一轮都要找当前频率最高的 pair。我们每轮都通过遍历 pairs_counter 来取最大值，这一步是
（O(n),n是 pair 的数量）。而这个操作正好符合堆（heap）的使用场景：用堆维护“当前最大的元素”，就能把“取最大”降到
O(1)。
具体做法是把每个 pair 作为堆元素，并把“排序依据”设计成：

频次越大优先级越高
频次相同则按 pair 的字典序更大者优先
在 Python 的 heapq 是最小堆，因此我们可以用负号把它变成“最大堆”，例如存成：

key = (-freq, a, b)
这样每一轮我们都能快速拿到候选的“最常见 pair”。
"""


class HeapItem:
    def __init__(
        self, neg_freq: int, pair_bytes: tuple[bytes, bytes], pair: tuple[int, int]
    ):
        """
        把频率取负数。比如频率 100 变成 -100，频率 50 变成 -50。在小顶堆里，-100 比 -50 小，所以 100 会先被弹出。
        """
        self.neg_freq = neg_freq
        self.pair_bytes = pair_bytes
        self.pair = pair

    def __lt__(self, other: "HeapItem") -> bool:
        if self.neg_freq != other.neg_freq:
            return self.neg_freq < other.neg_freq
        return self.pair_bytes > other.pair_bytes  # reverse order for max-heap behavior


def build_pair_heap(pairs_freqs: Counter, vocab: dict[int, bytes]):
    heap = []
    for (a, b), f in pairs_freqs.items():
        if f > 0:
            item = HeapItem(-f, (vocab[a], vocab[b]), (a, b))
            heapq.heappush(heap, item)
    return heap


"""
核心思路：延迟更新 (Lazy Update)
堆优化的核心难点在于：当两个 ID 合并时，周围相邻 pair 的频率会发生变化，但我们不能立即去堆里修改它们（因为在 Python 的 heapq 中修改中间元素非常慢）。

解决办法：

不管它：频率变了就让它在堆里待着。

校验：每次从堆顶弹出（Pop）最强 pair 时，检查一下它的频率是否和当前 pairs_counter 里的真实频率一致。如果不一致，说明这个数据“过期”了，直接扔掉，看下一个。
"""


def pop_most_frequent_pair(heap, pairs_counter: Counter) -> tuple[int, int]:
    while heap:
        item = heap[0]  # Peek at the top item
        neg_f = item.neg_freq
        pair = item.pair
        # 真实的次数
        cur_f = pairs_counter.get(pair, 0)
        # 堆顶元素的次数是负数，所以取反得到真实频率
        if (
            cur_f <= 0 or -neg_f != cur_f
        ):  # frequency changed, which means the pair we store in heap is stale
            heapq.heappop(heap)
            continue
        return pair

    raise ValueError("No positive-frequency pairs remain")


def string_to_bytes(s: str, return_int: bool = False) -> list[int] | list[bytes]:
    byte_array = s.encode("utf-8")
    return (
        list(map(int, byte_array)) if return_int else [bytes([b]) for b in byte_array]
    )


def utf8_bytes_to_string(byte_indices: list[bytes]) -> str:
    return b"".join(byte_indices).decode("utf-8")


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def print_color(message: str, color: str = "green"):
    # 定义颜色代码映射
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m",  # 必须有重置符，否则后续打印全是这个颜色
    }

    # 获取对应颜色，如果不存在则默认为无颜色
    c = colors.get(color.lower(), "")
    reset = colors["reset"]

    print(f"{c}{message}{reset}")


import os
import json


def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: str | os.PathLike,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vocab_filepath = os.path.join(output_dir, "vocab.json")
    merges_filepath = os.path.join(output_dir, "merges.txt")

    # Save vocab
    vocab_inv = {v.decode("latin1"): k for k, v in vocab.items()}
    with open(vocab_filepath, "w") as vf:
        json.dump(vocab_inv, vf, ensure_ascii=False, indent=2)

    # Save merges
    with open(merges_filepath, "w") as mf:
        mf.write("#version: 0.2\n")
        for a, b in merges:
            mf.write(f"{a.decode('latin1')} {b.decode('latin1')}\n")


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIME] {func.__name__} took {end - start:.2f}s")

        return result

    return wrapper


# ================================== BPE 训练相关工具函数 ==================================
