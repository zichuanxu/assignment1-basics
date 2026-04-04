import gc
import os
import random
import typing
from contextlib import nullcontext
import numpy as np
import torch
from cs336_basics.utils import print_color


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


def get_device(verbose: bool = True) -> torch.device:
    if torch.cuda.is_available():
        if verbose:
            print_color("Using CUDA device", "blue")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        if verbose:
            print_color("Using MPS device", "blue")
        return torch.device("mps")
    else:
        if verbose:
            print_color("Using CPU device", "blue")
        return torch.device("cpu")


def get_ctx(use_mixed: bool, device: torch.device, verbose: bool = True):
    if use_mixed and device.type == "cuda":
        if verbose:
            print_color("Using mixed precision on CUDA with BFloat16", "blue")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        if verbose:
            print_color("Not using mixed precision", "blue")
        return nullcontext()

"""
Checkpoint 的目标是：让你能从中断处无缝继续训练。

因此至少要存这三类东西：

1. 模型参数（model weights）
没有它，就没有模型本体了
2. 优化器状态（optimizer state）
例如 AdamW 的一阶/二阶动量（moment estimates）
不存优化器状态，恢复后训练轨迹会变（因为动量没了）
3. 当前迭代步数（iteration / step）
用来恢复学习率 schedule
否则学习率会从头开始或错位
"""
def save_checkpoint(
    model: torch.nn.Module,
    optimizer,
    iteration,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    verbose: bool = False,
) -> None:
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(state, out)

    if verbose:
        print_color(f"Checkpoint saved to {out}", "blue")


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model, optimizer, verbose: bool = False
) -> int:
    state = torch.load(src, map_location=get_device())

    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])

    if verbose:
        print_color(f"Checkpoint loaded from {src}", "blue")

    return state["iteration"]