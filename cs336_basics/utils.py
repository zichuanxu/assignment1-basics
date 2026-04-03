import warnings
from functools import wraps

# ================================ 通用工具函数 ==================================

def print_color(content: str, color: str = "green"):
    print(f"[{color}]{content}[/{color}]")


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


# ================================ 通用工具函数 ==================================
