from __future__ import annotations
from typing import IO, Callable

def get_fileno(file_like: IO[str]) -> int | None:
    if False:
        for i in range(10):
            print('nop')
    'Get fileno() from a file, accounting for poorly implemented file-like objects.\n\n    Args:\n        file_like (IO): A file-like object.\n\n    Returns:\n        int | None: The result of fileno if available, or None if operation failed.\n    '
    fileno: Callable[[], int] | None = getattr(file_like, 'fileno', None)
    if fileno is not None:
        try:
            return fileno()
        except Exception:
            return None
    return None