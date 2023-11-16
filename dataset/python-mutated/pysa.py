from typing import TypeVar
T = TypeVar('T')

def mark_sanitized(arg: T) -> T:
    if False:
        return 10
    return arg