import functools
from typing import Callable, TypeVar
T = TypeVar('T')

def lru_cache(maxsize: int=128, typed: bool=False) -> Callable[[T], T]:
    if False:
        i = 10
        return i + 15
    "\n    fix: lru_cache annotation doesn't work with a property\n    this hack is only needed for the property, so type annotations are as they are\n    "

    def wrapper(func: T) -> T:
        if False:
            i = 10
            return i + 15
        return functools.lru_cache(maxsize, typed)(func)
    return wrapper