from __future__ import annotations
from typing import Iterable

def coerce_bool_value(value: str | bool) -> bool:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(value, bool):
        return value
    elif not value:
        return False
    else:
        return value[0].lower() in ['t', 'y']

def one_or_none_set(iterable: Iterable[bool]) -> bool:
    if False:
        i = 10
        return i + 15
    return 0 <= sum((1 for i in iterable if i)) <= 1