from __future__ import annotations
import collections.abc

class CustomClass:

    def __init__(self, number: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.number = number

def add_custom_classes(c1: CustomClass, c2: CustomClass | None=None) -> CustomClass:
    if False:
        return 10
    if c2 is None:
        return CustomClass(c1.number)
    return CustomClass(c1.number + c2.number)

def merge_dicts(map1: collections.abc.Mapping[str, int], map2: collections.abc.Mapping[str, int]) -> collections.abc.Mapping[str, int]:
    if False:
        return 10
    return {**map1, **map2}

def invalid_types(attr1: int, attr2: UnknownClass, attr3: str) -> None:
    if False:
        i = 10
        return i + 15
    pass