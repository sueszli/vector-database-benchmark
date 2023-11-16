from __future__ import annotations
from typing import overload
from typing_extensions import assert_type

class CustomIndex:

    def __index__(self) -> int:
        if False:
            i = 10
            return i + 15
        return 1
assert_type(round(5.5), int)
assert_type(round(5.5, None), int)
assert_type(round(5.5, 0), float)
assert_type(round(5.5, 1), float)
assert_type(round(5.5, 5), float)
assert_type(round(5.5, CustomIndex()), float)
assert_type(round(1), int)
assert_type(round(1, 1), int)
assert_type(round(1, None), int)
assert_type(round(1, CustomIndex()), int)

class WithCustomRound1:

    def __round__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'a'
assert_type(round(WithCustomRound1()), str)
assert_type(round(WithCustomRound1(), None), str)
round(WithCustomRound1(), 1)
round(WithCustomRound1(), CustomIndex())

class WithCustomRound2:

    def __round__(self, digits: int) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'a'
assert_type(round(WithCustomRound2(), 1), str)
assert_type(round(WithCustomRound2(), CustomIndex()), str)
round(WithCustomRound2(), None)
round(WithCustomRound2())

class WithOverloadedRound:

    @overload
    def __round__(self, ndigits: None=...) -> str:
        if False:
            print('Hello World!')
        ...

    @overload
    def __round__(self, ndigits: int) -> bytes:
        if False:
            while True:
                i = 10
        ...

    def __round__(self, ndigits: int | None=None) -> str | bytes:
        if False:
            return 10
        return b'' if ndigits is None else ''
assert_type(round(WithOverloadedRound()), str)
assert_type(round(WithOverloadedRound(), None), str)
assert_type(round(WithOverloadedRound(), 1), bytes)