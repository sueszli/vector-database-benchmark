import math
import os
import sys
from math import inf
import numpy as np

def f12(x, y: str=os.pathsep) -> None:
    if False:
        return 10
    ...

def f11(*, x: str='x') -> None:
    if False:
        return 10
    ...

def f13(x: list[str]=['foo', 'bar', 'baz']) -> None:
    if False:
        return 10
    ...

def f14(x: tuple[str, ...]=('foo', 'bar', 'baz')) -> None:
    if False:
        while True:
            i = 10
    ...

def f15(x: set[str]={'foo', 'bar', 'baz'}) -> None:
    if False:
        return 10
    ...

def f151(x: dict[int, int]={1: 2}) -> None:
    if False:
        return 10
    ...

def f152(x: dict[int, int]={1: 2, **{3: 4}}) -> None:
    if False:
        while True:
            i = 10
    ...

def f153(x: list[int]=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) -> None:
    if False:
        return 10
    ...

def f154(x: tuple[str, tuple[str, ...]]=('foo', ('bar', 'baz'))) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f141(x: list[int]=[*range(10)]) -> None:
    if False:
        while True:
            i = 10
    ...

def f142(x: list[int]=list(range(10))) -> None:
    if False:
        print('Hello World!')
    ...

def f16(x: frozenset[bytes]=frozenset({b'foo', b'bar', b'baz'})) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f17(x: str='foo' + 'bar') -> None:
    if False:
        while True:
            i = 10
    ...

def f18(x: str=b'foo' + b'bar') -> None:
    if False:
        while True:
            i = 10
    ...

def f19(x: object='foo' + 4) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f20(x: int=5 + 5) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f21(x: complex=3j - 3j) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f22(x: complex=-42.5j + 4.3j) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f23(x: bool=True) -> None:
    if False:
        return 10
    ...

def f24(x: float=3.14) -> None:
    if False:
        return 10
    ...

def f25(x: float=-3.14) -> None:
    if False:
        return 10
    ...

def f26(x: complex=-3.14j) -> None:
    if False:
        return 10
    ...

def f27(x: complex=-3 - 3.14j) -> None:
    if False:
        return 10
    ...

def f28(x: float=math.tau) -> None:
    if False:
        print('Hello World!')
    ...

def f29(x: float=math.inf) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f30(x: float=-math.inf) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f31(x: float=inf) -> None:
    if False:
        while True:
            i = 10
    ...

def f32(x: float=np.inf) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f33(x: float=math.nan) -> None:
    if False:
        i = 10
        return i + 15
    ...

def f34(x: float=-math.nan) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f35(x: complex=math.inf + 1j) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def f36(*, x: str=sys.version) -> None:
    if False:
        i = 10
        return i + 15
    ...

def f37(*, x: str='' + '') -> None:
    if False:
        print('Hello World!')
    ...