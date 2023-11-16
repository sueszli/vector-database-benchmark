from inspect import Signature
from numbers import Integral
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

def f0(x: int, y: Integral) -> None:
    if False:
        return 10
    pass

def f1(x: list[int]) -> List[int]:
    if False:
        return 10
    pass
T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
T_contra = TypeVar('T_contra', contravariant=True)

def f2(x: List[T], y: List[T_co], z: T) -> List[T_contra]:
    if False:
        for i in range(10):
            print('nop')
    pass

def f3(x: Union[str, Integral]) -> None:
    if False:
        for i in range(10):
            print('nop')
    pass
MyStr = str

def f4(x: 'MyStr', y: MyStr) -> None:
    if False:
        while True:
            i = 10
    pass

def f5(x: int, *, y: str, z: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    pass

def f6(x: int, *args, y: str, z: str) -> None:
    if False:
        while True:
            i = 10
    pass

def f7(x: int=None, y: dict={}) -> None:
    if False:
        i = 10
        return i + 15
    pass

def f8(x: Callable[[int, str], int]) -> None:
    if False:
        i = 10
        return i + 15
    pass

def f9(x: Callable) -> None:
    if False:
        while True:
            i = 10
    pass

def f10(x: Tuple[int, str], y: Tuple[int, ...]) -> None:
    if False:
        for i in range(10):
            print('nop')
    pass

class CustomAnnotation:

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'CustomAnnotation'

def f11(x: CustomAnnotation(), y: 123) -> None:
    if False:
        return 10
    pass

def f12() -> Tuple[int, str, int]:
    if False:
        return 10
    pass

def f13() -> Optional[str]:
    if False:
        i = 10
        return i + 15
    pass

def f14() -> Any:
    if False:
        print('Hello World!')
    pass

def f15(x: 'Unknown', y: 'int') -> Any:
    if False:
        for i in range(10):
            print('nop')
    pass

def f16(arg1, arg2, *, arg3=None, arg4=None):
    if False:
        while True:
            i = 10
    pass

def f17(*, arg3, arg4):
    if False:
        for i in range(10):
            print('nop')
    pass

def f18(self, arg1: Union[int, Tuple]=10) -> List[Dict]:
    if False:
        print('Hello World!')
    pass

def f19(*args: int, **kwargs: str):
    if False:
        print('Hello World!')
    pass

def f20() -> Optional[Union[int, str]]:
    if False:
        i = 10
        return i + 15
    pass

def f21(arg1='whatever', arg2=Signature.empty):
    if False:
        for i in range(10):
            print('nop')
    pass

def f22(*, a, b):
    if False:
        i = 10
        return i + 15
    pass

def f23(a, b, /, c, d):
    if False:
        return 10
    pass

def f24(a, /, *, b):
    if False:
        print('Hello World!')
    pass

def f25(a, b, /):
    if False:
        while True:
            i = 10
    pass

class Node:

    def __init__(self, parent: Optional['Node']) -> None:
        if False:
            return 10
        pass

    def children(self) -> List['Node']:
        if False:
            print('Hello World!')
        pass