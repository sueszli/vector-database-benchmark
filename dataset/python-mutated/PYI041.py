from typing import Union
from typing_extensions import TypeAlias
TA0: TypeAlias = int
TA1: TypeAlias = int | float | bool
TA2: TypeAlias = Union[int, float, bool]

def good1(arg: int) -> int | bool:
    if False:
        print('Hello World!')
    ...

def good2(arg: int, arg2: int | bool) -> None:
    if False:
        print('Hello World!')
    ...

def f0(arg1: float | int) -> None:
    if False:
        return 10
    ...

def f1(arg1: float, *, arg2: float | list[str] | type[bool] | complex) -> None:
    if False:
        i = 10
        return i + 15
    ...

def f2(arg1: int, /, arg2: int | int | float) -> None:
    if False:
        print('Hello World!')
    ...

def f3(arg1: int, *args: Union[int | int | float]) -> None:
    if False:
        print('Hello World!')
    ...

async def f4(**kwargs: int | int | float) -> None:
    ...

class Foo:

    def good(self, arg: int) -> None:
        if False:
            while True:
                i = 10
        ...

    def bad(self, arg: int | float | complex) -> None:
        if False:
            i = 10
            return i + 15
        ...