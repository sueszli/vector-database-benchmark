import typing
from typing import Annotated, Any, Literal, Optional, Tuple, Union

def f(arg: int):
    if False:
        for i in range(10):
            print('nop')
    pass

def f(arg=None):
    if False:
        print('Hello World!')
    pass

def f(arg: Any=None):
    if False:
        return 10
    pass

def f(arg: object=None):
    if False:
        return 10
    pass

def f(arg: int=None):
    if False:
        return 10
    pass

def f(arg: str=None):
    if False:
        for i in range(10):
            print('nop')
    pass

def f(arg: typing.List[str]=None):
    if False:
        i = 10
        return i + 15
    pass

def f(arg: Tuple[str]=None):
    if False:
        print('Hello World!')
    pass

def f(arg: Optional[int]=None):
    if False:
        for i in range(10):
            print('nop')
    pass

def f(arg: typing.Optional[int]=None):
    if False:
        i = 10
        return i + 15
    pass

def f(arg: Union[None]=None):
    if False:
        while True:
            i = 10
    pass

def f(arg: Union[None, int]=None):
    if False:
        print('Hello World!')
    pass

def f(arg: Union[str, None]=None):
    if False:
        return 10
    pass

def f(arg: typing.Union[int, str, None]=None):
    if False:
        print('Hello World!')
    pass

def f(arg: Union[int, str, Any]=None):
    if False:
        return 10
    pass

def f(arg: Union=None):
    if False:
        return 10
    pass

def f(arg: Union[int]=None):
    if False:
        print('Hello World!')
    pass

def f(arg: Union[int, str]=None):
    if False:
        print('Hello World!')
    pass

def f(arg: typing.Union[int, str]=None):
    if False:
        for i in range(10):
            print('nop')
    pass

def f(arg: None | int=None):
    if False:
        return 10
    pass

def f(arg: int | None=None):
    if False:
        i = 10
        return i + 15
    pass

def f(arg: int | float | str | None=None):
    if False:
        i = 10
        return i + 15
    pass

def f(arg: int | float=None):
    if False:
        print('Hello World!')
    pass

def f(arg: int | float | str | bytes=None):
    if False:
        print('Hello World!')
    pass

def f(arg: None=None):
    if False:
        print('Hello World!')
    pass

def f(arg: Literal[None]=None):
    if False:
        for i in range(10):
            print('nop')
    pass

def f(arg: Literal[1, 2, None, 3]=None):
    if False:
        print('Hello World!')
    pass

def f(arg: Literal[1]=None):
    if False:
        return 10
    pass

def f(arg: Literal[1, 'foo']=None):
    if False:
        i = 10
        return i + 15
    pass

def f(arg: typing.Literal[1, 'foo', True]=None):
    if False:
        i = 10
        return i + 15
    pass

def f(arg: Annotated[Optional[int], ...]=None):
    if False:
        return 10
    pass

def f(arg: Annotated[Union[int, None], ...]=None):
    if False:
        for i in range(10):
            print('nop')
    pass

def f(arg: Annotated[Any, ...]=None):
    if False:
        return 10
    pass

def f(arg: Annotated[int, ...]=None):
    if False:
        i = 10
        return i + 15
    pass

def f(arg: Annotated[Annotated[int | str, ...], ...]=None):
    if False:
        i = 10
        return i + 15
    pass

def f(arg1: Optional[int]=None, arg2: Union[int, None]=None, arg3: Literal[1, 2, None, 3]=None):
    if False:
        print('Hello World!')
    pass

def f(arg1: int=None, arg2: Union[int, float]=None, arg3: Literal[1, 2, 3]=None):
    if False:
        while True:
            i = 10
    pass

def f(arg: Literal[1, 'foo', Literal[True, None]]=None):
    if False:
        return 10
    pass

def f(arg: Union[int, Union[float, Union[str, None]]]=None):
    if False:
        while True:
            i = 10
    pass

def f(arg: Union[int, Union[float, Optional[str]]]=None):
    if False:
        while True:
            i = 10
    pass

def f(arg: Union[int, Literal[True, None]]=None):
    if False:
        print('Hello World!')
    pass

def f(arg: Union[Annotated[int, ...], Annotated[Optional[float], ...]]=None):
    if False:
        while True:
            i = 10
    pass

def f(arg: Union[Annotated[int, ...], Union[str, bytes]]=None):
    if False:
        print('Hello World!')
    pass

def f(arg: 'int'=None):
    if False:
        i = 10
        return i + 15
    pass

def f(arg: 'str'=None):
    if False:
        for i in range(10):
            print('nop')
    pass

def f(arg: 'str'=None):
    if False:
        while True:
            i = 10
    pass

def f(arg: 'Optional[int]'=None):
    if False:
        while True:
            i = 10
    pass

def f(arg: Union['int', 'str']=None):
    if False:
        for i in range(10):
            print('nop')
    pass

def f(arg: Union['int', 'None']=None):
    if False:
        while True:
            i = 10
    pass

def f(arg: Union['None', 'int']=None):
    if False:
        print('Hello World!')
    pass

def f(arg: Union['<>', 'int']=None):
    if False:
        print('Hello World!')
    pass
Text = str | bytes

def f(arg: Text=None):
    if False:
        return 10
    pass

def f(arg: 'Text'=None):
    if False:
        while True:
            i = 10
    pass
from custom_typing import MaybeInt

def f(arg: MaybeInt=None):
    if False:
        i = 10
        return i + 15
    pass