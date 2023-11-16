import sys
import typing
from typing import Annotated, Literal, TypeAlias, TypeVar
import typing_extensions

def f(x: 'int'):
    if False:
        for i in range(10):
            print('nop')
    ...

def g(x: list['int']):
    if False:
        while True:
            i = 10
    ...
_T = TypeVar('_T', bound='int')

def h(w: Literal['a', 'b'], x: typing.Literal['c'], y: typing_extensions.Literal['d'], z: _T) -> _T:
    if False:
        i = 10
        return i + 15
    ...

def j() -> 'int':
    if False:
        i = 10
        return i + 15
    ...
Alias: TypeAlias = list['int']

class Child(list['int']):
    """Documented and guaranteed useful."""
if sys.platform == 'linux':
    f: 'int'
elif sys.platform == 'win32':
    f: 'str'
else:
    f: 'bytes'
k = ''
el = ''