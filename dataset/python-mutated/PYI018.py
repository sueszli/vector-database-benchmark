import typing
from typing import TypeVar
_T = typing.TypeVar('_T')
_P = TypeVar('_P')
_UsedTypeVar = TypeVar('_UsedTypeVar')

def func(arg: _UsedTypeVar) -> _UsedTypeVar:
    if False:
        for i in range(10):
            print('nop')
    ...
(_A, _B) = (TypeVar('_A'), TypeVar('_B'))
_C = _D = TypeVar('_C')