from typing import ParamSpec, TypeVar, TypeVarTuple
T = TypeVar('T')
TTuple = TypeVarTuple('TTuple')
P = ParamSpec('P')
_T = TypeVar('_T')
_TTuple = TypeVarTuple('_TTuple')
_P = ParamSpec('_P')

def f():
    if False:
        return 10
    T = TypeVar('T')