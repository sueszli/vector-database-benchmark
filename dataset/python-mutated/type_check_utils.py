from typing import Any, Literal, Type, Union
from typing_extensions import Literal as ExtLiteral
try:
    from types import UnionType
except ImportError:
    UnionType = Union
from typing_extensions import get_args, get_origin

def safe_is_subclass(cls: Any, possible_parent_cls: Type) -> bool:
    if False:
        while True:
            i = 10
    'Version of issubclass that returns False if cls is not a Type.'
    if not isinstance(cls, type):
        return False
    try:
        return issubclass(cls, possible_parent_cls)
    except TypeError:
        return False

def is_optional(annotation: Type) -> bool:
    if False:
        i = 10
        return i + 15
    'Returns true if the annotation signifies an Optional type.\n\n    In particular, this can be:\n    - Optional[T]\n    - Union[T, None]\n    - Union[None, T]\n    - T | None (in Python 3.10+)\n    - None | T (in Python 3.10+).\n\n    '
    if get_origin(annotation) == Union:
        return len(get_args(annotation)) == 2 and type(None) in get_args(annotation)
    if get_origin(annotation) == UnionType:
        return len(get_args(annotation)) == 2 and type(None) in get_args(annotation)
    return False

def is_literal(annotation: Type) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return get_origin(annotation) in (Literal, ExtLiteral)