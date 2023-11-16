"""Library for testing backwards compatibility.

Especially testing argument type information that has been changing after RF 4.
Examples are only using features compatible with all tested versions.
"""
from enum import Enum
from typing import Union
try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict
ROBOT_LIBRARY_VERSION = '1.0'
__all__ = ['simple', 'arguments', 'types', 'special_types', 'union']

class Color(Enum):
    """RGB colors."""
    RED = 'R'
    GREEN = 'G'
    BLUE = 'B'

class Size(TypedDict):
    """Some size."""
    width: int
    height: int

def simple():
    if False:
        while True:
            i = 10
    'Some doc.\n\n    Tags: example\n    '
    pass

def arguments(a, b=2, *c, d=4, e, **f):
    if False:
        while True:
            i = 10
    pass

def types(a: int, b: bool=True):
    if False:
        while True:
            i = 10
    pass

def special_types(a: Color, b: Size):
    if False:
        i = 10
        return i + 15
    pass

def union(a: Union[int, float]):
    if False:
        while True:
            i = 10
    pass