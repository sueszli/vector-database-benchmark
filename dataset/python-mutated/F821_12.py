"""Test case: strings used within calls within type annotations."""
from __future__ import annotations
from typing import Callable
import bpy
from mypy_extensions import VarArg
from foo import Bar

class LightShow(bpy.types.Operator):
    label = 'Create Character'
    name = 'lightshow.letter_creation'
    filepath: bpy.props.StringProperty(subtype='FILE_PATH')

def f(x: Callable[[VarArg('os')], None]):
    if False:
        i = 10
        return i + 15
    pass
f(Callable[['Bar'], None])
f(Callable[['Baz'], None])