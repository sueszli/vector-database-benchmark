from __future__ import annotations

import pytest

from pyupgrade._data import Settings
from pyupgrade._main import _fix_plugins


@pytest.mark.parametrize(
    ('s',),
    (
        pytest.param(
            'from typing import Unpack\n'
            'foo(Unpack())',
            id='Not a subscript',
        ),
        pytest.param(
            'from typing import TypeVarTuple, Unpack\n'
            'Shape = TypeVarTuple("Shape")\n'
            'class Foo(Unpack[Shape]):\n'
            '    pass',
            id='Not inside a subscript',
        ),
    ),
)
def test_fix_pep646_noop(s):
    assert _fix_plugins(s, settings=Settings(min_version=(3, 11))) == s
    assert _fix_plugins(s, settings=Settings(min_version=(3, 10))) == s


@pytest.mark.parametrize(
    ('s', 'expected'),
    (
        (
            'from typing import Generic, TypeVarTuple, Unpack\n'
            "Shape = TypeVarTuple('Shape')\n"
            'class C(Generic[Unpack[Shape]]):\n'
            '    pass',

            'from typing import Generic, TypeVarTuple, Unpack\n'
            "Shape = TypeVarTuple('Shape')\n"
            'class C(Generic[*Shape]):\n'
            '    pass',
        ),
        (
            'from typing import Generic, TypeVarTuple, Unpack\n'
            "Shape = TypeVarTuple('Shape')\n"
            'class C(Generic[Unpack  [Shape]]):\n'
            '    pass',

            'from typing import Generic, TypeVarTuple, Unpack\n'
            "Shape = TypeVarTuple('Shape')\n"
            'class C(Generic[*Shape]):\n'
            '    pass',
        ),
    ),
)
def test_typing_unpack(s, expected):
    assert _fix_plugins(s, settings=Settings(min_version=(3, 11))) == expected
    assert _fix_plugins(s, settings=Settings(min_version=(3, 10))) == s
