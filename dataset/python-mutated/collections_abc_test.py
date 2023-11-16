from __future__ import annotations
import pytest
from pyupgrade._data import Settings
from pyupgrade._main import _fix_plugins

def test_collections_abc_noop():
    if False:
        for i in range(10):
            print('nop')
    src = 'if isinstance(x, collections.defaultdict): pass\n'
    assert _fix_plugins(src, settings=Settings()) == src

@pytest.mark.parametrize(('src', 'expected'), (pytest.param('if isinstance(x, collections.Sized):\n    print(len(x))\n', 'if isinstance(x, collections.abc.Sized):\n    print(len(x))\n', id='Attribute reference for Sized class'),))
def test_collections_abc_rewrite(src, expected):
    if False:
        print('Hello World!')
    assert _fix_plugins(src, settings=Settings()) == expected