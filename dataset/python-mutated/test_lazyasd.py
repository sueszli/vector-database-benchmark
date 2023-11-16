"""Tests lazy and self destruictive objects."""
from xonsh.lazyasd import LazyObject

def test_lazyobject_getitem():
    if False:
        return 10
    lo = LazyObject(lambda : {'x': 1}, {}, 'lo')
    assert 1 == lo['x']