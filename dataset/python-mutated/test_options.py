from __future__ import annotations
import pytest
pytest
from bokeh.core.properties import Int, Nullable, String
from bokeh.util.options import Options

class DummyOpts(Options):
    foo = String(default='thing')
    bar = Nullable(Int())

def test_empty() -> None:
    if False:
        while True:
            i = 10
    empty = dict()
    o = DummyOpts(empty)
    assert o.foo == 'thing'
    assert o.bar is None
    assert empty == {}

def test_exact() -> None:
    if False:
        i = 10
        return i + 15
    exact = dict(foo='stuff', bar=10)
    o = DummyOpts(exact)
    assert o.foo == 'stuff'
    assert o.bar == 10
    assert exact == {}

def test_extra() -> None:
    if False:
        print('Hello World!')
    extra = dict(foo='stuff', bar=10, baz=22.2)
    o = DummyOpts(extra)
    assert o.foo == 'stuff'
    assert o.bar == 10
    assert extra == {'baz': 22.2}

def test_mixed() -> None:
    if False:
        print('Hello World!')
    mixed = dict(foo='stuff', baz=22.2)
    o = DummyOpts(mixed)
    assert o.foo == 'stuff'
    assert o.bar is None
    assert mixed == {'baz': 22.2}