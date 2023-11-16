from hscommon.path import pathify
from pathlib import Path

def test_pathify():
    if False:
        return 10

    @pathify
    def foo(a: Path, b, c: Path):
        if False:
            while True:
                i = 10
        return (a, b, c)
    (a, b, c) = foo('foo', 0, c=Path('bar'))
    assert isinstance(a, Path)
    assert a == Path('foo')
    assert b == 0
    assert isinstance(c, Path)
    assert c == Path('bar')

def test_pathify_preserve_none():
    if False:
        i = 10
        return i + 15

    @pathify
    def foo(a: Path):
        if False:
            print('Hello World!')
        return a
    a = foo(None)
    assert a is None