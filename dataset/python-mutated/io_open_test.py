from __future__ import annotations
import pytest
from pyupgrade._data import Settings
from pyupgrade._main import _fix_plugins

def test_fix_io_open_noop():
    if False:
        while True:
            i = 10
    src = 'from io import open\nwith open("f.txt") as f:\n    print(f.read())\n'
    expected = 'with open("f.txt") as f:\n    print(f.read())\n'
    ret = _fix_plugins(src, settings=Settings())
    assert ret == expected

@pytest.mark.parametrize(('s', 'expected'), (('import io\n\nwith io.open("f.txt", mode="r", buffering=-1, **kwargs) as f:\n   print(f.read())\n', 'import io\n\nwith open("f.txt", mode="r", buffering=-1, **kwargs) as f:\n   print(f.read())\n'),))
def test_fix_io_open(s, expected):
    if False:
        for i in range(10):
            print('nop')
    ret = _fix_plugins(s, settings=Settings())
    assert ret == expected