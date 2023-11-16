from __future__ import annotations
import pytest
from pyupgrade._data import Settings
from pyupgrade._main import _fix_plugins

@pytest.mark.parametrize('s', ('x = (', 'class C(B): pass'))
def test_fix_classes_noop(s):
    if False:
        i = 10
        return i + 15
    assert _fix_plugins(s, settings=Settings()) == s

@pytest.mark.parametrize(('s', 'expected'), (('class C(object): pass', 'class C: pass'), ('class C(\n    object,\n): pass', 'class C: pass'), ('class C(B, object): pass', 'class C(B): pass'), ('class C(B, (object)): pass', 'class C(B): pass'), ('class C(B, ( object )): pass', 'class C(B): pass'), ('class C((object)): pass', 'class C: pass'), ('class C(\n    B,\n    object,\n): pass\n', 'class C(\n    B,\n): pass\n'), ('class C(\n    B,\n    object\n): pass\n', 'class C(\n    B\n): pass\n'), ('class C(object, B): pass', 'class C(B): pass'), ('class C((object), B): pass', 'class C(B): pass'), ('class C(( object ), B): pass', 'class C(B): pass'), ('class C(\n    object,\n    B,\n): pass', 'class C(\n    B,\n): pass'), ('class C(\n    object,  # comment!\n    B,\n): pass', 'class C(\n    B,\n): pass'), ('class C(object, metaclass=ABCMeta): pass', 'class C(metaclass=ABCMeta): pass')))
def test_fix_classes(s, expected):
    if False:
        i = 10
        return i + 15
    ret = _fix_plugins(s, settings=Settings())
    assert ret == expected