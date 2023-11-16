from __future__ import annotations
import pytest
from pyupgrade._data import Settings
from pyupgrade._main import _fix_plugins

@pytest.mark.parametrize('s', ('x = 5', 'dict()', '(', 'dict ((a, b) for a, b in y)', 'dict(((a, b) for a, b in y), x=1)', 'dict(((a, b) for a, b in y), **kwargs)'))
def test_fix_dict_noop(s):
    if False:
        i = 10
        return i + 15
    assert _fix_plugins(s, settings=Settings()) == s

@pytest.mark.parametrize(('s', 'expected'), (('dict((a, b) for a, b in y)', '{a: b for a, b in y}'), ('dict((a, b,) for a, b in y)', '{a: b for a, b in y}'), ('dict((a, b, ) for a, b in y)', '{a: b for a, b in y}'), ('dict([a, b] for a, b in y)', '{a: b for a, b in y}'), ('dict(((a, b)) for a, b in y)', '{a: b for a, b in y}'), ('dict([(a, b) for a, b in y])', '{a: b for a, b in y}'), ('dict([(a, b), c] for a, b, c in y)', '{(a, b): c for a, b, c in y}'), ('dict(((a), b) for a, b in y)', '{(a): b for a, b in y}'), ('dict((k, dict((k2, v2) for k2, v2 in y2)) for k, y2 in y)', '{k: {k2: v2 for k2, v2 in y2} for k, y2 in y}'), ('dict((a, b)for a, b in y)', '{a: b for a, b in y}'), ('dict(\n    (\n        a,\n        b,\n    )\n    for a, b in y\n)', '{\n        a:\n        b\n    for a, b in y\n}'), ('x(\n    dict(\n        (a, b) for a, b in y\n    )\n)', 'x(\n    {\n        a: b for a, b in y\n    }\n)')))
def test_dictcomps(s, expected):
    if False:
        i = 10
        return i + 15
    ret = _fix_plugins(s, settings=Settings())
    assert ret == expected