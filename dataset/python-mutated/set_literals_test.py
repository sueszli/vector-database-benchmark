from __future__ import annotations
import pytest
from pyupgrade._data import Settings
from pyupgrade._main import _fix_plugins

@pytest.mark.parametrize('s', ('set()', 'set ((1, 2))'))
def test_fix_sets_noop(s):
    if False:
        for i in range(10):
            print('nop')
    assert _fix_plugins(s, settings=Settings()) == s

@pytest.mark.parametrize(('s', 'expected'), (('set(())', 'set()'), ('set([])', 'set()'), pytest.param('set (())', 'set ()', id='empty, weird ws'), ('set(( ))', 'set()'), ('set((1, 2))', '{1, 2}'), ('set([1, 2])', '{1, 2}'), ('set(x for x in y)', '{x for x in y}'), ('set([x for x in y])', '{x for x in y}'), ('set((x for x in y))', '{x for x in y}'), ('set(((1, 2)))', '{1, 2}'), ('set((a, b) for a, b in y)', '{(a, b) for a, b in y}'), ('set(((1, 2), (3, 4)))', '{(1, 2), (3, 4)}'), ('set([(1, 2), (3, 4)])', '{(1, 2), (3, 4)}'), ('set(\n    [(1, 2)]\n)', '{\n    (1, 2)\n}'), ('set([((1, 2)), (3, 4)])', '{((1, 2)), (3, 4)}'), ('set((((1, 2),),))', '{((1, 2),)}'), ('set(\n(1, 2))', '{\n1, 2}'), ('set((\n1,\n2,\n))\n', '{\n1,\n2,\n}\n'), ('set((frozenset(set((1, 2))), frozenset(set((3, 4)))))', '{frozenset({1, 2}), frozenset({3, 4})}'), ('set((1,))', '{1}'), ('set((1, ))', '{1}'), ('set([1, 2, 3,],)', '{1, 2, 3}'), ('set((x for x in y),)', '{x for x in y}'), ('set(\n    (x for x in y),\n)', '{\n    x for x in y\n}'), ('set(\n    [\n        99, 100,\n    ],\n)\n', '{\n        99, 100,\n}\n'), pytest.param('set((\n))', 'set()', id='empty literal with newline'), pytest.param('set((f"{x}(",))', '{f"{x}("}', id='3.12 fstring containing open brace'), pytest.param('set((f"{x})",))', '{f"{x})"}', id='3.12 fstring containing close brace')))
def test_sets(s, expected):
    if False:
        return 10
    ret = _fix_plugins(s, settings=Settings())
    assert ret == expected