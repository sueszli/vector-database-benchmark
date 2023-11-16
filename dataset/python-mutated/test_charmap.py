import os
import sys
import tempfile
import time
import unicodedata
from typing import get_args
from hypothesis import given, strategies as st
from hypothesis.internal import charmap as cm
from hypothesis.internal.intervalsets import IntervalSet
from hypothesis.strategies._internal.core import CategoryName

def test_charmap_contains_all_unicode():
    if False:
        return 10
    n = 0
    for vs in cm.charmap().values():
        for (u, v) in vs:
            n += v - u + 1
    assert n == sys.maxunicode + 1

def test_charmap_has_right_categories():
    if False:
        return 10
    for (cat, intervals) in cm.charmap().items():
        for (u, v) in intervals:
            for i in range(u, v + 1):
                real = unicodedata.category(chr(i))
                assert real == cat, f'{i} is {real} but reported in {cat}'

def assert_valid_range_list(ls):
    if False:
        return 10
    for (u, v) in ls:
        assert u <= v
    for i in range(len(ls) - 1):
        assert ls[i] <= ls[i + 1]
        assert ls[i][-1] < ls[i + 1][0]

@given(st.sets(st.sampled_from(cm.categories())))
def test_query_matches_categories(cats):
    if False:
        while True:
            i = 10
    values = cm.query(categories=cats).intervals
    assert_valid_range_list(values)
    for (u, v) in values:
        for i in (u, v, (u + v) // 2):
            assert unicodedata.category(chr(i)) in cats

@given(st.sets(st.sampled_from(cm.categories())) | st.none(), st.integers(0, sys.maxunicode), st.integers(0, sys.maxunicode))
def test_query_matches_categories_codepoints(cats, m1, m2):
    if False:
        while True:
            i = 10
    (m1, m2) = sorted((m1, m2))
    values = cm.query(categories=cats, min_codepoint=m1, max_codepoint=m2).intervals
    assert_valid_range_list(values)
    for (u, v) in values:
        assert m1 <= u
        assert v <= m2

def test_reload_charmap():
    if False:
        return 10
    x = cm.charmap()
    assert x is cm.charmap()
    cm._charmap = None
    y = cm.charmap()
    assert x is not y
    assert x == y

def test_recreate_charmap():
    if False:
        i = 10
        return i + 15
    x = cm.charmap()
    assert x is cm.charmap()
    cm._charmap = None
    cm.charmap_file().unlink()
    y = cm.charmap()
    assert x is not y
    assert x == y

def test_uses_cached_charmap():
    if False:
        for i in range(10):
            print('nop')
    cm.charmap()
    mtime = int(time.time() - 1000)
    os.utime(cm.charmap_file(), (mtime, mtime))
    statinfo = cm.charmap_file().stat()
    assert statinfo.st_mtime == mtime
    cm._charmap = None
    cm.charmap()
    statinfo = cm.charmap_file().stat()
    assert statinfo.st_mtime == mtime

def _union_intervals(x, y):
    if False:
        print('Hello World!')
    return IntervalSet(x).union(IntervalSet(y)).intervals

def test_union_empty():
    if False:
        i = 10
        return i + 15
    assert _union_intervals([], []) == ()
    assert _union_intervals([], [[1, 2]]) == ((1, 2),)
    assert _union_intervals([[1, 2]], []) == ((1, 2),)

def test_union_handles_totally_overlapped_gap():
    if False:
        i = 10
        return i + 15
    assert _union_intervals([[2, 3]], [[1, 2], [4, 5]]) == ((1, 5),)

def test_union_handles_partially_overlapped_gap():
    if False:
        while True:
            i = 10
    assert _union_intervals([[3, 3]], [[1, 2], [5, 5]]) == ((1, 3), (5, 5))

def test_successive_union():
    if False:
        while True:
            i = 10
    x = []
    for v in cm.charmap().values():
        x = _union_intervals(x, v)
    assert x == ((0, sys.maxunicode),)

def test_can_handle_race_between_exist_and_create(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    x = cm.charmap()
    cm._charmap = None
    monkeypatch.setattr(os.path, 'exists', lambda p: False)
    y = cm.charmap()
    assert x is not y
    assert x == y

def test_exception_in_write_does_not_lead_to_broken_charmap(monkeypatch):
    if False:
        while True:
            i = 10

    def broken(*args, **kwargs):
        if False:
            while True:
                i = 10
        raise ValueError
    cm._charmap = None
    monkeypatch.setattr(os.path, 'exists', lambda p: False)
    monkeypatch.setattr(os, 'rename', broken)
    cm.charmap()
    cm.charmap()

def test_regenerate_broken_charmap_file():
    if False:
        i = 10
        return i + 15
    cm.charmap()
    cm.charmap_file().write_bytes(b'')
    cm._charmap = None
    cm.charmap()

def test_exclude_characters_are_included_in_key():
    if False:
        while True:
            i = 10
    assert cm.query().intervals != cm.query(exclude_characters='0').intervals

def test_error_writing_charmap_file_is_suppressed(monkeypatch):
    if False:
        return 10

    def broken_mkstemp(dir):
        if False:
            i = 10
            return i + 15
        raise RuntimeError
    monkeypatch.setattr(tempfile, 'mkstemp', broken_mkstemp)
    try:
        saved = cm._charmap
        cm._charmap = None
        cm.charmap_file().unlink()
        cm.charmap()
    finally:
        cm._charmap = saved

def test_categoryname_literal_is_correct():
    if False:
        i = 10
        return i + 15
    minor_categories = set(cm.categories())
    major_categories = {c[0] for c in minor_categories}
    assert set(get_args(CategoryName)) == minor_categories | major_categories