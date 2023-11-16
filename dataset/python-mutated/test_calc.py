from math import sin, cos
from datetime import timedelta
import pytest
from funcy.calc import *

def test_memoize():
    if False:
        while True:
            i = 10

    @memoize
    def inc(x):
        if False:
            for i in range(10):
                print('nop')
        calls.append(x)
        return x + 1
    calls = []
    assert inc(0) == 1
    assert inc(1) == 2
    assert inc(0) == 1
    assert calls == [0, 1]
    assert inc(x=0) == 1
    assert inc(x=1) == 2
    assert inc(x=0) == 1
    assert calls == [0, 1, 0, 1]

def test_memoize_args_kwargs():
    if False:
        print('Hello World!')

    @memoize
    def mul(x, by=1):
        if False:
            print('Hello World!')
        calls.append((x, by))
        return x * by
    calls = []
    assert mul(0) == 0
    assert mul(1) == 1
    assert mul(0) == 0
    assert calls == [(0, 1), (1, 1)]
    assert mul(0, 1) == 0
    assert mul(1, 1) == 1
    assert mul(0, 1) == 0
    assert calls == [(0, 1), (1, 1), (0, 1), (1, 1)]

def test_memoize_memory():
    if False:
        return 10

    @memoize
    def inc(x):
        if False:
            print('Hello World!')
        calls.append(x)
        return x + 1
    calls = []
    inc(0)
    inc.memory.clear()
    inc(0)
    assert calls == [0, 0]

def test_memoize_key_func():
    if False:
        print('Hello World!')

    @memoize(key_func=len)
    def inc(s):
        if False:
            for i in range(10):
                print('nop')
        calls.append(s)
        return s * 2
    calls = []
    assert inc('a') == 'aa'
    assert inc('b') == 'aa'
    inc('ab')
    assert calls == ['a', 'ab']

def test_make_lookuper():
    if False:
        while True:
            i = 10

    @make_lookuper
    def letter_index():
        if False:
            return 10
        return ((c, i) for (i, c) in enumerate('abcdefghij'))
    assert letter_index('c') == 2
    with pytest.raises(LookupError):
        letter_index('_')

def test_make_lookuper_nested():
    if False:
        print('Hello World!')
    tables_built = [0]

    @make_lookuper
    def function_table(f):
        if False:
            i = 10
            return i + 15
        tables_built[0] += 1
        return ((x, f(x)) for x in range(10))
    assert function_table(sin)(5) == sin(5)
    assert function_table(cos)(3) == cos(3)
    assert function_table(sin)(3) == sin(3)
    assert tables_built[0] == 2
    with pytest.raises(LookupError):
        function_table(cos)(-1)

def test_silent_lookuper():
    if False:
        while True:
            i = 10

    @silent_lookuper
    def letter_index():
        if False:
            while True:
                i = 10
        return ((c, i) for (i, c) in enumerate('abcdefghij'))
    assert letter_index('c') == 2
    assert letter_index('_') is None

def test_silnent_lookuper_nested():
    if False:
        while True:
            i = 10

    @silent_lookuper
    def function_table(f):
        if False:
            while True:
                i = 10
        return ((x, f(x)) for x in range(10))
    assert function_table(sin)(5) == sin(5)
    assert function_table(cos)(-1) is None

@pytest.mark.parametrize('typ', [pytest.param(int, id='int'), pytest.param(lambda s: timedelta(seconds=s), id='timedelta')])
def test_cache(typ):
    if False:
        return 10
    calls = []

    @cache(timeout=typ(60))
    def inc(x):
        if False:
            print('Hello World!')
        calls.append(x)
        return x + 1
    assert inc(0) == 1
    assert inc(1) == 2
    assert inc(0) == 1
    assert calls == [0, 1]

def test_cache_mixed_args():
    if False:
        for i in range(10):
            print('nop')

    @cache(timeout=60)
    def add(x, y):
        if False:
            return 10
        return x + y
    assert add(1, y=2) == 3

def test_cache_timedout():
    if False:
        while True:
            i = 10
    calls = []

    @cache(timeout=0)
    def inc(x):
        if False:
            for i in range(10):
                print('nop')
        calls.append(x)
        return x + 1
    assert inc(0) == 1
    assert inc(1) == 2
    assert inc(0) == 1
    assert calls == [0, 1, 0]
    assert len(inc.memory) == 1

def test_cache_invalidate():
    if False:
        while True:
            i = 10
    calls = []

    @cache(timeout=60)
    def inc(x):
        if False:
            return 10
        calls.append(x)
        return x + 1
    assert inc(0) == 1
    assert inc(1) == 2
    assert inc(0) == 1
    assert calls == [0, 1]
    inc.invalidate_all()
    assert inc(0) == 1
    assert inc(1) == 2
    assert inc(0) == 1
    assert calls == [0, 1, 0, 1]
    inc.invalidate(1)
    assert inc(0) == 1
    assert inc(1) == 2
    assert inc(0) == 1
    assert calls == [0, 1, 0, 1, 1]
    inc.invalidate(0)
    inc.invalidate(0)