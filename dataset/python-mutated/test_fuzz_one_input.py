import io
import unittest
from operator import attrgetter
import pytest
from hypothesis import Phase, given, settings, strategies as st
from hypothesis.database import InMemoryExampleDatabase
from hypothesis.errors import InvalidArgument
from hypothesis.internal.conjecture.shrinker import sort_key
try:
    from random import randbytes
except ImportError:
    from secrets import token_bytes as randbytes

@pytest.mark.parametrize('buffer_type', [bytes, bytearray, memoryview, io.BytesIO], ids=attrgetter('__name__'))
def test_fuzz_one_input(buffer_type):
    if False:
        return 10
    db = InMemoryExampleDatabase()
    seen = []
    seeds = []

    @given(st.text())
    @settings(database=db, phases=[Phase.reuse, Phase.shrink])
    def test(s):
        if False:
            return 10
        seen.append(s)
        assert '\x00' not in s, repr(s)
    with pytest.raises(unittest.SkipTest):
        test()
    assert len(seen) == 0
    with pytest.raises(AssertionError):
        for _ in range(1000):
            buf = randbytes(1000)
            seeds.append(buf)
            test.hypothesis.fuzz_one_input(buffer_type(buf))
    assert len(seen) <= len(seeds)
    (saved_examples,) = db.data.values()
    assert len(saved_examples) == 1
    assert sort_key(seeds[-1]) >= sort_key(next(iter(saved_examples)))
    with pytest.raises(AssertionError):
        test()
    assert seen[-1] == '\x00'

def test_can_fuzz_with_database_eq_None():
    if False:
        while True:
            i = 10

    @given(st.none())
    @settings(database=None)
    def test(s):
        if False:
            i = 10
            return i + 15
        raise AssertionError
    with pytest.raises(AssertionError):
        test.hypothesis.fuzz_one_input(b'\x00\x00')

def test_fuzzing_unsatisfiable_test_always_returns_None():
    if False:
        return 10

    @given(st.none().filter(bool))
    @settings(database=None)
    def test(s):
        if False:
            print('Hello World!')
        raise AssertionError('Unreachable because there are no valid examples')
    for _ in range(100):
        buf = randbytes(3)
        ret = test.hypothesis.fuzz_one_input(buf)
        assert ret is None

def test_autopruning_of_returned_buffer():
    if False:
        while True:
            i = 10

    @given(st.binary(min_size=4, max_size=4))
    @settings(database=None)
    def test(s):
        if False:
            return 10
        pass
    assert test.hypothesis.fuzz_one_input(b'deadbeef') == b'dead'
STRAT = st.builds(object)

@given(x=STRAT)
def addx(x, y):
    if False:
        return 10
    pass

@given(STRAT)
def addy(x, y):
    if False:
        i = 10
        return i + 15
    pass

def test_can_access_strategy_for_wrapped_test():
    if False:
        return 10
    assert addx.hypothesis._given_kwargs == {'x': STRAT}
    assert addy.hypothesis._given_kwargs == {'y': STRAT}

@pytest.mark.parametrize('buffers,db_size', [([b'aa', b'bb', b'cc', b'dd'], 1), ([b'dd', b'cc', b'bb', b'aa'], 4), ([b'cc', b'dd', b'aa', b'bb'], 2), ([b'aa', b'bb', b'cc', b'XX'], 2)])
def test_fuzz_one_input_does_not_add_redundant_entries_to_database(buffers, db_size):
    if False:
        return 10
    db = InMemoryExampleDatabase()
    seen = []

    @given(st.binary(min_size=2, max_size=2))
    @settings(database=db)
    def test(s):
        if False:
            i = 10
            return i + 15
        seen.append(s)
        assert s != b'XX'
        raise AssertionError
    for buf in buffers:
        with pytest.raises(AssertionError):
            test.hypothesis.fuzz_one_input(buf)
    (saved_examples,) = db.data.values()
    assert seen == buffers
    assert len(saved_examples) == db_size

def test_fuzzing_invalid_test_raises_error():
    if False:
        i = 10
        return i + 15

    @given(st.integers(), st.integers())
    def invalid_test(s):
        if False:
            i = 10
            return i + 15
        pass
    with pytest.raises(InvalidArgument, match='Too many positional arguments'):
        invalid_test.hypothesis.fuzz_one_input