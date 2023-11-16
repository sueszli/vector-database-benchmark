import pytest
from hypothesis import HealthCheck, assume, example, given, settings, strategies as st
from hypothesis.internal.intervalsets import IntervalSet

def build_intervals(ls):
    if False:
        while True:
            i = 10
    ls.sort()
    result = []
    for (u, l) in ls:
        v = u + l
        if result:
            (a, b) = result[-1]
            if u <= b + 1:
                result[-1] = (a, v)
                continue
        result.append((u, v))
    return result

def IntervalLists(min_size=0):
    if False:
        for i in range(10):
            print('nop')
    return st.lists(st.tuples(st.integers(0, 200), st.integers(0, 20)), min_size=min_size).map(build_intervals)
Intervals = st.builds(IntervalSet, IntervalLists())

@given(Intervals)
def test_intervals_are_equivalent_to_their_lists(intervals):
    if False:
        print('Hello World!')
    ls = list(intervals)
    assert len(ls) == len(intervals)
    for i in range(len(ls)):
        assert ls[i] == intervals[i]
    for i in range(1, len(ls) - 1):
        assert ls[-i] == intervals[-i]

@given(Intervals)
def test_intervals_match_indexes(intervals):
    if False:
        print('Hello World!')
    ls = list(intervals)
    for v in ls:
        assert ls.index(v) == intervals.index(v)

@example(intervals=IntervalSet(((1, 1),)), v=0)
@example(intervals=IntervalSet(()), v=0)
@given(Intervals, st.integers(0, 1114111))
def test_error_for_index_of_not_present_value(intervals, v):
    if False:
        while True:
            i = 10
    assume(v not in intervals)
    with pytest.raises(ValueError):
        intervals.index(v)

def test_validates_index():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(IndexError):
        IntervalSet([])[1]
    with pytest.raises(IndexError):
        IntervalSet([[1, 10]])[11]
    with pytest.raises(IndexError):
        IntervalSet([[1, 10]])[-11]

def test_index_above_is_index_if_present():
    if False:
        while True:
            i = 10
    assert IntervalSet([[1, 10]]).index_above(1) == 0
    assert IntervalSet([[1, 10]]).index_above(2) == 1

def test_index_above_is_length_if_higher():
    if False:
        return 10
    assert IntervalSet([[1, 10]]).index_above(100) == 10

def intervals_to_set(ints):
    if False:
        i = 10
        return i + 15
    return set(IntervalSet(ints))

@settings(suppress_health_check=[HealthCheck.filter_too_much])
@example(x=[(0, 1), (3, 3)], y=[(1, 3)])
@example(x=[(0, 1)], y=[(0, 0), (1, 1)])
@example(x=[(0, 1)], y=[(1, 1)])
@given(IntervalLists(min_size=1), IntervalLists(min_size=1))
def test_subtraction_of_intervals(x, y):
    if False:
        i = 10
        return i + 15
    xs = intervals_to_set(x)
    ys = intervals_to_set(y)
    assume(not xs.isdisjoint(ys))
    z = IntervalSet(x).difference(IntervalSet(y)).intervals
    assert z == tuple(sorted(z))
    for (a, b) in z:
        assert a <= b
    assert intervals_to_set(z) == intervals_to_set(x) - intervals_to_set(y)

@given(Intervals, Intervals)
def test_interval_intersection(x, y):
    if False:
        print('Hello World!')
    print(f'set(x)={set(x)!r} set(y)={set(y)!r} set(x)-(set(y)-set(x))={set(x) - (set(y) - set(x))!r}')
    assert set(x & y) == set(x) & set(y)
    assert set(x.intersection(y)) == set(x).intersection(y)