"""This file demonstrates testing a binary search.

It's a useful example because the result of the binary search is so clearly
determined by the invariants it must satisfy, so we can simply test for those
invariants.

It also demonstrates the useful testing technique of testing how the answer
should change (or not) in response to movements in the underlying data.
"""
from hypothesis import given, strategies as st

def binary_search(ls, v):
    if False:
        i = 10
        return i + 15
    'Take a list ls and a value v such that ls is sorted and v is comparable\n    with the elements of ls.\n\n    Return an index i such that 0 <= i <= len(v) with the properties:\n\n    1. ls.insert(i, v) is sorted\n    2. ls.insert(j, v) is not sorted for j < i\n    '
    if not ls:
        return 0
    if v <= ls[0]:
        return 0
    lo = 0
    hi = len(ls)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if v > ls[mid]:
            lo = mid
        else:
            hi = mid
    assert lo + 1 == hi
    return hi

def is_sorted(ls):
    if False:
        i = 10
        return i + 15
    'Is this list sorted?'
    return all((x <= y for (x, y) in zip(ls, ls[1:])))
Values = st.integers()
SortedLists = st.lists(Values).map(sorted)

@given(ls=SortedLists, v=Values)
def test_insert_is_sorted(ls, v):
    if False:
        while True:
            i = 10
    'We test the first invariant: binary_search should return an index such\n    that inserting the value provided at that index would result in a sorted\n    set.'
    ls.insert(binary_search(ls, v), v)
    assert is_sorted(ls)

@given(ls=SortedLists, v=Values)
def test_is_minimal(ls, v):
    if False:
        while True:
            i = 10
    'We test the second invariant: binary_search should return an index such\n    that no smaller index is a valid insertion point for v.'
    for i in range(binary_search(ls, v)):
        ls2 = list(ls)
        ls2.insert(i, v)
        assert not is_sorted(ls2)

@given(ls=SortedLists, v=Values)
def test_inserts_into_same_place_twice(ls, v):
    if False:
        print('Hello World!')
    'In this we test a *consequence* of the second invariant: When we insert\n    a value into a list twice, the insertion point should be the same both\n    times. This is because we know that v is > the previous element and == the\n    next element.\n\n    In theory if the former passes, this should always pass. In practice,\n    failures are detected by this test with much higher probability because it\n    deliberately puts the data into a shape that is likely to trigger a\n    failure.\n\n    This is an instance of a good general category of test: Testing how the\n    function moves in responses to changes in the underlying data.\n    '
    i = binary_search(ls, v)
    ls.insert(i, v)
    assert binary_search(ls, v) == i