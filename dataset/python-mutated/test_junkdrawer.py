import inspect
import pytest
from hypothesis import example, given, strategies as st
from hypothesis.internal.conjecture.junkdrawer import IntList, LazySequenceCopy, NotFound, SelfOrganisingList, binary_search, clamp, replace_all, stack_depth_of_caller

def test_out_of_range():
    if False:
        i = 10
        return i + 15
    x = LazySequenceCopy([1, 2, 3])
    with pytest.raises(IndexError):
        x[3]
    with pytest.raises(IndexError):
        x[-4]

def test_pass_through():
    if False:
        i = 10
        return i + 15
    x = LazySequenceCopy([1, 2, 3])
    assert x[0] == 1
    assert x[1] == 2
    assert x[2] == 3

def test_can_assign_without_changing_underlying():
    if False:
        print('Hello World!')
    underlying = [1, 2, 3]
    x = LazySequenceCopy(underlying)
    x[1] = 10
    assert x[1] == 10
    assert underlying[1] == 2

def test_pop():
    if False:
        print('Hello World!')
    x = LazySequenceCopy([2, 3])
    assert x.pop() == 3
    assert x.pop() == 2
    with pytest.raises(IndexError):
        x.pop()

@example(1, 5, 10)
@example(1, 10, 5)
@example(5, 10, 5)
@example(5, 1, 10)
@given(st.integers(), st.integers(), st.integers())
def test_clamp(lower, value, upper):
    if False:
        for i in range(10):
            print('nop')
    (lower, upper) = sorted((lower, upper))
    clamped = clamp(lower, value, upper)
    assert lower <= clamped <= upper
    if lower <= value <= upper:
        assert value == clamped
    if lower > value:
        assert clamped == lower
    if value > upper:
        assert clamped == upper

def test_pop_without_mask():
    if False:
        return 10
    y = [1, 2, 3]
    x = LazySequenceCopy(y)
    x.pop()
    assert list(x) == [1, 2]
    assert y == [1, 2, 3]

def test_pop_with_mask():
    if False:
        for i in range(10):
            print('nop')
    y = [1, 2, 3]
    x = LazySequenceCopy(y)
    x[-1] = 5
    t = x.pop()
    assert t == 5
    assert list(x) == [1, 2]
    assert y == [1, 2, 3]

def test_assignment():
    if False:
        for i in range(10):
            print('nop')
    y = [1, 2, 3]
    x = LazySequenceCopy(y)
    x[-1] = 5
    assert list(x) == [1, 2, 5]
    x[-1] = 7
    assert list(x) == [1, 2, 7]

def test_replacement():
    if False:
        while True:
            i = 10
    result = replace_all(bytes([1, 1, 1, 1]), [(1, 3, bytes([3, 4]))])
    assert result == bytes([1, 3, 4, 1])

def test_int_list_cannot_contain_negative():
    if False:
        return 10
    with pytest.raises(ValueError):
        IntList([-1])

def test_int_list_can_contain_arbitrary_size():
    if False:
        return 10
    n = 2 ** 65
    assert list(IntList([n])) == [n]

def test_int_list_of_length():
    if False:
        for i in range(10):
            print('nop')
    assert list(IntList.of_length(10)) == [0] * 10

def test_int_list_equality():
    if False:
        i = 10
        return i + 15
    ls = [1, 2, 3]
    x = IntList(ls)
    y = IntList(ls)
    assert ls != x
    assert x != ls
    assert not x == ls
    assert x == x
    assert x == y

def test_int_list_extend():
    if False:
        while True:
            i = 10
    x = IntList.of_length(3)
    n = 2 ** 64 - 1
    x.extend([n])
    assert list(x) == [0, 0, 0, n]

@pytest.mark.parametrize('n', [0, 1, 30, 70])
def test_binary_search(n):
    if False:
        i = 10
        return i + 15
    i = binary_search(0, 100, lambda i: i <= n)
    assert i == n

def recur(i):
    if False:
        for i in range(10):
            print('nop')
    assert len(inspect.stack(0)) == stack_depth_of_caller()
    if i >= 1:
        recur(i - 1)

def test_stack_size_detection():
    if False:
        i = 10
        return i + 15
    recur(100)

def test_self_organising_list_raises_not_found_when_none_satisfy():
    if False:
        return 10
    with pytest.raises(NotFound):
        SelfOrganisingList(range(20)).find(lambda x: False)

def test_self_organising_list_moves_to_front():
    if False:
        while True:
            i = 10
    count = [0]

    def zero(n):
        if False:
            return 10
        count[0] += 1
        return n == 0
    x = SelfOrganisingList(range(20))
    assert x.find(zero) == 0
    assert count[0] == 20
    assert x.find(zero) == 0
    assert count[0] == 21