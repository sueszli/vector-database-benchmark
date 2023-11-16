from random import Random
from hypothesis import example, given, strategies as st
from hypothesis.internal.conjecture.shrinking import Ordering

@example([0, 1, 1, 1, 1, 1, 1, 0])
@example([0, 0])
@example([0, 1, -1])
@given(st.lists(st.integers()))
def test_shrinks_down_to_sorted_the_slow_way(ls):
    if False:
        i = 10
        return i + 15
    shrinker = Ordering(ls, lambda ls: True, random=Random(0), full=False)
    shrinker.run_step()
    assert list(shrinker.current) == sorted(ls)

def test_can_partially_sort_a_list():
    if False:
        for i in range(10):
            print('nop')
    finish = Ordering.shrink([5, 4, 3, 2, 1, 0], lambda x: x[0] > x[-1], random=Random(0))
    assert finish == (1, 2, 3, 4, 5, 0)

def test_can_partially_sort_a_list_2():
    if False:
        for i in range(10):
            print('nop')
    finish = Ordering.shrink([5, 4, 3, 2, 1, 0], lambda x: x[0] > x[2], random=Random(0), full=True)
    assert finish <= (1, 2, 0, 3, 4, 5)

def test_adaptively_shrinks_around_hole():
    if False:
        return 10
    initial = list(range(1000, 0, -1))
    initial[500] = 2000
    intended_result = sorted(initial)
    intended_result.insert(500, intended_result.pop())
    shrinker = Ordering(initial, lambda ls: ls[500] == 2000, random=Random(0), full=True)
    shrinker.run()
    assert shrinker.current[500] == 2000
    assert list(shrinker.current) == intended_result
    assert shrinker.calls <= 60