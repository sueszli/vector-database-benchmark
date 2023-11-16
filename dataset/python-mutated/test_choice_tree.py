from random import Random
from hypothesis import given, strategies as st
from hypothesis.internal.conjecture.choicetree import ChoiceTree, prefix_selection_order, random_selection_order

def select(*args):
    if False:
        for i in range(10):
            print('nop')
    return prefix_selection_order(args)

def exhaust(f):
    if False:
        print('Hello World!')
    tree = ChoiceTree()
    results = []
    prefix = ()
    while not tree.exhausted:
        prefix = tree.step(prefix_selection_order(prefix), lambda chooser: results.append(f(chooser)))
    return results

@given(st.lists(st.integers()))
def test_can_enumerate_a_shallow_set(ls):
    if False:
        for i in range(10):
            print('nop')
    results = exhaust(lambda chooser: chooser.choose(ls))
    assert sorted(results) == sorted(ls)

def test_can_enumerate_a_nested_set():
    if False:
        for i in range(10):
            print('nop')

    @exhaust
    def nested(chooser):
        if False:
            for i in range(10):
                print('nop')
        i = chooser.choose(range(10))
        j = chooser.choose(range(10), condition=lambda j: j > i)
        return (i, j)
    assert sorted(nested) == [(i, j) for i in range(10) for j in range(i + 1, 10)]

def test_can_enumerate_empty():
    if False:
        while True:
            i = 10

    @exhaust
    def empty(chooser):
        if False:
            i = 10
            return i + 15
        return 1
    assert empty == [1]

def test_all_filtered_child():
    if False:
        return 10

    @exhaust
    def all_filtered(chooser):
        if False:
            while True:
                i = 10
        chooser.choose(range(10), condition=lambda j: False)
    assert all_filtered == []

def test_skips_over_exhausted_children():
    if False:
        while True:
            i = 10
    results = []

    def f(chooser):
        if False:
            print('Hello World!')
        results.append((chooser.choose(range(3), condition=lambda x: x > 0), chooser.choose(range(2))))
    tree = ChoiceTree()
    tree.step(select(1, 0), f)
    tree.step(select(1, 1), f)
    tree.step(select(0, 0), f)
    assert results == [(1, 0), (1, 1), (2, 0)]

def test_extends_prefix_from_right():
    if False:
        i = 10
        return i + 15

    def f(chooser):
        if False:
            i = 10
            return i + 15
        chooser.choose(range(4))
    tree = ChoiceTree()
    result = tree.step(select(), f)
    assert result == (3,)

def test_starts_from_the_end():
    if False:
        print('Hello World!')

    def f(chooser):
        if False:
            print('Hello World!')
        chooser.choose(range(3))
    tree = ChoiceTree()
    assert tree.step(select(), f) == (2,)

def test_skips_over_exhausted_subtree():
    if False:
        i = 10
        return i + 15

    def f(chooser):
        if False:
            for i in range(10):
                print('nop')
        chooser.choose(range(10))
    tree = ChoiceTree()
    assert tree.step(select(8), f) == (8,)
    assert tree.step(select(8), f) == (7,)

def test_exhausts_randomly():
    if False:
        print('Hello World!')

    def f(chooser):
        if False:
            print('Hello World!')
        chooser.choose(range(10))
    tree = ChoiceTree()
    random = Random()
    seen = set()
    for _ in range(10):
        seen.add(tree.step(random_selection_order(random), f))
    assert len(seen) == 10
    assert tree.exhausted

def test_exhausts_randomly_when_filtering():
    if False:
        while True:
            i = 10

    def f(chooser):
        if False:
            for i in range(10):
                print('nop')
        chooser.choose(range(10), lambda x: False)
    tree = ChoiceTree()
    random = Random()
    tree.step(random_selection_order(random), f)
    assert tree.exhausted