from collections import Counter
from random import Random
from hypothesis.internal.conjecture.shrinking import Lexical

def test_shrink_to_zero():
    if False:
        print('Hello World!')
    assert Lexical.shrink(bytes([255] * 8), lambda x: True, random=Random(0)) == bytes(8)

def test_shrink_to_smallest():
    if False:
        print('Hello World!')
    assert Lexical.shrink(bytes([255] * 8), lambda x: sum(x) > 10, random=Random(0)) == bytes([0] * 7 + [11])

def test_float_hack_fails():
    if False:
        i = 10
        return i + 15
    assert Lexical.shrink(bytes([255] * 8), lambda x: x[0] >> 7, random=Random(0)) == bytes([128] + [0] * 7)

def test_can_sort_bytes_by_reordering():
    if False:
        for i in range(10):
            print('nop')
    start = bytes([5, 4, 3, 2, 1, 0])
    finish = Lexical.shrink(start, lambda x: set(x) == set(start), random=Random(0))
    assert finish == bytes([0, 1, 2, 3, 4, 5])

def test_can_sort_bytes_by_reordering_partially():
    if False:
        for i in range(10):
            print('nop')
    start = bytes([5, 4, 3, 2, 1, 0])
    finish = Lexical.shrink(start, lambda x: set(x) == set(start) and x[0] > x[-1], random=Random(0))
    assert finish == bytes([1, 2, 3, 4, 5, 0])

def test_can_sort_bytes_by_reordering_partially2():
    if False:
        return 10
    start = bytes([5, 4, 3, 2, 1, 0])
    finish = Lexical.shrink(start, lambda x: Counter(x) == Counter(start) and x[0] > x[2], random=Random(0), full=True)
    assert finish <= bytes([1, 2, 0, 3, 4, 5])

def test_can_sort_bytes_by_reordering_partially_not_cross_stationary_element():
    if False:
        for i in range(10):
            print('nop')
    start = bytes([5, 3, 0, 2, 1, 4])
    finish = Lexical.shrink(start, lambda x: set(x) == set(start) and x[3] == 2, random=Random(0))
    assert finish <= bytes([0, 3, 5, 2, 1, 4])