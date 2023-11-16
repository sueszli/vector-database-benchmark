from __future__ import annotations
from typing import NamedTuple
import pytest
from textual._layout_resolve import layout_resolve

class Edge(NamedTuple):
    size: int | None = None
    fraction: int = 1
    min_size: int = 1

def test_empty():
    if False:
        for i in range(10):
            print('nop')
    assert layout_resolve(10, []) == []

def test_total_zero():
    if False:
        i = 10
        return i + 15
    assert layout_resolve(0, [Edge(10)]) == [10]

def test_single():
    if False:
        print('Hello World!')
    assert layout_resolve(100, [Edge(10)]) == [10]
    assert layout_resolve(100, [Edge(None, 1)]) == [100]
    assert layout_resolve(100, [Edge(None, 2)]) == [100]
    assert layout_resolve(100, [Edge(None, 1, 20)]) == [100]
    assert layout_resolve(100, [Edge(None, 1, 120)]) == [120]

def test_two():
    if False:
        i = 10
        return i + 15
    assert layout_resolve(100, [Edge(10), Edge(20)]) == [10, 20]
    assert layout_resolve(100, [Edge(120), Edge(None, 1)]) == [120, 1]
    assert layout_resolve(100, [Edge(None, 1), Edge(None, 1)]) == [50, 50]
    assert layout_resolve(100, [Edge(None, 2), Edge(None, 1)]) == [66, 34]
    assert layout_resolve(100, [Edge(None, 2), Edge(None, 2)]) == [50, 50]
    assert layout_resolve(100, [Edge(None, 3), Edge(None, 1)]) == [75, 25]
    assert layout_resolve(100, [Edge(None, 3), Edge(None, 1, 30)]) == [70, 30]
    assert layout_resolve(100, [Edge(None, 1, 30), Edge(None, 3)]) == [30, 70]

@pytest.mark.parametrize('size, edges, result', [(10, [Edge(8), Edge(None, 0, 2), Edge(4)], [8, 2, 4]), (10, [Edge(None, 1), Edge(None, 1), Edge(None, 1)], [3, 3, 4]), (10, [Edge(5), Edge(None, 1), Edge(None, 1)], [5, 2, 3]), (10, [Edge(None, 2), Edge(None, 1), Edge(None, 1)], [5, 2, 3]), (10, [Edge(None, 2), Edge(3), Edge(None, 1)], [4, 3, 3]), (10, [Edge(None, 2), Edge(None, 1), Edge(None, 1), Edge(None, 1)], [4, 2, 2, 2]), (10, [Edge(None, 4), Edge(None, 1), Edge(None, 1), Edge(None, 1)], [5, 2, 1, 2]), (2, [Edge(None, 1), Edge(None, 1), Edge(None, 1)], [1, 1, 1]), (2, [Edge(None, 1, min_size=5), Edge(None, 1, min_size=4), Edge(None, 1, min_size=3)], [5, 4, 3]), (18, [Edge(None, 1, min_size=1), Edge(3), Edge(None, 1, min_size=1), Edge(4), Edge(None, 1, min_size=1), Edge(5), Edge(None, 1, min_size=1)], [1, 3, 2, 4, 1, 5, 2])])
def test_multiple(size, edges, result):
    if False:
        print('Hello World!')
    assert layout_resolve(size, edges) == result