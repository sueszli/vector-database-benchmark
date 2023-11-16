import pytest
import networkx as nx
from networkx.utils import BinaryHeap, PairingHeap

class X:

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        raise self is other

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        raise self is not other

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        raise TypeError('cannot compare')

    def __le__(self, other):
        if False:
            while True:
                i = 10
        raise TypeError('cannot compare')

    def __ge__(self, other):
        if False:
            return 10
        raise TypeError('cannot compare')

    def __gt__(self, other):
        if False:
            while True:
                i = 10
        raise TypeError('cannot compare')

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(id(self))
x = X()
data = [('min', nx.NetworkXError), ('pop', nx.NetworkXError), ('get', 0, None), ('get', x, None), ('get', None, None), ('insert', x, 1, True), ('get', x, 1), ('min', (x, 1)), ('min', (x, 1)), ('insert', 1, -2.0, True), ('min', (1, -2.0)), ('insert', 3, -10 ** 100, True), ('insert', 4, 5, True), ('pop', (3, -10 ** 100)), ('pop', (1, -2.0)), ('insert', 4, -50, True), ('insert', 4, -60, False, True), ('pop', (4, -60)), ('pop', (x, 1)), ('min', nx.NetworkXError), ('pop', nx.NetworkXError), ('insert', x, 0, True), ('insert', x, 0, False, False), ('min', (x, 0)), ('insert', x, 0, True, False), ('min', (x, 0)), ('pop', (x, 0)), ('pop', nx.NetworkXError), ('insert', None, 0, True), ('insert', 2, -1, True), ('min', (2, -1)), ('insert', 2, 1, True, False), ('min', (None, 0)), ('insert', None, 2, False, False), ('min', (None, 0)), ('pop', (None, 0)), ('pop', (2, 1)), ('min', nx.NetworkXError), ('pop', nx.NetworkXError)]

def _test_heap_class(cls, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    heap = cls(*args, **kwargs)
    for op in data:
        if op[-1] is not nx.NetworkXError:
            assert op[-1] == getattr(heap, op[0])(*op[1:-1])
        else:
            pytest.raises(op[-1], getattr(heap, op[0]), *op[1:-1])
    for i in range(99, -1, -1):
        assert heap.insert(i, i)
    for i in range(50):
        assert heap.pop() == (i, i)
    for i in range(100):
        assert heap.insert(i, i) == (i < 50)
    for i in range(100):
        assert not heap.insert(i, i + 1)
    for i in range(50):
        assert heap.pop() == (i, i)
    for i in range(100):
        assert heap.insert(i, i + 1) == (i < 50)
    for i in range(49):
        assert heap.pop() == (i, i + 1)
    assert sorted([heap.pop(), heap.pop()]) == [(49, 50), (50, 50)]
    for i in range(51, 100):
        assert not heap.insert(i, i + 1, True)
    for i in range(51, 70):
        assert heap.pop() == (i, i + 1)
    for i in range(100):
        assert heap.insert(i, i)
    for i in range(100):
        assert heap.pop() == (i, i)
    pytest.raises(nx.NetworkXError, heap.pop)

def test_PairingHeap():
    if False:
        for i in range(10):
            print('nop')
    _test_heap_class(PairingHeap)

def test_BinaryHeap():
    if False:
        return 10
    _test_heap_class(BinaryHeap)