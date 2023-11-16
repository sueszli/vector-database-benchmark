import pytest
import random
from annoy import AnnoyIndex

def test_get_item_vector():
    if False:
        for i in range(10):
            print('nop')
    f = 10
    i = AnnoyIndex(f, 'euclidean')
    i.add_item(0, [random.gauss(0, 1) for x in range(f)])
    for j in range(100):
        print(j, '...')
        for k in range(1000 * 1000):
            i.get_item_vector(0)

def test_get_lots_of_nns():
    if False:
        i = 10
        return i + 15
    f = 10
    i = AnnoyIndex(f, 'euclidean')
    i.add_item(0, [random.gauss(0, 1) for x in range(f)])
    i.build(10)
    for j in range(100):
        assert i.get_nns_by_item(0, 999999999) == [0]

def test_build_unbuid():
    if False:
        i = 10
        return i + 15
    f = 10
    i = AnnoyIndex(f, 'euclidean')
    for j in range(1000):
        i.add_item(j, [random.gauss(0, 1) for x in range(f)])
    i.build(10)
    for j in range(100):
        i.unbuild()
        i.build(10)
    assert i.get_n_items() == 1000

def test_include_distances():
    if False:
        print('Hello World!')
    f = 10
    i = AnnoyIndex(f, 'euclidean')
    for j in range(10000):
        i.add_item(j, [random.gauss(0, 1) for x in range(f)])
    i.build(10)
    v = [random.gauss(0, 1) for x in range(f)]
    for _ in range(10000000):
        (indices, distances) = i.get_nns_by_vector(v, 1, include_distances=True)