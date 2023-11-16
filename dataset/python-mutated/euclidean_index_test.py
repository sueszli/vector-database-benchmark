import random
import numpy
import pytest
from annoy import AnnoyIndex

def test_get_nns_by_vector():
    if False:
        return 10
    f = 2
    i = AnnoyIndex(f, 'euclidean')
    i.add_item(0, [2, 2])
    i.add_item(1, [3, 2])
    i.add_item(2, [3, 3])
    i.build(10)
    assert i.get_nns_by_vector([4, 4], 3) == [2, 1, 0]
    assert i.get_nns_by_vector([1, 1], 3) == [0, 1, 2]
    assert i.get_nns_by_vector([4, 2], 3) == [1, 2, 0]

def test_get_nns_by_item():
    if False:
        print('Hello World!')
    f = 2
    i = AnnoyIndex(f, 'euclidean')
    i.add_item(0, [2, 2])
    i.add_item(1, [3, 2])
    i.add_item(2, [3, 3])
    i.build(10)
    assert i.get_nns_by_item(0, 3) == [0, 1, 2]
    assert i.get_nns_by_item(2, 3) == [2, 1, 0]

def test_dist():
    if False:
        print('Hello World!')
    f = 2
    i = AnnoyIndex(f, 'euclidean')
    i.add_item(0, [0, 1])
    i.add_item(1, [1, 1])
    i.add_item(2, [0, 0])
    assert i.get_distance(0, 1) == pytest.approx(1.0 ** 0.5)
    assert i.get_distance(1, 2) == pytest.approx(2.0 ** 0.5)

def test_large_index():
    if False:
        for i in range(10):
            print('nop')
    f = 10
    [random.gauss(0, 10) for z in range(f)]
    i = AnnoyIndex(f, 'euclidean')
    for j in range(0, 10000, 2):
        p = [random.gauss(0, 1) for z in range(f)]
        x = [1 + pi + random.gauss(0, 0.01) for pi in p]
        y = [1 + pi + random.gauss(0, 0.01) for pi in p]
        i.add_item(j, x)
        i.add_item(j + 1, y)
    i.build(10)
    for j in range(0, 10000, 2):
        assert i.get_nns_by_item(j, 2) == [j, j + 1]
        assert i.get_nns_by_item(j + 1, 2) == [j + 1, j]

def precision(n, n_trees=10, n_points=10000, n_rounds=10):
    if False:
        while True:
            i = 10
    found = 0
    for r in range(n_rounds):
        f = 10
        i = AnnoyIndex(f, 'euclidean')
        for j in range(n_points):
            p = [random.gauss(0, 1) for z in range(f)]
            norm = sum([pi ** 2 for pi in p]) ** 0.5
            x = [pi / norm * j for pi in p]
            i.add_item(j, x)
        i.build(n_trees)
        nns = i.get_nns_by_vector([0] * f, n)
        assert nns == sorted(nns)
        found += len([x for x in nns if x < n])
    return 1.0 * found / (n * n_rounds)

def test_precision_1():
    if False:
        i = 10
        return i + 15
    assert precision(1) >= 0.98

def test_precision_10():
    if False:
        for i in range(10):
            print('nop')
    assert precision(10) >= 0.98

def test_precision_100():
    if False:
        print('Hello World!')
    assert precision(100) >= 0.98

def test_precision_1000():
    if False:
        i = 10
        return i + 15
    assert precision(1000) >= 0.98

def test_get_nns_with_distances():
    if False:
        i = 10
        return i + 15
    f = 3
    i = AnnoyIndex(f, 'euclidean')
    i.add_item(0, [0, 0, 2])
    i.add_item(1, [0, 1, 1])
    i.add_item(2, [1, 0, 0])
    i.build(10)
    (l, d) = i.get_nns_by_item(0, 3, -1, True)
    assert l == [0, 1, 2]
    assert d[0] ** 2 == pytest.approx(0)
    assert d[1] ** 2 == pytest.approx(2)
    assert d[2] ** 2 == pytest.approx(5)
    (l, d) = i.get_nns_by_vector([2, 2, 2], 3, -1, True)
    assert l == [1, 0, 2]
    assert d[0] ** 2 == pytest.approx(6)
    assert d[1] ** 2 == pytest.approx(8)
    assert d[2] ** 2 == pytest.approx(9)

def test_include_dists():
    if False:
        i = 10
        return i + 15
    f = 40
    i = AnnoyIndex(f, 'euclidean')
    v = numpy.random.normal(size=f)
    i.add_item(0, v)
    i.add_item(1, -v)
    i.build(10)
    (indices, dists) = i.get_nns_by_item(0, 2, 10, True)
    assert indices == [0, 1]
    assert dists[0] == pytest.approx(0)

def test_distance_consistency():
    if False:
        while True:
            i = 10
    (n, f) = (1000, 3)
    i = AnnoyIndex(f, 'euclidean')
    for j in range(n):
        i.add_item(j, numpy.random.normal(size=f))
    i.build(10)
    for a in random.sample(range(n), 100):
        (indices, dists) = i.get_nns_by_item(a, 100, include_distances=True)
        for (b, dist) in zip(indices, dists):
            assert dist == pytest.approx(i.get_distance(a, b))
            u = numpy.array(i.get_item_vector(a))
            v = numpy.array(i.get_item_vector(b))
            assert dist == pytest.approx(numpy.dot(u - v, u - v) ** 0.5)
            assert dist == pytest.approx(sum([(x - y) ** 2 for (x, y) in zip(u, v)]) ** 0.5)

def test_rounding_error():
    if False:
        return 10
    i = AnnoyIndex(1, 'euclidean')
    i.add_item(0, [0.712593])
    i.add_item(1, [0.7123166])
    assert i.get_distance(0, 1) >= 0.0