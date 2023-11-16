import random
import numpy
import pytest
from annoy import AnnoyIndex

def test_get_nns_by_vector():
    if False:
        while True:
            i = 10
    f = 3
    i = AnnoyIndex(f, 'angular')
    i.add_item(0, [0, 0, 1])
    i.add_item(1, [0, 1, 0])
    i.add_item(2, [1, 0, 0])
    i.build(10)
    assert i.get_nns_by_vector([3, 2, 1], 3) == [2, 1, 0]
    assert i.get_nns_by_vector([1, 2, 3], 3) == [0, 1, 2]
    assert i.get_nns_by_vector([2, 0, 1], 3) == [2, 0, 1]

def test_get_nns_by_item():
    if False:
        i = 10
        return i + 15
    f = 3
    i = AnnoyIndex(f, 'angular')
    i.add_item(0, [2, 1, 0])
    i.add_item(1, [1, 2, 0])
    i.add_item(2, [0, 0, 1])
    i.build(10)
    assert i.get_nns_by_item(0, 3) == [0, 1, 2]
    assert i.get_nns_by_item(1, 3) == [1, 0, 2]
    assert i.get_nns_by_item(2, 3) in [[2, 0, 1], [2, 1, 0]]

def test_dist():
    if False:
        return 10
    f = 2
    i = AnnoyIndex(f, 'angular')
    i.add_item(0, [0, 1])
    i.add_item(1, [1, 1])
    assert i.get_distance(0, 1) == pytest.approx((2 * (1.0 - 2 ** (-0.5))) ** 0.5)

def test_dist_2():
    if False:
        return 10
    f = 2
    i = AnnoyIndex(f, 'angular')
    i.add_item(0, [1000, 0])
    i.add_item(1, [10, 0])
    assert i.get_distance(0, 1) == pytest.approx(0)

def test_dist_3():
    if False:
        return 10
    f = 2
    i = AnnoyIndex(f, 'angular')
    i.add_item(0, [97, 0])
    i.add_item(1, [42, 42])
    dist = ((1 - 2 ** (-0.5)) ** 2 + (2 ** (-0.5)) ** 2) ** 0.5
    assert i.get_distance(0, 1) == pytest.approx(dist)

def test_dist_degen():
    if False:
        print('Hello World!')
    f = 2
    i = AnnoyIndex(f, 'angular')
    i.add_item(0, [1, 0])
    i.add_item(1, [0, 0])
    assert i.get_distance(0, 1) == pytest.approx(2.0 ** 0.5)

def test_large_index():
    if False:
        return 10
    f = 10
    i = AnnoyIndex(f, 'angular')
    for j in range(0, 10000, 2):
        p = [random.gauss(0, 1) for z in range(f)]
        f1 = random.random() + 1
        f2 = random.random() + 1
        x = [f1 * pi + random.gauss(0, 0.01) for pi in p]
        y = [f2 * pi + random.gauss(0, 0.01) for pi in p]
        i.add_item(j, x)
        i.add_item(j + 1, y)
    i.build(10)
    for j in range(0, 10000, 2):
        assert i.get_nns_by_item(j, 2) == [j, j + 1]
        assert i.get_nns_by_item(j + 1, 2) == [j + 1, j]

def precision(n, n_trees=10, n_points=10000, n_rounds=10, search_k=100000):
    if False:
        i = 10
        return i + 15
    found = 0
    for r in range(n_rounds):
        f = 10
        i = AnnoyIndex(f, 'angular')
        for j in range(n_points):
            p = [random.gauss(0, 1) for z in range(f - 1)]
            norm = sum([pi ** 2 for pi in p]) ** 0.5
            x = [1000] + [pi / norm * j for pi in p]
            i.add_item(j, x)
        i.build(n_trees)
        nns = i.get_nns_by_vector([1000] + [0] * (f - 1), n, search_k)
        assert nns == sorted(nns)
        found += len([x for x in nns if x < n])
    return 1.0 * found / (n * n_rounds)

def test_precision_1():
    if False:
        print('Hello World!')
    assert precision(1) >= 0.98

def test_precision_10():
    if False:
        print('Hello World!')
    assert precision(10) >= 0.98

def test_precision_100():
    if False:
        for i in range(10):
            print('nop')
    assert precision(100) >= 0.98

def test_precision_1000():
    if False:
        for i in range(10):
            print('nop')
    assert precision(1000) >= 0.98

def test_load_save_get_item_vector():
    if False:
        print('Hello World!')
    f = 3
    i = AnnoyIndex(f, 'angular')
    i.add_item(0, [1.1, 2.2, 3.3])
    i.add_item(1, [4.4, 5.5, 6.6])
    i.add_item(2, [7.7, 8.8, 9.9])
    numpy.testing.assert_array_almost_equal(i.get_item_vector(0), [1.1, 2.2, 3.3])
    assert i.build(10)
    assert i.save('blah.ann')
    numpy.testing.assert_array_almost_equal(i.get_item_vector(1), [4.4, 5.5, 6.6])
    j = AnnoyIndex(f, 'angular')
    assert j.load('blah.ann')
    numpy.testing.assert_array_almost_equal(j.get_item_vector(2), [7.7, 8.8, 9.9])

def test_get_nns_search_k():
    if False:
        print('Hello World!')
    f = 3
    i = AnnoyIndex(f, 'angular')
    i.add_item(0, [0, 0, 1])
    i.add_item(1, [0, 1, 0])
    i.add_item(2, [1, 0, 0])
    i.build(10)
    assert i.get_nns_by_item(0, 3, 10) == [0, 1, 2]
    assert i.get_nns_by_vector([3, 2, 1], 3, 10) == [2, 1, 0]

def test_include_dists():
    if False:
        for i in range(10):
            print('nop')
    f = 40
    i = AnnoyIndex(f, 'angular')
    v = numpy.random.normal(size=f)
    i.add_item(0, v)
    i.add_item(1, -v)
    i.build(10)
    (indices, dists) = i.get_nns_by_item(0, 2, 10, True)
    assert indices == [0, 1]
    assert dists[0] == pytest.approx(0.0)
    assert dists[1] == pytest.approx(2.0)

def test_include_dists_check_ranges():
    if False:
        return 10
    f = 3
    i = AnnoyIndex(f, 'angular')
    for j in range(100000):
        i.add_item(j, numpy.random.normal(size=f))
    i.build(10)
    (indices, dists) = i.get_nns_by_item(0, 100000, include_distances=True)
    assert max(dists) <= 2.0
    assert min(dists) == pytest.approx(0.0)

def test_distance_consistency():
    if False:
        for i in range(10):
            print('nop')
    (n, f) = (1000, 3)
    i = AnnoyIndex(f, 'angular')
    for j in range(n):
        while True:
            v = numpy.random.normal(size=f)
            if numpy.dot(v, v) > 0.1:
                break
        i.add_item(j, v)
    i.build(10)
    for a in random.sample(range(n), 100):
        (indices, dists) = i.get_nns_by_item(a, 100, include_distances=True)
        for (b, dist) in zip(indices, dists):
            u = i.get_item_vector(a)
            v = i.get_item_vector(b)
            assert dist == pytest.approx(i.get_distance(a, b), rel=0.001, abs=0.001)
            u_norm = numpy.array(u) * numpy.dot(u, u) ** (-0.5)
            v_norm = numpy.array(v) * numpy.dot(v, v) ** (-0.5)
            assert dist ** 2 == pytest.approx(numpy.dot(u_norm - v_norm, u_norm - v_norm), rel=0.001, abs=0.001)
            assert dist ** 2 == pytest.approx(sum([(x - y) ** 2 for (x, y) in zip(u_norm, v_norm)]), rel=0.001, abs=0.001)

def test_only_one_item():
    if False:
        while True:
            i = 10
    idx = AnnoyIndex(100, 'angular')
    idx.add_item(0, numpy.random.randn(100))
    idx.build(n_trees=10)
    idx.save('foo.idx')
    idx = AnnoyIndex(100, 'angular')
    idx.load('foo.idx')
    assert idx.get_n_items() == 1
    assert idx.get_nns_by_vector(vector=numpy.random.randn(100), n=50, include_distances=False) == [0]

def test_no_items():
    if False:
        print('Hello World!')
    idx = AnnoyIndex(100, 'angular')
    idx.build(n_trees=10)
    idx.save('foo.idx')
    idx = AnnoyIndex(100, 'angular')
    idx.load('foo.idx')
    assert idx.get_n_items() == 0
    assert idx.get_nns_by_vector(vector=numpy.random.randn(100), n=50, include_distances=False) == []

def test_single_vector():
    if False:
        print('Hello World!')
    a = AnnoyIndex(3, 'angular')
    a.add_item(0, [1, 0, 0])
    a.build(10)
    a.save('1.ann')
    (indices, dists) = a.get_nns_by_vector([1, 0, 0], 3, include_distances=True)
    assert indices == [0]
    assert dists[0] ** 2 == pytest.approx(0.0)