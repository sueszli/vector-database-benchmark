import random
import numpy
from annoy import AnnoyIndex

def test_random_holes():
    if False:
        print('Hello World!')
    f = 10
    index = AnnoyIndex(f, 'angular')
    valid_indices = random.sample(range(2000), 1000)
    for i in valid_indices:
        v = numpy.random.normal(size=(f,))
        index.add_item(i, v)
    index.build(10)
    for i in valid_indices:
        js = index.get_nns_by_item(i, 10000)
        for j in js:
            assert j in valid_indices
    for i in range(1000):
        v = numpy.random.normal(size=(f,))
        js = index.get_nns_by_vector(v, 10000)
        for j in js:
            assert j in valid_indices

def _test_holes_base(n, f=100, base_i=100000):
    if False:
        while True:
            i = 10
    annoy = AnnoyIndex(f, 'angular')
    for i in range(n):
        annoy.add_item(base_i + i, numpy.random.normal(size=(f,)))
    annoy.build(100)
    res = annoy.get_nns_by_item(base_i, n)
    assert set(res) == set([base_i + i for i in range(n)])

def test_root_one_child():
    if False:
        for i in range(10):
            print('nop')
    _test_holes_base(1)

def test_root_two_children():
    if False:
        print('Hello World!')
    _test_holes_base(2)

def test_root_some_children():
    if False:
        for i in range(10):
            print('nop')
    _test_holes_base(10)

def test_root_many_children():
    if False:
        return 10
    _test_holes_base(1000)