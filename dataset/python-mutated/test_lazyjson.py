"""Tests lazy json functionality."""
from io import StringIO
from xonsh.lazyjson import LazyJSON, LJNode, index, ljdump

def test_index_int():
    if False:
        while True:
            i = 10
    exp = {'offsets': 0, 'sizes': 2}
    (s, obs) = index(42)
    assert exp == obs

def test_index_str():
    if False:
        return 10
    exp = {'offsets': 0, 'sizes': 7}
    (s, obs) = index('wakka')
    assert exp == obs

def test_index_list_ints():
    if False:
        while True:
            i = 10
    exp = {'offsets': [1, 4, 0], 'sizes': [1, 2, 8]}
    (s, obs) = index([1, 42])
    assert exp == obs

def test_index_list_str():
    if False:
        i = 10
        return i + 15
    exp = {'offsets': [1, 10, 0], 'sizes': [7, 8, 20]}
    (s, obs) = index(['wakka', 'jawaka'])
    assert exp == obs

def test_index_list_str_int():
    if False:
        i = 10
        return i + 15
    exp = {'offsets': [1, 10, 0], 'sizes': [7, 2, 14]}
    (s, obs) = index(['wakka', 42])
    assert exp == obs

def test_index_list_int_str():
    if False:
        return 10
    exp = {'offsets': [1, 5, 14, 0], 'sizes': [2, 7, 8, 24]}
    (s, obs) = index([42, 'wakka', 'jawaka'])
    assert exp == obs

def test_index_dict_int():
    if False:
        return 10
    exp = {'offsets': {'wakka': 10, '__total__': 0}, 'sizes': {'wakka': 2, '__total__': 14}}
    (s, obs) = index({'wakka': 42})
    assert exp == obs

def test_index_dict_str():
    if False:
        return 10
    exp = {'offsets': {'wakka': 10, '__total__': 0}, 'sizes': {'wakka': 8, '__total__': 20}}
    (s, obs) = index({'wakka': 'jawaka'})
    assert exp == obs

def test_index_dict_dict_int():
    if False:
        while True:
            i = 10
    exp = {'offsets': {'wakka': {'jawaka': 21, '__total__': 10}, '__total__': 0}, 'sizes': {'wakka': {'jawaka': 2, '__total__': 15}, '__total__': 27}}
    (s, obs) = index({'wakka': {'jawaka': 42}})
    assert exp == obs

def test_lazy_load_index():
    if False:
        print('Hello World!')
    f = StringIO()
    ljdump({'wakka': 42}, f)
    f.seek(0)
    lj = LazyJSON(f)
    assert {'wakka': 10, '__total__': 0} == lj.offsets
    assert {'wakka': 2, '__total__': 14} == lj.sizes

def test_lazy_int():
    if False:
        i = 10
        return i + 15
    f = StringIO()
    ljdump(42, f)
    f.seek(0)
    lj = LazyJSON(f)
    assert 42 == lj.load()

def test_lazy_str():
    if False:
        for i in range(10):
            print('nop')
    f = StringIO()
    ljdump('wakka', f)
    f.seek(0)
    lj = LazyJSON(f)
    assert 'wakka' == lj.load()

def test_lazy_list_empty():
    if False:
        i = 10
        return i + 15
    x = []
    f = StringIO()
    ljdump(x, f)
    f.seek(0)
    lj = LazyJSON(f)
    assert 0 == len(lj)
    assert x == lj.load()

def test_lazy_list_ints():
    if False:
        while True:
            i = 10
    x = [0, 1, 6, 28, 496, 8128]
    f = StringIO()
    ljdump(x, f)
    f.seek(0)
    lj = LazyJSON(f)
    assert 28 == lj[3]
    assert x[:2:-2] == lj[:2:-2]
    assert x == [_ for _ in lj]
    assert x == lj.load()

def test_lazy_list_str():
    if False:
        for i in range(10):
            print('nop')
    x = ['I', 'have', 'seen', 'the', 'wind', 'blow']
    f = StringIO()
    ljdump(x, f)
    f.seek(0)
    lj = LazyJSON(f)
    assert 'the' == lj[3]
    assert x[:2:-2] == lj[:2:-2]
    assert x == [_ for _ in lj]
    assert x == lj.load()

def test_lazy_list_list_ints():
    if False:
        while True:
            i = 10
    x = [[0, 1], [6, 28], [496, 8128]]
    f = StringIO()
    ljdump(x, f)
    f.seek(0)
    lj = LazyJSON(f)
    assert isinstance(lj[1], LJNode)
    assert 28 == lj[1][1]
    assert [6 == 28], lj[1].load()
    assert x == lj.load()

def test_lazy_dict_empty():
    if False:
        while True:
            i = 10
    x = {}
    f = StringIO()
    ljdump(x, f)
    f.seek(0)
    lj = LazyJSON(f)
    assert 0 == len(lj)
    assert x == lj.load()

def test_lazy_dict():
    if False:
        return 10
    f = StringIO()
    ljdump({'wakka': 42}, f)
    f.seek(0)
    lj = LazyJSON(f)
    assert ['wakka'] == list(lj.keys())
    assert 42 == lj['wakka']
    assert 1 == len(lj)
    assert {'wakka': 42} == lj.load()

def test_lazy_dict_dict_int():
    if False:
        while True:
            i = 10
    x = {'wakka': {'jawaka': 42}}
    f = StringIO()
    ljdump(x, f)
    f.seek(0)
    lj = LazyJSON(f)
    assert ['wakka'] == list(lj.keys())
    assert isinstance(lj['wakka'], LJNode)
    assert 42 == lj['wakka']['jawaka']
    assert 1 == len(lj)
    assert x == lj.load()