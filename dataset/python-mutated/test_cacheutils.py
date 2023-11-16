import string
import sys
from abc import abstractmethod, ABCMeta
import pytest
from boltons.cacheutils import LRU, LRI, cached, cachedmethod, cachedproperty, MinIDMap, ThresholdCounter

class CountingCallable(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.call_count = 0

    def __call__(self, *a, **kw):
        if False:
            while True:
                i = 10
        self.call_count += 1
        return self.call_count

def test_lru_add():
    if False:
        for i in range(10):
            print('nop')
    cache = LRU(max_size=3)
    for i in range(4):
        cache[i] = i
    assert len(cache) == 3
    assert 0 not in cache

def test_lri():
    if False:
        print('Hello World!')
    cache_size = 10
    bc = LRI(cache_size, on_miss=lambda k: k.upper())
    for (idx, char) in enumerate(string.ascii_letters):
        x = bc[char]
        assert x == char.upper()
        least_recent_insert_index = idx - cache_size
        if least_recent_insert_index >= 0:
            assert len(bc) == cache_size
            for char in string.ascii_letters[least_recent_insert_index + 1:idx]:
                assert char in bc
    bc[string.ascii_letters[-cache_size + 1]] = 'new value'
    least_recently_inserted_key = string.ascii_letters[-cache_size + 2]
    bc['unreferenced_key'] = 'value'
    keys_in_cache = [string.ascii_letters[i] for i in range(-cache_size + 1, 0) if string.ascii_letters[i] != least_recently_inserted_key]
    keys_in_cache.append('unreferenced_key')
    assert len(bc) == cache_size
    for k in keys_in_cache:
        assert k in bc

def test_lri_cache_eviction():
    if False:
        for i in range(10):
            print('nop')
    '\n    Regression test\n    Original LRI implementation had a bug where the specified cache\n    size only supported `max_size` number of inserts to the cache,\n    rather than support `max_size` number of keys in the cache. This\n    would result in some unintuitive behavior, where a key is evicted\n    recently inserted value would be evicted from the cache if the key\n    inserted was inserted `max_size` keys earlier.\n    '
    test_cache = LRI(2)
    test_cache['key1'] = 'value1'
    test_cache['key1'] = 'value1'
    test_cache['key2'] = 'value2'
    test_cache['key3'] = 'value3'
    test_cache['key3'] = 'value3'

def test_cache_sizes_on_repeat_insertions():
    if False:
        while True:
            i = 10
    '\n    Regression test\n    Original LRI implementation had an unbounded size of memory\n    regardless of the value for its `max_size` parameter due to a naive\n    insertion algorithm onto an underlying deque data structure. To\n    prevent memory leaks, this test will assert that a cache does not\n    grow past its max size given values of a uniform memory footprint\n    '
    caches_to_test = (LRU, LRI)
    for cache_type in caches_to_test:
        test_cache = cache_type(2)
        test_cache['key1'] = '1'
        test_cache['key2'] = '1'
        initial_list_size = len(test_cache._get_flattened_ll())
        for k in test_cache:
            for __ in range(100):
                test_cache[k] = '1'
        list_size_after_inserts = len(test_cache._get_flattened_ll())
        assert initial_list_size == list_size_after_inserts

def test_lru_basic():
    if False:
        for i in range(10):
            print('nop')
    lru = LRU(max_size=1)
    repr(lru)
    lru['hi'] = 0
    lru['bye'] = 1
    assert len(lru) == 1
    lru['bye']
    assert lru.get('hi') is None
    del lru['bye']
    assert 'bye' not in lru
    assert len(lru) == 0
    assert not lru
    try:
        lru.pop('bye')
    except KeyError:
        pass
    else:
        assert False
    default = object()
    assert lru.pop('bye', default) is default
    try:
        lru.popitem()
    except KeyError:
        pass
    else:
        assert False
    lru['another'] = 1
    assert lru.popitem() == ('another', 1)
    lru['yet_another'] = 2
    assert lru.pop('yet_another') == 2
    lru['yet_another'] = 3
    assert lru.pop('yet_another', default) == 3
    lru['yet_another'] = 4
    lru.clear()
    assert not lru
    lru['yet_another'] = 5
    second_lru = LRU(max_size=1)
    assert lru.copy() == lru
    second_lru['yet_another'] = 5
    assert second_lru == lru
    assert lru == second_lru
    lru.update(LRU(max_size=2, values=[('a', 1), ('b', 2)]))
    assert len(lru) == 1
    assert 'yet_another' not in lru
    lru.setdefault('x', 2)
    assert dict(lru) == {'x': 2}
    lru.setdefault('x', 3)
    assert dict(lru) == {'x': 2}
    assert lru != second_lru
    assert second_lru != lru

@pytest.mark.parametrize('lru_class', [LRU, LRI])
def test_lru_dict_replacement(lru_class):
    if False:
        print('Hello World!')
    cache = lru_class()
    cache['a'] = 1
    assert cache['a'] == 1
    assert dict(cache) == {'a': 1}
    assert list(cache.values())[0] == 1
    cache['a'] = 200
    assert cache['a'] == 200
    assert dict(cache) == {'a': 200}
    assert list(cache.values())[0] == 200

def test_lru_with_dupes():
    if False:
        while True:
            i = 10
    SIZE = 2
    lru = LRU(max_size=SIZE)
    for i in [0, 0, 1, 1, 2, 2]:
        lru[i] = i
        assert _test_linkage(lru._anchor, SIZE + 1), 'linked list invalid'

def test_lru_with_dupes_2():
    if False:
        print('Hello World!')
    'From Issue #55, h/t github.com/mt'
    SIZE = 3
    lru = LRU(max_size=SIZE)
    keys = ['A', 'A', 'B', 'A', 'C', 'B', 'D', 'E']
    for (i, k) in enumerate(keys):
        lru[k] = 'HIT'
        assert _test_linkage(lru._anchor, SIZE + 1), 'linked list invalid'
    return

def _test_linkage(dll, max_count=10000, prev_idx=0, next_idx=1):
    if False:
        return 10
    'A function to test basic invariants of doubly-linked lists (with\n    links made of Python lists).\n\n    1. Test that the list is not longer than a certain length\n    2. That the forward links (indicated by `next_idx`) correspond to\n    the backward links (indicated by `prev_idx`).\n\n    The `dll` parameter is the root/anchor link of the list.\n    '
    start = cur = dll
    i = 0
    prev = None
    while 1:
        if i > max_count:
            raise Exception('did not return to anchor link after %r rounds' % max_count)
        if prev is not None and cur is start:
            break
        prev = cur
        cur = cur[next_idx]
        if cur[prev_idx] is not prev:
            raise Exception('prev_idx does not point to prev at i = %r' % i)
        i += 1
    return True

def test_cached_dec():
    if False:
        while True:
            i = 10
    lru = LRU()
    inner_func = CountingCallable()
    func = cached(lru)(inner_func)
    assert inner_func.call_count == 0
    func()
    assert inner_func.call_count == 1
    func()
    assert inner_func.call_count == 1
    func('man door hand hook car door')
    assert inner_func.call_count == 2
    return

def test_unscoped_cached_dec():
    if False:
        print('Hello World!')
    lru = LRU()
    inner_func = CountingCallable()
    func = cached(lru)(inner_func)
    other_inner_func = CountingCallable()
    other_func = cached(lru)(other_inner_func)
    assert inner_func.call_count == 0
    func('a')
    assert inner_func.call_count == 1
    func('a')
    other_func('a')
    assert other_inner_func.call_count == 0
    return

def test_callable_cached_dec():
    if False:
        while True:
            i = 10
    lru = LRU()
    get_lru = lambda : lru
    inner_func = CountingCallable()
    func = cached(get_lru)(inner_func)
    assert inner_func.call_count == 0
    func()
    assert inner_func.call_count == 1
    func()
    assert inner_func.call_count == 1
    lru.clear()
    func()
    assert inner_func.call_count == 2
    func()
    assert inner_func.call_count == 2
    print(repr(func))
    return

def test_cachedmethod():
    if False:
        return 10

    class Car(object):

        def __init__(self, cache=None):
            if False:
                while True:
                    i = 10
            self.h_cache = LRI() if cache is None else cache
            self.door_count = 0
            self.hook_count = 0
            self.hand_count = 0

        @cachedmethod('h_cache')
        def hand(self, *a, **kw):
            if False:
                while True:
                    i = 10
            self.hand_count += 1

        @cachedmethod(lambda obj: obj.h_cache)
        def hook(self, *a, **kw):
            if False:
                return 10
            self.hook_count += 1

        @cachedmethod('h_cache', scoped=False)
        def door(self, *a, **kw):
            if False:
                for i in range(10):
                    print('nop')
            self.door_count += 1
    car = Car()
    assert car.hand_count == 0
    car.hand('h', a='nd')
    assert car.hand_count == 1
    car.hand('h', a='nd')
    assert car.hand_count == 1
    assert car.hook_count == 0
    car.hook()
    assert car.hook_count == 1
    car.hook()
    assert car.hook_count == 1
    lru = LRU()
    car_one = Car(cache=lru)
    assert car_one.door_count == 0
    car_one.door('bob')
    assert car_one.door_count == 1
    car_one.door('bob')
    assert car_one.door_count == 1
    car_two = Car(cache=lru)
    assert car_two.door_count == 0
    car_two.door('bob')
    assert car_two.door_count == 0
    Car.door(Car(), 'bob')
    print(repr(car_two.door))
    print(repr(Car.door))
    return

def test_cachedmethod_maintains_func_abstraction():
    if False:
        while True:
            i = 10
    ABC = ABCMeta('ABC', (object,), {})

    class Car(ABC):

        def __init__(self, cache=None):
            if False:
                return 10
            self.h_cache = LRI() if cache is None else cache
            self.hand_count = 0

        @cachedmethod('h_cache')
        @abstractmethod
        def hand(self, *a, **kw):
            if False:
                print('Hello World!')
            self.hand_count += 1
    with pytest.raises(TypeError):
        Car()

def test_cachedproperty():
    if False:
        for i in range(10):
            print('nop')

    class Proper(object):

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.expensive_func = CountingCallable()

        @cachedproperty
        def useful_attr(self):
            if False:
                i = 10
                return i + 15
            'Useful DocString'
            return self.expensive_func()
    prop = Proper()
    assert prop.expensive_func.call_count == 0
    assert prop.useful_attr == 1
    assert prop.expensive_func.call_count == 1
    assert prop.useful_attr == 1
    assert prop.expensive_func.call_count == 1
    assert Proper.useful_attr.__doc__ == 'Useful DocString'
    prop.useful_attr += 1
    assert prop.useful_attr == 2
    delattr(prop, 'useful_attr')
    assert prop.expensive_func.call_count == 1
    assert prop.useful_attr
    assert prop.expensive_func.call_count == 2
    repr(Proper.useful_attr)

def test_cachedproperty_maintains_func_abstraction():
    if False:
        while True:
            i = 10
    ABC = ABCMeta('ABC', (object,), {})

    class AbstractExpensiveCalculator(ABC):

        @cachedproperty
        @abstractmethod
        def calculate(self):
            if False:
                return 10
            pass
    with pytest.raises(TypeError):
        AbstractExpensiveCalculator()

def test_min_id_map():
    if False:
        while True:
            i = 10
    import sys
    if '__pypy__' in sys.builtin_module_names:
        return
    midm = MinIDMap()

    class Foo(object):

        def __init__(self, val):
            if False:
                return 10
            self.val = val
    ref_wheel = [None, None, None]
    for i in range(1000):
        nxt = Foo(i)
        ref_wheel[i % len(ref_wheel)] = nxt
        assert midm.get(nxt) <= len(ref_wheel)
        if i % 10 == 0:
            midm.drop(nxt)
    assert sorted([f.val for f in list(midm)[:10]]) == list(range(1000 - len(ref_wheel), 1000))
    items = list(midm.iteritems())
    assert isinstance(items[0][0], Foo)
    assert sorted((item[1] for item in items)) == list(range(0, len(ref_wheel)))

def test_threshold_counter():
    if False:
        return 10
    tc = ThresholdCounter(threshold=0.1)
    tc.add(1)
    assert tc.items() == [(1, 1)]
    tc.update([2] * 10)
    assert tc.get(1) == 0
    tc.add(5)
    assert 5 in tc
    assert len(list(tc.elements())) == 11
    assert tc.threshold == 0.1
    assert tc.get_common_count() == 11
    assert tc.get_uncommon_count() == 1
    assert round(tc.get_commonality(), 2) == 0.92
    assert tc.most_common(2) == [(2, 10), (5, 1)]
    assert list(tc.elements()) == [2] * 10 + [5]
    assert tc[2] == 10
    assert len(tc) == 2
    assert sorted(tc.keys()) == [2, 5]
    assert sorted(tc.values()) == [1, 10]
    assert sorted(tc.items()) == [(2, 10), (5, 1)]