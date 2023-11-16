import pytest
from ray.tune.utils.object_cache import _ObjectCache

@pytest.mark.parametrize('eager', [False, True])
def test_no_may_keep_one(eager):
    if False:
        for i in range(10):
            print('nop')
    'Test object caching.\n\n    - After init, no objects are cached (as max cached is 0), except when eager caching\n    - After increasing max to 2, up to 2 objects are cached\n    - Decreasing max objects will evict them on flush\n    '
    cache = _ObjectCache(may_keep_one=eager)
    assert cache.cache_object('A', 1) == eager
    assert cache.num_cached_objects == int(eager)
    cache.increase_max('A', 2)
    if not eager:
        assert cache.cache_object('A', 1)
    assert cache.cache_object('A', 2)
    assert not cache.cache_object('A', 3)
    assert cache.num_cached_objects == 2
    assert not list(cache.flush_cached_objects())
    cache.decrease_max('A', 1)
    assert list(cache.flush_cached_objects()) == [1]
    assert cache.num_cached_objects == 1
    cache.decrease_max('A', 1)
    assert list(cache.flush_cached_objects()) == ([2] if not eager else [])
    assert cache.num_cached_objects == (0 if not eager else 1)

@pytest.mark.parametrize('eager', [False, True])
def test_multi(eager):
    if False:
        return 10
    'Test caching with multiple objects'
    cache = _ObjectCache(may_keep_one=eager)
    assert cache.cache_object('A', 1) == eager
    assert cache.num_cached_objects == int(eager)
    assert not cache.cache_object('B', 5)
    assert cache.num_cached_objects == int(eager)
    cache.increase_max('A', 1)
    cache.increase_max('B', 1)
    assert cache.cache_object('A', 1) != eager
    assert cache.cache_object('B', 5)
    assert not cache.cache_object('A', 2)
    assert not cache.cache_object('B', 6)
    assert cache.num_cached_objects == 2
    cache.decrease_max('A', 1)
    assert list(cache.flush_cached_objects()) == [1]
    cache.decrease_max('B', 1)
    assert list(cache.flush_cached_objects()) == ([5] if not eager else [])
    assert cache.num_cached_objects == (0 if not eager else 1)

def test_multi_eager_other():
    if False:
        i = 10
        return i + 15
    "On eager caching, only cache an object if no other object is expected.\n\n    - Expect up to one cached A object\n    - Try to cache object B --> doesn't get cached\n    - Remove expectation for A object\n    - Try to cache object B --> get's cached\n    "
    cache = _ObjectCache(may_keep_one=True)
    cache.increase_max('A', 1)
    assert not cache.cache_object('B', 2)
    cache.decrease_max('A', 1)
    assert cache.cache_object('B', 3)

@pytest.mark.parametrize('eager', [False, True])
def test_force_all(eager):
    if False:
        i = 10
        return i + 15
    'Assert that force_all=True will always evict all object.'
    cache = _ObjectCache(may_keep_one=eager)
    cache.increase_max('A', 2)
    assert cache.cache_object('A', 1)
    assert cache.cache_object('A', 2)
    assert list(cache.flush_cached_objects(force_all=True)) == [1, 2]
    assert cache.num_cached_objects == 0
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))