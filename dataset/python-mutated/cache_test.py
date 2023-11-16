import pytest
import numpy as np
import vaex.cache
from unittest.mock import MagicMock, call

def passes(df):
    if False:
        for i in range(10):
            print('nop')
    return df.executor.passes if df.is_local() else df.executor.remote_calls

def reset(df):
    if False:
        while True:
            i = 10
    if df.is_local():
        df.executor.passes = 0
    else:
        df.executor.remote_calls = 0

def test_memory():
    if False:
        print('Hello World!')
    with vaex.cache.off():
        assert vaex.cache.cache is None
        with vaex.cache.memory_infinite():
            assert isinstance(vaex.cache.cache, dict)
        assert vaex.cache.cache is None
        vaex.cache.memory_infinite()
        assert isinstance(vaex.cache.cache, dict)

def test_on():
    if False:
        for i in range(10):
            print('nop')
    with vaex.cache.off():
        assert vaex.cache.cache is None
        with vaex.cache.on():
            assert isinstance(vaex.cache.cache, dict)
        assert vaex.cache.cache is None
        vaex.cache.on()
        assert isinstance(vaex.cache.cache, dict)
        vaex.cache.off()
        assert vaex.cache.cache is None
    with vaex.cache.on('memory_infinite,disk'):
        import diskcache
        assert isinstance(vaex.cache.cache, vaex.cache.MultiLevelCache)
        assert isinstance(vaex.cache.cache.maps[0], dict)
        assert isinstance(vaex.cache.cache.maps[1], diskcache.Cache)
    assert not vaex.cache.is_on()

def test_cached_result(df_local):
    if False:
        print('Hello World!')
    with vaex.cache.memory_infinite(clear=True):
        assert vaex.cache.is_on()
        df = df_local._future()
        reset(df)
        sum0 = df.sum('x', delay=True)
        mean0 = df.mean('x', delay=True)
        df.execute()
        assert passes(df) == 1
        sum0 = sum0.get()
        reset(df)
        sum0b = df.sum('x')
        assert sum0 == sum0b
        assert passes(df) == 0
        reset(df)
        df = df[df.x < 4]
        total = 1 + 2 + 3
        sum1_filtered = df.sum('x')
        assert passes(df) == 1
        assert sum1_filtered == total
        reset(df)
        sum1b_filtered = df.sum('x')
        assert passes(df) == 0
        assert sum1b_filtered == total
        df['foo'] = df.x * 2
        sum1b_filtered = df.sum('x')
        assert passes(df) == 0
        assert sum1b_filtered == total

def test_cached_result_array(df_local):
    if False:
        print('Hello World!')
    with vaex.cache.memory_infinite(clear=True):
        assert vaex.cache.is_on()
        df = df_local._future()
        reset(df)
        sum0 = df.sum('x', binby='y', delay=True)
        sum0b = df.sum('x', binby='y', delay=True)
        df.execute()
        assert passes(df) == 2, 'one pass for min/max, one for the aggregation'
        reset(df)
        sum0 = df.sum('x', binby='y')
        assert passes(df) == 0

def test_cache_length_without_messing_up_filter_mask(df_local):
    if False:
        i = 10
        return i + 15
    df = df_local
    with vaex.cache.memory_infinite(clear=True):
        dff = df[df.x < 4]
        passes0 = passes(df)
        len(dff)
        assert passes(df) == passes0 + 1
        dff = df[df.x < 4]
        len(dff)
        assert passes(df) == passes0 + 1
        dffs = dff[1:2]
        assert passes(df) == passes0 + 2

def test_cache_set():
    if False:
        print('Hello World!')
    df = vaex.from_arrays(x=[0, 1, 2, 2])
    with vaex.cache.memory_infinite(clear=True):
        passes0 = passes(df)
        df._set('x')
        assert passes(df) == passes0 + 1
        df._set('x')
        assert passes(df) == passes0 + 1

def test_nunique():
    if False:
        i = 10
        return i + 15
    df = vaex.from_arrays(x=[0, 1, 2, 2])
    with vaex.cache.memory_infinite(clear=True):
        vaex.cache._cache_hit = 0
        vaex.cache._cache_miss = 0
        df.x.nunique()
        assert vaex.cache._cache_miss == 3
        assert vaex.cache._cache_hit == 0
        df.x.nunique()
        assert vaex.cache._cache_miss == 3
        assert vaex.cache._cache_hit == 1

@pytest.mark.parametrize('copy', [True, False])
def test_cache_groupby(copy):
    if False:
        while True:
            i = 10
    df = vaex.from_arrays(x=[0, 1, 2, 2], y=['a', 'b', 'c', 'd'])
    df2 = vaex.from_arrays(x=[0, 1, 2, 2], y=['a', 'b', 'c', 'd'])
    fp = df.fingerprint()
    with vaex.cache.memory_infinite(clear=True):
        passes0 = passes(df)
        df.groupby('x', agg='count', copy=copy)
        assert passes(df) == passes0 + 2
        if copy:
            assert df.fingerprint() == fp
        df.groupby('y', agg='count', copy=copy)
        assert passes(df) == passes0 + 4
        if copy:
            assert df.fingerprint() == fp
        vaex.execution.logger.debug('HERE IT GOES')
        df2.groupby('y', agg='count', copy=copy)
        assert passes(df) == passes0 + 4
        if copy:
            assert df.fingerprint() == fp
        df.groupby('x', agg='count', copy=copy)
        assert passes(df) == passes0 + 4
        df.groupby('y', agg='count', copy=copy)
        assert passes(df) == passes0 + 4
        if copy:
            assert df.fingerprint() == fp
        df.groupby(['x', 'y'], agg='count', copy=copy)
        assert passes(df) == passes0 + 7
        if copy:
            assert df.fingerprint() == fp
        df.groupby(['x', 'y'], agg='count', copy=copy)
        assert passes(df) == passes0 + 7
        if copy:
            assert df.fingerprint() == fp
        df.groupby(['y', 'x'], agg='count', copy=copy)
        assert passes(df) == passes0 + 7 + 3
        if copy:
            assert df.fingerprint() == fp

def test_cache_selections():
    if False:
        i = 10
        return i + 15
    df = vaex.from_arrays(x=[0, 1, 2, 2], y=['a', 'b', 'c', 'd'])
    df2 = vaex.from_arrays(x=[0, 1, 2, 2], y=['a', 'b', 'c', 'd'])
    fp = df.fingerprint()
    with vaex.cache.memory_infinite(clear=True):
        df.executor.passes = 0
        assert df.x.sum(selection=df.y == 'b') == 1
        assert passes(df) == 1
        assert df2.x.sum(selection=df2.y == 'b') == 1
        assert passes(df) == 1
        df.executor.passes = 0
        df.select(df.y == 'c', name='a')
        assert df.x.sum(selection='a') == 2
        assert passes(df) == 1
        df2.select(df2.y == 'c', name='a')
        assert df2.x.sum(selection='a') == 2
        assert passes(df2) == 1
        df['z'] = df.x * 2
        df2['z'] = df2.x * 2
        df.executor.passes = 0
        df.select(df.z == 4, name='a')
        assert df.x.sum(selection='a') == 4
        assert passes(df) == 1
        df2.select(df2.z == 4, name='a')
        assert df2.x.sum(selection='a') == 4
        assert passes(df2) == 1
        df['z'] = df.x * 3
        df2['z'] = df2.x * 1
        df.executor.passes = 0
        df.select(df.z == 3, name='a')
        assert df.x.sum(selection='a') == 1
        assert passes(df) == 1
        df2.select(df2.z == 3, name='a')
        assert df2.x.sum(selection='a') == 0
        assert passes(df2) == 2

def test_multi_level_cache():
    if False:
        for i in range(10):
            print('nop')
    l1 = {}
    l2 = {}
    cache = vaex.cache.MultiLevelCache(l1, l2)
    with pytest.raises(KeyError):
        value = cache['key1']
    assert l1 == {}
    assert l2 == {}
    cache['key1'] = 1
    assert l1 == {'key1': 1}
    assert l2 == {'key1': 1}
    assert cache['key1'] == 1
    del l1['key1']
    assert l1 == {}
    assert l2 == {'key1': 1}
    assert cache['key1'] == 1
    assert l1 == {'key1': 1}
    assert l2 == {'key1': 1}

def test_memoize():
    if False:
        print('Hello World!')
    f1 = f1_mock = MagicMock()
    f1b = f1b_mock = MagicMock()
    f2 = f2_mock = MagicMock()
    f1 = vaex.cache._memoize(f1, key_function=lambda : 'same')
    f1b = vaex.cache._memoize(f1b, key_function=lambda : 'same')
    f2 = vaex.cache._memoize(f2, key_function=lambda : 'different')
    with vaex.cache.memory_infinite(clear=True):
        f1()
        f1_mock.assert_called_once()
        f1b_mock.assert_not_called()
        f2_mock.assert_not_called()
        f1b()
        f1_mock.assert_called_once()
        f1b_mock.assert_not_called()
        f2_mock.assert_not_called()
        f2()
        f1_mock.assert_called_once()
        f1b_mock.assert_not_called()
        f2_mock.assert_called_once()
    with vaex.cache.off():
        f1()
        f1_mock.assert_has_calls([call(), call()])
        f1b_mock.assert_not_called()
        f2_mock.assert_called_once()
        f1b()
        f1_mock.assert_has_calls([call(), call()])
        f1b_mock.assert_called_once()
        f2_mock.assert_called_once()
        f2()
        f1_mock.assert_has_calls([call(), call()])
        f1b_mock.assert_called_once()
        f2_mock.assert_has_calls([call(), call()])

def test_memoize_with_delay():
    if False:
        print('Hello World!')
    df = vaex.from_arrays(x=[0, 1, 2, 2])
    with vaex.cache.memory_infinite(clear=True):
        vaex.cache._cache_hit = 0
        vaex.cache._cache_miss = 0
        value = df.x.nunique(delay=True)
        assert vaex.cache._cache_miss == 2
        assert vaex.cache._cache_hit == 0
        df.execute()
        assert value.get() == 3
        assert vaex.cache._cache_miss == 3
        assert vaex.cache._cache_hit == 0
        value = df.x.nunique()
        assert value == 3
        assert vaex.cache._cache_miss == 3
        assert vaex.cache._cache_hit == 1
        value = df.x.nunique(delay=True)
        assert value.get() == 3
        assert vaex.cache._cache_miss == 3
        assert vaex.cache._cache_hit == 2

def test_value_counts():
    if False:
        for i in range(10):
            print('nop')
    df = vaex.from_arrays(x=[0, 1, 2, 2])
    with vaex.cache.memory_infinite(clear=True):
        vaex.cache._cache_hit = 0
        vaex.cache._cache_miss = 0
        df.x.value_counts()
        assert vaex.cache._cache_miss == 2
        assert vaex.cache._cache_hit == 0
        df.x.value_counts()
        assert vaex.cache._cache_miss == 2
        assert vaex.cache._cache_hit == 1