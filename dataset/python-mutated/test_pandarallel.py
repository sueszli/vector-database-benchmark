import importlib
import math
import numpy as np
import pandas as pd
import pytest
from pandarallel import pandarallel

@pytest.fixture(params=(1000, 1))
def df_size(request):
    if False:
        print('Hello World!')
    return request.param

@pytest.fixture(params=(False, True))
def progress_bar(request):
    if False:
        print('Hello World!')
    return request.param

@pytest.fixture(params=(None, False))
def use_memory_fs(request):
    if False:
        return 10
    return request.param

@pytest.fixture(params=(RuntimeError, AttributeError, ZeroDivisionError))
def exception(request):
    if False:
        while True:
            i = 10
    return request.param

@pytest.fixture(params=('named', 'anonymous'))
def func_dataframe_apply_axis_0(request):
    if False:
        for i in range(10):
            print('nop')

    def func(x):
        if False:
            i = 10
            return i + 15
        return max(x) - min(x)
    return dict(named=func, anonymous=lambda x: max(x) - min(x))[request.param]

@pytest.fixture(params=('named', 'anonymous'))
def func_dataframe_apply_axis_1(request):
    if False:
        print('Hello World!')

    def func(x):
        if False:
            return 10
        return math.sin(x.a ** 2) + math.sin(x.b ** 2)
    return dict(named=func, anonymous=lambda x: math.sin(x.a ** 2) + math.sin(x.b ** 2))[request.param]

@pytest.fixture(params=('named', 'anonymous'))
def func_dataframe_applymap(request):
    if False:
        for i in range(10):
            print('nop')

    def func(x):
        if False:
            return 10
        return math.sin(x ** 2) - math.cos(x ** 2)
    return dict(named=func, anonymous=lambda x: math.sin(x ** 2) - math.cos(x ** 2))[request.param]

@pytest.fixture(params=('named', 'anonymous'))
def func_series_map(request):
    if False:
        print('Hello World!')

    def func(x):
        if False:
            print('Hello World!')
        return math.log10(math.sqrt(math.exp(x ** 2)))
    return dict(named=func, anonymous=lambda x: math.log10(math.sqrt(math.exp(x ** 2))))[request.param]

@pytest.fixture(params=('named', 'anonymous'))
def func_series_apply(request):
    if False:
        while True:
            i = 10

    def func(x, power, bias=0):
        if False:
            for i in range(10):
                print('nop')
        return math.log10(math.sqrt(math.exp(x ** power))) + bias
    return dict(named=func, anonymous=lambda x, power, bias=0: math.log10(math.sqrt(math.exp(x ** power))) + bias)[request.param]

@pytest.fixture(params=('named', 'anonymous'))
def func_series_rolling_apply(request):
    if False:
        while True:
            i = 10

    def func(x):
        if False:
            return 10
        return x.iloc[0] + x.iloc[1] ** 2 + x.iloc[2] ** 3 + x.iloc[3] ** 4
    return dict(named=func, anonymous=lambda x: x.iloc[0] + x.iloc[1] ** 2 + x.iloc[2] ** 3 + x.iloc[3] ** 4)[request.param]

@pytest.fixture()
def func_dataframe_groupby_apply():
    if False:
        return 10

    def func(df):
        if False:
            while True:
                i = 10
        dum = 0
        for item in df.b:
            dum += math.log10(math.sqrt(math.exp(item ** 2)))
        return dum / len(df.b)
    return func

@pytest.fixture()
def func_dataframe_groupby_apply_complex():
    if False:
        i = 10
        return i + 15

    def func(df):
        if False:
            return 10
        return pd.DataFrame([[df.b.mean(), df.b.min(), df.b.max()]], columns=['b_mean', 'b_min', 'b_max'])
    return func

@pytest.fixture(params=('named', 'anonymous'))
def func_dataframe_groupby_rolling_apply(request):
    if False:
        return 10

    def func(x):
        if False:
            i = 10
            return i + 15
        return x.iloc[0] + x.iloc[1] ** 2 + x.iloc[2] ** 3 + x.iloc[3] ** 4
    return dict(named=func, anonymous=lambda x: x.iloc[0] + x.iloc[1] ** 2 + x.iloc[2] ** 3 + x.iloc[3] ** 4)[request.param]

@pytest.fixture(params=('named', 'anonymous'))
def func_dataframe_groupby_expanding_apply(request):
    if False:
        i = 10
        return i + 15

    def func(x):
        if False:
            return 10
        return x.multiply(pd.Series(range(1, len(x)), dtype='float')).sum()
    return dict(named=func, anonymous=lambda x: x.multiply(pd.Series(range(1, len(x)), dtype='float')).sum())[request.param]

@pytest.fixture(params=('named', 'anonymous'))
def func_dataframe_apply_axis_0_no_reduce(request):
    if False:
        print('Hello World!')

    def func(x):
        if False:
            print('Hello World!')
        return x
    return dict(named=func, anonymous=lambda x: x)[request.param]

@pytest.fixture(params=('named', 'anonymous'))
def func_dataframe_apply_axis_1_no_reduce(request):
    if False:
        print('Hello World!')

    def func(x):
        if False:
            for i in range(10):
                print('nop')
        return x ** 2
    return dict(named=func, anonymous=lambda x: x ** 2)[request.param]

@pytest.fixture
def pandarallel_init(progress_bar, use_memory_fs):
    if False:
        i = 10
        return i + 15
    pandarallel.initialize(progress_bar=progress_bar, use_memory_fs=use_memory_fs, nb_workers=2)

def test_dataframe_apply_invalid_function(pandarallel_init, exception):
    if False:
        return 10

    def f(_):
        if False:
            while True:
                i = 10
        raise exception
    df = pd.DataFrame(dict(a=[1, 2, 3, 4]))
    with pytest.raises(exception):
        df.parallel_apply(f)

def test_dataframe_apply_axis_0(pandarallel_init, func_dataframe_apply_axis_0, df_size):
    if False:
        return 10
    df = pd.DataFrame(dict(a=np.random.randint(1, 8, df_size), b=np.random.rand(df_size), c=np.random.randint(1, 8, df_size), d=np.random.rand(df_size), e=np.random.randint(1, 8, df_size), f=np.random.rand(df_size), g=np.random.randint(1, 8, df_size), h=np.random.rand(df_size)))
    df.index = [item / 10 for item in df.index]
    res = df.apply(func_dataframe_apply_axis_0)
    res_parallel = df.parallel_apply(func_dataframe_apply_axis_0)
    assert res.equals(res_parallel)

def test_dataframe_apply_axis_1(pandarallel_init, func_dataframe_apply_axis_1, df_size):
    if False:
        while True:
            i = 10
    df = pd.DataFrame(dict(a=np.random.randint(1, 8, df_size), b=np.random.rand(df_size)))
    df.index = [item / 10 for item in df.index]
    res = df.apply(func_dataframe_apply_axis_1, axis=1)
    res_parallel = df.parallel_apply(func_dataframe_apply_axis_1, axis=1)
    assert res.equals(res_parallel)

def test_dataframe_apply_invalid_axis(pandarallel_init):
    if False:
        i = 10
        return i + 15
    df = pd.DataFrame(dict(a=[1, 2, 3, 4]))
    with pytest.raises(ValueError):
        df.parallel_apply(lambda x: x, axis='invalid')

def test_dataframe_applymap(pandarallel_init, func_dataframe_applymap, df_size):
    if False:
        print('Hello World!')
    df = pd.DataFrame(dict(a=np.random.randint(1, 8, df_size), b=np.random.rand(df_size)))
    df.index = [item / 10 for item in df.index]
    res = df.applymap(func_dataframe_applymap)
    res_parallel = df.parallel_applymap(func_dataframe_applymap)
    assert res.equals(res_parallel)

def test_series_map(pandarallel_init, func_series_map, df_size):
    if False:
        print('Hello World!')
    df = pd.DataFrame(dict(a=np.random.rand(df_size) + 1))
    res = df.a.map(func_series_map)
    res_parallel = df.a.parallel_map(func_series_map)
    assert res.equals(res_parallel)

def test_series_apply(pandarallel_init, func_series_apply, df_size):
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame(dict(a=np.random.rand(df_size) + 1))
    res = df.a.apply(func_series_apply, args=(2,), bias=3)
    res_parallel = df.a.parallel_apply(func_series_apply, args=(2,), bias=3)
    assert res.equals(res_parallel)

def test_series_rolling_apply(pandarallel_init, func_series_rolling_apply, df_size):
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame(dict(a=np.random.randint(1, 8, df_size), b=list(range(df_size))))
    res = df.b.rolling(4).apply(func_series_rolling_apply, raw=False)
    res_parallel = df.b.rolling(4).parallel_apply(func_series_rolling_apply, raw=False)
    assert res.equals(res_parallel)

def test_dataframe_groupby_apply(pandarallel_init, func_dataframe_groupby_apply, df_size):
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame(dict(a=np.random.randint(1, 8, df_size), b=np.random.rand(df_size), c=np.random.rand(df_size)))
    res = df.groupby('a').apply(func_dataframe_groupby_apply)
    res_parallel = df.groupby('a').parallel_apply(func_dataframe_groupby_apply)
    assert res.equals(res_parallel)
    res = df.groupby(['a']).apply(func_dataframe_groupby_apply)
    res_parallel = df.groupby(['a']).parallel_apply(func_dataframe_groupby_apply)
    assert res.equals(res_parallel)
    res = df.groupby(['a', 'b']).apply(func_dataframe_groupby_apply)
    res_parallel = df.groupby(['a', 'b']).parallel_apply(func_dataframe_groupby_apply)
    assert res.equals(res_parallel)

def test_dataframe_groupby_apply_complex(pandarallel_init, func_dataframe_groupby_apply_complex, df_size):
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame(dict(a=np.random.randint(1, 100, df_size), b=np.random.rand(df_size)))
    res = df.groupby('a').apply(func_dataframe_groupby_apply_complex)
    res_parallel = df.groupby('a').parallel_apply(func_dataframe_groupby_apply_complex)
    res.equals(res_parallel)

def test_dataframe_groupby_rolling_apply(pandarallel_init, func_dataframe_groupby_rolling_apply, df_size):
    if False:
        print('Hello World!')
    df = pd.DataFrame(dict(a=np.random.randint(1, 10, df_size), b=np.random.rand(df_size)))
    res = df.groupby('a').b.rolling(4).apply(func_dataframe_groupby_rolling_apply, raw=False)
    res_parallel = df.groupby('a').b.rolling(4).parallel_apply(func_dataframe_groupby_rolling_apply, raw=False)
    assert res.equals(res_parallel)

def test_dataframe_groupby_expanding_apply(pandarallel_init, func_dataframe_groupby_expanding_apply, df_size):
    if False:
        return 10
    df = pd.DataFrame(dict(a=np.random.randint(1, 10, df_size), b=np.random.rand(df_size)))
    res = df.groupby('a').b.expanding().apply(func_dataframe_groupby_expanding_apply, raw=False)
    res_parallel = df.groupby('a').b.expanding().parallel_apply(func_dataframe_groupby_expanding_apply, raw=False)
    res.equals(res_parallel)

def test_dataframe_axis_0_no_reduction(pandarallel_init, func_dataframe_apply_axis_0_no_reduce, df_size):
    if False:
        return 10
    df = pd.DataFrame(dict(a=np.random.randint(1, 10, df_size), b=np.random.randint(1, 10, df_size), c=np.random.randint(1, 10, df_size)))
    res = df.apply(func_dataframe_apply_axis_0_no_reduce)
    res_parallel = df.parallel_apply(func_dataframe_apply_axis_0_no_reduce)
    assert res.equals(res_parallel)

def test_dataframe_axis_1_no_reduction(pandarallel_init, func_dataframe_apply_axis_1_no_reduce, df_size):
    if False:
        print('Hello World!')
    df = pd.DataFrame(dict(a=np.random.randint(1, 10, df_size), b=np.random.randint(1, 10, df_size), c=np.random.randint(1, 10, df_size)))
    res = df.apply(func_dataframe_apply_axis_1_no_reduce, axis=1)
    res_parallel = df.parallel_apply(func_dataframe_apply_axis_1_no_reduce, axis=1)
    assert res.equals(res_parallel)

def test_memory_fs_root_environment_variable(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setenv('MEMORY_FS_ROOT', '/test')
    from pandarallel import core
    importlib.reload(core)
    assert core.MEMORY_FS_ROOT == '/test'