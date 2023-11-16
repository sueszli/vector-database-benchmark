import pytest
import numpy as np
from common import *
from random import random

def test_value_counts():
    if False:
        while True:
            i = 10
    ds = create_base_ds()
    assert len(ds.x.value_counts()) == 21
    assert len(ds.y.value_counts()) == 19
    assert len(ds.m.value_counts(dropmissing=True)) == 19
    assert len(ds.m.value_counts()) == 20
    assert len(ds.n.value_counts(dropna=False)) == 20
    assert len(ds.n.value_counts(dropna=True)) == 19
    assert len(ds.nm.value_counts(dropnan=True, dropmissing=True)) == 21 - 4
    assert len(ds.nm.value_counts(dropnan=True, dropmissing=False)) == 21 - 3
    assert len(ds.nm.value_counts(dropna=False, dropmissing=True)) == 21 - 3
    assert len(ds.nm.value_counts(dropna=False, dropmissing=False)) == 21 - 2
    assert len(ds.mi.value_counts(dropmissing=True)) == 21 - 2
    assert len(ds.mi.value_counts(dropmissing=False)) == 21 - 1
    v_counts_name = ds['name'].value_counts()
    v_counts_name_arrow = ds.name_arrow.value_counts()
    assert np.all(v_counts_name == v_counts_name_arrow)

def test_value_counts_object():
    if False:
        i = 10
        return i + 15
    ds = create_base_ds()
    assert len(ds.obj.value_counts(dropmissing=True)) == 17
    assert len(ds.obj.value_counts(dropmissing=False)) == 18

@pytest.mark.parametrize('dropna', [True, False])
def test_value_counts_with_pandas(ds_local, dropna):
    if False:
        for i in range(10):
            print('nop')
    ds = ds_local
    df = ds.to_pandas_df()
    assert df.x.value_counts(dropna=dropna).values.tolist() == ds.x.value_counts(dropna=dropna).values.tolist()

def test_value_counts_simple():
    if False:
        print('Hello World!')
    x = np.array([0, 1, 1, 2, 2, 2, np.nan])
    y = np.ma.array(x, mask=[True, True, False, False, False, False, False])
    s = np.array(list(map(str, x)))
    ds = vaex.from_arrays(x=x, y=y, s=s)
    df = ds.to_pandas_df()
    assert ds.x.value_counts(dropna=True, ascending=True).values.tolist() == [1, 2, 3]
    assert ds.x.value_counts(dropna=False, ascending=True).values.tolist() == [1, 1, 2, 3]
    assert set(ds.s.value_counts(dropna=True, ascending=True).index.tolist()) == {'0.0', 'nan', '1.0', '2.0'}
    assert set(ds.s.value_counts(dropna=True, ascending=True).values.tolist()) == {1, 1.0, 2, 3}
    assert set(ds.y.value_counts(dropna=True, ascending=True).index.tolist()) == {1, 2}
    assert set(ds.y.value_counts(dropna=True, ascending=True).values.tolist()) == {1, 3}
    assert ds.y.value_counts(dropna=False, dropmissing=True, ascending=True).values.tolist() == [1, 1, 3]
    assert ds.y.value_counts(dropna=False, dropmissing=False, ascending=True).values.tolist() == [2, 1, 1, 3]
    assert set(df.x.value_counts(dropna=False).values.tolist()) == set(ds.x.value_counts(dropna=False).values.tolist())
    assert set(df.x.value_counts(dropna=True).values.tolist()) == set(ds.x.value_counts(dropna=True).values.tolist())

def test_value_counts_object_missing():
    if False:
        print('Hello World!')
    x = np.array([None, 'A', 'B', -1, 0, 2, '', '', None, None, None, np.nan, np.nan])
    df = vaex.from_arrays(x=x)
    assert len(df.x.value_counts(dropnan=False, dropmissing=False)) == 8
    assert len(df.x.value_counts(dropnan=True, dropmissing=True)) == 6

def test_value_counts_masked_str():
    if False:
        for i in range(10):
            print('nop')
    x = np.ma.MaskedArray(data=['A', 'A', 'A', 'B', 'B', 'B', '', '', ''], mask=[False, True, False, False, True, True, False, True, False])
    df = vaex.from_arrays(x=x)
    value_counts = df.x.value_counts()
    assert len(value_counts) == 4
    assert value_counts['A'] == 2
    assert value_counts['B'] == 1
    assert value_counts[''] == 2
    assert value_counts['missing'] == 4
    value_counts = df.x.value_counts(dropmissing=True)
    assert len(value_counts) == 3
    assert value_counts['A'] == 2
    assert value_counts['B'] == 1
    assert value_counts[''] == 2
    value_counts = df.x.value_counts(dropna=True)
    assert len(value_counts) == 3
    assert value_counts['A'] == 2
    assert value_counts['B'] == 1
    assert value_counts[''] == 2

def test_value_counts_add_strings():
    if False:
        i = 10
        return i + 15
    x = ['car', 'car', 'boat']
    y = ['red', 'red', 'blue']
    df = vaex.from_arrays(x=x, y=y)
    df['z'] = df.x + '-' + df.y
    value_counts = df.z.value_counts()
    assert list(value_counts.index) == ['car-red', 'boat-blue']
    assert value_counts.values.tolist() == [2, 1]

def test_value_counts_list(df_types):
    if False:
        for i in range(10):
            print('nop')
    df = df_types
    vc = df.string_list.value_counts()
    assert vc['aap'] == 2
    assert vc['mies'] == 1
    vc = df.int_list.value_counts()
    assert vc[1] == 2
    assert vc[2] == 1

def test_value_counts_small_chunk_size(buffer_size):
    if False:
        for i in range(10):
            print('nop')
    df = vaex.datasets.iris()
    with buffer_size(df, 3):
        result = df[df.petal_width > 1].class_.value_counts()
        assert result.tolist() == [50, 43]

def test_value_counts_chunked_array():
    if False:
        while True:
            i = 10
    df = vaex.from_arrays(id=list(range(1000)), text=[f'some text here {random()} text {random()} text {random()}' for _ in range(1000)])
    x = df.text.values
    df['text'] = pa.chunked_array([x[:100], x[100:500], x[500:]])
    res = df.text.str.split(' ').value_counts()
    assert list(res.items())[0] == ('text', 3000)

@pytest.mark.parametrize('high', [100, 1000, 10000, 100000])
def test_value_counts_high_cardinality(high):
    if False:
        while True:
            i = 10
    x = np.random.randint(low=0, high=high, size=100000)
    s = [str(i) for i in x]
    df = vaex.from_arrays(x=x, s=s)
    assert df.x.value_counts().sum() == 100000
    assert df.s.value_counts().sum() == 100000