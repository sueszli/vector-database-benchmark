import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from arctic.serialization.numpy_arrays import FrameConverter, FrametoArraySerializer
from tests.util import assert_frame_equal_

def test_frame_converter():
    if False:
        while True:
            i = 10
    f = FrameConverter()
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    assert_frame_equal_(f.objify(f.docify(df)), df)

def test_with_strings():
    if False:
        for i in range(10):
            print('nop')
    f = FrameConverter()
    df = pd.DataFrame(data={'one': ['a', 'b', 'c']})
    assert_frame_equal_(f.objify(f.docify(df)), df)

def test_with_objects_raises():
    if False:
        print('Hello World!')

    class Example(object):

        def __init__(self, data):
            if False:
                i = 10
                return i + 15
            self.data = data

        def get(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.data
    f = FrameConverter()
    df = pd.DataFrame(data={'one': [Example(444)]})
    with pytest.raises(Exception):
        f.docify(df)

def test_without_index():
    if False:
        print('Hello World!')
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    n = FrametoArraySerializer()
    a = n.serialize(df)
    assert_frame_equal_(df, n.deserialize(a))

def test_with_index():
    if False:
        print('Hello World!')
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    df = df.set_index(['A'])
    n = FrametoArraySerializer()
    a = n.serialize(df)
    assert_frame_equal_(df, n.deserialize(a))

def test_with_nans():
    if False:
        print('Hello World!')
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    df['A'] = np.NaN
    n = FrametoArraySerializer()
    a = n.serialize(df)
    assert_frame_equal_(df, n.deserialize(a))

def test_empty_dataframe():
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame()
    n = FrametoArraySerializer()
    a = n.serialize(df)
    assert_frame_equal(df, n.deserialize(a))

def test_empty_columns():
    if False:
        while True:
            i = 10
    df = pd.DataFrame(data={'A': [], 'B': [], 'C': []})
    n = FrametoArraySerializer()
    a = n.serialize(df)
    assert_frame_equal_(df, n.deserialize(a))

def test_string_cols_with_nans():
    if False:
        while True:
            i = 10
    f = FrameConverter()
    df = pd.DataFrame(data={'one': ['a', 'b', 'c', np.NaN]})
    assert df.equals(f.objify(f.docify(df)))

def test_objify_with_missing_columns():
    if False:
        while True:
            i = 10
    f = FrameConverter()
    df = pd.DataFrame(data={'one': ['a', 'b', 'c', np.NaN]})
    res = f.objify(f.docify(df), columns=['one', 'two'])
    assert res['one'].equals(df['one'])
    assert all(res['two'].isnull())

def test_multi_column_fail():
    if False:
        while True:
            i = 10
    df = pd.DataFrame(data={'A': [1, 2, 3], 'B': [2, 3, 4], 'C': [3, 4, 5]})
    df = df.set_index(['A'])
    n = FrametoArraySerializer()
    a = n.serialize(df)
    with pytest.raises(Exception) as e:
        n.deserialize(a, columns=['A', 'B'])
    assert 'Duplicate' in str(e.value)

def test_dataframe_writable_after_objify():
    if False:
        while True:
            i = 10
    f = FrameConverter()
    df = pd.DataFrame(data={'one': [5, 6, 2]})
    df = f.objify(f.docify(df))
    df['one'] = 7
    assert np.all(df['one'].values == np.array([7, 7, 7]))