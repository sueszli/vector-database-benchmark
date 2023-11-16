from common import *

def test_fillna_column(df_local_non_arrow):
    if False:
        i = 10
        return i + 15
    if isinstance(df_local_non_arrow.dataset['obj'], vaex.column.ColumnConcatenatedLazy):
        return
    df = df_local_non_arrow
    df['ok'] = df['obj'].fillna(value='NA')
    assert df.ok.values[5] == 'NA'
    df['obj'] = df['obj'].fillna(value='NA')
    assert df.obj.values[5] == 'NA'

def test_fillna(ds_local):
    if False:
        print('Hello World!')
    df = ds_local
    df_copy = df.copy()
    df_string_filled = df.fillna(value='NA')
    assert df_string_filled.obj.values[5] == 'NA'
    df_filled = df.fillna(value=0)
    assert df_filled.obj.values[5] == 0
    assert df_filled.to_pandas_df(virtual=True).isna().any().any() == False
    assert df_filled.to_pandas_df(virtual=True).isna().any().any() == False
    df_filled = df.fillna(value=10, fill_masked=False)
    assert df_filled.n.values[6] == 10.0
    assert df_filled.nm.values[6] == 10.0
    df_filled = df.fillna(value=-15, fill_nan=False)
    assert df_filled.m.values[7] == -15.0
    assert df_filled.nm.values[7] == -15.0
    assert df_filled.mi.values[7] == -15.0
    df_filled = df.fillna(value=-11, column_names=['nm', 'mi'])
    assert df_filled.to_pandas_df(virtual=True).isna().any().any() == True
    assert df_filled.to_pandas_df(column_names=['nm', 'mi']).isna().any().any() == False
    state = df_filled.state_get()
    df_copy.state_set(state)
    np.testing.assert_array_equal(df_copy['nm'].values, df_filled['nm'].values)
    np.testing.assert_array_equal(df_copy['mi'].values, df_filled['mi'].values)

def test_fillna_virtual():
    if False:
        return 10
    x = [1, 2, 3, 5, np.nan, -1, -7, 10]
    df = vaex.from_arrays(x=x)
    df['r'] = np.log(df.x)
    df['r'] = df.r.fillna(value=3735928559)
    np.testing.assert_almost_equal(df.r.tolist()[:4], [0.0, 0.6931471805599453, 1.0986122886681098, 1.6094379124341003])
    assert df.r.tolist()[4:7] == [3735928559, 3735928559, 3735928559]

def test_fillna_missing():
    if False:
        i = 10
        return i + 15
    x = np.array(['A', 'B', -1, 0, 2, '', '', None, None, None, np.nan, np.nan, np.nan, np.nan])
    df = vaex.from_arrays(x=x)
    assert df.x.fillna(value=-5).tolist() == ['A', 'B', -1, 0, 2, '', '', -5, -5, -5, -5, -5, -5, -5]

def test_fillmissing():
    if False:
        for i in range(10):
            print('nop')
    s = vaex.string_column(['aap', None, 'noot', 'mies'])
    o = ['aap', None, False, np.nan]
    x = np.arange(4, dtype=np.float64)
    x[2] = x[3] = np.nan
    m = np.ma.array(x, mask=[0, 1, 0, 1])
    df = vaex.from_arrays(x=x, m=m, s=s, o=o)
    x = df.x.fillmissing(9).tolist()
    assert 9 not in x
    assert np.any(np.isnan(x)), 'nan is not a missing value'
    m = df.m.fillmissing(9).tolist()
    assert m[:2] == [0, 9]
    assert np.isnan(m[2])
    assert m[3] == 9
    assert df.s.fillmissing('kees').tolist() == ['aap', 'kees', 'noot', 'mies']
    assert df.o.fillmissing({'a': 1}).tolist()[:3] == ['aap', {'a': 1}, False]
    assert np.isnan(df.o.fillmissing([1]).tolist()[3])

def test_fillmissing_upcast(df_factory):
    if False:
        i = 10
        return i + 15
    df = df_factory(x=[1, 2, None])
    df['x'] = df['x'].astype('int8')
    assert df.x.dtype == np.dtype('int8')
    df['y'] = df['x'].fillmissing(127)
    assert df.y.dtype == np.dtype('int8')
    df['z'] = df['x'].fillmissing(128)
    assert df.z.dtype != np.dtype('int8')
    assert df.z.dtype == np.dtype('int16')
    df['z'] = df['x'].fillmissing(-129)
    assert df.z.dtype != np.dtype('int8')
    assert df.z.dtype == np.dtype('int16')
    df = df_factory(x=[1, 2, None])
    df['x'] = df['x'].astype('uint8')
    assert df.x.dtype == np.dtype('uint8')
    df['z'] = df['x'].fillmissing(256)
    assert df.z.dtype != np.dtype('int8')
    assert df.z.dtype == np.dtype('uint16')
    df['z'] = df['x'].fillmissing(-129)
    assert df.z.dtype != np.dtype('int8')
    assert df.z.dtype == np.dtype('int16')

def test_fillnan():
    if False:
        while True:
            i = 10
    s = vaex.string_column(['aap', None, 'noot', 'mies'])
    o = ['aap', None, False, np.nan]
    x = np.arange(4, dtype=np.float64)
    x[2] = x[3] = np.nan
    m = np.ma.array(x, mask=[0, 1, 0, 1])
    df = vaex.from_arrays(x=x, m=m, s=s, o=o)
    x = df.x.fillnan(9).tolist()
    assert x == [0, 1, 9, 9]
    m = df.m.fillnan(9).tolist()
    assert m == [0, None, 9, None]
    assert df.s.fillnan('kees').tolist() == ['aap', None, 'noot', 'mies']
    assert df.o.fillnan({'a': 1}).tolist() == ['aap', None, False, {'a': 1}]

def test_fillna():
    if False:
        print('Hello World!')
    s = vaex.string_column(['aap', None, 'noot', 'mies'])
    o = ['aap', None, False, np.nan]
    x = np.arange(4, dtype=np.float64)
    x[2] = x[3] = np.nan
    m = np.ma.array(x, mask=[0, 1, 0, 1])
    df = vaex.from_arrays(x=x, m=m, s=s, o=o)
    x = df.x.fillna(9).tolist()
    assert x == [0, 1, 9, 9]
    m = df.m.fillna(9).tolist()
    assert m == [0, 9, 9, 9]
    assert df.s.fillna('kees').tolist() == ['aap', 'kees', 'noot', 'mies']
    assert df.o.fillna({'a': 1}).tolist() == ['aap', {'a': 1}, False, {'a': 1}]

def test_fillna_array():
    if False:
        return 10
    x = np.array([1, 2, 3, np.nan])
    df = vaex.from_arrays(x=x)
    df['x_2'] = df.x.fillna(np.array(2.0))
    assert df.x_2.tolist() == [1, 2, 3, 2]

def test_fillna_dataframe(df_factory):
    if False:
        while True:
            i = 10
    x = np.array([3, 1, np.nan, 10, np.nan])
    y = np.array([None, 1, True, '10street', np.nan], dtype='object')
    z = np.ma.array(data=[5, 7, 3, 1, -10], mask=[False, False, True, False, True])
    df = df_factory(x=x, y=y, z=z)
    df_filled = df.fillna(value=-1)
    assert df_filled.x.tolist() == [3, 1, -1, 10, -1]
    assert df_filled.y.tolist() == [-1, 1, True, '10street', -1]
    assert df_filled.z.tolist() == [5, 7, -1, 1, -1]

def test_fillna_string_dtype():
    if False:
        print('Hello World!')
    name = ['Maria', 'Adam', None, None, 'Dan']
    age = [28, 15, 34, 55, 41]
    weight = [np.nan, np.nan, 77.5, 65, 95]
    df = vaex.from_arrays(name=name, age=age, weight=weight)
    assert df['name'].is_string()
    df['name'] = df['name'].fillna('missing')
    assert df['name'].is_string()

def test_fillna_num_to_string_dtype():
    if False:
        while True:
            i = 10
    inp = vaex.from_arrays(int1=np.ma.array([1, 0], mask=[0, 1], dtype=int), float1=np.ma.array([3.14, 0], mask=[0, 1], dtype=float))
    inp['int1'] = inp['int1'].astype('string')
    inp['float1'] = inp['float1'].astype('string')
    assert inp['int1'].is_string
    assert inp['float1'].is_string
    assert inp['int1'].fillna('').tolist() == ['1', '']
    assert inp['float1'].fillna('').tolist() == ['3.140000', '']