from common import *

def test_first_no_data(df_filtered):
    if False:
        while True:
            i = 10
    df = df_filtered
    assert df.first([df.y, df.y], df.x * 1, selection='x > 100').tolist() == [None, None]

def test_first(df_filtered):
    if False:
        for i in range(10):
            print('nop')
    df = df_filtered
    with small_buffer(df, 3):
        assert df.first(df.y, df.x * 1).tolist() == 0
        assert df.first(df.y, df.x * 1, binby=[df.x], limits=[0, 10], shape=2).tolist() == [0, 5 ** 2]
        assert df.first(df.y, -df.x, binby=[df.x], limits=[0, 10], shape=2).tolist() == [4 ** 2, 9 ** 2]
        assert df.first(df.y, -df.x, binby=[df.x, df.x + 5], limits=[[0, 10], [5, 15]], shape=[2, 1]).tolist() == [[4 ** 2], [9 ** 2]]
        assert df.first([df.y, df.y], df.x * 1).tolist() == [0, 0]
        assert df.first([df.y, df.y], df.x * 1, selection='x > 100').tolist() == [None, None]

def test_last(df_filtered):
    if False:
        while True:
            i = 10
    df = df_filtered
    with small_buffer(df, 3):
        assert df.last(df.y, df.x * 1).tolist() == 81
        assert df.last(df.y, df.x * 1, binby=[df.x], limits=[0, 10], shape=2).tolist() == [4 ** 2, 9 ** 2]
        assert df.last(df.y, -df.x, binby=[df.x], limits=[0, 10], shape=2).tolist() == [0, 25]
        assert df.last(df.y, -df.x, binby=[df.x, df.x + 5], limits=[[0, 10], [5, 15]], shape=[2, 1]).tolist() == [[0], [25]]
        assert df.last(df.y, -df.x, binby=[df.x, df.x + 5], limits=[[0, 10], [5, 15]], shape=[2, 1]).tolist() == [[0], [25]]
        assert df.last([df.y, df.y], df.x * 1).tolist() == [9 ** 2, 9 ** 2]
        assert df.last(df.y, -df.x, binby=[df.x, df.x + 5], limits=[[0, 10], [5, 15]], shape=[2, 1]).tolist() == [[0], [25]]

@pytest.mark.parametrize('dtype1', ['float64', 'int32'])
@pytest.mark.parametrize('dtype2', ['float32', 'int16'])
def test_first_mixed(dtype1, dtype2):
    if False:
        print('Hello World!')
    x = np.arange(10, dtype=dtype1)
    y = (x ** 2).astype(dtype=dtype2)
    df = vaex.from_arrays(x=x, y=y)
    values = df.first(df.y, -df.x, binby=[df.x], limits=[0, 10], shape=2)
    assert values.tolist() == [4 ** 2, 9 ** 2]
    assert values.dtype == dtype2
    values = df.first(df.y, binby=[df.x], limits=[0, 10], shape=2)
    assert values.tolist() == [0, 5 ** 2]
    assert values.dtype == dtype2
    values = df.last(df.y, df.x, binby=[df.x], limits=[0, 10], shape=2)
    assert values.tolist() == [4 ** 2, 9 ** 2]
    assert values.dtype == dtype2
    values = df.last(df.y, binby=[df.x], limits=[0, 10], shape=2)
    assert values.tolist() == [4 ** 2, 9 ** 2]
    assert values.dtype == dtype2

def test_first_groupby_agg():
    if False:
        print('Hello World!')
    d = {'ex': [0, 0, 0, 1, 1, 1, 2, 2], 'why': [1, 2, 3, 4, 5, 6, 7, 8], 'zet': [4, 1, 3, 2, 7, 0, 1, 1], 'word': ['yes', 'no', 'foo', 'bar', 'NL', 'MK', '?!', 'other']}
    df = vaex.from_dict(d)
    result = df.groupby('ex', sort=True).agg({'f': vaex.agg.first('why'), 'l': vaex.agg.last('why'), 'fo': vaex.agg.first('why', order_expression='zet'), 'lo': vaex.agg.last(df.why, order_expression=df.zet)})
    assert result.ex.tolist() == [0, 1, 2]
    assert result.f.tolist() == [1, 4, 7]
    assert result.l.tolist() == [3, 6, 8]
    assert result.fo.tolist() == [2, 6, 7]
    assert result.lo.tolist() == [1, 5, 7]

def test_first_missing():
    if False:
        while True:
            i = 10
    d = {'x': [0, 0, 0, 1, 1, 1, 2, 2]}
    df = vaex.from_dict(d)
    assert df.first('x', selection=[None, 'x>0']).tolist() == [0, 1]

def test_first_selection():
    if False:
        for i in range(10):
            print('nop')
    d = {'x': [0, 0, 0, 1, 1, 1, 2, 2], 'z': [4, 1, 3, 2, 7, 0, 1, 1]}
    df = vaex.from_dict(d)
    assert df.first('x', selection=[None, 'x>0']).tolist() == [0, 1]
    assert df.first('x', order_expression='x', selection=[None, 'x>0']).tolist() == [0, 1]
    assert df.first('z', order_expression='x', selection=[None, 'x>0']).tolist() == [4, 2]