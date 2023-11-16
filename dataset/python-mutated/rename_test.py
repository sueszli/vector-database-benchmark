from common import *

def test_rename(ds_filtered):
    if False:
        return 10
    ds = ds_filtered
    ds['r'] = ds.x
    ds['q'] = np.exp(ds.x)
    qvalues = ds.q.values.tolist()
    xvalues = ds.x.tolist()
    qexpr = ds.q.expand().expression
    ds['x'] = ds.y
    assert ds.r.values.tolist() == xvalues
    assert ds.q.values.tolist() == qvalues

def test_rename_existing_expression(df_local):
    if False:
        i = 10
        return i + 15
    df = df_local
    x_old = df['x']
    xvalues = x_old.tolist()
    df['x'] = df.x * 2
    assert x_old.tolist() == xvalues

def test_reassign_virtual(ds_local):
    if False:
        i = 10
        return i + 15
    df = ds_local
    x = df.x.to_numpy()
    df['r'] = df.x + 1
    assert df.r.tolist() == (x + 1).tolist()
    df['r'] = df.r + 1
    assert df.r.tolist() == (x + 2).tolist()

def test_reassign_column(ds_filtered):
    if False:
        return 10
    df = ds_filtered.extract()
    x = df.x.to_numpy()
    df['r'] = (df.x + 1).evaluate()
    df['r'] = (df.r + 1).evaluate()
    assert df.r.tolist() == (x + 2).tolist()

def test_rename_state_transfer():
    if False:
        i = 10
        return i + 15
    ds = vaex.from_scalars(x=3, y=4)
    ds['r'] = (ds.x ** 2 + ds.y ** 2) ** 0.5
    ds['x'] = ds.x + 1
    ds['q'] = ds.x + 10
    assert ds.r.tolist() == [5]
    assert ds.q.tolist() == [14]
    ds2 = vaex.from_scalars(x=3, y=4)
    ds2.state_set(ds.state_get())
    assert ds2.r.tolist() == [5]
    assert ds2.q.tolist() == [14]

def test_rename_access(ds_local):
    if False:
        print('Hello World!')
    df = ds_local
    df.rename(name='x', new_name='x2')
    assert df['x2'].tolist() == df.x2.tolist()

def test_rename_twice(ds_local):
    if False:
        for i in range(10):
            print('nop')
    df = ds_local
    df['z'] = df.x + df.y
    zvalues = df.z.values
    df.rename('x', 'xx')
    assert zvalues.tolist() == df.z.tolist()
    df.rename('xx', 'xxx')
    assert zvalues.tolist() == df.z.tolist()

def test_rename_order(ds_local):
    if False:
        for i in range(10):
            print('nop')
    df = ds_local[['x', 'y']]
    assert df.get_column_names() == ['x', 'y']
    df.rename('x', 'xx')
    assert df.get_column_names() == ['xx', 'y']

def test_rename_aliased():
    if False:
        i = 10
        return i + 15
    df = vaex.from_dict({'1': [1, 2], '#': [2, 3]})
    df['x'] = df['1'] + df['#']
    assert df.x.tolist() == [3, 5]
    assert df['#'].tolist() == [2, 3]
    df.rename('#', '$')
    assert df['$'].tolist() == [2, 3]
    df['x'] = df['1'] + df['$']
    assert df.x.tolist() == [3, 5]
    df['$'] = df['$'] * 2
    assert df.x.tolist() == [3, 5]
    df['x'] = df['1'] + df['$']
    assert df.x.tolist() == [5, 8]

def test_rename_expression():
    if False:
        while True:
            i = 10
    df = vaex.from_dict({'1': [1, 2], '#': [2, 3]})
    expr = vaex.expression.Expression(df, "df['1']")
    assert expr.tolist() == [1, 2]
    expr2 = expr._rename('1', '#')
    assert expr2.tolist() == [2, 3]

def test_rename_function():
    if False:
        for i in range(10):
            print('nop')
    df = vaex.from_arrays(x=[1, 2], y=[2, 4])

    def add(a, b):
        if False:
            print('Hello World!')
        return a + b
    df.add_function('add', add)
    assert 'add' in df.functions
    df['z'] = df.func.add(df.x, df.y)
    assert df.z.tolist() == [3, 6]
    df.rename('add', 'add_test')
    assert df.z.tolist() == [3, 6]
    assert 'add' not in df.functions