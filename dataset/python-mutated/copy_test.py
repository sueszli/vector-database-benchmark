from common import *
import pytest

def test_copy(df):
    if False:
        return 10
    df = df
    df['v'] = df.x + 1
    df.add_variable('myvar', 2)
    dfc = df.copy()
    assert 'x' in dfc.get_column_names()
    assert 'v' in dfc.get_column_names()
    assert 'v' in dfc.virtual_columns
    assert 'myvar' in dfc.variables
    dfc.x.values

def test_non_existing_column(df_local):
    if False:
        for i in range(10):
            print('nop')
    df = df_local
    with pytest.raises(NameError, match='.*Did you.*'):
        df.copy(column_names=['x', 'x_'])

def test_copy_alias(df_local):
    if False:
        for i in range(10):
            print('nop')
    df = df_local
    df['alias'] = df.x
    dfc = df.copy(['alias'])
    assert set(dfc.get_column_names(hidden=True)) == {'alias', '__x'}

def test_copy_dependencies():
    if False:
        for i in range(10):
            print('nop')
    df = vaex.from_scalars(x=1)
    df['y'] = df.x + 1
    df2 = df.copy(['y'])
    assert df2.get_column_names(hidden=True) == ['y', '__x']

def test_copy_dependencies_invalid_identifier():
    if False:
        print('Hello World!')
    df = vaex.from_dict({'#': [1]})
    df['y'] = df['#'] + 1
    df2 = df.copy(['y'])
    assert df2.get_column_names(hidden=True) == ['y', '__#']
    df['$'] = df['#'] + df['y']
    df['z'] = df['y'] + df['$']
    df2 = df.copy(['z'])
    assert set(df2.get_column_names(hidden=True)) == {'z', '__#', '__$', '__y'}
    df['@'] = df['y'] + df['$']
    df2 = df.copy(["df['@'] * 2"])
    assert set(df2.get_column_names(hidden=True)) == {"df['@'] * 2", '__@', '__#', '__$', '__y'}

def test_copy_filter():
    if False:
        for i in range(10):
            print('nop')
    df = vaex.from_arrays(x=[0, 1, 2], z=[2, 3, 4])
    df['y'] = df.x + 1
    dff = df[df.x < 2]
    assert dff.x.tolist() == [0, 1]
    assert dff.y.tolist() == [1, 2]
    dffy = dff[['z']]
    assert dffy.z.tolist() == [2, 3]
    assert dffy.get_column_names() == ['z']

def test_copy_filter_boolean_column():
    if False:
        for i in range(10):
            print('nop')
    df = vaex.from_scalars(x=1, ok=True)
    dff = df[df.ok]
    assert dff.get_column_names(hidden=True) == ['x', 'ok']
    dff2 = dff[['x']]
    assert dff2.get_column_names(hidden=True) == ['x', '__ok']

def test_copy_selection():
    if False:
        while True:
            i = 10
    df = vaex.from_arrays(x=[0, 1, 2], z=[2, 3, 4])
    df['y'] = df.x + 1
    df.select(df.x < 2, name='selection_a')
    dfc = df.copy(['z'])
    dfc._invalidate_caches()
    assert dfc.z.sum(selection='selection_a') == 5
    df = vaex.from_arrays(x=[0, 1, 2], z=[2, 3, 4])
    df['y'] = df.x + 1
    df.select(df.y < 3, name='selection_a')
    dfc = df.copy(['z'])
    dfc._invalidate_caches()
    assert dfc.z.sum(selection='selection_a') == 5

def test_copy_tree_shake():
    if False:
        for i in range(10):
            print('nop')
    df = vaex.from_arrays(x=[0, 1, 2], y=[2, 3, 4])
    df.add_variable('t', 1)
    assert 't' in df.copy().variables
    df.add_variable('t', 1)
    assert 't' not in df.copy(treeshake=True).variables
    df.add_virtual_column('z', 'x*t')
    assert 't' in df.copy().variables
    assert 't' in df.copy(treeshake=True).variables
    assert 't' in df.copy(['y']).variables
    assert 't' not in df.copy(['y'], treeshake=True).variables