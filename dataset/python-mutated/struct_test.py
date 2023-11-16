"""This module contains tests regarding struct array support.

"""
import vaex
import pyarrow as pa
import pytest

@pytest.fixture
def df():
    if False:
        i = 10
        return i + 15
    arrays = [[1, 2, 3], ['a', 'b', 'c'], [4, 5, 6]]
    names = ['col1', 'col2', 'col3']
    array = pa.StructArray.from_arrays(arrays=arrays, names=names)
    return vaex.from_arrays(array=array, integer=[8, 9, 10])

@pytest.fixture
def df_duplicated():
    if False:
        print('Hello World!')
    'Contains a struct with duplicated labels.'
    arrays = [[1, 2, 3], ['a', 'b', 'c'], [4, 5, 6]]
    names = ['col1', 'col1', 'col3']
    array = pa.StructArray.from_arrays(arrays=arrays, names=names)
    return vaex.from_arrays(array=array, integer=[8, 9, 10])

def test_struct_get_label(df):
    if False:
        while True:
            i = 10
    expr = df.array.struct.get('col1')
    assert expr.tolist() == [1, 2, 3]
    expr = df.array.struct.get('col2')
    assert expr.tolist() == ['a', 'b', 'c']

def test_struct_get_label_duplicated_raise(df_duplicated):
    if False:
        print('Hello World!')
    with pytest.raises(LookupError):
        df_duplicated.array.struct.get('col1').tolist()

def test_struct_get_label_duplicated_raise_bracket_notation(df_duplicated):
    if False:
        while True:
            i = 10
    with pytest.raises(LookupError):
        df_duplicated.array.struct['col1'].tolist()

def test_struct_get_index(df):
    if False:
        return 10
    expr = df.array.struct.get(0)
    assert expr.tolist() == [1, 2, 3]
    expr = df.array.struct.get(1)
    assert expr.tolist() == ['a', 'b', 'c']

def test_struct_get_index_duplicated(df_duplicated):
    if False:
        print('Hello World!')
    expr = df_duplicated.array.struct.get(0)
    assert expr.tolist() == [1, 2, 3]
    expr = df_duplicated.array.struct.get(1)
    assert expr.tolist() == ['a', 'b', 'c']

def test_struct_get_getitem_notation(df):
    if False:
        return 10
    assert df.array[:, 'col1'].tolist() == [1, 2, 3]
    assert df.array[:, 'col2'].tolist() == ['a', 'b', 'c']
    assert df.array[:, 0].tolist() == [1, 2, 3]
    assert df.array[:, 1].tolist() == ['a', 'b', 'c']

def test_struct_get_invalid_field(df):
    if False:
        print('Hello World!')
    with pytest.raises(LookupError):
        df.array.struct.get('doesNotExist').tolist()

def test_struct_get_invalid_dtype(df):
    if False:
        return 10
    'Ensure that struct function is only applied to correct dtype.'
    with pytest.raises(TypeError):
        df.integer.struct.get('col1').tolist()
    with pytest.raises(TypeError):
        df.integer.struct.get(0).tolist()

def test_struct_project_label(df):
    if False:
        i = 10
        return i + 15
    expr = df.array.struct.project(['col3', 'col1'])
    assert expr.tolist() == [{'col3': 4, 'col1': 1}, {'col3': 5, 'col1': 2}, {'col3': 6, 'col1': 3}]

def test_struct_project_index(df):
    if False:
        while True:
            i = 10
    expr = df.array.struct.project([2, 0])
    assert expr.tolist() == [{'col3': 4, 'col1': 1}, {'col3': 5, 'col1': 2}, {'col3': 6, 'col1': 3}]

def test_struct_project_mixed_label_index(df):
    if False:
        return 10
    expr = df.array.struct.project(['col3', 0])
    assert expr.tolist() == [{'col3': 4, 'col1': 1}, {'col3': 5, 'col1': 2}, {'col3': 6, 'col1': 3}]

def test_struct_project_getitem_notation(df):
    if False:
        while True:
            i = 10
    assert df.array[:, ['col3', 'col1']].tolist() == [{'col3': 4, 'col1': 1}, {'col3': 5, 'col1': 2}, {'col3': 6, 'col1': 3}]
    assert df.array[:, [2, 0]].tolist() == [{'col3': 4, 'col1': 1}, {'col3': 5, 'col1': 2}, {'col3': 6, 'col1': 3}]

def test_struct_project_invalid_field(df):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(LookupError):
        df.array.struct.project(['doesNotExist']).tolist()

def test_struct_project_invalid_dtype(df):
    if False:
        return 10
    'Ensure that struct function is only applied to correct dtype.'
    with pytest.raises(TypeError):
        df.integer.struct.project(['col1']).tolist()
    with pytest.raises(TypeError):
        df.integer.struct.project([0]).tolist()

def test_struct_keys(df):
    if False:
        return 10
    assert df.array.struct.keys() == ['col1', 'col2', 'col3']

def test_struct_keys_duplicated(df_duplicated):
    if False:
        print('Hello World!')
    assert df_duplicated.array.struct.keys() == ['col1', 'col1', 'col3']

def test_struct_keys_invalid_dtypes(df):
    if False:
        for i in range(10):
            print('nop')
    'Ensure that struct function is only applied to correct dtype.'
    with pytest.raises(TypeError):
        df.integer.struct.keys()

def test_struct_values(df):
    if False:
        return 10
    assert df.array.struct.values() == [df.array.struct[0], df.array.struct[1], df.array.struct[2]]

def test_struct_values_duplicated(df_duplicated):
    if False:
        return 10
    assert df_duplicated.array.struct.values() == [df_duplicated.array.struct[0], df_duplicated.array.struct[1], df_duplicated.array.struct[2]]

def test_struct_values_invalid_dtypes(df):
    if False:
        for i in range(10):
            print('nop')
    'Ensure that struct function is only applied to correct dtype.'
    with pytest.raises(TypeError):
        df.integer.struct.values()

def test_struct_items(df):
    if False:
        return 10
    items = [('col1', df.array.struct[0]), ('col2', df.array.struct[1]), ('col3', df.array.struct[2])]
    assert df.array.struct.items() == items

def test_struct_items_duplicated(df_duplicated):
    if False:
        i = 10
        return i + 15
    items = [('col1', df_duplicated.array.struct[0]), ('col1', df_duplicated.array.struct[1]), ('col3', df_duplicated.array.struct[2])]
    assert df_duplicated.array.struct.items() == items

def test_struct_items_invalid_dtypes(df):
    if False:
        print('Hello World!')
    'Ensure that struct function is only applied to correct dtype.'
    with pytest.raises(TypeError):
        df.integer.struct.items()

def test_struct_len(df):
    if False:
        for i in range(10):
            print('nop')
    assert len(df.array.struct) == 3

def test_struct_len_invalid_dtypes(df):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError):
        len(df.integer.struct)

def test_struct_iter(df):
    if False:
        for i in range(10):
            print('nop')
    assert list(df.array.struct) == ['col1', 'col2', 'col3']

def test_struct_dtypes(df):
    if False:
        while True:
            i = 10
    types = df.array.struct.dtypes
    assert types['col1'] == vaex.datatype.DataType(pa.int64())
    assert types['col2'] == vaex.datatype.DataType(pa.string())
    assert types['col3'] == vaex.datatype.DataType(pa.int64())

def test_struct_dtypes_duplicated(df_duplicated):
    if False:
        print('Hello World!')
    types = df_duplicated.array.struct.dtypes
    assert types.iloc[0] == vaex.datatype.DataType(pa.int64())
    assert types.iloc[1] == vaex.datatype.DataType(pa.string())
    assert types.iloc[2] == vaex.datatype.DataType(pa.int64())

def test_struct_dtypes_invalid_dtypes(df):
    if False:
        print('Hello World!')
    'Ensure that struct function is only applied to correct dtype.'
    with pytest.raises(TypeError):
        df.integer.struct.dtypes

def test_struct_repr(df):
    if False:
        return 10
    'Ensure that `repr` works without failing and contains correct dtype information.'
    string = repr(df.array)
    assert 'dtype: struct' in string
    assert 'array' in string

def test_struct_repr_duplicated(df_duplicated):
    if False:
        for i in range(10):
            print('nop')
    'Ensure that `repr` works without failing and contains correct dtype information.'
    string = repr(df_duplicated.array)
    assert 'dtype: struct' in string
    assert 'array' in string

def test_struct_correct_df_dtypes(df):
    if False:
        for i in range(10):
            print('nop')
    'Ensure that `dtypes` works correctly on vaex dataframe containing a struct.'
    assert 'array' in df.dtypes
    assert df.dtypes['array'].is_struct

def test_struct_correct_df_duplicated_dtypes(df_duplicated):
    if False:
        for i in range(10):
            print('nop')
    'Ensure that `dtypes` works correctly on vaex dataframe containing a struct.'
    assert 'array' in df_duplicated.dtypes
    assert df_duplicated.dtypes['array'].is_struct

def test_struct_correct_expression_dtype(df):
    if False:
        i = 10
        return i + 15
    'Ensure that `dtype` works correctly on vaex expression containing a struct.'
    assert df.array.dtype.is_struct

def test_struct_correct_expression_dtype_duplicated(df_duplicated):
    if False:
        print('Hello World!')
    'Ensure that `dtype` works correctly on vaex expression containing a struct.'
    assert df_duplicated.array.dtype.is_struct

def test_struct_flatten(df):
    if False:
        return 10
    assert df.get_column_names() == ['array', 'integer']
    df_flat = df.struct.flatten()
    assert df_flat.get_column_names() == ['array_col1', 'array_col2', 'array_col3', 'integer']
    assert df.get_column_names() == ['array', 'integer']
    df_flat = df.struct.flatten('integer')
    assert df_flat.get_column_names() == ['array', 'integer']
    df_flat = df.struct.flatten(['integer'])
    assert df_flat.get_column_names() == ['array', 'integer']

def test_struct_flatten_recursive(df):
    if False:
        i = 10
        return i + 15
    ar = df.array.values
    array = pa.StructArray.from_arrays(arrays=[ar, ar], names=['copy1', 'copy2'])
    df = vaex.from_arrays(array=array, integer=[8, 9, 10])
    df_flat = df.struct.flatten()
    assert df_flat.get_column_names() == ['array_copy1_col1', 'array_copy1_col2', 'array_copy1_col3', 'array_copy2_col1', 'array_copy2_col2', 'array_copy2_col3', 'integer']
    df_flat = df.struct.flatten(recursive=False)
    assert df_flat.get_column_names() == ['array_copy1', 'array_copy2', 'integer']