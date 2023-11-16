import datetime
import io
import pathlib
from typing import Optional
import fastavro
import numpy as np
import pandas as pd
import pytest
import cudf
from cudf.testing._utils import assert_eq
from cudf.testing.dataset_generator import rand_dataframe

def cudf_from_avro_util(schema: dict, records: list) -> cudf.DataFrame:
    if False:
        i = 10
        return i + 15
    schema = [] if schema is None else fastavro.parse_schema(schema)
    buffer = io.BytesIO()
    fastavro.writer(buffer, schema, records)
    buffer.seek(0)
    return cudf.read_avro(buffer)
avro_type_params = [('boolean', 'bool'), ('int', 'int32'), ('long', 'int64'), ('float', 'float32'), ('double', 'float64'), ('bytes', 'str'), ('string', 'str')]

@pytest.mark.parametrize('avro_type, expected_dtype', avro_type_params)
@pytest.mark.parametrize('namespace', [None, 'root_ns'])
@pytest.mark.parametrize('nullable', [True, False])
def test_can_detect_dtype_from_avro_type(avro_type, expected_dtype, namespace, nullable):
    if False:
        while True:
            i = 10
    avro_type = avro_type if not nullable else ['null', avro_type]
    schema = fastavro.parse_schema({'type': 'record', 'name': 'test', 'namespace': namespace, 'fields': [{'name': 'prop', 'type': avro_type}]})
    actual = cudf_from_avro_util(schema, [])
    expected = cudf.DataFrame({'prop': cudf.Series(None, None, expected_dtype)})
    assert_eq(expected, actual)

@pytest.mark.parametrize('avro_type, expected_dtype', avro_type_params)
@pytest.mark.parametrize('namespace', [None, 'root_ns'])
@pytest.mark.parametrize('nullable', [True, False])
def test_can_detect_dtype_from_avro_type_nested(avro_type, expected_dtype, namespace, nullable):
    if False:
        i = 10
        return i + 15
    avro_type = avro_type if not nullable else ['null', avro_type]
    schema_leaf = {'name': 'leaf', 'type': 'record', 'fields': [{'name': 'prop3', 'type': avro_type}]}
    schema_child = {'name': 'child', 'type': 'record', 'fields': [{'name': 'prop2', 'type': schema_leaf}]}
    schema_root = {'name': 'root', 'type': 'record', 'namespace': namespace, 'fields': [{'name': 'prop1', 'type': schema_child}]}
    actual = cudf_from_avro_util(schema_root, [])
    col_name = '{ns}child.{ns}leaf.prop3'.format(ns='' if namespace is None else namespace + '.')
    expected = cudf.DataFrame({col_name: cudf.Series(None, None, expected_dtype)})
    assert_eq(expected, actual)

@pytest.mark.parametrize('avro_type, cudf_type, avro_val, cudf_val', [('boolean', 'bool', True, True), ('boolean', 'bool', False, False), ('int', 'int32', 1234, 1234), ('long', 'int64', 1234, 1234), ('float', 'float32', 12.34, 12.34), ('double', 'float64', 12.34, 12.34), ('string', 'str', 'heyϴ', 'heyϴ')])
def test_can_parse_single_value(avro_type, cudf_type, avro_val, cudf_val):
    if False:
        for i in range(10):
            print('nop')
    schema_root = {'name': 'root', 'type': 'record', 'fields': [{'name': 'prop', 'type': ['null', avro_type]}]}
    records = [{'prop': avro_val}]
    actual = cudf_from_avro_util(schema_root, records)
    expected = cudf.DataFrame({'prop': cudf.Series(data=[cudf_val], dtype=cudf_type)})
    assert_eq(expected, actual)

@pytest.mark.parametrize('avro_type, cudf_type', avro_type_params)
def test_can_parse_single_null(avro_type, cudf_type):
    if False:
        for i in range(10):
            print('nop')
    schema_root = {'name': 'root', 'type': 'record', 'fields': [{'name': 'prop', 'type': ['null', avro_type]}]}
    records = [{'prop': None}]
    actual = cudf_from_avro_util(schema_root, records)
    expected = cudf.DataFrame({'prop': cudf.Series(data=[None], dtype=cudf_type)})
    assert_eq(expected, actual)

@pytest.mark.parametrize('avro_type, cudf_type', avro_type_params)
def test_can_parse_no_data(avro_type, cudf_type):
    if False:
        return 10
    schema_root = {'name': 'root', 'type': 'record', 'fields': [{'name': 'prop', 'type': ['null', avro_type]}]}
    records = []
    actual = cudf_from_avro_util(schema_root, records)
    expected = cudf.DataFrame({'prop': cudf.Series(data=[], dtype=cudf_type)})
    assert_eq(expected, actual)

@pytest.mark.xfail(reason='cudf avro reader is unable to parse zero-field metadata.')
@pytest.mark.parametrize('avro_type, cudf_type', avro_type_params)
def test_can_parse_no_fields(avro_type, cudf_type):
    if False:
        for i in range(10):
            print('nop')
    schema_root = {'name': 'root', 'type': 'record', 'fields': []}
    records = []
    actual = cudf_from_avro_util(schema_root, records)
    expected = cudf.DataFrame()
    assert_eq(expected, actual)

def test_can_parse_no_schema():
    if False:
        while True:
            i = 10
    schema_root = None
    records = []
    actual = cudf_from_avro_util(schema_root, records)
    expected = cudf.DataFrame()
    assert_eq(expected, actual)

@pytest.mark.parametrize('rows', [0, 1, 10, 1000])
@pytest.mark.parametrize('codec', ['null', 'deflate', 'snappy'])
def test_avro_compression(rows, codec):
    if False:
        print('Hello World!')
    schema = {'name': 'root', 'type': 'record', 'fields': [{'name': '0', 'type': 'int'}, {'name': '1', 'type': 'string'}]}
    df = rand_dataframe([{'dtype': 'int32', 'null_frequency': 0, 'cardinality': 1000}, {'dtype': 'str', 'null_frequency': 0, 'cardinality': 100, 'max_string_length': 10}], rows)
    expected_df = cudf.DataFrame.from_arrow(df)
    records = df.to_pandas().to_dict(orient='records')
    buffer = io.BytesIO()
    fastavro.writer(buffer, schema, records, codec=codec)
    buffer.seek(0)
    got_df = cudf.read_avro(buffer)
    assert_eq(expected_df, got_df)
avro_logical_type_params = [('date', 'int', 'datetime64[s]')]

@pytest.mark.parametrize('logical_type, primitive_type, expected_dtype', avro_logical_type_params)
@pytest.mark.parametrize('namespace', [None, 'root_ns'])
@pytest.mark.parametrize('nullable', [True, False])
@pytest.mark.parametrize('prepend_null', [True, False])
def test_can_detect_dtypes_from_avro_logical_type(logical_type, primitive_type, expected_dtype, namespace, nullable, prepend_null):
    if False:
        i = 10
        return i + 15
    avro_type = [{'logicalType': logical_type, 'type': primitive_type}]
    if nullable:
        if prepend_null:
            avro_type.insert(0, 'null')
        else:
            avro_type.append('null')
    schema = fastavro.parse_schema({'type': 'record', 'name': 'test', 'namespace': namespace, 'fields': [{'name': 'prop', 'type': avro_type}]})
    actual = cudf_from_avro_util(schema, [])
    expected = cudf.DataFrame({'prop': cudf.Series(None, None, expected_dtype)})
    assert_eq(expected, actual)

def get_days_from_epoch(date: Optional[datetime.date]) -> Optional[int]:
    if False:
        while True:
            i = 10
    if date is None:
        return None
    return (date - datetime.date(1970, 1, 1)).days

@pytest.mark.parametrize('namespace', [None, 'root_ns'])
@pytest.mark.parametrize('nullable', [True, False])
@pytest.mark.parametrize('prepend_null', [True, False])
def test_can_parse_avro_date_logical_type(namespace, nullable, prepend_null):
    if False:
        print('Hello World!')
    avro_type = {'logicalType': 'date', 'type': 'int'}
    if nullable:
        if prepend_null:
            avro_type = ['null', avro_type]
        else:
            avro_type = [avro_type, 'null']
    schema_dict = {'type': 'record', 'name': 'test', 'fields': [{'name': 'o_date', 'type': avro_type}]}
    if namespace:
        schema_dict['namespace'] = namespace
    schema = fastavro.parse_schema(schema_dict)
    dates = [None, datetime.date(1970, 1, 1), datetime.date(1970, 1, 2), datetime.date(1981, 10, 25), None, None, datetime.date(2012, 5, 18), None, datetime.date(2019, 9, 3), None, datetime.date(9999, 12, 31)]
    if not nullable:
        dates = [date for date in dates if date is not None]
    days_from_epoch = [get_days_from_epoch(date) for date in dates]
    records = [{'o_date': day} for day in days_from_epoch]
    actual = cudf_from_avro_util(schema, records)
    expected = cudf.DataFrame({'o_date': cudf.Series(dates, dtype='datetime64[s]')})
    assert_eq(expected, actual)

def test_alltypes_plain_avro():
    if False:
        print('Hello World!')
    relpath = '../../../../java/src/test/resources/alltypes_plain.avro'
    path = pathlib.Path(__file__).parent.joinpath(relpath).resolve()
    assert path.is_file(), path
    path = str(path)
    with open(path, 'rb') as f:
        reader = fastavro.reader(f)
        records = [record for record in reader]
    columns = ['bool_col', 'int_col', 'timestamp_col']
    actual = cudf.read_avro(path, columns=columns)
    data = [{column: row[column] for column in columns} for row in records]
    expected = pd.DataFrame(data)
    expected['timestamp_col'].dt.tz_localize(None)
    timestamps = expected['timestamp_col'].astype('int64')
    timestamps //= 1000
    expected['timestamp_col'] = timestamps
    expected['int_col'] = expected['int_col'].astype('int32')
    assert_eq(actual, expected)

def multiblock_testname_ids(param):
    if False:
        for i in range(10):
            print('nop')
    (total_rows, num_rows, skip_rows, sync_interval) = param
    return f'total_rows={total_rows!r}-num_rows={num_rows!r}-skip_rows={skip_rows!r}-sync_interval={sync_interval!r}'

@pytest.fixture(ids=multiblock_testname_ids, params=[(10, 10, 9, 9), (10, 10, 9, 5), (10, 10, 9, 3), (10, 10, 9, 2), (10, 10, 9, 10), (10, 10, 8, 2), (10, 10, 5, 5), (10, 10, 2, 9), (10, 10, 2, 2), (10, 10, 1, 9), (10, 10, 1, 5), (10, 10, 1, 2), (10, 10, 1, 10), (10, 10, 10, 9), (10, 10, 10, 5), (10, 10, 10, 2), (10, 10, 10, 10), (10, 10, 0, 9), (10, 10, 0, 5), (10, 10, 0, 2), (10, 10, 0, 10), (100, 100, 99, 10), (100, 100, 90, 90), (100, 100, 90, 89), (100, 100, 90, 88), (100, 100, 90, 87), (100, 100, 90, 5), (100, 100, 89, 90), (100, 100, 87, 90), (100, 100, 50, 7), (100, 100, 50, 31), (10, 1, 8, 9), (100, 1, 99, 10), (100, 1, 98, 10), (100, 1, 97, 10), (100, 3, 90, 87), (100, 4, 90, 5), (100, 2, 89, 90), (100, 9, 87, 90), (100, 20, 50, 7), (100, 10, 50, 31), (100, 20, 50, 31), (100, 30, 50, 31), (256, 256, 0, 256), (256, 256, 0, 32), (256, 256, 0, 31), (256, 256, 0, 33), (256, 256, 31, 32), (256, 256, 32, 31), (256, 256, 31, 33), (512, 512, 0, 32), (512, 512, 0, 31), (512, 512, 0, 33), (512, 512, 31, 32), (512, 512, 32, 31), (512, 512, 31, 33), (1024, 1024, 0, 1), (1024, 1024, 0, 3), (1024, 1024, 0, 7), (1024, 1024, 0, 8), (1024, 1024, 0, 9), (1024, 1024, 0, 15), (1024, 1024, 0, 16), (1024, 1024, 0, 17), (1024, 1024, 0, 32), (1024, 1024, 0, 31), (1024, 1024, 0, 33), (1024, 1024, 31, 32), (1024, 1024, 32, 31), (1024, 1024, 31, 33), (16384, 16384, 0, 31), (16384, 16384, 0, 32), (16384, 16384, 0, 33), (16384, 16384, 0, 16384)])
def total_rows_and_num_rows_and_skip_rows_and_rows_per_block(request):
    if False:
        return 10
    return request.param

@pytest.mark.parametrize('dtype', ['str', 'float32', 'float64'])
@pytest.mark.parametrize('use_sync_interval', [True, False], ids=['use_sync_interval', 'ignore_sync_interval'])
@pytest.mark.parametrize('codec', ['null', 'deflate', 'snappy'])
def test_avro_reader_multiblock(dtype, codec, use_sync_interval, total_rows_and_num_rows_and_skip_rows_and_rows_per_block):
    if False:
        print('Hello World!')
    (total_rows, num_rows, skip_rows, rows_per_block) = total_rows_and_num_rows_and_skip_rows_and_rows_per_block
    assert total_rows >= num_rows
    assert rows_per_block <= total_rows
    limit_rows = num_rows != total_rows
    if limit_rows:
        assert total_rows >= num_rows + skip_rows
    if dtype == 'str':
        avro_type = 'string'
        values = [f'{i:0>6}' for i in range(0, total_rows)]
        bytes_per_row = len(values[0]) + 1
        assert bytes_per_row == 7, bytes_per_row
    else:
        assert dtype in ('float32', 'float64')
        avro_type = 'float' if dtype == 'float32' else 'double'
        values = np.random.rand(total_rows).astype(dtype)
        bytes_per_row = values.dtype.itemsize
    total_bytes_per_block = rows_per_block * bytes_per_row
    sync_interval = total_bytes_per_block
    source_df = cudf.DataFrame({'0': pd.Series(values)})
    if limit_rows:
        expected_df = source_df[skip_rows:skip_rows + num_rows].reset_index(drop=True)
    else:
        expected_df = source_df[skip_rows:].reset_index(drop=True)
    records = source_df.to_pandas().to_dict(orient='records')
    schema = {'name': 'root', 'type': 'record', 'fields': [{'name': '0', 'type': avro_type}]}
    if use_sync_interval:
        kwds = {'sync_interval': sync_interval}
    else:
        kwds = {}
    kwds['codec'] = codec
    buffer = io.BytesIO()
    fastavro.writer(buffer, schema, records, **kwds)
    buffer.seek(0)
    if not limit_rows:
        num_rows = None
    actual_df = cudf.read_avro(buffer, skiprows=skip_rows, num_rows=num_rows)
    assert_eq(expected_df, actual_df)