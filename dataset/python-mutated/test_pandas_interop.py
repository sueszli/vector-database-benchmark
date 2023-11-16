from __future__ import annotations
import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.impala.pandas_interop import DataFrameWriter
pytest.importorskip('impala')

@pytest.fixture
def exhaustive_df():
    if False:
        while True:
            i = 10
    return pd.DataFrame({'bigint_col': np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype='i8'), 'bool_col': np.array([True, False, True, False, True, None, True, False, True, False], dtype=np.bool_), 'date_string_col': ['11/01/10', None, '11/01/10', '11/01/10', '11/01/10', '11/01/10', '11/01/10', '11/01/10', '11/01/10', '11/01/10'], 'double_col': np.array([0.0, 10.1, np.nan, 30.299999999999997, 40.4, 50.5, 60.599999999999994, 70.7, 80.8, 90.89999999999999], dtype=np.float64), 'floatcol': np.array([np.nan, 1.100000023841858, 2.200000047683716, 3.299999952316284, 4.400000095367432, 5.5, 6.599999904632568, 7.699999809265137, 8.800000190734863, 9.899999618530273], dtype='f8'), 'int_col': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='i4'), 'month': [11, 11, 11, 11, 2, 11, 11, 11, 11, 11], 'smallint_col': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='i2'), 'string_col': ['0', '1', None, 'double , whammy', '4', '5', '6', '7', '8', '9'], 'timestamp_col': [pd.Timestamp('2010-11-01 00:00:00'), None, pd.Timestamp('2010-11-01 00:02:00.100000'), pd.Timestamp('2010-11-01 00:03:00.300000'), pd.Timestamp('2010-11-01 00:04:00.600000'), pd.Timestamp('2010-11-01 00:05:00.100000'), pd.Timestamp('2010-11-01 00:06:00.150000'), pd.Timestamp('2010-11-01 00:07:00.210000'), pd.Timestamp('2010-11-01 00:08:00.280000'), pd.Timestamp('2010-11-01 00:09:00.360000')], 'tinyint_col': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='i1'), 'year': [2010, 2010, 2010, 2010, 2010, 2009, 2009, 2009, 2009, 2009]})

def test_alltypes_roundtrip(con, alltypes_df):
    if False:
        i = 10
        return i + 15
    _check_roundtrip(con, alltypes_df)

@pytest.mark.hdfs
def test_writer_cleanup_deletes_hdfs_dir(con, hdfs, alltypes_df):
    if False:
        for i in range(10):
            print('nop')
    writer = DataFrameWriter(con, alltypes_df)
    path = writer.write_temp_csv()
    assert hdfs.exists(path)
    writer.cleanup()
    assert not hdfs.exists(path)
    writer.cleanup()
    assert not hdfs.exists(path)

@pytest.mark.hdfs
def test_writer_cleanup_context_manager(con, hdfs, alltypes_df):
    if False:
        for i in range(10):
            print('nop')
    with DataFrameWriter(con, alltypes_df) as writer:
        path = writer.write_temp_csv()
        assert hdfs.exists(path)
    assert not hdfs.exists(path)

def test_create_table_from_dataframe(con, alltypes_df, temp_table_db):
    if False:
        return 10
    (tmp_db, tname) = temp_table_db
    con.create_table(tname, alltypes_df, database=tmp_db)
    table = con.table(tname, database=tmp_db)
    df = table.execute()
    tm.assert_frame_equal(df, alltypes_df)

def test_insert(con, temp_table_db, exhaustive_df):
    if False:
        return 10
    (tmp_db, table_name) = temp_table_db
    schema = sch.infer(exhaustive_df)
    con.create_table(table_name, database=tmp_db, schema=schema)
    con.insert(table_name, exhaustive_df.iloc[:4], database=tmp_db)
    con.insert(table_name, exhaustive_df.iloc[4:], database=tmp_db)
    table = con.table(table_name, database=tmp_db)
    result = table.execute().sort_values(by='tinyint_col').reset_index(drop=True)
    tm.assert_frame_equal(result.fillna(pd.NA), exhaustive_df.fillna(pd.NA))

def test_round_trip_exhaustive(con, exhaustive_df):
    if False:
        while True:
            i = 10
    _check_roundtrip(con, exhaustive_df)

def _check_roundtrip(con, df):
    if False:
        for i in range(10):
            print('nop')
    writer = DataFrameWriter(con, df)
    path = writer.write_temp_csv()
    table = writer.delimited_table(path)
    df2 = table.execute()
    tm.assert_frame_equal(df2.fillna(pd.NA), df.fillna(pd.NA))

def test_timestamp_with_timezone():
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame({'A': pd.date_range('20130101', periods=3, tz='US/Eastern')})
    schema = sch.infer(df)
    expected = ibis.schema([('A', "timestamp('US/Eastern')")])
    assert schema.equals(expected)
    assert schema.types[0].equals(dt.Timestamp('US/Eastern'))