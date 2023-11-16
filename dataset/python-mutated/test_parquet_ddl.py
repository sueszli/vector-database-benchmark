from __future__ import annotations
import gc
from posixpath import join as pjoin
import pytest
import ibis
from ibis.tests.util import assert_equal
pytest.importorskip('impala')
from ibis.backends.impala.compat import HS2Error

def test_cleanup_tmp_table_on_gc(con, test_data_dir):
    if False:
        return 10
    hdfs_path = pjoin(test_data_dir, 'impala/parquet/region')
    table = con.parquet_file(hdfs_path)
    name = table.op().name
    table = None
    gc.collect()
    assert name not in con.list_tables()

def test_persist_parquet_file_with_name(con, test_data_dir, temp_table_db):
    if False:
        print('Hello World!')
    hdfs_path = pjoin(test_data_dir, 'impala/parquet/region')
    (tmp_db, name) = temp_table_db
    schema = ibis.schema([('r_regionkey', 'int16'), ('r_name', 'string'), ('r_comment', 'string')])
    con.parquet_file(hdfs_path, schema=schema, name=name, database=tmp_db, persist=True)
    gc.collect()
    con.table(name, database=tmp_db)

def test_query_parquet_file_with_schema(con, test_data_dir):
    if False:
        i = 10
        return i + 15
    hdfs_path = pjoin(test_data_dir, 'impala/parquet/region')
    schema = ibis.schema([('r_regionkey', 'int16'), ('r_name', 'string'), ('r_comment', 'string')])
    table = con.parquet_file(hdfs_path, schema=schema)
    name = table._qualified_name
    con.table(name)
    expr = table.r_name.value_counts()
    expr.execute()
    assert table.count().execute() == 5

def test_query_parquet_file_like_table(con, test_data_dir):
    if False:
        while True:
            i = 10
    hdfs_path = pjoin(test_data_dir, 'impala/parquet/region')
    ex_schema = ibis.schema([('r_regionkey', 'int32'), ('r_name', 'string'), ('r_comment', 'string')])
    table = con.parquet_file(hdfs_path, like_table='region')
    assert_equal(table.schema(), ex_schema)

def test_query_parquet_infer_schema(con, test_data_dir):
    if False:
        for i in range(10):
            print('nop')
    hdfs_path = pjoin(test_data_dir, 'impala/parquet/region')
    table = con.parquet_file(hdfs_path)
    ex_schema = ibis.schema([('r_regionkey', 'int32'), ('r_name', 'string'), ('r_comment', 'string')])
    assert_equal(table.schema(), ex_schema)

def test_create_table_persist_fails_if_called_twice(con, temp_table_db, test_data_dir):
    if False:
        return 10
    (tmp_db, tname) = temp_table_db
    hdfs_path = pjoin(test_data_dir, 'impala/parquet/region')
    con.parquet_file(hdfs_path, name=tname, persist=True, database=tmp_db)
    with pytest.raises(HS2Error):
        con.parquet_file(hdfs_path, name=tname, persist=True, database=tmp_db)