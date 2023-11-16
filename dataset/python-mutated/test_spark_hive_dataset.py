import gc
import re
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from psutil import Popen
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from kedro.extras.datasets.spark import SparkHiveDataSet
from kedro.io import DatasetError
TESTSPARKDIR = 'test_spark_dir'

@pytest.fixture(scope='module')
def spark_session():
    if False:
        while True:
            i = 10
    try:
        with TemporaryDirectory(TESTSPARKDIR) as tmpdir:
            spark = SparkSession.builder.config('spark.local.dir', (Path(tmpdir) / 'spark_local').absolute()).config('spark.sql.warehouse.dir', (Path(tmpdir) / 'warehouse').absolute()).config('javax.jdo.option.ConnectionURL', f"jdbc:derby:;databaseName={(Path(tmpdir) / 'warehouse_db').absolute()};create=true").enableHiveSupport().getOrCreate()
            spark.sparkContext.setCheckpointDir(str((Path(tmpdir) / 'spark_checkpoint').absolute()))
            yield spark
            spark.stop()
    except PermissionError:
        pass
    SparkContext._jvm = None
    SparkContext._gateway = None
    for obj in gc.get_objects():
        try:
            if isinstance(obj, Popen) and 'pyspark' in obj.args[0]:
                obj.terminate()
        except ReferenceError:
            pass

@pytest.fixture(scope='module', autouse=True)
def spark_test_databases(spark_session):
    if False:
        return 10
    'Setup spark test databases for all tests in this module.'
    dataset = _generate_spark_df_one()
    dataset.createOrReplaceTempView('tmp')
    databases = ['default_1', 'default_2']
    for database in databases:
        spark_session.sql(f'create database {database}')
    spark_session.sql('use default_1')
    spark_session.sql('create table table_1 as select * from tmp')
    yield spark_session
    for database in databases:
        spark_session.sql(f'drop database {database} cascade')

def assert_df_equal(expected, result):
    if False:
        i = 10
        return i + 15

    def indexRDD(data_frame):
        if False:
            while True:
                i = 10
        return data_frame.rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
    index_expected = indexRDD(expected)
    index_result = indexRDD(result)
    assert index_expected.cogroup(index_result).map(lambda x: tuple(map(list, x[1]))).filter(lambda x: x[0] != x[1]).take(1) == []

def _generate_spark_df_one():
    if False:
        i = 10
        return i + 15
    schema = StructType([StructField('name', StringType(), True), StructField('age', IntegerType(), True)])
    data = [('Alex', 31), ('Bob', 12), ('Clarke', 65), ('Dave', 29)]
    return SparkSession.builder.getOrCreate().createDataFrame(data, schema).coalesce(1)

def _generate_spark_df_upsert():
    if False:
        print('Hello World!')
    schema = StructType([StructField('name', StringType(), True), StructField('age', IntegerType(), True)])
    data = [('Alex', 99), ('Jeremy', 55)]
    return SparkSession.builder.getOrCreate().createDataFrame(data, schema).coalesce(1)

def _generate_spark_df_upsert_expected():
    if False:
        i = 10
        return i + 15
    schema = StructType([StructField('name', StringType(), True), StructField('age', IntegerType(), True)])
    data = [('Alex', 99), ('Bob', 12), ('Clarke', 65), ('Dave', 29), ('Jeremy', 55)]
    return SparkSession.builder.getOrCreate().createDataFrame(data, schema).coalesce(1)

class TestSparkHiveDataSet:

    def test_cant_pickle(self):
        if False:
            return 10
        import pickle
        with pytest.raises(pickle.PicklingError):
            pickle.dumps(SparkHiveDataSet(database='default_1', table='table_1', write_mode='overwrite'))

    def test_read_existing_table(self):
        if False:
            while True:
                i = 10
        dataset = SparkHiveDataSet(database='default_1', table='table_1', write_mode='overwrite', save_args={})
        assert_df_equal(_generate_spark_df_one(), dataset.load())

    def test_overwrite_empty_table(self, spark_session):
        if False:
            for i in range(10):
                print('nop')
        spark_session.sql('create table default_1.test_overwrite_empty_table (name string, age integer)').take(1)
        dataset = SparkHiveDataSet(database='default_1', table='test_overwrite_empty_table', write_mode='overwrite')
        dataset.save(_generate_spark_df_one())
        assert_df_equal(dataset.load(), _generate_spark_df_one())

    def test_overwrite_not_empty_table(self, spark_session):
        if False:
            while True:
                i = 10
        spark_session.sql('create table default_1.test_overwrite_full_table (name string, age integer)').take(1)
        dataset = SparkHiveDataSet(database='default_1', table='test_overwrite_full_table', write_mode='overwrite')
        dataset.save(_generate_spark_df_one())
        dataset.save(_generate_spark_df_one())
        assert_df_equal(dataset.load(), _generate_spark_df_one())

    def test_insert_not_empty_table(self, spark_session):
        if False:
            return 10
        spark_session.sql('create table default_1.test_insert_not_empty_table (name string, age integer)').take(1)
        dataset = SparkHiveDataSet(database='default_1', table='test_insert_not_empty_table', write_mode='append')
        dataset.save(_generate_spark_df_one())
        dataset.save(_generate_spark_df_one())
        assert_df_equal(dataset.load(), _generate_spark_df_one().union(_generate_spark_df_one()))

    def test_upsert_config_err(self):
        if False:
            print('Hello World!')
        with pytest.raises(DatasetError, match="'table_pk' must be set to utilise 'upsert' read mode"):
            SparkHiveDataSet(database='default_1', table='table_1', write_mode='upsert')

    def test_upsert_empty_table(self, spark_session):
        if False:
            i = 10
            return i + 15
        spark_session.sql('create table default_1.test_upsert_empty_table (name string, age integer)').take(1)
        dataset = SparkHiveDataSet(database='default_1', table='test_upsert_empty_table', write_mode='upsert', table_pk=['name'])
        dataset.save(_generate_spark_df_one())
        assert_df_equal(dataset.load().sort('name'), _generate_spark_df_one().sort('name'))

    def test_upsert_not_empty_table(self, spark_session):
        if False:
            return 10
        spark_session.sql('create table default_1.test_upsert_not_empty_table (name string, age integer)').take(1)
        dataset = SparkHiveDataSet(database='default_1', table='test_upsert_not_empty_table', write_mode='upsert', table_pk=['name'])
        dataset.save(_generate_spark_df_one())
        dataset.save(_generate_spark_df_upsert())
        assert_df_equal(dataset.load().sort('name'), _generate_spark_df_upsert_expected().sort('name'))

    def test_invalid_pk_provided(self):
        if False:
            return 10
        _test_columns = ['column_doesnt_exist']
        dataset = SparkHiveDataSet(database='default_1', table='table_1', write_mode='upsert', table_pk=_test_columns)
        with pytest.raises(DatasetError, match=re.escape(f'Columns {str(_test_columns)} selected as primary key(s) not found in table default_1.table_1')):
            dataset.save(_generate_spark_df_one())

    def test_invalid_write_mode_provided(self):
        if False:
            for i in range(10):
                print('nop')
        pattern = "Invalid 'write_mode' provided: not_a_write_mode. 'write_mode' must be one of: append, error, errorifexists, upsert, overwrite"
        with pytest.raises(DatasetError, match=re.escape(pattern)):
            SparkHiveDataSet(database='default_1', table='table_1', write_mode='not_a_write_mode', table_pk=['name'])

    def test_invalid_schema_insert(self, spark_session):
        if False:
            for i in range(10):
                print('nop')
        spark_session.sql('create table default_1.test_invalid_schema_insert (name string, additional_column_on_hive integer)').take(1)
        dataset = SparkHiveDataSet(database='default_1', table='test_invalid_schema_insert', write_mode='append')
        with pytest.raises(DatasetError, match="Dataset does not match hive table schema\\.\\nPresent on insert only: \\[\\('age', 'int'\\)\\]\\nPresent on schema only: \\[\\('additional_column_on_hive', 'int'\\)\\]"):
            dataset.save(_generate_spark_df_one())

    def test_insert_to_non_existent_table(self):
        if False:
            print('Hello World!')
        dataset = SparkHiveDataSet(database='default_1', table='table_not_yet_created', write_mode='append')
        dataset.save(_generate_spark_df_one())
        assert_df_equal(dataset.load().sort('name'), _generate_spark_df_one().sort('name'))

    def test_read_from_non_existent_table(self):
        if False:
            return 10
        dataset = SparkHiveDataSet(database='default_1', table='table_doesnt_exist', write_mode='append')
        with pytest.raises(DatasetError, match='Failed while loading data from data set SparkHiveDataSet|table_doesnt_exist|UnresolvedRelation'):
            dataset.load()

    def test_save_delta_format(self, mocker):
        if False:
            return 10
        dataset = SparkHiveDataSet(database='default_1', table='delta_table', save_args={'format': 'delta'})
        mocked_save = mocker.patch('pyspark.sql.DataFrameWriter.saveAsTable')
        dataset.save(_generate_spark_df_one())
        mocked_save.assert_called_with('default_1.delta_table', mode='errorifexists', format='delta')
        assert dataset._format == 'delta'