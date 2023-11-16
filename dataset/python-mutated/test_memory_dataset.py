import pytest
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from kedro.io import MemoryDataset

def _update_spark_df(data, idx, jdx, value):
    if False:
        print('Hello World!')
    session = SparkSession.builder.getOrCreate()
    data = session.createDataFrame(data.rdd.zipWithIndex()).select(col('_1.*'), col('_2').alias('__id'))
    cname = data.columns[idx]
    return data.withColumn(cname, when(col('__id') == jdx, value).otherwise(col(cname))).drop('__id')

def _check_equals(data1, data2):
    if False:
        while True:
            i = 10
    if isinstance(data1, SparkDataFrame) and isinstance(data2, SparkDataFrame):
        return data1.toPandas().equals(data2.toPandas())
    return False

@pytest.fixture
def spark_data_frame(spark_session):
    if False:
        print('Hello World!')
    return spark_session.createDataFrame([(1, 4, 5), (2, 5, 6)], ['col1', 'col2', 'col3'])

@pytest.fixture
def memory_dataset(spark_data_frame):
    if False:
        while True:
            i = 10
    return MemoryDataset(data=spark_data_frame)

def test_load_modify_original_data(memory_dataset, spark_data_frame):
    if False:
        return 10
    'Check that the data set object is not updated when the original\n    SparkDataFrame is changed.'
    spark_data_frame = _update_spark_df(spark_data_frame, 1, 1, -5)
    assert not _check_equals(memory_dataset.load(), spark_data_frame)

def test_save_modify_original_data(spark_data_frame):
    if False:
        while True:
            i = 10
    'Check that the data set object is not updated when the original\n    SparkDataFrame is changed.'
    memory_dataset = MemoryDataset()
    memory_dataset.save(spark_data_frame)
    spark_data_frame = _update_spark_df(spark_data_frame, 1, 1, 'new value')
    assert not _check_equals(memory_dataset.load(), spark_data_frame)

def test_load_returns_same_spark_object(memory_dataset, spark_data_frame):
    if False:
        return 10
    'Test that consecutive loads point to the same object in case of\n    a SparkDataFrame'
    loaded_data = memory_dataset.load()
    reloaded_data = memory_dataset.load()
    assert _check_equals(loaded_data, spark_data_frame)
    assert _check_equals(reloaded_data, spark_data_frame)
    assert loaded_data is reloaded_data

def test_str_representation(memory_dataset):
    if False:
        print('Hello World!')
    'Test string representation of the data set'
    assert 'MemoryDataset(data=<DataFrame>)' in str(memory_dataset)