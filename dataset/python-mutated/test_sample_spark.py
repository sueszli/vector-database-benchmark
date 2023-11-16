import pandas as pd
import pytest
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from ydata_profiling.config import Settings
from ydata_profiling.model.spark.sample_spark import spark_get_sample

@pytest.fixture()
def df(spark_session):
    if False:
        return 10
    data_pandas = pd.DataFrame({'make': ['Jaguar', 'MG', 'MINI', 'Rover', 'Lotus'] * 50, 'registration': ['AB98ABCD', 'BC99BCDF', 'CD00CDE', 'DE01DEF', 'EF02EFG'] * 50, 'year': [1998, 1999, 2000, 2001, 2002] * 50})
    data_spark = spark_session.createDataFrame(data_pandas)
    return data_spark

@pytest.fixture()
def df_empty(spark_session):
    if False:
        for i in range(10):
            print('nop')
    data_pandas = pd.DataFrame({'make': [], 'registration': [], 'year': []})
    schema = StructType({StructField('make', StringType(), True), StructField('registration', StringType(), True), StructField('year', IntegerType(), True)})
    data_spark = spark_session.createDataFrame(data_pandas, schema=schema)
    return data_spark

def test_spark_get_sample(df):
    if False:
        return 10
    config = Settings()
    config.samples.head = 17
    config.samples.random = 0
    config.samples.tail = 0
    res = spark_get_sample(config, df)
    assert len(res) == 1
    assert res[0].id == 'head'
    assert len(res[0].data) == 17
    config = Settings()
    config.samples.head = 0
    config.samples.random = 0
    config.samples.tail = 0
    res = spark_get_sample(config, df)
    assert len(res) == 0

def test_spark_sample_empty(df_empty):
    if False:
        while True:
            i = 10
    config = Settings()
    config.samples.head = 5
    config.samples.random = 0
    config.samples.tail = 0
    res = spark_get_sample(config, df_empty)
    assert len(res) == 0