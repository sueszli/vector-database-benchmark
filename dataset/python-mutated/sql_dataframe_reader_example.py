from pyspark.sql.functions import *
from pyspark.sql import Row, Window, SparkSession, SQLContext
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F
from pyspark.sql.functions import rank, min, col, mean
import random

def sql_dataframe_reader_api(spark):
    if False:
        while True:
            i = 10
    print('Start running dataframe reader API')
    sc = spark.sparkContext
    sqlContext = SQLContext(sc)
    df = spark.read.csv('/ppml/spark-3.1.3/python/test_support/sql/ages.csv')
    print(df.dtypes)
    rdd = sc.textFile('/ppml/spark-3.1.3/python/test_support/sql/ages.csv')
    df2 = spark.read.option('header', 'true').csv(rdd)
    print(df2.dtypes)
    print('csv and option API finished')
    df = spark.read.format('json').load('/ppml/spark-3.1.3/python/test_support/sql/people.json')
    print(df.dtypes)
    print('format API finished')
    df1 = spark.read.json('/ppml/spark-3.1.3/python/test_support/sql/people.json')
    print(df1.dtypes)
    rdd = sc.textFile('/ppml/spark-3.1.3/python/test_support/sql/people.json')
    df2 = spark.read.json(rdd)
    print(df2.dtypes)
    print('json API finished')
    df = spark.read.format('json').load(['/ppml/spark-3.1.3/python/test_support/sql/people.json', '/ppml/spark-3.1.3/python/test_support/sql/people1.json'])
    print(df.dtypes)
    df = spark.read.orc('/ppml/spark-3.1.3/python/test_support/sql/orc_partitioned')
    print(df.dtypes)
    print('orc API finished')
    df = spark.read.parquet('/ppml/spark-3.1.3/python/test_support/sql/parquet_partitioned')
    print(df.dtypes)
    print('parquet API finished')
    s = spark.read.schema('col0 INT, col1 DOUBLE')
    print(s)
    print('schema API finished')
    df = spark.read.parquet('/ppml/spark-3.1.3/python/test_support/sql/parquet_partitioned')
    df.createOrReplaceTempView('tmpTable')
    res = spark.read.table('tmpTable').dtypes
    print(res)
    print('table API finished')
    df = spark.read.text('/ppml/spark-3.1.3/python/test_support/sql/text-test.txt')
    df.show()
    df = spark.read.text('/ppml/spark-3.1.3/python/test_support/sql/text-test.txt', wholetext=True)
    df.show()
    print('text API finished')
    print('Finish running dataframe reader API')
if __name__ == '__main__':
    spark = SparkSession.builder.appName('Python Spark SQL Dataframe Reader example').config('spark.some.config.option', 'some-value').getOrCreate()
    sql_dataframe_reader_api(spark)