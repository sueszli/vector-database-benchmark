from __future__ import annotations
import typing
import pendulum
if typing.TYPE_CHECKING:
    import pandas as pd
    from pyspark import SparkContext
    from pyspark.sql import SparkSession
from airflow.decorators import dag, task

@dag(schedule=None, start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, tags=['example'])
def example_pyspark():
    if False:
        i = 10
        return i + 15
    '\n    ### Example Pyspark DAG\n    This is an example DAG which uses pyspark\n    '

    @task.pyspark(conn_id='spark-local')
    def spark_task(spark: SparkSession, sc: SparkContext) -> pd.DataFrame:
        if False:
            return 10
        df = spark.createDataFrame([(1, 'John Doe', 21), (2, 'Jane Doe', 22), (3, 'Joe Bloggs', 23)], ['id', 'name', 'age'])
        df.show()
        return df.toPandas()

    @task
    def print_df(df: pd.DataFrame):
        if False:
            return 10
        print(df)
    df = spark_task()
    print_df(df)
dag = example_pyspark()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)