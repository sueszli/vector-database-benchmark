from __future__ import annotations
from unittest import mock
import pytest
from airflow.decorators import task
from airflow.models import Connection
from airflow.utils import db, timezone
DEFAULT_DATE = timezone.datetime(2021, 9, 1)

class TestPysparkDecorator:

    def setup_method(self):
        if False:
            return 10
        db.merge_conn(Connection(conn_id='pyspark_local', conn_type='spark', host='spark://none', extra=''))

    @pytest.mark.db_test
    @mock.patch('pyspark.SparkConf.setAppName')
    @mock.patch('pyspark.sql.SparkSession')
    def test_pyspark_decorator_with_connection(self, spark_mock, conf_mock, dag_maker):
        if False:
            for i in range(10):
                print('nop')

        @task.pyspark(conn_id='pyspark_local', config_kwargs={'spark.executor.memory': '2g'})
        def f(spark, sc):
            if False:
                i = 10
                return i + 15
            import random
            return [random.random() for _ in range(100)]
        with dag_maker():
            ret = f()
        dr = dag_maker.create_dagrun()
        ret.operator.run(start_date=dr.execution_date, end_date=dr.execution_date)
        ti = dr.get_task_instances()[0]
        assert len(ti.xcom_pull()) == 100
        conf_mock().set.assert_called_with('spark.executor.memory', '2g')
        conf_mock().setMaster.assert_called_once_with('spark://none')
        spark_mock.builder.config.assert_called_once_with(conf=conf_mock())

    @pytest.mark.db_test
    @mock.patch('pyspark.SparkConf.setAppName')
    @mock.patch('pyspark.sql.SparkSession')
    def test_simple_pyspark_decorator(self, spark_mock, conf_mock, dag_maker):
        if False:
            i = 10
            return i + 15
        e = 2

        @task.pyspark
        def f():
            if False:
                print('Hello World!')
            return e
        with dag_maker():
            ret = f()
        dr = dag_maker.create_dagrun()
        ret.operator.run(start_date=dr.execution_date, end_date=dr.execution_date)
        ti = dr.get_task_instances()[0]
        assert ti.xcom_pull() == e
        conf_mock().setMaster.assert_called_once_with('local[*]')
        spark_mock.builder.config.assert_called_once_with(conf=conf_mock())