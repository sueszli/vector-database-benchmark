from __future__ import annotations
import pytest
from airflow.models import Connection
from airflow.models.dag import DAG
from airflow.providers.mongo.hooks.mongo import MongoHook
from airflow.providers.mongo.sensors.mongo import MongoSensor
from airflow.utils import db, timezone
DEFAULT_DATE = timezone.datetime(2017, 1, 1)

@pytest.mark.integration('mongo')
class TestMongoSensor:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        db.merge_conn(Connection(conn_id='mongo_test', conn_type='mongo', host='mongo', port=27017, schema='test'))
        args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
        self.dag = DAG('test_dag_id', default_args=args)
        hook = MongoHook('mongo_test')
        hook.insert_one('foo', {'bar': 'baz'})
        self.sensor = MongoSensor(task_id='test_task', mongo_conn_id='mongo_test', dag=self.dag, collection='foo', query={'bar': 'baz'})

    def test_poke(self):
        if False:
            return 10
        assert self.sensor.poke(None)

    def test_sensor_with_db(self):
        if False:
            return 10
        hook = MongoHook('mongo_test')
        hook.insert_one('nontest', {'1': '2'}, mongo_db='nontest')
        sensor = MongoSensor(task_id='test_task2', mongo_conn_id='mongo_test', dag=self.dag, collection='nontest', query={'1': '2'}, mongo_db='nontest')
        assert sensor.poke(None)