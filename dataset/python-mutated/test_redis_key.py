from __future__ import annotations
import pytest
from airflow.models.dag import DAG
from airflow.providers.redis.hooks.redis import RedisHook
from airflow.providers.redis.sensors.redis_key import RedisKeySensor
from airflow.utils import timezone
DEFAULT_DATE = timezone.datetime(2017, 1, 1)

@pytest.mark.integration('celery')
class TestRedisSensor:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
        self.dag = DAG('test_dag_id', default_args=args)
        self.sensor = RedisKeySensor(task_id='test_task', redis_conn_id='redis_default', dag=self.dag, key='test_key')

    def test_poke(self):
        if False:
            return 10
        hook = RedisHook(redis_conn_id='redis_default')
        redis = hook.get_conn()
        redis.set('test_key', 'test_value')
        assert self.sensor.poke(None), 'Key exists on first call.'
        redis.delete('test_key')
        assert not self.sensor.poke(None), 'Key does NOT exists on second call.'