from __future__ import annotations
from time import sleep
from unittest.mock import MagicMock, call
import pytest
from airflow.models.dag import DAG
from airflow.providers.redis.hooks.redis import RedisHook
from airflow.providers.redis.sensors.redis_pub_sub import RedisPubSubSensor
from airflow.utils import timezone
DEFAULT_DATE = timezone.datetime(2017, 1, 1)

@pytest.mark.integration('celery')
class TestRedisPubSubSensor:

    def setup_method(self):
        if False:
            print('Hello World!')
        args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
        self.dag = DAG('test_dag_id', default_args=args)
        self.mock_context = MagicMock()

    def test_poke_true(self):
        if False:
            return 10
        sensor = RedisPubSubSensor(task_id='test_task', dag=self.dag, channels='test', redis_conn_id='redis_default')
        hook = RedisHook(redis_conn_id='redis_default')
        redis = hook.get_conn()
        result = sensor.poke(self.mock_context)
        assert not result
        redis.publish('test', 'message')
        for _ in range(1, 10):
            result = sensor.poke(self.mock_context)
            if result:
                break
            sleep(0.1)
        assert result
        context_calls = [call.xcom_push(key='message', value={'type': 'message', 'pattern': None, 'channel': b'test', 'data': b'message'})]
        assert self.mock_context['ti'].method_calls == context_calls, 'context calls should be same'
        result = sensor.poke(self.mock_context)
        assert not result

    def test_poke_false(self):
        if False:
            for i in range(10):
                print('nop')
        sensor = RedisPubSubSensor(task_id='test_task', dag=self.dag, channels='test', redis_conn_id='redis_default')
        result = sensor.poke(self.mock_context)
        assert not result
        assert self.mock_context['ti'].method_calls == [], 'context calls should be same'
        result = sensor.poke(self.mock_context)
        assert not result
        assert self.mock_context['ti'].method_calls == [], 'context calls should be same'