from __future__ import annotations
from unittest.mock import MagicMock
import pytest
from airflow.models.dag import DAG
from airflow.providers.redis.hooks.redis import RedisHook
from airflow.providers.redis.operators.redis_publish import RedisPublishOperator
from airflow.utils import timezone
DEFAULT_DATE = timezone.datetime(2017, 1, 1)

@pytest.mark.integration('celery')
class TestRedisPublishOperator:

    def setup_method(self):
        if False:
            return 10
        args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
        self.dag = DAG('test_redis_dag_id', default_args=args)
        self.mock_context = MagicMock()
        self.channel = 'test'

    def test_execute_hello(self):
        if False:
            i = 10
            return i + 15
        operator = RedisPublishOperator(task_id='test_task', dag=self.dag, message='hello', channel=self.channel, redis_conn_id='redis_default')
        hook = RedisHook(redis_conn_id='redis_default')
        pubsub = hook.get_conn().pubsub()
        pubsub.subscribe(self.channel)
        operator.execute(self.mock_context)
        context_calls = []
        assert self.mock_context['ti'].method_calls == context_calls, 'context calls should be same'
        message = pubsub.get_message()
        assert message['type'] == 'subscribe'
        message = pubsub.get_message()
        assert message['type'] == 'message'
        assert message['data'] == b'hello'
        pubsub.unsubscribe(self.channel)