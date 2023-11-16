from __future__ import annotations
import json
import pytest
from confluent_kafka import Producer
from airflow.models import Connection
from airflow.providers.apache.kafka.triggers.await_message import AwaitMessageTrigger
from airflow.utils import db
GROUP = 'trigger.await_message.test.integration.test_1'
TOPIC = 'trigger.await_message.test.integration.test_1'

def _apply_function(message):
    if False:
        for i in range(10):
            print('nop')
    if message.value() == bytes(TOPIC, 'utf-8'):
        return message

@pytest.mark.integration('kafka')
class TestTrigger:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        for num in [1]:
            db.merge_conn(Connection(conn_id=f'trigger.await_message.test.integration.test_{num}', conn_type='kafka', extra=json.dumps({'socket.timeout.ms': 10, 'bootstrap.servers': 'broker:29092', 'group.id': f'trigger.await_message.test.integration.test_{num}', 'enable.auto.commit': False, 'auto.offset.reset': 'beginning'})))

    @pytest.mark.asyncio
    async def test_trigger_await_message_test_1(self):
        """
        Await message waits for a message that returns truthy
        """
        TOPIC = 'trigger.await_message.test.integration.test_1'
        p = Producer(**{'bootstrap.servers': 'broker:29092'})
        for x in range(20):
            p.produce(TOPIC, 'not_this')
        p.produce(TOPIC, TOPIC)
        assert len(p) == 21
        x = p.flush()
        assert x == 0
        trigger = AwaitMessageTrigger(topics=[TOPIC], apply_function='tests.integration.providers.apache.kafka.triggers.test_await_message._apply_function', apply_function_args=None, apply_function_kwargs=None, kafka_config_id='trigger.await_message.test.integration.test_1', poll_timeout=0, poll_interval=1)
        generator = trigger.run()
        actual = await generator.__anext__()
        assert actual.payload.value() == bytes(TOPIC, 'utf-8')