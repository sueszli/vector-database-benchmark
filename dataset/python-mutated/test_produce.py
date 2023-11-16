from __future__ import annotations
import json
import logging
import pytest
from confluent_kafka import Consumer
from airflow.models import Connection
from airflow.providers.apache.kafka.operators.produce import ProduceToTopicOperator
from airflow.utils import db
log = logging.getLogger(__name__)

def _producer_function():
    if False:
        while True:
            i = 10
    for i in range(20):
        yield (json.dumps(i), json.dumps(i + 1))

@pytest.mark.integration('kafka')
class TestProduceToTopic:
    """
    test ProduceToTopicOperator
    """

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        db.merge_conn(Connection(conn_id='kafka_default', conn_type='kafka', extra=json.dumps({'socket.timeout.ms': 10, 'message.timeout.ms': 10, 'bootstrap.servers': 'broker:29092'})))

    def test_producer_operator_test_1(self):
        if False:
            for i in range(10):
                print('nop')
        GROUP = 'operator.producer.test.integration.test_1'
        TOPIC = 'operator.producer.test.integration.test_1'
        t = ProduceToTopicOperator(kafka_config_id='kafka_default', task_id='produce_to_topic', topic=TOPIC, producer_function='tests.integration.providers.apache.kafka.operators.test_produce._producer_function')
        t.execute(context={})
        config = {'bootstrap.servers': 'broker:29092', 'group.id': GROUP, 'enable.auto.commit': False, 'auto.offset.reset': 'beginning'}
        c = Consumer(config)
        c.subscribe([TOPIC])
        msg = c.consume()
        assert msg[0].key() == b'0'
        assert msg[0].value() == b'1'

    def test_producer_operator_test_2(self):
        if False:
            while True:
                i = 10
        GROUP = 'operator.producer.test.integration.test_2'
        TOPIC = 'operator.producer.test.integration.test_2'
        t = ProduceToTopicOperator(kafka_config_id='kafka_default', task_id='produce_to_topic', topic=TOPIC, producer_function=_producer_function)
        t.execute(context={})
        config = {'bootstrap.servers': 'broker:29092', 'group.id': GROUP, 'enable.auto.commit': False, 'auto.offset.reset': 'beginning'}
        c = Consumer(config)
        c.subscribe([TOPIC])
        msg = c.consume()
        assert msg[0].key() == b'0'
        assert msg[0].value() == b'1'