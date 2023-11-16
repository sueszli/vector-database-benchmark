from __future__ import annotations
import json
import logging
import pytest
from airflow.models import Connection
from airflow.providers.apache.kafka.hooks.produce import KafkaProducerHook
from airflow.utils import db
log = logging.getLogger(__name__)
config = {'bootstrap.servers': 'broker:29092', 'group.id': 'hook.producer.integration.test'}

@pytest.mark.integration('kafka')
class TestProducerHook:
    """
    Test consumer hook.
    """

    def setup_method(self):
        if False:
            return 10
        db.merge_conn(Connection(conn_id='kafka_default', conn_type='kafka', extra=json.dumps(config)))

    def test_produce(self):
        if False:
            print('Hello World!')
        'test producer hook functionality'
        topic = 'producer_hook_integration_test'

        def acked(err, msg):
            if False:
                print('Hello World!')
            if err is not None:
                raise Exception(f'{err}')
            else:
                assert msg.topic() == topic
                assert msg.partition() == 0
                assert msg.offset() == 0
        p_hook = KafkaProducerHook(kafka_config_id='kafka_default')
        producer = p_hook.get_producer()
        producer.produce(topic, key='p1', value='p2', on_delivery=acked)
        producer.poll(0)
        producer.flush()