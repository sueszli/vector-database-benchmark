from __future__ import annotations
import json
import pytest
from airflow.models import Connection
from airflow.providers.apache.kafka.hooks.client import KafkaAdminClientHook
from airflow.utils import db
client_config = {'socket.timeout.ms': 1000, 'bootstrap.servers': 'broker:29092'}

@pytest.mark.integration('kafka')
class TestKafkaAdminClientHook:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        db.merge_conn(Connection(conn_id='kafka_d', conn_type='kafka', extra=json.dumps(client_config)))

    def test_hook(self):
        if False:
            i = 10
            return i + 15
        'test the creation of topics'
        hook = KafkaAdminClientHook(kafka_config_id='kafka_d')
        hook.create_topic(topics=[('test_1', 1, 1), ('test_2', 1, 1)])
        kadmin = hook.get_conn
        t = kadmin.list_topics(timeout=10).topics
        assert t.get('test_2')