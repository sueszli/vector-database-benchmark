from __future__ import annotations
import json
import logging
import pytest
from airflow.exceptions import TaskDeferred
from airflow.models import Connection
from airflow.providers.apache.kafka.sensors.kafka import AwaitMessageSensor, AwaitMessageTriggerFunctionSensor
from airflow.utils import db
pytestmark = pytest.mark.db_test
log = logging.getLogger(__name__)

def _return_true(message):
    if False:
        i = 10
        return i + 15
    return True

class TestSensors:
    """
    Test Sensors
    """

    def setup_method(self):
        if False:
            print('Hello World!')
        db.merge_conn(Connection(conn_id='kafka_d', conn_type='kafka', extra=json.dumps({'socket.timeout.ms': 10, 'bootstrap.servers': 'localhost:9092', 'group.id': 'test_group'})))

    def test_await_message_good(self):
        if False:
            while True:
                i = 10
        sensor = AwaitMessageSensor(kafka_config_id='kafka_d', topics=['test'], task_id='test', apply_function=_return_true)
        with pytest.raises(TaskDeferred):
            sensor.execute(context={})

    def test_await_execute_complete(self):
        if False:
            for i in range(10):
                print('nop')
        sensor = AwaitMessageSensor(kafka_config_id='kafka_d', topics=['test'], task_id='test', apply_function=_return_true)
        assert 'test' == sensor.execute_complete(context={}, event='test')

    def test_await_message_trigger_event(self):
        if False:
            print('Hello World!')
        sensor = AwaitMessageTriggerFunctionSensor(kafka_config_id='kafka_d', topics=['test'], task_id='test', apply_function=_return_true, event_triggered_function=_return_true)
        with pytest.raises(TaskDeferred):
            sensor.execute(context={})

    def test_await_message_trigger_event_execute_complete(self):
        if False:
            print('Hello World!')
        sensor = AwaitMessageTriggerFunctionSensor(kafka_config_id='kafka_d', topics=['test'], task_id='test', apply_function=_return_true, event_triggered_function=_return_true)
        with pytest.raises(TaskDeferred):
            sensor.execute_complete(context={})