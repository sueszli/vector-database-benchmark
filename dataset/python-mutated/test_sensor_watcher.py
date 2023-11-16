from __future__ import absolute_import
from kombu.message import Message
import mock
import unittest2
from st2common.services.sensor_watcher import SensorWatcher
from st2common.models.db.sensor import SensorTypeDB
from st2common.transport.publishers import PoolPublisher
MOCK_SENSOR_DB = SensorTypeDB(name='foo', pack='test')

class SensorWatcherTests(unittest2.TestCase):

    @mock.patch.object(Message, 'ack', mock.MagicMock())
    @mock.patch.object(PoolPublisher, 'publish', mock.MagicMock())
    def test_assert_handlers_called(self):
        if False:
            i = 10
            return i + 15
        handler_vars = {'create_handler_called': False, 'update_handler_called': False, 'delete_handler_called': False}

        def create_handler(sensor_db):
            if False:
                print('Hello World!')
            handler_vars['create_handler_called'] = True

        def update_handler(sensor_db):
            if False:
                while True:
                    i = 10
            handler_vars['update_handler_called'] = True

        def delete_handler(sensor_db):
            if False:
                i = 10
                return i + 15
            handler_vars['delete_handler_called'] = True
        sensor_watcher = SensorWatcher(create_handler, update_handler, delete_handler)
        message = Message(None, delivery_info={'routing_key': 'create'})
        sensor_watcher.process_task(MOCK_SENSOR_DB, message)
        self.assertTrue(handler_vars['create_handler_called'], 'create handler should be called.')
        message = Message(None, delivery_info={'routing_key': 'update'})
        sensor_watcher.process_task(MOCK_SENSOR_DB, message)
        self.assertTrue(handler_vars['update_handler_called'], 'update handler should be called.')
        message = Message(None, delivery_info={'routing_key': 'delete'})
        sensor_watcher.process_task(MOCK_SENSOR_DB, message)
        self.assertTrue(handler_vars['delete_handler_called'], 'delete handler should be called.')