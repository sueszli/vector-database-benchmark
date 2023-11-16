from __future__ import absolute_import
import random
import eventlet
from kombu import Exchange
from kombu import Queue
from unittest2 import TestCase
from st2common.transport.consumers import ActionsQueueConsumer
from st2common.transport.publishers import PoolPublisher
from st2common.transport import utils as transport_utils
from st2common.models.db.liveaction import LiveActionDB
__all__ = ['ActionsQueueConsumerTestCase']

class ActionsQueueConsumerTestCase(TestCase):
    message_count = 0
    message_type = LiveActionDB

    def test_stop_consumption_on_shutdown(self):
        if False:
            i = 10
            return i + 15
        exchange = Exchange('st2.execution.test', type='topic')
        queue_name = 'test-' + str(random.randint(1, 10000))
        queue = Queue(name=queue_name, exchange=exchange, routing_key='#', auto_delete=True)
        publisher = PoolPublisher()
        with transport_utils.get_connection() as connection:
            connection.connect()
            watcher = ActionsQueueConsumer(connection=connection, queues=queue, handler=self)
            watcher_thread = eventlet.greenthread.spawn(watcher.run)
        eventlet.sleep(0.5)
        body = LiveActionDB(status='scheduled', action='core.local', action_is_workflow=False)
        publisher.publish(payload=body, exchange=exchange)
        eventlet.sleep(0.2)
        self.assertEqual(self.message_count, 1)
        body = LiveActionDB(status='scheduled', action='core.local', action_is_workflow=True)
        watcher.shutdown()
        eventlet.sleep(1)
        publisher.publish(payload=body, exchange=exchange)
        self.assertEqual(self.message_count, 1)
        watcher_thread.kill()

    def process(self, liveaction):
        if False:
            i = 10
            return i + 15
        self.message_count = self.message_count + 1