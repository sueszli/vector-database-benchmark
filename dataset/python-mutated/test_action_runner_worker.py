from __future__ import absolute_import
from unittest2 import TestCase
from mock import Mock
from st2common.transport.consumers import ActionsQueueConsumer
from st2common.models.db.liveaction import LiveActionDB
from st2tests import config as test_config
test_config.parse_args()
__all__ = ['ActionsQueueConsumerTestCase']

class ActionsQueueConsumerTestCase(TestCase):

    def test_process_right_dispatcher_is_used(self):
        if False:
            print('Hello World!')
        handler = Mock()
        handler.message_type = LiveActionDB
        consumer = ActionsQueueConsumer(connection=None, queues=None, handler=handler)
        consumer._workflows_dispatcher = Mock()
        consumer._actions_dispatcher = Mock()
        body = LiveActionDB(status='scheduled', action='core.local', action_is_workflow=False)
        message = Mock()
        consumer.process(body=body, message=message)
        self.assertEqual(consumer._workflows_dispatcher.dispatch.call_count, 0)
        self.assertEqual(consumer._actions_dispatcher.dispatch.call_count, 1)
        consumer._workflows_dispatcher = Mock()
        consumer._actions_dispatcher = Mock()
        body = LiveActionDB(status='scheduled', action='core.local', action_is_workflow=True)
        message = Mock()
        consumer.process(body=body, message=message)
        self.assertEqual(consumer._workflows_dispatcher.dispatch.call_count, 1)
        self.assertEqual(consumer._actions_dispatcher.dispatch.call_count, 0)