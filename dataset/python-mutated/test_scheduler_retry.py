import eventlet
import mock
import pymongo
import uuid
from st2tests import config as test_config
test_config.parse_args()
from st2actions.scheduler import handler
from st2common.models.db import execution_queue as ex_q_db
from st2common.persistence import execution_queue as ex_q_db_access
from st2tests.base import CleanDbTestCase
__all__ = ['SchedulerHandlerRetryTestCase']
MOCK_QUEUE_ITEM = ex_q_db.ActionExecutionSchedulingQueueItemDB(liveaction_id=uuid.uuid4().hex)

class SchedulerHandlerRetryTestCase(CleanDbTestCase):

    @mock.patch.object(handler.ActionExecutionSchedulingQueueHandler, '_get_next_execution', mock.MagicMock(side_effect=[pymongo.errors.ConnectionFailure(), MOCK_QUEUE_ITEM]))
    @mock.patch.object(eventlet.GreenPool, 'spawn', mock.MagicMock(return_value=None))
    def test_handler_retry_connection_error(self):
        if False:
            while True:
                i = 10
        scheduling_queue_handler = handler.ActionExecutionSchedulingQueueHandler()
        scheduling_queue_handler.process()
        calls = [mock.call(scheduling_queue_handler._handle_execution, MOCK_QUEUE_ITEM)]
        eventlet.GreenPool.spawn.assert_has_calls(calls)

    @mock.patch.object(handler.ActionExecutionSchedulingQueueHandler, '_get_next_execution', mock.MagicMock(side_effect=[pymongo.errors.ConnectionFailure()] * 3))
    @mock.patch.object(eventlet.GreenPool, 'spawn', mock.MagicMock(return_value=None))
    def test_handler_retries_exhausted(self):
        if False:
            i = 10
            return i + 15
        scheduling_queue_handler = handler.ActionExecutionSchedulingQueueHandler()
        self.assertRaises(pymongo.errors.ConnectionFailure, scheduling_queue_handler.process)
        self.assertEqual(eventlet.GreenPool.spawn.call_count, 0)

    @mock.patch.object(handler.ActionExecutionSchedulingQueueHandler, '_get_next_execution', mock.MagicMock(side_effect=KeyError()))
    @mock.patch.object(eventlet.GreenPool, 'spawn', mock.MagicMock(return_value=None))
    def test_handler_retry_unexpected_error(self):
        if False:
            for i in range(10):
                print('nop')
        scheduling_queue_handler = handler.ActionExecutionSchedulingQueueHandler()
        self.assertRaises(KeyError, scheduling_queue_handler.process)
        self.assertEqual(eventlet.GreenPool.spawn.call_count, 0)

    @mock.patch.object(ex_q_db_access.ActionExecutionSchedulingQueue, 'query', mock.MagicMock(side_effect=[pymongo.errors.ConnectionFailure(), [MOCK_QUEUE_ITEM]]))
    @mock.patch.object(ex_q_db_access.ActionExecutionSchedulingQueue, 'add_or_update', mock.MagicMock(return_value=None))
    def test_handler_gc_retry_connection_error(self):
        if False:
            while True:
                i = 10
        scheduling_queue_handler = handler.ActionExecutionSchedulingQueueHandler()
        scheduling_queue_handler._handle_garbage_collection()
        calls = [mock.call(MOCK_QUEUE_ITEM, publish=False)]
        ex_q_db_access.ActionExecutionSchedulingQueue.add_or_update.assert_has_calls(calls)

    @mock.patch.object(ex_q_db_access.ActionExecutionSchedulingQueue, 'query', mock.MagicMock(side_effect=[pymongo.errors.ConnectionFailure()] * 3))
    @mock.patch.object(ex_q_db_access.ActionExecutionSchedulingQueue, 'add_or_update', mock.MagicMock(return_value=None))
    def test_handler_gc_retries_exhausted(self):
        if False:
            for i in range(10):
                print('nop')
        scheduling_queue_handler = handler.ActionExecutionSchedulingQueueHandler()
        self.assertRaises(pymongo.errors.ConnectionFailure, scheduling_queue_handler._handle_garbage_collection)
        self.assertEqual(ex_q_db_access.ActionExecutionSchedulingQueue.add_or_update.call_count, 0)

    @mock.patch.object(ex_q_db_access.ActionExecutionSchedulingQueue, 'query', mock.MagicMock(side_effect=KeyError()))
    @mock.patch.object(ex_q_db_access.ActionExecutionSchedulingQueue, 'add_or_update', mock.MagicMock(return_value=None))
    def test_handler_gc_unexpected_error(self):
        if False:
            while True:
                i = 10
        scheduling_queue_handler = handler.ActionExecutionSchedulingQueueHandler()
        self.assertRaises(KeyError, scheduling_queue_handler._handle_garbage_collection)
        self.assertEqual(ex_q_db_access.ActionExecutionSchedulingQueue.add_or_update.call_count, 0)