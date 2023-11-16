import mock
import unittest2
from st2common.stream import listener

class MockBody(object):

    def __init__(self, id):
        if False:
            for i in range(10):
                print('nop')
        self.id = id
        self.status = 'succeeded'
INCLUDE = 'test'
END_EVENT = 'test_end_event'
END_ID = 'test_end_id'
EVENTS = [(INCLUDE, MockBody('notend')), (END_EVENT, MockBody(END_ID))]

class MockQueue:

    def __init__(self):
        if False:
            return 10
        self.items = EVENTS

    def get(self, *args, **kwargs):
        if False:
            print('Hello World!')
        if len(self.items) > 0:
            return self.items.pop(0)
        return None

    def put(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.items.append(event)

class MockListener(listener.BaseListener):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(MockListener, self).__init__(*args, **kwargs)

    def get_consumers(self, consumer, channel):
        if False:
            return 10
        pass

class TestStream(unittest2.TestCase):

    @mock.patch('st2common.stream.listener.BaseListener._get_action_ref_for_body')
    @mock.patch('eventlet.Queue')
    def test_generator(self, mock_queue, get_action_ref_for_body):
        if False:
            return 10
        get_action_ref_for_body.return_value = None
        mock_queue.return_value = MockQueue()
        mock_consumer = MockListener(connection=None)
        mock_consumer._stopped = False
        app_iter = mock_consumer.generator(events=INCLUDE, end_event=END_EVENT, end_statuses=['succeeded'], end_execution_id=END_ID)
        events = EVENTS.append('')
        for (index, val) in enumerate(app_iter):
            self.assertEquals(val, events[index])