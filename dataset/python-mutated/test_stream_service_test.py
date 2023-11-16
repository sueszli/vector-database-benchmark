import unittest
from unittest.mock import patch
import grpc
from apache_beam.portability.api import beam_interactive_api_pb2
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.portability.api import beam_runner_api_pb2_grpc
from apache_beam.testing.test_stream_service import TestStreamServiceController
beam_runner_api_pb2.TestStreamPayload.__test__ = False
beam_interactive_api_pb2.TestStreamFileHeader.__test__ = False
beam_interactive_api_pb2.TestStreamFileRecord.__test__ = False

class EventsReader:

    def __init__(self, expected_key):
        if False:
            return 10
        self._expected_key = expected_key

    def read_multiple(self, keys):
        if False:
            for i in range(10):
                print('nop')
        if keys != self._expected_key:
            raise ValueError('Expected key ({}) is not argument({})'.format(self._expected_key, keys))
        for i in range(10):
            e = beam_runner_api_pb2.TestStreamPayload.Event()
            e.element_event.elements.append(beam_runner_api_pb2.TestStreamPayload.TimestampedElement(timestamp=i))
            yield e
EXPECTED_KEY = 'key'
EXPECTED_KEYS = [EXPECTED_KEY]

class TestStreamServiceTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.controller = TestStreamServiceController(EventsReader(expected_key=[('full', EXPECTED_KEY)]))
        self.controller.start()
        channel = grpc.insecure_channel(self.controller.endpoint)
        self.stub = beam_runner_api_pb2_grpc.TestStreamServiceStub(channel)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.controller.stop()

    def test_normal_run(self):
        if False:
            return 10
        r = self.stub.Events(beam_runner_api_pb2.EventsRequest(output_ids=EXPECTED_KEYS))
        events = [e for e in r]
        expected_events = [e for e in EventsReader(expected_key=[EXPECTED_KEYS]).read_multiple([EXPECTED_KEYS])]
        self.assertEqual(events, expected_events)

    def test_multiple_sessions(self):
        if False:
            while True:
                i = 10
        resp_a = self.stub.Events(beam_runner_api_pb2.EventsRequest(output_ids=EXPECTED_KEYS))
        resp_b = self.stub.Events(beam_runner_api_pb2.EventsRequest(output_ids=EXPECTED_KEYS))
        events_a = []
        events_b = []
        done = False
        while not done:
            a_is_done = False
            b_is_done = False
            try:
                events_a.append(next(resp_a))
            except StopIteration:
                a_is_done = True
            try:
                events_b.append(next(resp_b))
            except StopIteration:
                b_is_done = True
            done = a_is_done and b_is_done
        expected_events = [e for e in EventsReader(expected_key=[EXPECTED_KEYS]).read_multiple([EXPECTED_KEYS])]
        self.assertEqual(events_a, expected_events)
        self.assertEqual(events_b, expected_events)

class TestStreamServiceStartStopTest(unittest.TestCase):
    from grpc import _server

    def setUp(self):
        if False:
            while True:
                i = 10
        self.controller = TestStreamServiceController(EventsReader(expected_key=[('full', EXPECTED_KEY)]))
        self.assertFalse(self.controller._server_started)
        self.assertFalse(self.controller._server_stopped)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.controller.stop()

    def test_start_when_never_started(self):
        if False:
            for i in range(10):
                print('nop')
        with patch.object(self._server._Server, 'start', wraps=self.controller._server.start) as mock_start:
            self.controller.start()
            mock_start.assert_called_once()
            self.assertTrue(self.controller._server_started)
            self.assertFalse(self.controller._server_stopped)

    def test_start_noop_when_already_started(self):
        if False:
            while True:
                i = 10
        with patch.object(self._server._Server, 'start', wraps=self.controller._server.start) as mock_start:
            self.controller.start()
            mock_start.assert_called_once()
            self.controller.start()
            mock_start.assert_called_once()

    def test_start_noop_when_already_stopped(self):
        if False:
            print('Hello World!')
        with patch.object(self._server._Server, 'start', wraps=self.controller._server.start) as mock_start:
            self.controller.start()
            self.controller.stop()
            mock_start.assert_called_once()
            self.controller.start()
            mock_start.assert_called_once()

    def test_stop_noop_when_not_started(self):
        if False:
            while True:
                i = 10
        with patch.object(self._server._Server, 'stop', wraps=self.controller._server.stop) as mock_stop:
            self.controller.stop()
            mock_stop.assert_not_called()

    def test_stop_when_already_started(self):
        if False:
            for i in range(10):
                print('nop')
        with patch.object(self._server._Server, 'stop', wraps=self.controller._server.stop) as mock_stop:
            self.controller.start()
            mock_stop.assert_not_called()
            self.controller.stop()
            mock_stop.assert_called_once()
            self.assertFalse(self.controller._server_started)
            self.assertTrue(self.controller._server_stopped)

    def test_stop_noop_when_already_stopped(self):
        if False:
            for i in range(10):
                print('nop')
        with patch.object(self._server._Server, 'stop', wraps=self.controller._server.stop) as mock_stop:
            self.controller.start()
            self.controller.stop()
            mock_stop.assert_called_once()
            self.controller.stop()
            mock_stop.assert_called_once()
if __name__ == '__main__':
    unittest.main()