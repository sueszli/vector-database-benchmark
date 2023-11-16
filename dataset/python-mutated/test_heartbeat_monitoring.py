"""Test the monitoring of the server heartbeats."""
from __future__ import annotations
import sys
sys.path[0:0] = ['']
from test import IntegrationTest, client_knobs, unittest
from test.utils import HeartbeatEventListener, MockPool, single_client, wait_until
from pymongo.errors import ConnectionFailure
from pymongo.hello import Hello, HelloCompat
from pymongo.monitor import Monitor

class TestHeartbeatMonitoring(IntegrationTest):

    def create_mock_monitor(self, responses, uri, expected_results):
        if False:
            print('Hello World!')
        listener = HeartbeatEventListener()
        with client_knobs(heartbeat_frequency=0.1, min_heartbeat_interval=0.1, events_queue_frequency=0.1):

            class MockMonitor(Monitor):

                def _check_with_socket(self, *args, **kwargs):
                    if False:
                        i = 10
                        return i + 15
                    if isinstance(responses[1], Exception):
                        raise responses[1]
                    return (Hello(responses[1]), 99)
            m = single_client(h=uri, event_listeners=(listener,), _monitor_class=MockMonitor, _pool_class=MockPool)
            expected_len = len(expected_results)
            wait_until(lambda : len(listener.events) >= expected_len, 'publish all events')
        try:
            for (expected, actual) in zip(expected_results, listener.events):
                self.assertEqual(expected, actual.__class__.__name__)
                self.assertEqual(actual.connection_id, responses[0])
                if expected != 'ServerHeartbeatStartedEvent':
                    if isinstance(actual.reply, Hello):
                        self.assertEqual(actual.duration, 99)
                        self.assertEqual(actual.reply._doc, responses[1])
                    else:
                        self.assertEqual(actual.reply, responses[1])
        finally:
            m.close()

    def test_standalone(self):
        if False:
            i = 10
            return i + 15
        responses = (('a', 27017), {HelloCompat.LEGACY_CMD: True, 'maxWireVersion': 4, 'minWireVersion': 0, 'ok': 1})
        uri = 'mongodb://a:27017'
        expected_results = ['ServerHeartbeatStartedEvent', 'ServerHeartbeatSucceededEvent']
        self.create_mock_monitor(responses, uri, expected_results)

    def test_standalone_error(self):
        if False:
            while True:
                i = 10
        responses = (('a', 27017), ConnectionFailure('SPECIAL MESSAGE'))
        uri = 'mongodb://a:27017'
        expected_results = ['ServerHeartbeatStartedEvent', 'ServerHeartbeatFailedEvent', 'ServerHeartbeatStartedEvent', 'ServerHeartbeatFailedEvent']
        self.create_mock_monitor(responses, uri, expected_results)
if __name__ == '__main__':
    unittest.main()