"""Test the database module."""
from __future__ import annotations
import sys
import time
sys.path[0:0] = ['']
from test import IntegrationTest, client_context, unittest
from test.utils import HeartbeatEventListener, ServerEventListener, rs_or_single_client, single_client, wait_until
from pymongo import monitoring
from pymongo.hello import HelloCompat

class TestStreamingProtocol(IntegrationTest):

    @client_context.require_failCommand_appName
    def test_failCommand_streaming(self):
        if False:
            while True:
                i = 10
        listener = ServerEventListener()
        hb_listener = HeartbeatEventListener()
        client = rs_or_single_client(event_listeners=[listener, hb_listener], heartbeatFrequencyMS=500, appName='failingHeartbeatTest')
        self.addCleanup(client.close)
        client.admin.command('ping')
        address = client.address
        listener.reset()
        fail_hello = {'configureFailPoint': 'failCommand', 'mode': {'times': 4}, 'data': {'failCommands': [HelloCompat.LEGACY_CMD, 'hello'], 'closeConnection': False, 'errorCode': 10107, 'appName': 'failingHeartbeatTest'}}
        with self.fail_point(fail_hello):

            def _marked_unknown(event):
                if False:
                    while True:
                        i = 10
                return event.server_address == address and (not event.new_description.is_server_type_known)

            def _discovered_node(event):
                if False:
                    while True:
                        i = 10
                return event.server_address == address and (not event.previous_description.is_server_type_known) and event.new_description.is_server_type_known

            def marked_unknown():
                if False:
                    while True:
                        i = 10
                return len(listener.matching(_marked_unknown)) >= 1

            def rediscovered():
                if False:
                    return 10
                return len(listener.matching(_discovered_node)) >= 1
            wait_until(marked_unknown, 'mark node unknown')
            wait_until(rediscovered, 'rediscover node')
        client.admin.command('ping')

    @client_context.require_failCommand_appName
    def test_streaming_rtt(self):
        if False:
            for i in range(10):
                print('nop')
        listener = ServerEventListener()
        hb_listener = HeartbeatEventListener()
        name = 'streamingRttTest'
        delay_hello: dict = {'configureFailPoint': 'failCommand', 'mode': {'times': 1000}, 'data': {'failCommands': [HelloCompat.LEGACY_CMD, 'hello'], 'blockConnection': True, 'blockTimeMS': 20}}
        with self.fail_point(delay_hello):
            client = rs_or_single_client(event_listeners=[listener, hb_listener], heartbeatFrequencyMS=500, appName=name)
            self.addCleanup(client.close)
            client.admin.command('ping')
            address = client.address
        delay_hello['data']['blockTimeMS'] = 500
        delay_hello['data']['appName'] = name
        with self.fail_point(delay_hello):

            def rtt_exceeds_250_ms():
                if False:
                    i = 10
                    return i + 15
                topology = client._topology
                sd = topology.description.server_descriptions()[address]
                assert sd.round_trip_time is not None
                return sd.round_trip_time > 0.25
            wait_until(rtt_exceeds_250_ms, 'exceed 250ms RTT')
        client.admin.command('ping')

        def changed_event(event):
            if False:
                print('Hello World!')
            return event.server_address == address and isinstance(event, monitoring.ServerDescriptionChangedEvent)
        events = listener.matching(changed_event)
        self.assertEqual(1, len(events))
        self.assertGreater(events[0].new_description.round_trip_time, 0)

    @client_context.require_version_min(4, 9, -1)
    @client_context.require_failCommand_appName
    def test_monitor_waits_after_server_check_error(self):
        if False:
            return 10
        fail_hello = {'mode': {'times': 5}, 'data': {'failCommands': [HelloCompat.LEGACY_CMD, 'hello'], 'errorCode': 1234, 'appName': 'SDAMMinHeartbeatFrequencyTest'}}
        with self.fail_point(fail_hello):
            start = time.time()
            client = single_client(appName='SDAMMinHeartbeatFrequencyTest', serverSelectionTimeoutMS=5000)
            self.addCleanup(client.close)
            client.admin.command('ping')
            duration = time.time() - start
            self.assertGreaterEqual(duration, 2)
            self.assertLessEqual(duration, 3.5)

    @client_context.require_failCommand_appName
    def test_heartbeat_awaited_flag(self):
        if False:
            i = 10
            return i + 15
        hb_listener = HeartbeatEventListener()
        client = single_client(event_listeners=[hb_listener], heartbeatFrequencyMS=500, appName='heartbeatEventAwaitedFlag')
        self.addCleanup(client.close)
        client.admin.command('ping')

        def hb_succeeded(event):
            if False:
                while True:
                    i = 10
            return isinstance(event, monitoring.ServerHeartbeatSucceededEvent)

        def hb_failed(event):
            if False:
                return 10
            return isinstance(event, monitoring.ServerHeartbeatFailedEvent)
        fail_heartbeat = {'mode': {'times': 2}, 'data': {'failCommands': [HelloCompat.LEGACY_CMD, 'hello'], 'closeConnection': True, 'appName': 'heartbeatEventAwaitedFlag'}}
        with self.fail_point(fail_heartbeat):
            wait_until(lambda : hb_listener.matching(hb_failed), 'published failed event')
        client.admin.command('ping')
        hb_succeeded_events = hb_listener.matching(hb_succeeded)
        hb_failed_events = hb_listener.matching(hb_failed)
        self.assertFalse(hb_succeeded_events[0].awaited)
        self.assertTrue(hb_failed_events[0].awaited)
        events = [type(e) for e in hb_listener.events[:4]]
        if events == [monitoring.ServerHeartbeatStartedEvent, monitoring.ServerHeartbeatSucceededEvent, monitoring.ServerHeartbeatStartedEvent, monitoring.ServerHeartbeatFailedEvent]:
            self.assertFalse(hb_succeeded_events[1].awaited)
        else:
            self.assertTrue(hb_succeeded_events[1].awaited)
if __name__ == '__main__':
    unittest.main()