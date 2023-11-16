from typing import Any, Dict
from unittest.mock import Mock
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import EventTypes
from synapse.rest import admin
from synapse.rest.client import login, room
from synapse.server import HomeServer
from synapse.types import JsonDict, create_requester
from synapse.util import Clock
from synapse.visibility import filter_events_for_client
from tests import unittest
from tests.unittest import override_config
one_hour_ms = 3600000
one_day_ms = one_hour_ms * 24

class RetentionTestCase(unittest.HomeserverTestCase):
    servlets = [admin.register_servlets, login.register_servlets, room.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            print('Hello World!')
        config = self.default_config()
        retention_config = {'enabled': True, 'default_policy': {'min_lifetime': one_day_ms, 'max_lifetime': one_day_ms * 3}, 'allowed_lifetime_min': one_day_ms, 'allowed_lifetime_max': one_day_ms * 3}
        retention_config.update(config.get('retention', {}))
        config['retention'] = retention_config
        self.hs = self.setup_test_homeserver(config=config)
        return self.hs

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.user_id = self.register_user('user', 'password')
        self.token = self.login('user', 'password')
        self.store = self.hs.get_datastores().main
        self.serializer = self.hs.get_event_client_serializer()
        self.clock = self.hs.get_clock()

    def test_retention_event_purged_with_state_event(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Tests that expired events are correctly purged when the room's retention policy\n        is defined by a state event.\n        "
        room_id = self.helper.create_room_as(self.user_id, tok=self.token)
        lifetime = one_day_ms * 2
        self.helper.send_state(room_id=room_id, event_type=EventTypes.Retention, body={'max_lifetime': lifetime}, tok=self.token)
        self._test_retention_event_purged(room_id, one_day_ms * 1.5)

    def test_retention_event_purged_with_state_event_outside_allowed(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that the server configuration can override the policy for a room when\n        running the purge jobs.\n        '
        room_id = self.helper.create_room_as(self.user_id, tok=self.token)
        self.helper.send_state(room_id=room_id, event_type=EventTypes.Retention, body={'max_lifetime': one_day_ms * 4}, tok=self.token)
        self._test_retention_event_purged(room_id, one_day_ms * 1.5)
        self.helper.send_state(room_id=room_id, event_type=EventTypes.Retention, body={'max_lifetime': one_hour_ms}, tok=self.token)
        self._test_retention_event_purged(room_id, one_day_ms * 0.5)

    def test_retention_event_purged_without_state_event(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Tests that expired events are correctly purged when the room's retention policy\n        is defined by the server's configuration's default retention policy.\n        "
        room_id = self.helper.create_room_as(self.user_id, tok=self.token)
        self._test_retention_event_purged(room_id, one_day_ms * 2)

    @override_config({'retention': {'purge_jobs': [{'interval': '5d'}]}})
    def test_visibility(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Tests that synapse.visibility.filter_events_for_client correctly filters out\n        outdated events, even if the purge job hasn't got to them yet.\n\n        We do this by setting a very long time between purge jobs.\n        "
        store = self.hs.get_datastores().main
        storage_controllers = self.hs.get_storage_controllers()
        room_id = self.helper.create_room_as(self.user_id, tok=self.token)
        resp = self.helper.send(room_id=room_id, body='1', tok=self.token)
        first_event_id = resp.get('event_id')
        assert isinstance(first_event_id, str)
        self.reactor.advance(one_day_ms * 2 / 1000)
        resp = self.helper.send(room_id=room_id, body='2', tok=self.token)
        valid_event_id = resp.get('event_id')
        assert isinstance(valid_event_id, str)
        self.reactor.advance(one_day_ms * 2 / 1000)
        events = self.get_success(store.get_events_as_list([first_event_id, valid_event_id]))
        self.assertEqual(2, len(events), 'events retrieved from database')
        filtered_events = self.get_success(filter_events_for_client(storage_controllers, self.user_id, events))
        self.assertEqual(len(filtered_events), 1, filtered_events)
        self.assertEqual(filtered_events[0].event_id, valid_event_id, filtered_events)

    def _test_retention_event_purged(self, room_id: str, increment: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Run the following test scenario to test the message retention policy support:\n\n        1. Send event 1\n        2. Increment time by `increment`\n        3. Send event 2\n        4. Increment time by `increment`\n        5. Check that event 1 has been purged\n        6. Check that event 2 has not been purged\n        7. Check that state events that were sent before event 1 aren't purged.\n        The main reason for sending a second event is because currently Synapse won't\n        purge the latest message in a room because it would otherwise result in a lack of\n        forward extremities for this room. It's also a good thing to ensure the purge jobs\n        aren't too greedy and purge messages they shouldn't.\n\n        Args:\n            room_id: The ID of the room to test retention in.\n            increment: The number of milliseconds to advance the clock each time. Must be\n                defined so that events in the room aren't purged if they are `increment`\n                old but are purged if they are `increment * 2` old.\n        "
        message_handler = self.hs.get_message_handler()
        create_event = self.get_success(message_handler.get_room_data(create_requester(self.user_id), room_id, EventTypes.Create, state_key=''))
        resp = self.helper.send(room_id=room_id, body='1', tok=self.token)
        expired_event_id = resp.get('event_id')
        assert expired_event_id is not None
        expired_event = self.get_event(expired_event_id)
        self.assertEqual(expired_event.get('content', {}).get('body'), '1', expired_event)
        self.reactor.advance(increment / 1000)
        resp = self.helper.send(room_id=room_id, body='2', tok=self.token)
        valid_event_id = resp.get('event_id')
        assert valid_event_id is not None
        self.reactor.advance(increment / 1000)
        self.get_event(expired_event_id, expect_none=True)
        valid_event = self.get_event(valid_event_id)
        self.assertEqual(valid_event.get('content', {}).get('body'), '2', valid_event)
        self.get_event(room_id, bool(create_event))

    def get_event(self, event_id: str, expect_none: bool=False) -> JsonDict:
        if False:
            while True:
                i = 10
        event = self.get_success(self.store.get_event(event_id, allow_none=True))
        if expect_none:
            self.assertIsNone(event)
            return {}
        assert event is not None
        time_now = self.clock.time_msec()
        serialized = self.get_success(self.serializer.serialize_event(event, time_now))
        return serialized

class RetentionNoDefaultPolicyTestCase(unittest.HomeserverTestCase):
    servlets = [admin.register_servlets, login.register_servlets, room.register_servlets]

    def default_config(self) -> Dict[str, Any]:
        if False:
            return 10
        config = super().default_config()
        retention_config = {'enabled': True}
        retention_config.update(config.get('retention', {}))
        config['retention'] = retention_config
        return config

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            print('Hello World!')
        mock_federation_client = Mock(spec=['backfill'])
        self.hs = self.setup_test_homeserver(federation_client=mock_federation_client)
        return self.hs

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.user_id = self.register_user('user', 'password')
        self.token = self.login('user', 'password')

    def test_no_default_policy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Tests that an event doesn't get expired if there is neither a default retention\n        policy nor a policy specific to the room.\n        "
        room_id = self.helper.create_room_as(self.user_id, tok=self.token)
        self._test_retention(room_id)

    def test_state_policy(self) -> None:
        if False:
            return 10
        "Tests that an event gets correctly expired if there is no default retention\n        policy but there's a policy specific to the room.\n        "
        room_id = self.helper.create_room_as(self.user_id, tok=self.token)
        self.helper.send_state(room_id=room_id, event_type=EventTypes.Retention, body={'max_lifetime': one_day_ms * 35}, tok=self.token)
        self._test_retention(room_id, expected_code_for_first_event=404)

    @unittest.override_config({'retention': {'enabled': False}})
    def test_visibility_when_disabled(self) -> None:
        if False:
            i = 10
            return i + 15
        'Retention policies should be ignored when the retention feature is disabled.'
        room_id = self.helper.create_room_as(self.user_id, tok=self.token)
        self.helper.send_state(room_id=room_id, event_type=EventTypes.Retention, body={'max_lifetime': one_day_ms}, tok=self.token)
        resp = self.helper.send(room_id=room_id, body='test', tok=self.token)
        self.reactor.advance(one_day_ms * 2 / 1000)
        self.get_event(room_id, resp['event_id'])

    def _test_retention(self, room_id: str, expected_code_for_first_event: int=200) -> None:
        if False:
            for i in range(10):
                print('nop')
        resp = self.helper.send(room_id=room_id, body='1', tok=self.token)
        first_event_id = resp.get('event_id')
        assert first_event_id is not None
        expired_event = self.get_event(room_id, first_event_id)
        self.assertEqual(expired_event.get('content', {}).get('body'), '1', expired_event)
        self.reactor.advance(one_day_ms * 30 / 1000)
        resp = self.helper.send(room_id=room_id, body='2', tok=self.token)
        second_event_id = resp.get('event_id')
        assert second_event_id is not None
        self.reactor.advance(one_day_ms * 30 / 1000)
        first_event = self.get_event(room_id, first_event_id, expected_code=expected_code_for_first_event)
        if expected_code_for_first_event == 200:
            self.assertEqual(first_event.get('content', {}).get('body'), '1', first_event)
        second_event = self.get_event(room_id, second_event_id)
        self.assertEqual(second_event.get('content', {}).get('body'), '2', second_event)

    def get_event(self, room_id: str, event_id: str, expected_code: int=200) -> JsonDict:
        if False:
            print('Hello World!')
        url = '/_matrix/client/r0/rooms/%s/event/%s' % (room_id, event_id)
        channel = self.make_request('GET', url, access_token=self.token)
        self.assertEqual(channel.code, expected_code, channel.result)
        return channel.json_body