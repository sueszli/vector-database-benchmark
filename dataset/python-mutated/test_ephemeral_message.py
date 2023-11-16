from http import HTTPStatus
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import EventContentFields, EventTypes
from synapse.rest import admin
from synapse.rest.client import room
from synapse.server import HomeServer
from synapse.types import JsonDict
from synapse.util import Clock
from tests import unittest

class EphemeralMessageTestCase(unittest.HomeserverTestCase):
    user_id = '@user:test'
    servlets = [admin.register_servlets, room.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            print('Hello World!')
        config = self.default_config()
        config['enable_ephemeral_messages'] = True
        self.hs = self.setup_test_homeserver(config=config)
        return self.hs

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.room_id = self.helper.create_room_as(self.user_id)

    def test_message_expiry_no_delay(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that sending a message sent with a m.self_destruct_after field set to the\n        past results in that event being deleted right away.\n        '
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'hello', EventContentFields.SELF_DESTRUCT_AFTER: 0})
        event_id = res['event_id']
        event_content = self.get_event(self.room_id, event_id)['content']
        self.assertFalse(bool(event_content), event_content)

    def test_message_expiry_delay(self) -> None:
        if False:
            return 10
        'Tests that sending a message with a m.self_destruct_after field set to the\n        future results in that event not being deleted right away, but advancing the\n        clock to after that expiry timestamp causes the event to be deleted.\n        '
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'hello', EventContentFields.SELF_DESTRUCT_AFTER: self.clock.time_msec() + 1000})
        event_id = res['event_id']
        event_content = self.get_event(self.room_id, event_id)['content']
        self.assertTrue(bool(event_content), event_content)
        self.reactor.advance(1)
        event_content = self.get_event(self.room_id, event_id)['content']
        self.assertFalse(bool(event_content), event_content)

    def get_event(self, room_id: str, event_id: str, expected_code: int=HTTPStatus.OK) -> JsonDict:
        if False:
            i = 10
            return i + 15
        url = '/_matrix/client/r0/rooms/%s/event/%s' % (room_id, event_id)
        channel = self.make_request('GET', url)
        self.assertEqual(channel.code, expected_code, channel.result)
        return channel.json_body