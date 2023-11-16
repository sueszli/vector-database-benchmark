from collections import OrderedDict
from typing import Any, Dict, List, Optional
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import EventTypes, JoinRules, Membership
from synapse.api.room_versions import RoomVersion, RoomVersions
from synapse.events import EventBase, builder
from synapse.events.snapshot import EventContext
from synapse.rest import admin
from synapse.rest.client import login, room
from synapse.server import HomeServer
from synapse.types import RoomAlias
from synapse.util import Clock
from tests.test_utils import event_injection
from tests.unittest import FederatingHomeserverTestCase, HomeserverTestCase

class KnockingStrippedStateEventHelperMixin(HomeserverTestCase):

    def send_example_state_events_to_room(self, hs: 'HomeServer', room_id: str, sender: str) -> OrderedDict:
        if False:
            return 10
        "Adds some state to a room. State events are those that should be sent to a knocking\n        user after they knock on the room, as well as some state that *shouldn't* be sent\n        to the knocking user.\n\n        Args:\n            hs: The homeserver of the sender.\n            room_id: The ID of the room to send state into.\n            sender: The ID of the user to send state as. Must be in the room.\n\n        Returns:\n            The OrderedDict of event types and content that a user is expected to see\n            after knocking on a room.\n        "
        canonical_alias = '#fancy_alias:test'
        self.get_success(self.hs.get_datastores().main.create_room_alias_association(RoomAlias.from_string(canonical_alias), room_id, ['test']))
        self.get_success(event_injection.inject_event(hs, room_version=RoomVersions.V7.identifier, room_id=room_id, sender=sender, type='com.example.secret', state_key='', content={'secret': 'password'}))
        room_state = OrderedDict([(EventTypes.JoinRules, {'content': {'join_rule': JoinRules.KNOCK}, 'state_key': ''}), (EventTypes.Name, {'content': {'name': 'A cool room'}, 'state_key': ''}), (EventTypes.RoomAvatar, {'content': {'info': {'h': 398, 'mimetype': 'image/jpeg', 'size': 31037, 'w': 394}, 'url': 'mxc://example.org/JWEIFJgwEIhweiWJE'}, 'state_key': ''}), (EventTypes.RoomEncryption, {'content': {'algorithm': 'm.megolm.v1.aes-sha2'}, 'state_key': ''}), (EventTypes.CanonicalAlias, {'content': {'alias': canonical_alias, 'alt_aliases': []}, 'state_key': ''}), (EventTypes.Topic, {'content': {'topic': 'A really cool room'}, 'state_key': ''})])
        for (event_type, event_dict) in room_state.items():
            event_content = event_dict['content']
            state_key = event_dict['state_key']
            self.get_success(event_injection.inject_event(hs, room_version=RoomVersions.V7.identifier, room_id=room_id, sender=sender, type=event_type, state_key=state_key, content=event_content))
        room_state[EventTypes.Create] = {'content': {'creator': sender, 'room_version': RoomVersions.V7.identifier}, 'state_key': ''}
        return room_state

    def check_knock_room_state_against_room_state(self, knock_room_state: List[Dict], expected_room_state: Dict) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test a list of stripped room state events received over federation against a\n        dict of expected state events.\n\n        Args:\n            knock_room_state: The list of room state that was received over federation.\n            expected_room_state: A dict containing the room state we expect to see in\n                `knock_room_state`.\n        '
        for event in knock_room_state:
            event_type = event['type']
            self.assertIn(event_type, expected_room_state)
            self.assertEqual(expected_room_state[event_type]['content'], event['content'])
            self.assertEqual(expected_room_state[event_type]['state_key'], event['state_key'])
            self.assertNotIn('signatures', event)
            expected_room_state.pop(event_type)
        self.assertEqual(len(expected_room_state), 0)

class FederationKnockingTestCase(FederatingHomeserverTestCase, KnockingStrippedStateEventHelperMixin):
    servlets = [admin.register_servlets, room.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, homeserver: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.store = homeserver.get_datastores().main

        async def approve_all_signature_checking(room_version: RoomVersion, pdu: EventBase, record_failure_callback: Any=None) -> EventBase:
            return pdu
        homeserver.get_federation_server()._check_sigs_and_hash = approve_all_signature_checking

        async def _check_event_auth(origin: Optional[str], event: EventBase, context: EventContext) -> None:
            pass
        homeserver.get_federation_event_handler()._check_event_auth = _check_event_auth
        return super().prepare(reactor, clock, homeserver)

    def test_room_state_returned_when_knocking(self) -> None:
        if False:
            return 10
        '\n        Tests that specific, stripped state events from a room are returned after\n        a remote homeserver successfully knocks on a local room.\n        '
        user_id = self.register_user('u1', 'you the one')
        user_token = self.login('u1', 'you the one')
        fake_knocking_user_id = '@user:other.example.com'
        room_id = self.helper.create_room_as('u1', is_public=False, room_version=RoomVersions.V7.identifier, tok=user_token)
        expected_room_state = self.send_example_state_events_to_room(self.hs, room_id, user_id)
        channel = self.make_signed_federation_request('GET', '/_matrix/federation/v1/make_knock/%s/%s?ver=%s' % (room_id, fake_knocking_user_id, RoomVersions.V7.identifier))
        self.assertEqual(200, channel.code, channel.result)
        knock_event = channel.json_body['event']
        self.assertEqual(knock_event['room_id'], room_id)
        self.assertEqual(knock_event['sender'], fake_knocking_user_id)
        self.assertEqual(knock_event['state_key'], fake_knocking_user_id)
        self.assertEqual(knock_event['type'], EventTypes.Member)
        self.assertEqual(knock_event['content']['membership'], Membership.KNOCK)
        signed_knock_event = builder.create_local_event_from_event_dict(self.clock, self.hs.hostname, self.hs.signing_key, room_version=RoomVersions.V7, event_dict=knock_event)
        signed_knock_event_json = signed_knock_event.get_pdu_json(self.clock.time_msec())
        channel = self.make_signed_federation_request('PUT', '/_matrix/federation/v1/send_knock/%s/%s' % (room_id, signed_knock_event.event_id), signed_knock_event_json)
        self.assertEqual(200, channel.code, channel.result)
        room_state_events = channel.json_body['knock_room_state']
        self.check_knock_room_state_against_room_state(room_state_events, expected_room_state)