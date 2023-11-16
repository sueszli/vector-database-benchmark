from unittest.mock import Mock, patch
from twisted.test.proto_helpers import MemoryReactor
import synapse.rest.admin
from synapse.api.constants import EduTypes, EventTypes
from synapse.rest.client import directory, login, profile, room, room_upgrade_rest_servlet
from synapse.server import HomeServer
from synapse.types import UserID, create_requester
from synapse.util import Clock
from tests import unittest

class _ShadowBannedBase(unittest.HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.banned_user_id = self.register_user('banned', 'test')
        self.banned_access_token = self.login('banned', 'test')
        self.store = self.hs.get_datastores().main
        self.get_success(self.store.set_shadow_banned(UserID.from_string(self.banned_user_id), True))
        self.other_user_id = self.register_user('otheruser', 'pass')
        self.other_access_token = self.login('otheruser', 'pass')

@patch('random.randint', new=lambda a, b: 0)
class RoomTestCase(_ShadowBannedBase):
    servlets = [synapse.rest.admin.register_servlets_for_client_rest_resource, directory.register_servlets, login.register_servlets, room.register_servlets, room_upgrade_rest_servlet.register_servlets]

    def test_invite(self) -> None:
        if False:
            while True:
                i = 10
        "Invites from shadow-banned users don't actually get sent."
        room_id = self.helper.create_room_as(self.banned_user_id, tok=self.banned_access_token)
        self.helper.invite(room=room_id, src=self.banned_user_id, tok=self.banned_access_token, targ=self.other_user_id)
        invited_rooms = self.get_success(self.store.get_invited_rooms_for_local_user(self.other_user_id))
        self.assertEqual(invited_rooms, [])

    def test_invite_3pid(self) -> None:
        if False:
            i = 10
            return i + 15
        'Ensure that a 3PID invite does not attempt to contact the identity server.'
        identity_handler = self.hs.get_identity_handler()
        identity_handler.lookup_3pid = Mock(side_effect=AssertionError('This should not get called'))
        room_id = self.helper.create_room_as(self.banned_user_id, tok=self.banned_access_token)
        channel = self.make_request('POST', '/rooms/%s/invite' % (room_id,), {'id_server': 'test', 'medium': 'email', 'address': 'test@test.test', 'id_access_token': 'anytoken'}, access_token=self.banned_access_token)
        self.assertEqual(200, channel.code, channel.result)
        identity_handler.lookup_3pid.assert_not_called()

    def test_create_room(self) -> None:
        if False:
            print('Hello World!')
        'Invitations during a room creation should be discarded, but the room still gets created.'
        channel = self.make_request('POST', '/_matrix/client/r0/createRoom', {'visibility': 'public', 'invite': [self.other_user_id]}, access_token=self.banned_access_token)
        self.assertEqual(200, channel.code, channel.result)
        room_id = channel.json_body['room_id']
        invited_rooms = self.get_success(self.store.get_invited_rooms_for_local_user(self.other_user_id))
        self.assertEqual(invited_rooms, [])
        self.helper.join(room_id, self.other_user_id, tok=self.other_access_token)
        users = self.get_success(self.store.get_users_in_room(room_id))
        self.assertCountEqual(users, ['@banned:test', '@otheruser:test'])

    def test_message(self) -> None:
        if False:
            while True:
                i = 10
        "Messages from shadow-banned users don't actually get sent."
        room_id = self.helper.create_room_as(self.other_user_id, tok=self.other_access_token)
        self.helper.join(room_id, self.banned_user_id, tok=self.banned_access_token)
        result = self.helper.send_event(room_id=room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'with right label'}, tok=self.banned_access_token)
        self.assertIn('event_id', result)
        event_id = result['event_id']
        latest_events = self.get_success(self.store.get_latest_event_ids_in_room(room_id))
        self.assertNotIn(event_id, latest_events)

    def test_upgrade(self) -> None:
        if False:
            while True:
                i = 10
        'A room upgrade should fail, but look like it succeeded.'
        room_id = self.helper.create_room_as(self.banned_user_id, tok=self.banned_access_token)
        channel = self.make_request('POST', '/_matrix/client/r0/rooms/%s/upgrade' % (room_id,), {'new_version': '6'}, access_token=self.banned_access_token)
        self.assertEqual(200, channel.code, channel.result)
        self.assertIn('replacement_room', channel.json_body)
        new_room_id = channel.json_body['replacement_room']
        summary = self.get_success(self.store.get_room_summary(new_room_id))
        self.assertEqual(summary, {})

    def test_typing(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Typing notifications should not be propagated into the room.'
        room_id = self.helper.create_room_as(self.banned_user_id, tok=self.banned_access_token)
        channel = self.make_request('PUT', '/rooms/%s/typing/%s' % (room_id, self.banned_user_id), {'typing': True, 'timeout': 30000}, access_token=self.banned_access_token)
        self.assertEqual(200, channel.code)
        event_source = self.hs.get_event_sources().sources.typing
        self.assertEqual(event_source.get_current_key(), 0)
        self.helper.join(room_id, self.other_user_id, tok=self.other_access_token)
        channel = self.make_request('PUT', '/rooms/%s/typing/%s' % (room_id, self.other_user_id), {'typing': True, 'timeout': 30000}, access_token=self.other_access_token)
        self.assertEqual(200, channel.code)
        self.assertEqual(event_source.get_current_key(), 1)
        events = self.get_success(event_source.get_new_events(user=UserID.from_string(self.other_user_id), from_key=0, limit=10, room_ids=[room_id], is_guest=False))
        self.assertEqual(events[0], [{'type': EduTypes.TYPING, 'room_id': room_id, 'content': {'user_ids': [self.other_user_id]}}])

@patch('random.randint', new=lambda a, b: 0)
class ProfileTestCase(_ShadowBannedBase):
    servlets = [synapse.rest.admin.register_servlets_for_client_rest_resource, login.register_servlets, profile.register_servlets, room.register_servlets]

    def test_displayname(self) -> None:
        if False:
            return 10
        "Profile changes should succeed, but don't end up in a room."
        original_display_name = 'banned'
        new_display_name = 'new name'
        room_id = self.helper.create_room_as(self.banned_user_id, tok=self.banned_access_token)
        channel = self.make_request('PUT', '/_matrix/client/r0/profile/%s/displayname' % (self.banned_user_id,), {'displayname': new_display_name}, access_token=self.banned_access_token)
        self.assertEqual(200, channel.code, channel.result)
        self.assertEqual(channel.json_body, {})
        channel = self.make_request('GET', '/profile/%s/displayname' % (self.banned_user_id,))
        self.assertEqual(channel.code, 200, channel.result)
        self.assertEqual(channel.json_body['displayname'], new_display_name)
        message_handler = self.hs.get_message_handler()
        event = self.get_success(message_handler.get_room_data(create_requester(self.banned_user_id), room_id, 'm.room.member', self.banned_user_id))
        assert event is not None
        self.assertEqual(event.content, {'membership': 'join', 'displayname': original_display_name})

    def test_room_displayname(self) -> None:
        if False:
            return 10
        'Changes to state events for a room should be processed, but not end up in the room.'
        original_display_name = 'banned'
        new_display_name = 'new name'
        room_id = self.helper.create_room_as(self.banned_user_id, tok=self.banned_access_token)
        channel = self.make_request('PUT', '/_matrix/client/r0/rooms/%s/state/m.room.member/%s' % (room_id, self.banned_user_id), {'membership': 'join', 'displayname': new_display_name}, access_token=self.banned_access_token)
        self.assertEqual(200, channel.code, channel.result)
        self.assertIn('event_id', channel.json_body)
        message_handler = self.hs.get_message_handler()
        event = self.get_success(message_handler.get_room_data(create_requester(self.banned_user_id), room_id, 'm.room.member', self.banned_user_id))
        assert event is not None
        self.assertEqual(event.content, {'membership': 'join', 'displayname': original_display_name})