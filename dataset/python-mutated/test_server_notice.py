from typing import List, Sequence
from twisted.test.proto_helpers import MemoryReactor
import synapse.rest.admin
from synapse.api.errors import Codes
from synapse.rest.client import login, room, sync
from synapse.server import HomeServer
from synapse.storage.roommember import RoomsForUser
from synapse.types import JsonDict
from synapse.util import Clock
from synapse.util.stringutils import random_string
from tests import unittest
from tests.unittest import override_config

class ServerNoticeTestCase(unittest.HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets, login.register_servlets, room.register_servlets, sync.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.store = hs.get_datastores().main
        self.room_shutdown_handler = hs.get_room_shutdown_handler()
        self.pagination_handler = hs.get_pagination_handler()
        self.server_notices_manager = self.hs.get_server_notices_manager()
        self.admin_user = self.register_user('admin', 'pass', admin=True)
        self.admin_user_tok = self.login('admin', 'pass')
        self.other_user = self.register_user('user', 'pass')
        self.other_user_token = self.login('user', 'pass')
        self.url = '/_synapse/admin/v1/send_server_notice'

    def test_no_auth(self) -> None:
        if False:
            print('Hello World!')
        'Try to send a server notice without authentication.'
        channel = self.make_request('POST', self.url)
        self.assertEqual(401, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.MISSING_TOKEN, channel.json_body['errcode'])

    def test_requester_is_no_admin(self) -> None:
        if False:
            while True:
                i = 10
        'If the user is not a server admin, an error is returned.'
        channel = self.make_request('POST', self.url, access_token=self.other_user_token)
        self.assertEqual(403, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.FORBIDDEN, channel.json_body['errcode'])

    @override_config({'server_notices': {'system_mxid_localpart': 'notices'}})
    def test_user_does_not_exist(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that a lookup for a user that does not exist returns a 404'
        channel = self.make_request('POST', self.url, access_token=self.admin_user_tok, content={'user_id': '@unknown_person:test', 'content': ''})
        self.assertEqual(404, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.NOT_FOUND, channel.json_body['errcode'])

    @override_config({'server_notices': {'system_mxid_localpart': 'notices'}})
    def test_user_is_not_local(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Tests that a lookup for a user that is not a local returns a 400\n        '
        channel = self.make_request('POST', self.url, access_token=self.admin_user_tok, content={'user_id': '@unknown_person:unknown_domain', 'content': ''})
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual('Server notices can only be sent to local users', channel.json_body['error'])

    @override_config({'server_notices': {'system_mxid_localpart': 'notices'}})
    def test_invalid_parameter(self) -> None:
        if False:
            while True:
                i = 10
        'If parameters are invalid, an error is returned.'
        channel = self.make_request('POST', self.url, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.NOT_JSON, channel.json_body['errcode'])
        channel = self.make_request('POST', self.url, access_token=self.admin_user_tok, content={'user_id': self.other_user})
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.MISSING_PARAM, channel.json_body['errcode'])
        channel = self.make_request('POST', self.url, access_token=self.admin_user_tok, content={'user_id': self.other_user, 'content': ''})
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.UNKNOWN, channel.json_body['errcode'])
        self.assertEqual("'body' not in content", channel.json_body['error'])
        channel = self.make_request('POST', self.url, access_token=self.admin_user_tok, content={'user_id': self.other_user, 'content': {'body': ''}})
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.UNKNOWN, channel.json_body['errcode'])
        self.assertEqual("'msgtype' not in content", channel.json_body['error'])

    @override_config({'server_notices': {'system_mxid_localpart': 'notices', 'system_mxid_avatar_url': 'somthingwrong'}, 'max_avatar_size': '10M'})
    def test_invalid_avatar_url(self) -> None:
        if False:
            print('Hello World!')
        'If avatar url in homeserver.yaml is invalid and\n        "check avatar size and mime type" is set, an error is returned.\n        TODO: Should be checked when reading the configuration.'
        channel = self.make_request('POST', self.url, access_token=self.admin_user_tok, content={'user_id': self.other_user, 'content': {'msgtype': 'm.text', 'body': 'test msg'}})
        self.assertEqual(500, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.UNKNOWN, channel.json_body['errcode'])

    @override_config({'server_notices': {'system_mxid_localpart': 'notices', 'system_mxid_display_name': 'test display name', 'system_mxid_avatar_url': None}, 'max_avatar_size': '10M'})
    def test_displayname_is_set_avatar_is_none(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Tests that sending a server notices is successfully,\n        if a display_name is set, avatar_url is `None` and\n        "check avatar size and mime type" is set.\n        '
        channel = self.make_request('POST', self.url, access_token=self.admin_user_tok, content={'user_id': self.other_user, 'content': {'msgtype': 'm.text', 'body': 'test msg'}})
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self._check_invite_and_join_status(self.other_user, 1, 0)

    def test_server_notice_disabled(self) -> None:
        if False:
            print('Hello World!')
        'Tests that server returns error if server notice is disabled'
        channel = self.make_request('POST', self.url, access_token=self.admin_user_tok, content={'user_id': self.other_user, 'content': ''})
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.UNKNOWN, channel.json_body['errcode'])
        self.assertEqual('Server notices are not enabled on this server', channel.json_body['error'])

    @override_config({'server_notices': {'system_mxid_localpart': 'notices'}})
    def test_send_server_notice(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that sending two server notices is successfully,\n        the server uses the same room and do not send messages twice.\n        '
        self._check_invite_and_join_status(self.other_user, 0, 0)
        channel = self.make_request('POST', self.url, access_token=self.admin_user_tok, content={'user_id': self.other_user, 'content': {'msgtype': 'm.text', 'body': 'test msg one'}})
        self.assertEqual(200, channel.code, msg=channel.json_body)
        invited_rooms = self._check_invite_and_join_status(self.other_user, 1, 0)
        room_id = invited_rooms[0].room_id
        self.helper.join(room=room_id, user=self.other_user, tok=self.other_user_token)
        self._check_invite_and_join_status(self.other_user, 0, 1)
        messages = self._sync_and_get_messages(room_id, self.other_user_token)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['content']['body'], 'test msg one')
        self.assertEqual(messages[0]['sender'], '@notices:test')
        self.server_notices_manager.get_or_create_notice_room_for_user.invalidate_all()
        channel = self.make_request('POST', self.url, access_token=self.admin_user_tok, content={'user_id': self.other_user, 'content': {'msgtype': 'm.text', 'body': 'test msg two'}})
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self._check_invite_and_join_status(self.other_user, 0, 1)
        messages = self._sync_and_get_messages(room_id, self.other_user_token)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['content']['body'], 'test msg one')
        self.assertEqual(messages[0]['sender'], '@notices:test')
        self.assertEqual(messages[1]['content']['body'], 'test msg two')
        self.assertEqual(messages[1]['sender'], '@notices:test')

    @override_config({'server_notices': {'system_mxid_localpart': 'notices'}})
    def test_send_server_notice_leave_room(self) -> None:
        if False:
            print('Hello World!')
        '\n        Tests that sending a server notices is successfully.\n        The user leaves the room and the second message appears\n        in a new room.\n        '
        self._check_invite_and_join_status(self.other_user, 0, 0)
        channel = self.make_request('POST', self.url, access_token=self.admin_user_tok, content={'user_id': self.other_user, 'content': {'msgtype': 'm.text', 'body': 'test msg one'}})
        self.assertEqual(200, channel.code, msg=channel.json_body)
        invited_rooms = self._check_invite_and_join_status(self.other_user, 1, 0)
        first_room_id = invited_rooms[0].room_id
        self.helper.join(room=first_room_id, user=self.other_user, tok=self.other_user_token)
        self._check_invite_and_join_status(self.other_user, 0, 1)
        messages = self._sync_and_get_messages(first_room_id, self.other_user_token)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['content']['body'], 'test msg one')
        self.assertEqual(messages[0]['sender'], '@notices:test')
        self.helper.leave(room=first_room_id, user=self.other_user, tok=self.other_user_token)
        self._check_invite_and_join_status(self.other_user, 0, 0)
        self.server_notices_manager.get_or_create_notice_room_for_user.invalidate_all()
        channel = self.make_request('POST', self.url, access_token=self.admin_user_tok, content={'user_id': self.other_user, 'content': {'msgtype': 'm.text', 'body': 'test msg two'}})
        self.assertEqual(200, channel.code, msg=channel.json_body)
        invited_rooms = self._check_invite_and_join_status(self.other_user, 1, 0)
        second_room_id = invited_rooms[0].room_id
        self.helper.join(room=second_room_id, user=self.other_user, tok=self.other_user_token)
        self._check_invite_and_join_status(self.other_user, 0, 1)
        messages = self._sync_and_get_messages(second_room_id, self.other_user_token)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['content']['body'], 'test msg two')
        self.assertEqual(messages[0]['sender'], '@notices:test')
        self.assertNotEqual(first_room_id, second_room_id)

    @override_config({'server_notices': {'system_mxid_localpart': 'notices'}})
    def test_send_server_notice_delete_room(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Tests that the user get server notice in a new room\n        after the first server notice room was deleted.\n        '
        self._check_invite_and_join_status(self.other_user, 0, 0)
        channel = self.make_request('POST', self.url, access_token=self.admin_user_tok, content={'user_id': self.other_user, 'content': {'msgtype': 'm.text', 'body': 'test msg one'}})
        self.assertEqual(200, channel.code, msg=channel.json_body)
        invited_rooms = self._check_invite_and_join_status(self.other_user, 1, 0)
        first_room_id = invited_rooms[0].room_id
        self.helper.join(room=first_room_id, user=self.other_user, tok=self.other_user_token)
        self._check_invite_and_join_status(self.other_user, 0, 1)
        messages = self._sync_and_get_messages(first_room_id, self.other_user_token)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['content']['body'], 'test msg one')
        self.assertEqual(messages[0]['sender'], '@notices:test')
        random_string(16)
        self.get_success(self.room_shutdown_handler.shutdown_room(first_room_id, {'requester_user_id': self.admin_user, 'new_room_user_id': None, 'new_room_name': None, 'message': None, 'block': False, 'purge': True, 'force_purge': False}))
        self.get_success(self.pagination_handler.purge_room(first_room_id, force=False))
        self._check_invite_and_join_status(self.other_user, 0, 0)
        summary = self.get_success(self.store.get_room_summary(first_room_id))
        self.assertEqual(summary, {})
        self.server_notices_manager.get_or_create_notice_room_for_user.invalidate_all()
        channel = self.make_request('POST', self.url, access_token=self.admin_user_tok, content={'user_id': self.other_user, 'content': {'msgtype': 'm.text', 'body': 'test msg two'}})
        self.assertEqual(200, channel.code, msg=channel.json_body)
        invited_rooms = self._check_invite_and_join_status(self.other_user, 1, 0)
        second_room_id = invited_rooms[0].room_id
        self.helper.join(room=second_room_id, user=self.other_user, tok=self.other_user_token)
        self._check_invite_and_join_status(self.other_user, 0, 1)
        messages = self._sync_and_get_messages(second_room_id, self.other_user_token)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['content']['body'], 'test msg two')
        self.assertEqual(messages[0]['sender'], '@notices:test')
        self.assertNotEqual(first_room_id, second_room_id)

    @override_config({'server_notices': {'system_mxid_localpart': 'notices'}})
    def test_update_notice_user_name_when_changed(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Tests that existing server notices user name in room is updated after\n        server notice config changes.\n        '
        server_notice_request_content = {'user_id': self.other_user, 'content': {'msgtype': 'm.text', 'body': 'test msg one'}}
        self.make_request('POST', self.url, access_token=self.admin_user_tok, content=server_notice_request_content)
        new_display_name = 'new display name'
        self.server_notices_manager._config.servernotices.server_notices_mxid_display_name = new_display_name
        self.server_notices_manager.get_or_create_notice_room_for_user.cache.invalidate_all()
        self.make_request('POST', self.url, access_token=self.admin_user_tok, content=server_notice_request_content)
        invited_rooms = self._check_invite_and_join_status(self.other_user, 1, 0)
        notice_room_id = invited_rooms[0].room_id
        self.helper.join(room=notice_room_id, user=self.other_user, tok=self.other_user_token)
        notice_user_state_in_room = self.helper.get_state(notice_room_id, 'm.room.member', self.other_user_token, state_key='@notices:test')
        self.assertEqual(notice_user_state_in_room['displayname'], new_display_name)

    @override_config({'server_notices': {'system_mxid_localpart': 'notices'}})
    def test_update_notice_user_avatar_when_changed(self) -> None:
        if False:
            print('Hello World!')
        '\n        Tests that existing server notices user avatar in room is updated when is\n        different from the one in homeserver config.\n        '
        server_notice_request_content = {'user_id': self.other_user, 'content': {'msgtype': 'm.text', 'body': 'test msg one'}}
        self.make_request('POST', self.url, access_token=self.admin_user_tok, content=server_notice_request_content)
        new_avatar_url = 'test/new-url'
        self.server_notices_manager._config.servernotices.server_notices_mxid_avatar_url = new_avatar_url
        self.server_notices_manager.get_or_create_notice_room_for_user.cache.invalidate_all()
        self.make_request('POST', self.url, access_token=self.admin_user_tok, content=server_notice_request_content)
        invited_rooms = self._check_invite_and_join_status(self.other_user, 1, 0)
        notice_room_id = invited_rooms[0].room_id
        self.helper.join(room=notice_room_id, user=self.other_user, tok=self.other_user_token)
        notice_user_state = self.helper.get_state(notice_room_id, 'm.room.member', self.other_user_token, state_key='@notices:test')
        self.assertEqual(notice_user_state['avatar_url'], new_avatar_url)

    def _check_invite_and_join_status(self, user_id: str, expected_invites: int, expected_memberships: int) -> Sequence[RoomsForUser]:
        if False:
            while True:
                i = 10
        'Check invite and room membership status of a user.\n\n        Args\n            user_id: user to check\n            expected_invites: number of expected invites of this user\n            expected_memberships: number of expected room memberships of this user\n        Returns\n            room_ids from the rooms that the user is invited\n        '
        invited_rooms = self.get_success(self.store.get_invited_rooms_for_local_user(user_id))
        self.assertEqual(expected_invites, len(invited_rooms))
        room_ids = self.get_success(self.store.get_rooms_for_user(user_id))
        self.assertEqual(expected_memberships, len(room_ids))
        return invited_rooms

    def _sync_and_get_messages(self, room_id: str, token: str) -> List[JsonDict]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Do a sync and get messages of a room.\n\n        Args\n            room_id: room that contains the messages\n            token: access token of user\n\n        Returns\n            list of messages contained in the room\n        '
        channel = self.make_request('GET', '/_matrix/client/r0/sync', access_token=token)
        self.assertEqual(channel.code, 200)
        room = channel.json_body['rooms']['join'][room_id]
        messages = [x for x in room['timeline']['events'] if x['type'] == 'm.room.message']
        return messages