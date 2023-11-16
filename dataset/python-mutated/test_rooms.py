"""Tests REST events for /rooms paths."""
import json
from http import HTTPStatus
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, Mock, call, patch
from urllib import parse as urlparse
from parameterized import param, parameterized
from typing_extensions import Literal
from twisted.test.proto_helpers import MemoryReactor
import synapse.rest.admin
from synapse.api.constants import EduTypes, EventContentFields, EventTypes, Membership, PublicRoomsFilterFields, RoomTypes
from synapse.api.errors import Codes, HttpResponseException
from synapse.appservice import ApplicationService
from synapse.events import EventBase
from synapse.events.snapshot import EventContext
from synapse.rest import admin
from synapse.rest.client import account, directory, login, profile, register, room, sync
from synapse.server import HomeServer
from synapse.types import JsonDict, RoomAlias, UserID, create_requester
from synapse.util import Clock
from synapse.util.stringutils import random_string
from tests import unittest
from tests.http.server._base import make_request_with_cancellation_test
from tests.storage.test_stream import PaginationTestCase
from tests.test_utils.event_injection import create_event
from tests.unittest import override_config
PATH_PREFIX = b'/_matrix/client/api/v1'

class RoomBase(unittest.HomeserverTestCase):
    rmcreator_id: Optional[str] = None
    servlets = [room.register_servlets, room.register_deprecated_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            while True:
                i = 10
        self.hs = self.setup_test_homeserver('red')
        self.hs.get_federation_handler = Mock()
        self.hs.get_federation_handler.return_value.maybe_backfill = AsyncMock(return_value=None)

        async def _insert_client_ip(*args: Any, **kwargs: Any) -> None:
            return None
        self.hs.get_datastores().main.insert_client_ip = _insert_client_ip
        return self.hs

class RoomPermissionsTestCase(RoomBase):
    """Tests room permissions."""
    user_id = '@sid1:red'
    rmcreator_id = '@notme:red'

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            while True:
                i = 10
        self.helper.auth_user_id = self.rmcreator_id
        self.uncreated_rmid = '!aa:test'
        self.created_rmid = self.helper.create_room_as(self.rmcreator_id, is_public=False)
        self.created_public_rmid = self.helper.create_room_as(self.rmcreator_id, is_public=True)
        self.created_rmid_msg_path = ('rooms/%s/send/m.room.message/a1' % self.created_rmid).encode('ascii')
        channel = self.make_request('PUT', self.created_rmid_msg_path, b'{"msgtype":"m.text","body":"test msg"}')
        self.assertEqual(HTTPStatus.OK, channel.code, channel.result)
        channel = self.make_request('PUT', ('rooms/%s/state/m.room.topic' % self.created_public_rmid).encode('ascii'), b'{"topic":"Public Room Topic"}')
        self.assertEqual(HTTPStatus.OK, channel.code, channel.result)
        self.helper.auth_user_id = self.user_id

    def test_can_do_action(self) -> None:
        if False:
            return 10
        msg_content = b'{"msgtype":"m.text","body":"hello"}'
        seq = iter(range(100))

        def send_msg_path() -> str:
            if False:
                i = 10
                return i + 15
            return '/rooms/%s/send/m.room.message/mid%s' % (self.created_rmid, str(next(seq)))
        channel = self.make_request('PUT', '/rooms/%s/send/m.room.message/mid2' % (self.uncreated_rmid,), msg_content)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', send_msg_path(), msg_content)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        self.helper.invite(room=self.created_rmid, src=self.rmcreator_id, targ=self.user_id)
        channel = self.make_request('PUT', send_msg_path(), msg_content)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        self.helper.join(room=self.created_rmid, user=self.user_id)
        channel = self.make_request('PUT', send_msg_path(), msg_content)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        self.helper.leave(room=self.created_rmid, user=self.user_id)
        channel = self.make_request('PUT', send_msg_path(), msg_content)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])

    def test_topic_perms(self) -> None:
        if False:
            return 10
        topic_content = b'{"topic":"My Topic Name"}'
        topic_path = '/rooms/%s/state/m.room.topic' % self.created_rmid
        channel = self.make_request('PUT', '/rooms/%s/state/m.room.topic' % self.uncreated_rmid, topic_content)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        channel = self.make_request('GET', '/rooms/%s/state/m.room.topic' % self.uncreated_rmid)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', topic_path, topic_content)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        channel = self.make_request('GET', topic_path)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        self.helper.invite(room=self.created_rmid, src=self.rmcreator_id, targ=self.user_id)
        channel = self.make_request('PUT', topic_path, topic_content)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        channel = self.make_request('GET', topic_path)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        self.helper.join(room=self.created_rmid, user=self.user_id)
        self.helper.auth_user_id = self.rmcreator_id
        channel = self.make_request('PUT', topic_path, topic_content)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        self.helper.auth_user_id = self.user_id
        channel = self.make_request('GET', topic_path)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        self.assert_dict(json.loads(topic_content.decode('utf8')), channel.json_body)
        self.helper.leave(room=self.created_rmid, user=self.user_id)
        channel = self.make_request('PUT', topic_path, topic_content)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        channel = self.make_request('GET', topic_path)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        channel = self.make_request('GET', '/rooms/%s/state/m.room.topic' % self.created_public_rmid)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', '/rooms/%s/state/m.room.topic' % self.created_public_rmid, topic_content)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])

    def _test_get_membership(self, room: str, members: Iterable=frozenset(), expect_code: int=200) -> None:
        if False:
            i = 10
            return i + 15
        for member in members:
            path = '/rooms/%s/state/m.room.member/%s' % (room, member)
            channel = self.make_request('GET', path)
            self.assertEqual(expect_code, channel.code)

    def test_membership_basic_room_perms(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        room = self.uncreated_rmid
        self._test_get_membership(members=[self.user_id, self.rmcreator_id], room=room, expect_code=403)
        self.helper.invite(room=room, src=self.user_id, targ=self.rmcreator_id, expect_code=403)
        for usr in [self.user_id, self.rmcreator_id]:
            self.helper.join(room=room, user=usr, expect_code=404)
            self.helper.leave(room=room, user=usr, expect_code=404)

    def test_membership_private_room_perms(self) -> None:
        if False:
            i = 10
            return i + 15
        room = self.created_rmid
        self.helper.invite(room=room, src=self.rmcreator_id, targ=self.user_id)
        self._test_get_membership(members=[self.user_id, self.rmcreator_id], room=room, expect_code=403)
        self.helper.join(room=room, user=self.user_id)
        self._test_get_membership(members=[self.user_id, self.rmcreator_id], room=room, expect_code=200)
        self.helper.leave(room=room, user=self.user_id)
        self._test_get_membership(members=[self.user_id, self.rmcreator_id], room=room, expect_code=200)

    def test_membership_public_room_perms(self) -> None:
        if False:
            while True:
                i = 10
        room = self.created_public_rmid
        self.helper.invite(room=room, src=self.rmcreator_id, targ=self.user_id)
        self._test_get_membership(members=[self.user_id, self.rmcreator_id], room=room, expect_code=403)
        self.helper.join(room=room, user=self.user_id)
        self._test_get_membership(members=[self.user_id, self.rmcreator_id], room=room, expect_code=200)
        self.helper.leave(room=room, user=self.user_id)
        self._test_get_membership(members=[self.user_id, self.rmcreator_id], room=room, expect_code=200)

    def test_invited_permissions(self) -> None:
        if False:
            print('Hello World!')
        room = self.created_rmid
        self.helper.invite(room=room, src=self.rmcreator_id, targ=self.user_id)
        self.helper.invite(room=room, src=self.user_id, targ=self.rmcreator_id, expect_code=403)
        self.helper.change_membership(room=room, src=self.user_id, targ=self.rmcreator_id, membership=Membership.JOIN, expect_code=HTTPStatus.FORBIDDEN)
        self.helper.change_membership(room=room, src=self.user_id, targ=self.rmcreator_id, membership=Membership.LEAVE, expect_code=HTTPStatus.FORBIDDEN)

    def test_joined_permissions(self) -> None:
        if False:
            return 10
        room = self.created_rmid
        self.helper.invite(room=room, src=self.rmcreator_id, targ=self.user_id)
        self.helper.join(room=room, user=self.user_id)
        self.helper.invite(room=room, src=self.user_id, targ=self.user_id, expect_code=403)
        self.helper.join(room=room, user=self.user_id)
        other = '@burgundy:red'
        self.helper.invite(room=room, src=self.user_id, targ=other, expect_code=200)
        self.helper.change_membership(room=room, src=self.user_id, targ=other, membership=Membership.JOIN, expect_code=HTTPStatus.FORBIDDEN)
        self.helper.change_membership(room=room, src=self.user_id, targ=other, membership=Membership.LEAVE, expect_code=HTTPStatus.FORBIDDEN)
        self.helper.leave(room=room, user=self.user_id)

    def test_leave_permissions(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        room = self.created_rmid
        self.helper.invite(room=room, src=self.rmcreator_id, targ=self.user_id)
        self.helper.join(room=room, user=self.user_id)
        self.helper.leave(room=room, user=self.user_id)
        for usr in [self.user_id, self.rmcreator_id]:
            self.helper.change_membership(room=room, src=self.user_id, targ=usr, membership=Membership.INVITE, expect_code=HTTPStatus.FORBIDDEN)
            self.helper.change_membership(room=room, src=self.user_id, targ=usr, membership=Membership.JOIN, expect_code=HTTPStatus.FORBIDDEN)
        self.helper.change_membership(room=room, src=self.user_id, targ=self.rmcreator_id, membership=Membership.LEAVE, expect_code=HTTPStatus.FORBIDDEN)

    def test_member_event_from_ban(self) -> None:
        if False:
            i = 10
            return i + 15
        room = self.created_rmid
        self.helper.invite(room=room, src=self.rmcreator_id, targ=self.user_id)
        self.helper.join(room=room, user=self.user_id)
        other = '@burgundy:red'
        self.helper.change_membership(room=room, src=self.user_id, targ=other, membership=Membership.BAN, expect_code=HTTPStatus.FORBIDDEN, expect_errcode=Codes.FORBIDDEN)
        self.helper.change_membership(room=room, src=self.rmcreator_id, targ=other, membership=Membership.BAN, expect_code=HTTPStatus.OK)
        self.helper.change_membership(room=room, src=self.rmcreator_id, targ=other, membership=Membership.INVITE, expect_code=HTTPStatus.FORBIDDEN, expect_errcode=Codes.BAD_STATE)
        self.helper.change_membership(room=room, src=other, targ=other, membership=Membership.JOIN, expect_code=HTTPStatus.FORBIDDEN, expect_errcode=Codes.BAD_STATE)
        self.helper.change_membership(room=room, src=self.rmcreator_id, targ=other, membership=Membership.BAN, expect_code=HTTPStatus.OK)
        self.helper.change_membership(room=room, src=self.rmcreator_id, targ=other, membership=Membership.KNOCK, expect_code=HTTPStatus.FORBIDDEN, expect_errcode=Codes.BAD_STATE)
        self.helper.change_membership(room=room, src=self.user_id, targ=other, membership=Membership.LEAVE, expect_code=HTTPStatus.FORBIDDEN, expect_errcode=Codes.FORBIDDEN)
        self.helper.change_membership(room=room, src=self.rmcreator_id, targ=other, membership=Membership.LEAVE, expect_code=HTTPStatus.OK)

class RoomStateTestCase(RoomBase):
    """Tests /rooms/$room_id/state."""
    user_id = '@sid1:red'

    def test_get_state_cancellation(self) -> None:
        if False:
            while True:
                i = 10
        'Test cancellation of a `/rooms/$room_id/state` request.'
        room_id = self.helper.create_room_as(self.user_id)
        channel = make_request_with_cancellation_test('test_state_cancellation', self.reactor, self.site, 'GET', '/rooms/%s/state' % room_id)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        self.assertCountEqual([state_event['type'] for state_event in channel.json_list], {'m.room.create', 'm.room.power_levels', 'm.room.join_rules', 'm.room.member', 'm.room.history_visibility'})

    def test_get_state_event_cancellation(self) -> None:
        if False:
            print('Hello World!')
        'Test cancellation of a `/rooms/$room_id/state/$event_type` request.'
        room_id = self.helper.create_room_as(self.user_id)
        channel = make_request_with_cancellation_test('test_state_cancellation', self.reactor, self.site, 'GET', '/rooms/%s/state/m.room.member/%s' % (room_id, self.user_id))
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        self.assertEqual(channel.json_body, {'membership': 'join'})

class RoomsMemberListTestCase(RoomBase):
    """Tests /rooms/$room_id/members/list REST events."""
    servlets = RoomBase.servlets + [sync.register_servlets]
    user_id = '@sid1:red'

    def test_get_member_list(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        room_id = self.helper.create_room_as(self.user_id)
        channel = self.make_request('GET', '/rooms/%s/members' % room_id)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])

    def test_get_member_list_no_room(self) -> None:
        if False:
            return 10
        channel = self.make_request('GET', '/rooms/roomdoesnotexist/members')
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])

    def test_get_member_list_no_permission(self) -> None:
        if False:
            return 10
        room_id = self.helper.create_room_as('@some_other_guy:red')
        channel = self.make_request('GET', '/rooms/%s/members' % room_id)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])

    def test_get_member_list_no_permission_with_at_token(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Tests that a stranger to the room cannot get the member list\n        (in the case that they use an at token).\n        '
        room_id = self.helper.create_room_as('@someone.else:red')
        channel = self.make_request('GET', '/sync')
        self.assertEqual(HTTPStatus.OK, channel.code)
        sync_token = channel.json_body['next_batch']
        channel = self.make_request('GET', f'/rooms/{room_id}/members?at={sync_token}')
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])

    def test_get_member_list_no_permission_former_member(self) -> None:
        if False:
            return 10
        '\n        Tests that a former member of the room can not get the member list.\n        '
        room_id = self.helper.create_room_as('@alice:red')
        self.helper.invite(room_id, '@alice:red', self.user_id)
        self.helper.join(room_id, self.user_id)
        channel = self.make_request('GET', '/rooms/%s/members' % room_id)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        self.helper.change_membership(room_id, '@alice:red', self.user_id, 'ban')
        channel = self.make_request('GET', '/rooms/%s/members' % room_id)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])

    def test_get_member_list_no_permission_former_member_with_at_token(self) -> None:
        if False:
            return 10
        '\n        Tests that a former member of the room can not get the member list\n        (in the case that they use an at token).\n        '
        room_id = self.helper.create_room_as('@alice:red')
        self.helper.invite(room_id, '@alice:red', self.user_id)
        self.helper.join(room_id, self.user_id)
        channel = self.make_request('GET', '/sync')
        self.assertEqual(HTTPStatus.OK, channel.code)
        sync_token = channel.json_body['next_batch']
        channel = self.make_request('GET', '/rooms/%s/members?at=%s' % (room_id, sync_token))
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        self.helper.change_membership(room_id, '@alice:red', self.user_id, 'ban')
        self.helper.invite(room_id, '@alice:red', '@bob:red')
        self.helper.join(room_id, '@bob:red')
        channel = self.make_request('GET', '/sync')
        self.assertEqual(HTTPStatus.OK, channel.code)
        sync_token = channel.json_body['next_batch']
        channel = self.make_request('GET', '/rooms/%s/members?at=%s' % (room_id, sync_token))
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])

    def test_get_member_list_mixed_memberships(self) -> None:
        if False:
            print('Hello World!')
        room_creator = '@some_other_guy:red'
        room_id = self.helper.create_room_as(room_creator)
        room_path = '/rooms/%s/members' % room_id
        self.helper.invite(room=room_id, src=room_creator, targ=self.user_id)
        channel = self.make_request('GET', room_path)
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        self.helper.join(room=room_id, user=self.user_id)
        channel = self.make_request('GET', room_path)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        self.helper.leave(room=room_id, user=self.user_id)
        channel = self.make_request('GET', room_path)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])

    def test_get_member_list_cancellation(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test cancellation of a `/rooms/$room_id/members` request.'
        room_id = self.helper.create_room_as(self.user_id)
        channel = make_request_with_cancellation_test('test_get_member_list_cancellation', self.reactor, self.site, 'GET', '/rooms/%s/members' % room_id)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        self.assertEqual(len(channel.json_body['chunk']), 1)
        self.assertLessEqual({'content': {'membership': 'join'}, 'room_id': room_id, 'sender': self.user_id, 'state_key': self.user_id, 'type': 'm.room.member', 'user_id': self.user_id}.items(), channel.json_body['chunk'][0].items())

    def test_get_member_list_with_at_token_cancellation(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test cancellation of a `/rooms/$room_id/members?at=<sync token>` request.'
        room_id = self.helper.create_room_as(self.user_id)
        channel = self.make_request('GET', '/sync')
        self.assertEqual(HTTPStatus.OK, channel.code)
        sync_token = channel.json_body['next_batch']
        channel = make_request_with_cancellation_test('test_get_member_list_with_at_token_cancellation', self.reactor, self.site, 'GET', '/rooms/%s/members?at=%s' % (room_id, sync_token))
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        self.assertEqual(len(channel.json_body['chunk']), 1)
        self.assertLessEqual({'content': {'membership': 'join'}, 'room_id': room_id, 'sender': self.user_id, 'state_key': self.user_id, 'type': 'm.room.member', 'user_id': self.user_id}.items(), channel.json_body['chunk'][0].items())

class RoomsCreateTestCase(RoomBase):
    """Tests /rooms and /rooms/$room_id REST events."""
    user_id = '@sid1:red'

    def test_post_room_no_keys(self) -> None:
        if False:
            return 10
        channel = self.make_request('POST', '/createRoom', '{}')
        self.assertEqual(HTTPStatus.OK, channel.code, channel.result)
        self.assertTrue('room_id' in channel.json_body)
        assert channel.resource_usage is not None
        self.assertEqual(32, channel.resource_usage.db_txn_count)

    def test_post_room_initial_state(self) -> None:
        if False:
            print('Hello World!')
        channel = self.make_request('POST', '/createRoom', b'{"initial_state":[{"type": "m.bridge", "content": {}}]}')
        self.assertEqual(HTTPStatus.OK, channel.code, channel.result)
        self.assertTrue('room_id' in channel.json_body)
        assert channel.resource_usage is not None
        self.assertEqual(34, channel.resource_usage.db_txn_count)

    def test_post_room_visibility_key(self) -> None:
        if False:
            while True:
                i = 10
        channel = self.make_request('POST', '/createRoom', b'{"visibility":"private"}')
        self.assertEqual(HTTPStatus.OK, channel.code)
        self.assertTrue('room_id' in channel.json_body)

    def test_post_room_custom_key(self) -> None:
        if False:
            i = 10
            return i + 15
        channel = self.make_request('POST', '/createRoom', b'{"custom":"stuff"}')
        self.assertEqual(HTTPStatus.OK, channel.code)
        self.assertTrue('room_id' in channel.json_body)

    def test_post_room_known_and_unknown_keys(self) -> None:
        if False:
            while True:
                i = 10
        channel = self.make_request('POST', '/createRoom', b'{"visibility":"private","custom":"things"}')
        self.assertEqual(HTTPStatus.OK, channel.code)
        self.assertTrue('room_id' in channel.json_body)

    def test_post_room_invalid_content(self) -> None:
        if False:
            i = 10
            return i + 15
        channel = self.make_request('POST', '/createRoom', b'{"visibili')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code)
        channel = self.make_request('POST', '/createRoom', b'["hello"]')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code)

    def test_post_room_invitees_invalid_mxid(self) -> None:
        if False:
            i = 10
            return i + 15
        channel = self.make_request('POST', '/createRoom', b'{"invite":["@alice:example.com "]}')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code)

    @unittest.override_config({'rc_invites': {'per_room': {'burst_count': 3}}})
    def test_post_room_invitees_ratelimit(self) -> None:
        if False:
            print('Hello World!')
        'Test that invites sent when creating a room are ratelimited by a RateLimiter,\n        which ratelimits them correctly, including by not limiting when the requester is\n        exempt from ratelimiting.\n        '
        content = {'invite': ['@alice1:red', '@alice2:red', '@alice3:red', '@alice4:red']}
        channel = self.make_request('POST', '/createRoom', content)
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code)
        self.assertEqual('Cannot invite so many users at once', channel.json_body['error'])
        self.get_success(self.hs.get_datastores().main.set_ratelimit_for_user(self.user_id, 0, 0))
        channel = self.make_request('POST', '/createRoom', content)
        self.assertEqual(HTTPStatus.OK, channel.code)

    def test_spam_checker_may_join_room_deprecated(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that the user_may_join_room spam checker callback is correctly bypassed\n        when creating a new room.\n\n        In this test, we use the deprecated API in which callbacks return a bool.\n        '

        async def user_may_join_room(mxid: str, room_id: str, is_invite: bool) -> bool:
            return False
        join_mock = Mock(side_effect=user_may_join_room)
        self.hs.get_module_api_callbacks().spam_checker._user_may_join_room_callbacks.append(join_mock)
        channel = self.make_request('POST', '/createRoom', {})
        self.assertEqual(channel.code, HTTPStatus.OK, channel.json_body)
        self.assertEqual(join_mock.call_count, 0)

    def test_spam_checker_may_join_room(self) -> None:
        if False:
            print('Hello World!')
        'Tests that the user_may_join_room spam checker callback is correctly bypassed\n        when creating a new room.\n\n        In this test, we use the more recent API in which callbacks return a `Union[Codes, Literal["NOT_SPAM"]]`.\n        '

        async def user_may_join_room_codes(mxid: str, room_id: str, is_invite: bool) -> Codes:
            return Codes.CONSENT_NOT_GIVEN
        join_mock = Mock(side_effect=user_may_join_room_codes)
        self.hs.get_module_api_callbacks().spam_checker._user_may_join_room_callbacks.append(join_mock)
        channel = self.make_request('POST', '/createRoom', {})
        self.assertEqual(channel.code, HTTPStatus.OK, channel.json_body)
        self.assertEqual(join_mock.call_count, 0)

        async def user_may_join_room_tuple(mxid: str, room_id: str, is_invite: bool) -> Tuple[Codes, dict]:
            return (Codes.INCOMPATIBLE_ROOM_VERSION, {})
        join_mock.side_effect = user_may_join_room_tuple
        channel = self.make_request('POST', '/createRoom', {})
        self.assertEqual(channel.code, HTTPStatus.OK, channel.json_body)
        self.assertEqual(join_mock.call_count, 0)

    def _create_basic_room(self) -> Tuple[int, object]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Tries to create a basic room and returns the response code.\n        '
        channel = self.make_request('POST', '/createRoom', {})
        return (channel.code, channel.json_body)

    @override_config({'rc_message': {'per_second': 0.2, 'burst_count': 10}})
    def test_room_creation_ratelimiting(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Regression test for https://github.com/matrix-org/synapse/issues/14312,\n        where ratelimiting was made too strict.\n        Clients should be able to create 10 rooms in a row\n        without hitting rate limits, using default rate limit config.\n        (We override rate limiting config back to its default value.)\n\n        To ensure we don't make ratelimiting too generous accidentally,\n        also check that we can't create an 11th room.\n        "
        for _ in range(10):
            (code, json_body) = self._create_basic_room()
            self.assertEqual(code, HTTPStatus.OK, json_body)
        (code, json_body) = self._create_basic_room()
        self.assertEqual(code, HTTPStatus.TOO_MANY_REQUESTS, json_body)

class RoomTopicTestCase(RoomBase):
    """Tests /rooms/$room_id/topic REST events."""
    user_id = '@sid1:red'

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.room_id = self.helper.create_room_as(self.user_id)
        self.path = '/rooms/%s/state/m.room.topic' % (self.room_id,)

    def test_invalid_puts(self) -> None:
        if False:
            i = 10
            return i + 15
        channel = self.make_request('PUT', self.path, '{}')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', self.path, '{"_name":"bo"}')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', self.path, '{"nao')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', self.path, '[{"_name":"bo"},{"_name":"jill"}]')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', self.path, 'text only')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', self.path, '')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        content = '{"topic":["Topic name"]}'
        channel = self.make_request('PUT', self.path, content)
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])

    def test_rooms_topic(self) -> None:
        if False:
            return 10
        channel = self.make_request('GET', self.path)
        self.assertEqual(HTTPStatus.NOT_FOUND, channel.code, msg=channel.result['body'])
        content = '{"topic":"Topic name"}'
        channel = self.make_request('PUT', self.path, content)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        channel = self.make_request('GET', self.path)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        self.assert_dict(json.loads(content), channel.json_body)

    def test_rooms_topic_with_extra_keys(self) -> None:
        if False:
            return 10
        content = '{"topic":"Seasons","subtopic":"Summer"}'
        channel = self.make_request('PUT', self.path, content)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        channel = self.make_request('GET', self.path)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        self.assert_dict(json.loads(content), channel.json_body)

class RoomMemberStateTestCase(RoomBase):
    """Tests /rooms/$room_id/members/$user_id/state REST events."""
    user_id = '@sid1:red'

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            while True:
                i = 10
        self.room_id = self.helper.create_room_as(self.user_id)

    def test_invalid_puts(self) -> None:
        if False:
            i = 10
            return i + 15
        path = '/rooms/%s/state/m.room.member/%s' % (self.room_id, self.user_id)
        channel = self.make_request('PUT', path, '{}')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', path, '{"_name":"bo"}')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', path, '{"nao')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', path, b'[{"_name":"bo"},{"_name":"jill"}]')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', path, 'text only')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', path, '')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        content = '{"membership":["%s","%s","%s"]}' % (Membership.INVITE, Membership.JOIN, Membership.LEAVE)
        channel = self.make_request('PUT', path, content.encode('ascii'))
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])

    def test_rooms_members_self(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        path = '/rooms/%s/state/m.room.member/%s' % (urlparse.quote(self.room_id), self.user_id)
        content = '{"membership":"%s"}' % Membership.JOIN
        channel = self.make_request('PUT', path, content.encode('ascii'))
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        channel = self.make_request('GET', path, content=b'')
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        expected_response = {'membership': Membership.JOIN}
        self.assertEqual(expected_response, channel.json_body)

    def test_rooms_members_other(self) -> None:
        if False:
            i = 10
            return i + 15
        self.other_id = '@zzsid1:red'
        path = '/rooms/%s/state/m.room.member/%s' % (urlparse.quote(self.room_id), self.other_id)
        content = '{"membership":"%s"}' % Membership.INVITE
        channel = self.make_request('PUT', path, content)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        channel = self.make_request('GET', path, content=b'')
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        self.assertEqual(json.loads(content), channel.json_body)

    def test_rooms_members_other_custom_keys(self) -> None:
        if False:
            print('Hello World!')
        self.other_id = '@zzsid1:red'
        path = '/rooms/%s/state/m.room.member/%s' % (urlparse.quote(self.room_id), self.other_id)
        content = '{"membership":"%s","invite_text":"%s"}' % (Membership.INVITE, 'Join us!')
        channel = self.make_request('PUT', path, content)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        channel = self.make_request('GET', path, content=b'')
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        self.assertEqual(json.loads(content), channel.json_body)

class RoomInviteRatelimitTestCase(RoomBase):
    user_id = '@sid1:red'
    servlets = [admin.register_servlets, profile.register_servlets, room.register_servlets]

    @unittest.override_config({'rc_invites': {'per_room': {'per_second': 0.5, 'burst_count': 3}}})
    def test_invites_by_rooms_ratelimit(self) -> None:
        if False:
            print('Hello World!')
        'Tests that invites in a room are actually rate-limited.'
        room_id = self.helper.create_room_as(self.user_id)
        for i in range(3):
            self.helper.invite(room_id, self.user_id, '@user-%s:red' % (i,))
        self.helper.invite(room_id, self.user_id, '@user-4:red', expect_code=429)

    @unittest.override_config({'rc_invites': {'per_user': {'per_second': 0.5, 'burst_count': 3}}})
    def test_invites_by_users_ratelimit(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that invites to a specific user are actually rate-limited.'
        for _ in range(3):
            room_id = self.helper.create_room_as(self.user_id)
            self.helper.invite(room_id, self.user_id, '@other-users:red')
        room_id = self.helper.create_room_as(self.user_id)
        self.helper.invite(room_id, self.user_id, '@other-users:red', expect_code=429)

class RoomJoinTestCase(RoomBase):
    servlets = [admin.register_servlets, login.register_servlets, room.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.user1 = self.register_user('thomas', 'hackme')
        self.tok1 = self.login('thomas', 'hackme')
        self.user2 = self.register_user('teresa', 'hackme')
        self.tok2 = self.login('teresa', 'hackme')
        self.room1 = self.helper.create_room_as(room_creator=self.user1, tok=self.tok1)
        self.room2 = self.helper.create_room_as(room_creator=self.user1, tok=self.tok1)
        self.room3 = self.helper.create_room_as(room_creator=self.user1, tok=self.tok1)

    def test_spam_checker_may_join_room_deprecated(self) -> None:
        if False:
            print('Hello World!')
        'Tests that the user_may_join_room spam checker callback is correctly called\n        and blocks room joins when needed.\n\n        This test uses the deprecated API, in which callbacks return booleans.\n        '
        return_value = True

        async def user_may_join_room(userid: str, room_id: str, is_invited: bool) -> bool:
            return return_value
        callback_mock = Mock(side_effect=user_may_join_room, spec=lambda *x: None)
        self.hs.get_module_api_callbacks().spam_checker._user_may_join_room_callbacks.append(callback_mock)
        self.helper.join(self.room1, self.user2, tok=self.tok2)
        expected_call_args = ((self.user2, self.room1, False),)
        self.assertEqual(callback_mock.call_args, expected_call_args, callback_mock.call_args)
        self.helper.invite(self.room2, self.user1, self.user2, tok=self.tok1)
        self.helper.join(self.room2, self.user2, tok=self.tok2)
        expected_call_args = ((self.user2, self.room2, True),)
        self.assertEqual(callback_mock.call_args, expected_call_args, callback_mock.call_args)
        return_value = False
        self.helper.join(self.room3, self.user2, expect_code=HTTPStatus.FORBIDDEN, tok=self.tok2)

    def test_spam_checker_may_join_room(self) -> None:
        if False:
            return 10
        'Tests that the user_may_join_room spam checker callback is correctly called\n        and blocks room joins when needed.\n\n        This test uses the latest API to this day, in which callbacks return `NOT_SPAM` or `Codes`.\n        '
        return_value: Union[Literal['NOT_SPAM'], Tuple[Codes, dict], Codes] = synapse.module_api.NOT_SPAM

        async def user_may_join_room(userid: str, room_id: str, is_invited: bool) -> Union[Literal['NOT_SPAM'], Tuple[Codes, dict], Codes]:
            return return_value
        callback_mock = Mock(side_effect=user_may_join_room, spec=lambda *x: None)
        self.hs.get_module_api_callbacks().spam_checker._user_may_join_room_callbacks.append(callback_mock)
        self.helper.join(self.room1, self.user2, tok=self.tok2)
        expected_call_args = ((self.user2, self.room1, False),)
        self.assertEqual(callback_mock.call_args, expected_call_args, callback_mock.call_args)
        self.helper.invite(self.room2, self.user1, self.user2, tok=self.tok1)
        self.helper.join(self.room2, self.user2, tok=self.tok2)
        expected_call_args = ((self.user2, self.room2, True),)
        self.assertEqual(callback_mock.call_args, expected_call_args, callback_mock.call_args)
        return_value = Codes.CONSENT_NOT_GIVEN
        self.helper.invite(self.room3, self.user1, self.user2, tok=self.tok1)
        self.helper.join(self.room3, self.user2, expect_code=HTTPStatus.FORBIDDEN, expect_errcode=return_value, tok=self.tok2)
        return_value = (Codes.BAD_ALIAS, {'another_field': '12345'})
        self.helper.join(self.room3, self.user2, expect_code=HTTPStatus.FORBIDDEN, expect_errcode=return_value[0], tok=self.tok2, expect_additional_fields=return_value[1])

class RoomAppserviceTsParamTestCase(unittest.HomeserverTestCase):
    servlets = [room.register_servlets, synapse.rest.admin.register_servlets, register.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            i = 10
            return i + 15
        (self.appservice_user, _) = self.register_appservice_user('as_user_potato', self.appservice.token)
        args = {'access_token': self.appservice.token, 'user_id': self.appservice_user}
        channel = self.make_request('POST', f'/_matrix/client/r0/createRoom?{urlparse.urlencode(args)}', content={'visibility': 'public'})
        assert channel.code == 200
        self.room = channel.json_body['room_id']
        self.main_store = self.hs.get_datastores().main

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            return 10
        config = self.default_config()
        self.appservice = ApplicationService(token='i_am_an_app_service', id='1234', namespaces={'users': [{'regex': '@as_user.*', 'exclusive': True}]}, sender='@as_main:test')
        mock_load_appservices = Mock(return_value=[self.appservice])
        with patch('synapse.storage.databases.main.appservice.load_appservices', mock_load_appservices):
            hs = self.setup_test_homeserver(config=config)
        return hs

    def test_send_event_ts(self) -> None:
        if False:
            print('Hello World!')
        'Test sending a non-state event with a custom timestamp.'
        ts = 1
        url_params = {'user_id': self.appservice_user, 'ts': ts}
        channel = self.make_request('PUT', path=f'/_matrix/client/r0/rooms/{self.room}/send/m.room.message/1234?' + urlparse.urlencode(url_params), content={'body': 'test', 'msgtype': 'm.text'}, access_token=self.appservice.token)
        self.assertEqual(channel.code, 200, channel.json_body)
        event_id = channel.json_body['event_id']
        res = self.get_success(self.main_store.get_event(event_id))
        self.assertEqual(ts, res.origin_server_ts)

    def test_send_state_event_ts(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test sending a state event with a custom timestamp.'
        ts = 1
        url_params = {'user_id': self.appservice_user, 'ts': ts}
        channel = self.make_request('PUT', path=f'/_matrix/client/r0/rooms/{self.room}/state/m.room.name?' + urlparse.urlencode(url_params), content={'name': 'test'}, access_token=self.appservice.token)
        self.assertEqual(channel.code, 200, channel.json_body)
        event_id = channel.json_body['event_id']
        res = self.get_success(self.main_store.get_event(event_id))
        self.assertEqual(ts, res.origin_server_ts)

    def test_send_membership_event_ts(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test sending a membership event with a custom timestamp.'
        ts = 1
        url_params = {'user_id': self.appservice_user, 'ts': ts}
        channel = self.make_request('PUT', path=f'/_matrix/client/r0/rooms/{self.room}/state/m.room.member/{self.appservice_user}?' + urlparse.urlencode(url_params), content={'membership': 'join', 'display_name': 'test'}, access_token=self.appservice.token)
        self.assertEqual(channel.code, 200, channel.json_body)
        event_id = channel.json_body['event_id']
        res = self.get_success(self.main_store.get_event(event_id))
        self.assertEqual(ts, res.origin_server_ts)

class RoomJoinRatelimitTestCase(RoomBase):
    user_id = '@sid1:red'
    servlets = [admin.register_servlets, profile.register_servlets, room.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        super().prepare(reactor, clock, hs)
        user = UserID.from_string(self.user_id)
        self.register_user(user.localpart, 'supersecretpassword')

    @unittest.override_config({'rc_joins': {'local': {'per_second': 0.5, 'burst_count': 3}}})
    def test_join_local_ratelimit(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that local joins are actually rate-limited.'
        room_ids = [self.helper.create_room_as(self.user_id, is_public=True) for _ in range(4)]
        joiner_user_id = self.register_user('joiner', 'secret')
        for room_id in room_ids[0:3]:
            self.helper.join(room_id, joiner_user_id)
        self.helper.join(room_ids[3], joiner_user_id, expect_code=HTTPStatus.TOO_MANY_REQUESTS)

    @unittest.override_config({'rc_joins': {'local': {'per_second': 0.5, 'burst_count': 3}}})
    def test_join_attempts_local_ratelimit(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests that unsuccessful joins that end up being denied are rate-limited.'
        room_ids = [self.helper.create_room_as(self.user_id, is_public=True) for _ in range(4)]
        joiner_user_id = self.register_user('joiner', 'secret')
        for room_id in room_ids:
            self.helper.ban(room_id, self.user_id, joiner_user_id)
        for room_id in room_ids[0:3]:
            self.helper.join(room_id, joiner_user_id, expect_code=HTTPStatus.FORBIDDEN)
        self.helper.join(room_ids[3], joiner_user_id, expect_code=HTTPStatus.TOO_MANY_REQUESTS)

    @unittest.override_config({'rc_joins': {'local': {'per_second': 0.5, 'burst_count': 3}}})
    def test_join_local_ratelimit_profile_change(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Tests that sending a profile update into all of the user's joined rooms isn't\n        rate-limited by the rate-limiter on joins."
        room_ids = [self.helper.create_room_as(self.user_id), self.helper.create_room_as(self.user_id), self.helper.create_room_as(self.user_id)]
        self.reactor.advance(2)
        room_ids.append(self.helper.create_room_as(self.user_id))
        path = '/_matrix/client/r0/profile/%s/displayname' % self.user_id
        channel = self.make_request('PUT', path, {'displayname': 'John Doe'})
        self.assertEqual(channel.code, HTTPStatus.OK, channel.json_body)
        for room_id in room_ids:
            path = '/_matrix/client/r0/rooms/%s/state/m.room.member/%s' % (room_id, self.user_id)
            channel = self.make_request('GET', path)
            self.assertEqual(channel.code, 200)
            self.assertIn('displayname', channel.json_body)
            self.assertEqual(channel.json_body['displayname'], 'John Doe')

    @unittest.override_config({'rc_joins': {'local': {'per_second': 0.5, 'burst_count': 3}}})
    def test_join_local_ratelimit_idempotent(self) -> None:
        if False:
            return 10
        'Tests that the room join endpoints remain idempotent despite rate-limiting\n        on room joins.'
        room_id = self.helper.create_room_as(self.user_id)
        paths_to_test = ['/_matrix/client/r0/rooms/%s/join', '/_matrix/client/r0/join/%s']
        for path in paths_to_test:
            for _ in range(4):
                channel = self.make_request('POST', path % room_id, {})
                self.assertEqual(channel.code, 200)

    @unittest.override_config({'rc_joins': {'local': {'per_second': 0.5, 'burst_count': 3}}, 'auto_join_rooms': ['#room:red', '#room2:red', '#room3:red', '#room4:red'], 'autocreate_auto_join_rooms': True})
    def test_autojoin_rooms(self) -> None:
        if False:
            return 10
        user_id = self.register_user('testuser', 'password')
        rooms = self.get_success(self.hs.get_datastores().main.get_rooms_for_user(user_id))
        self.assertEqual(len(rooms), 4)

class RoomMessagesTestCase(RoomBase):
    """Tests /rooms/$room_id/messages/$user_id/$msg_id REST events."""
    user_id = '@sid1:red'

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            i = 10
            return i + 15
        self.room_id = self.helper.create_room_as(self.user_id)

    def test_invalid_puts(self) -> None:
        if False:
            print('Hello World!')
        path = '/rooms/%s/send/m.room.message/mid1' % urlparse.quote(self.room_id)
        channel = self.make_request('PUT', path, b'{}')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', path, b'{"_name":"bo"}')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', path, b'{"nao')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', path, b'[{"_name":"bo"},{"_name":"jill"}]')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', path, b'text only')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        channel = self.make_request('PUT', path, b'')
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])

    def test_rooms_messages_sent(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        path = '/rooms/%s/send/m.room.message/mid1' % urlparse.quote(self.room_id)
        content = b'{"body":"test","msgtype":{"type":"a"}}'
        channel = self.make_request('PUT', path, content)
        self.assertEqual(HTTPStatus.BAD_REQUEST, channel.code, msg=channel.result['body'])
        content = b'{"body":"test","msgtype":"test.custom.text"}'
        channel = self.make_request('PUT', path, content)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])
        path = '/rooms/%s/send/m.room.message/mid2' % urlparse.quote(self.room_id)
        content = b'{"body":"test2","msgtype":"m.text"}'
        channel = self.make_request('PUT', path, content)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])

    @parameterized.expand([param(name='NOT_SPAM', value='NOT_SPAM', expected_code=HTTPStatus.OK, expected_fields={}), param(name='False', value=False, expected_code=HTTPStatus.OK, expected_fields={}), param(name='scalene string', value='ANY OTHER STRING', expected_code=HTTPStatus.FORBIDDEN, expected_fields={'errcode': 'M_FORBIDDEN'}), param(name='True', value=True, expected_code=HTTPStatus.FORBIDDEN, expected_fields={'errcode': 'M_FORBIDDEN'}), param(name='Code', value=Codes.LIMIT_EXCEEDED, expected_code=HTTPStatus.FORBIDDEN, expected_fields={'errcode': 'M_LIMIT_EXCEEDED'}), param(name='Tuple', value=(Codes.SERVER_NOT_TRUSTED, {'additional_field': '12345'}), expected_code=HTTPStatus.FORBIDDEN, expected_fields={'errcode': 'M_SERVER_NOT_TRUSTED', 'additional_field': '12345'})])
    def test_spam_checker_check_event_for_spam(self, name: str, value: Union[str, bool, Codes, Tuple[Codes, JsonDict]], expected_code: int, expected_fields: dict) -> None:
        if False:
            i = 10
            return i + 15

        class SpamCheck:
            mock_return_value: Union[str, bool, Codes, Tuple[Codes, JsonDict], bool] = 'NOT_SPAM'
            mock_content: Optional[JsonDict] = None

            async def check_event_for_spam(self, event: synapse.events.EventBase) -> Union[str, Codes, Tuple[Codes, JsonDict], bool]:
                self.mock_content = event.content
                return self.mock_return_value
        spam_checker = SpamCheck()
        self.hs.get_module_api_callbacks().spam_checker._check_event_for_spam_callbacks.append(spam_checker.check_event_for_spam)
        spam_checker.mock_return_value = value
        path = '/rooms/%s/send/m.room.message/check_event_for_spam_%s' % (urlparse.quote(self.room_id), urlparse.quote(name))
        body = 'test-%s' % name
        content = '{"body":"%s","msgtype":"m.text"}' % body
        channel = self.make_request('PUT', path, content)
        self.assertIsNotNone(spam_checker.mock_content)
        if spam_checker.mock_content is not None:
            self.assertEqual(spam_checker.mock_content['body'], body, spam_checker.mock_content)
        self.assertEqual(expected_code, channel.code, msg=channel.result['body'])
        for (expected_key, expected_value) in expected_fields.items():
            self.assertEqual(channel.json_body.get(expected_key, None), expected_value, 'Field %s absent or invalid ' % expected_key)

class RoomPowerLevelOverridesTestCase(RoomBase):
    """Tests that the power levels can be overridden with server config."""
    user_id = '@sid1:red'
    servlets = [admin.register_servlets, room.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.admin_user_id = self.register_user('admin', 'pass')
        self.admin_access_token = self.login('admin', 'pass')

    def power_levels(self, room_id: str) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return self.helper.get_state(room_id, 'm.room.power_levels', self.admin_access_token)

    def test_default_power_levels_with_room_override(self) -> None:
        if False:
            return 10
        "\n        Create a room, providing power level overrides.\n        Confirm that the room's power levels reflect the overrides.\n\n        See https://github.com/matrix-org/matrix-spec/issues/492\n        - currently we overwrite each key of power_level_content_override\n        completely.\n        "
        room_id = self.helper.create_room_as(self.user_id, extra_content={'power_level_content_override': {'events': {'custom.event': 0}}})
        self.assertEqual({'custom.event': 0}, self.power_levels(room_id)['events'])

    @unittest.override_config({'default_power_level_content_override': {'public_chat': {'events': {'custom.event': 0}}}})
    def test_power_levels_with_server_override(self) -> None:
        if False:
            print('Hello World!')
        "\n        With a server configured to modify the room-level defaults,\n        Create a room, without providing any extra power level overrides.\n        Confirm that the room's power levels reflect the server-level overrides.\n\n        Similar to https://github.com/matrix-org/matrix-spec/issues/492,\n        we overwrite each key of power_level_content_override completely.\n        "
        room_id = self.helper.create_room_as(self.user_id)
        self.assertEqual({'custom.event': 0}, self.power_levels(room_id)['events'])

    @unittest.override_config({'default_power_level_content_override': {'public_chat': {'events': {'server.event': 0}, 'ban': 13}}})
    def test_power_levels_with_server_and_room_overrides(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        With a server configured to modify the room-level defaults,\n        create a room, providing different overrides.\n        Confirm that the room's power levels reflect both overrides, and\n        choose the room overrides where they clash.\n        "
        room_id = self.helper.create_room_as(self.user_id, extra_content={'power_level_content_override': {'events': {'room.event': 0}}})
        self.assertEqual({'room.event': 0}, self.power_levels(room_id)['events'])
        self.assertEqual(13, self.power_levels(room_id)['ban'])

class RoomPowerLevelOverridesInPracticeTestCase(RoomBase):
    """
    Tests that we can really do various otherwise-prohibited actions
    based on overriding the power levels in config.
    """
    user_id = '@sid1:red'

    def test_creator_can_post_state_event(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        room_id = self.helper.create_room_as(self.user_id)
        path = '/rooms/{room_id}/state/custom.event/my_state_key'.format(room_id=urlparse.quote(room_id))
        channel = self.make_request('PUT', path, '{}')
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])

    def test_normal_user_can_not_post_state_event(self) -> None:
        if False:
            while True:
                i = 10
        room_id = self.helper.create_room_as('@some_other_guy:red')
        self.helper.join(room=room_id, user=self.user_id)
        path = '/rooms/{room_id}/state/custom.event/my_state_key'.format(room_id=urlparse.quote(room_id))
        channel = self.make_request('PUT', path, '{}')
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        self.assertEqual("You don't have permission to post that to the room. user_level (0) < send_level (50)", channel.json_body['error'])

    @unittest.override_config({'default_power_level_content_override': {'public_chat': {'events': {'custom.event': 0}}}})
    def test_with_config_override_normal_user_can_post_state_event(self) -> None:
        if False:
            print('Hello World!')
        room_id = self.helper.create_room_as('@some_other_guy:red')
        self.helper.join(room=room_id, user=self.user_id)
        path = '/rooms/{room_id}/state/custom.event/my_state_key'.format(room_id=urlparse.quote(room_id))
        channel = self.make_request('PUT', path, '{}')
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.result['body'])

    @unittest.override_config({'default_power_level_content_override': {'public_chat': {'events': {'custom.event': 0}}}})
    def test_any_room_override_defeats_config_override(self) -> None:
        if False:
            i = 10
            return i + 15
        extra_content: Dict[str, Any] = {'power_level_content_override': {'events': {}}}
        room_id = self.helper.create_room_as('@some_other_guy:red', extra_content=extra_content)
        self.helper.join(room=room_id, user=self.user_id)
        path = '/rooms/{room_id}/state/custom.event/my_state_key'.format(room_id=urlparse.quote(room_id))
        channel = self.make_request('PUT', path, '{}')
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])

    @unittest.override_config({'default_power_level_content_override': {'public_chat': {'events': {'custom.event': 0}}}})
    def test_specific_room_override_defeats_config_override(self) -> None:
        if False:
            print('Hello World!')
        extra_content = {'power_level_content_override': {'events': {'custom.event': 1}}}
        room_id = self.helper.create_room_as('@some_other_guy:red', extra_content=extra_content)
        self.helper.join(room=room_id, user=self.user_id)
        path = '/rooms/{room_id}/state/custom.event/my_state_key'.format(room_id=urlparse.quote(room_id))
        channel = self.make_request('PUT', path, '{}')
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        self.assertEqual("You don't have permission to post that to the room. " + 'user_level (0) < send_level (1)', channel.json_body['error'])

    @unittest.override_config({'default_power_level_content_override': {'public_chat': {'events': {'custom.event': 0}}, 'private_chat': None, 'trusted_private_chat': None}})
    def test_config_override_applies_only_to_specific_preset(self) -> None:
        if False:
            while True:
                i = 10
        room_id = self.helper.create_room_as('@some_other_guy:red', is_public=False)
        self.helper.invite(room=room_id, src='@some_other_guy:red', targ=self.user_id)
        self.helper.join(room=room_id, user=self.user_id)
        path = '/rooms/{room_id}/state/custom.event/my_state_key'.format(room_id=urlparse.quote(room_id))
        channel = self.make_request('PUT', path, '{}')
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        self.assertEqual("You don't have permission to post that to the room. " + 'user_level (0) < send_level (50)', channel.json_body['error'])

    @unittest.override_config({'default_power_level_content_override': {'private_chat': {'events': {'m.room.avatar': 50, 'm.room.canonical_alias': 50, 'm.room.encryption': 999, 'm.room.history_visibility': 100, 'm.room.name': 50, 'm.room.power_levels': 100, 'm.room.server_acl': 100, 'm.room.tombstone': 100}, 'events_default': 0}}})
    def test_config_override_blocks_encrypted_room(self) -> None:
        if False:
            print('Hello World!')
        channel = self.make_request('POST', '/createRoom', '{"creation_content": {"m.federate": false},"name": "Secret Private Room","preset": "private_chat","initial_state": [{"type": "m.room.encryption","state_key": "","content": {"algorithm": "m.megolm.v1.aes-sha2"}}]}')
        self.assertEqual(HTTPStatus.FORBIDDEN, channel.code, msg=channel.result['body'])
        self.assertEqual('You cannot create an encrypted room. ' + 'user_level (100) < send_level (999)', channel.json_body['error'])

class RoomInitialSyncTestCase(RoomBase):
    """Tests /rooms/$room_id/initialSync."""
    user_id = '@sid1:red'

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.room_id = self.helper.create_room_as(self.user_id)

    def test_initial_sync(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        channel = self.make_request('GET', '/rooms/%s/initialSync' % self.room_id)
        self.assertEqual(HTTPStatus.OK, channel.code)
        self.assertEqual(self.room_id, channel.json_body['room_id'])
        self.assertEqual('join', channel.json_body['membership'])
        state: JsonDict = {}
        for event in channel.json_body['state']:
            if 'state_key' not in event:
                continue
            t = event['type']
            if t not in state:
                state[t] = []
            state[t].append(event)
        self.assertTrue('m.room.create' in state)
        self.assertTrue('messages' in channel.json_body)
        self.assertTrue('chunk' in channel.json_body['messages'])
        self.assertTrue('end' in channel.json_body['messages'])
        self.assertTrue('presence' in channel.json_body)
        presence_by_user = {e['content']['user_id']: e for e in channel.json_body['presence']}
        self.assertTrue(self.user_id in presence_by_user)
        self.assertEqual(EduTypes.PRESENCE, presence_by_user[self.user_id]['type'])

class RoomMessageListTestCase(RoomBase):
    """Tests /rooms/$room_id/messages REST events."""
    user_id = '@sid1:red'

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.room_id = self.helper.create_room_as(self.user_id)

    def test_topo_token_is_accepted(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        token = 't1-0_0_0_0_0_0_0_0_0_0'
        channel = self.make_request('GET', '/rooms/%s/messages?access_token=x&from=%s' % (self.room_id, token))
        self.assertEqual(HTTPStatus.OK, channel.code)
        self.assertTrue('start' in channel.json_body)
        self.assertEqual(token, channel.json_body['start'])
        self.assertTrue('chunk' in channel.json_body)
        self.assertTrue('end' in channel.json_body)

    def test_stream_token_is_accepted_for_fwd_pagianation(self) -> None:
        if False:
            return 10
        token = 's0_0_0_0_0_0_0_0_0_0'
        channel = self.make_request('GET', '/rooms/%s/messages?access_token=x&from=%s' % (self.room_id, token))
        self.assertEqual(HTTPStatus.OK, channel.code)
        self.assertTrue('start' in channel.json_body)
        self.assertEqual(token, channel.json_body['start'])
        self.assertTrue('chunk' in channel.json_body)
        self.assertTrue('end' in channel.json_body)

    def test_room_messages_purge(self) -> None:
        if False:
            print('Hello World!')
        store = self.hs.get_datastores().main
        pagination_handler = self.hs.get_pagination_handler()
        first_event_id = self.helper.send(self.room_id, 'message 1')['event_id']
        first_token = self.get_success(store.get_topological_token_for_event(first_event_id))
        first_token_str = self.get_success(first_token.to_string(store))
        second_event_id = self.helper.send(self.room_id, 'message 2')['event_id']
        second_token = self.get_success(store.get_topological_token_for_event(second_event_id))
        second_token_str = self.get_success(second_token.to_string(store))
        self.helper.send(self.room_id, 'message 3')
        channel = self.make_request('GET', '/rooms/%s/messages?access_token=x&from=%s&dir=b&filter=%s' % (self.room_id, second_token_str, json.dumps({'types': [EventTypes.Message]})))
        self.assertEqual(channel.code, HTTPStatus.OK, channel.json_body)
        chunk = channel.json_body['chunk']
        self.assertEqual(len(chunk), 2, [event['content'] for event in chunk])
        self.get_success(pagination_handler.purge_history(room_id=self.room_id, token=second_token_str, delete_local_events=True))
        channel = self.make_request('GET', '/rooms/%s/messages?access_token=x&from=%s&dir=b&filter=%s' % (self.room_id, second_token_str, json.dumps({'types': [EventTypes.Message]})))
        self.assertEqual(channel.code, HTTPStatus.OK, channel.json_body)
        chunk = channel.json_body['chunk']
        self.assertEqual(len(chunk), 1, [event['content'] for event in chunk])
        channel = self.make_request('GET', '/rooms/%s/messages?access_token=x&from=%s&dir=b&filter=%s' % (self.room_id, first_token_str, json.dumps({'types': [EventTypes.Message]})))
        self.assertEqual(channel.code, HTTPStatus.OK, channel.json_body)
        chunk = channel.json_body['chunk']
        self.assertEqual(len(chunk), 0, [event['content'] for event in chunk])

class RoomSearchTestCase(unittest.HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets_for_client_rest_resource, room.register_servlets, login.register_servlets]
    user_id = True
    hijack_auth = False

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.user_id2 = self.register_user('user', 'pass')
        self.access_token = self.login('user', 'pass')
        self.other_user_id = self.register_user('otheruser', 'pass')
        self.other_access_token = self.login('otheruser', 'pass')
        self.room = self.helper.create_room_as(self.user_id2, tok=self.access_token)
        self.helper.invite(room=self.room, src=self.user_id2, tok=self.access_token, targ=self.other_user_id)
        self.helper.join(room=self.room, user=self.other_user_id, tok=self.other_access_token)

    def test_finds_message(self) -> None:
        if False:
            print('Hello World!')
        '\n        The search functionality will search for content in messages if asked to\n        do so.\n        '
        self.helper.send(self.room, body='Hi!', tok=self.other_access_token)
        self.helper.send(self.room, body='There!', tok=self.other_access_token)
        channel = self.make_request('POST', '/search?access_token=%s' % (self.access_token,), {'search_categories': {'room_events': {'keys': ['content.body'], 'search_term': 'Hi'}}})
        self.assertEqual(channel.code, 200)
        results = channel.json_body['search_categories']['room_events']
        self.assertEqual(results['count'], 1)
        self.assertEqual(results['results'][0]['result']['content']['body'], 'Hi!')
        self.assertEqual(results['results'][0]['context'], {})

    def test_include_context(self) -> None:
        if False:
            while True:
                i = 10
        '\n        When event_context includes include_profile, profile information will be\n        included in the search response.\n        '
        self.helper.send(self.room, body='Hi!', tok=self.other_access_token)
        self.helper.send(self.room, body='There!', tok=self.other_access_token)
        channel = self.make_request('POST', '/search?access_token=%s' % (self.access_token,), {'search_categories': {'room_events': {'keys': ['content.body'], 'search_term': 'Hi', 'event_context': {'include_profile': True}}}})
        self.assertEqual(channel.code, 200)
        results = channel.json_body['search_categories']['room_events']
        self.assertEqual(results['count'], 1)
        self.assertEqual(results['results'][0]['result']['content']['body'], 'Hi!')
        context = results['results'][0]['context']
        self.assertEqual(len(context['profile_info'].keys()), 2)
        self.assertEqual(context['profile_info'][self.other_user_id]['displayname'], 'otheruser')

class PublicRoomsRestrictedTestCase(unittest.HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets_for_client_rest_resource, room.register_servlets, login.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            for i in range(10):
                print('nop')
        self.url = b'/_matrix/client/r0/publicRooms'
        config = self.default_config()
        config['allow_public_rooms_without_auth'] = False
        self.hs = self.setup_test_homeserver(config=config)
        return self.hs

    def test_restricted_no_auth(self) -> None:
        if False:
            print('Hello World!')
        channel = self.make_request('GET', self.url)
        self.assertEqual(channel.code, HTTPStatus.UNAUTHORIZED, channel.result)

    def test_restricted_auth(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.register_user('user', 'pass')
        tok = self.login('user', 'pass')
        channel = self.make_request('GET', self.url, access_token=tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)

class PublicRoomsRoomTypeFilterTestCase(unittest.HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets_for_client_rest_resource, room.register_servlets, login.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            while True:
                i = 10
        config = self.default_config()
        config['allow_public_rooms_without_auth'] = True
        self.hs = self.setup_test_homeserver(config=config)
        self.url = b'/_matrix/client/r0/publicRooms'
        return self.hs

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            i = 10
            return i + 15
        user = self.register_user('alice', 'pass')
        self.token = self.login(user, 'pass')
        self.helper.create_room_as(user, is_public=True, extra_content={'visibility': 'public'}, tok=self.token)
        self.helper.create_room_as(user, is_public=True, extra_content={'visibility': 'public', 'creation_content': {EventContentFields.ROOM_TYPE: RoomTypes.SPACE}}, tok=self.token)

    def make_public_rooms_request(self, room_types: Optional[List[Union[str, None]]], instance_id: Optional[str]=None) -> Tuple[List[Dict[str, Any]], int]:
        if False:
            while True:
                i = 10
        body: JsonDict = {'filter': {PublicRoomsFilterFields.ROOM_TYPES: room_types}}
        if instance_id:
            body['third_party_instance_id'] = 'test|test'
        channel = self.make_request('POST', self.url, body, self.token)
        self.assertEqual(channel.code, 200)
        chunk = channel.json_body['chunk']
        count = channel.json_body['total_room_count_estimate']
        self.assertEqual(len(chunk), count)
        return (chunk, count)

    def test_returns_both_rooms_and_spaces_if_no_filter(self) -> None:
        if False:
            while True:
                i = 10
        (chunk, count) = self.make_public_rooms_request(None)
        self.assertEqual(count, 2)
        channel = self.make_request('POST', self.url, {}, self.token)
        self.assertEqual(channel.code, 200)
        self.assertEqual(len(channel.json_body['chunk']), 2)
        self.assertEqual(channel.json_body['total_room_count_estimate'], 2)
        (chunk, count) = self.make_public_rooms_request(None, 'test|test')
        self.assertEqual(count, 0)

    def test_returns_only_rooms_based_on_filter(self) -> None:
        if False:
            i = 10
            return i + 15
        (chunk, count) = self.make_public_rooms_request([None])
        self.assertEqual(count, 1)
        self.assertEqual(chunk[0].get('room_type', None), None)
        (chunk, count) = self.make_public_rooms_request([None], 'test|test')
        self.assertEqual(count, 0)

    def test_returns_only_space_based_on_filter(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (chunk, count) = self.make_public_rooms_request(['m.space'])
        self.assertEqual(count, 1)
        self.assertEqual(chunk[0].get('room_type', None), 'm.space')
        (chunk, count) = self.make_public_rooms_request(['m.space'], 'test|test')
        self.assertEqual(count, 0)

    def test_returns_both_rooms_and_space_based_on_filter(self) -> None:
        if False:
            return 10
        (chunk, count) = self.make_public_rooms_request(['m.space', None])
        self.assertEqual(count, 2)
        (chunk, count) = self.make_public_rooms_request(['m.space', None], 'test|test')
        self.assertEqual(count, 0)

    def test_returns_both_rooms_and_spaces_if_array_is_empty(self) -> None:
        if False:
            i = 10
            return i + 15
        (chunk, count) = self.make_public_rooms_request([])
        self.assertEqual(count, 2)
        (chunk, count) = self.make_public_rooms_request([], 'test|test')
        self.assertEqual(count, 0)

class PublicRoomsTestRemoteSearchFallbackTestCase(unittest.HomeserverTestCase):
    """Test that we correctly fallback to local filtering if a remote server
    doesn't support search.
    """
    servlets = [synapse.rest.admin.register_servlets_for_client_rest_resource, room.register_servlets, login.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            i = 10
            return i + 15
        return self.setup_test_homeserver(federation_client=AsyncMock())

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.register_user('user', 'pass')
        self.token = self.login('user', 'pass')
        self.federation_client = hs.get_federation_client()

    def test_simple(self) -> None:
        if False:
            while True:
                i = 10
        'Simple test for searching rooms over federation'
        self.federation_client.get_public_rooms.return_value = {}
        search_filter = {PublicRoomsFilterFields.GENERIC_SEARCH_TERM: 'foobar'}
        channel = self.make_request('POST', b'/_matrix/client/r0/publicRooms?server=testserv', content={'filter': search_filter}, access_token=self.token)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self.federation_client.get_public_rooms.assert_called_once_with('testserv', limit=100, since_token=None, search_filter=search_filter, include_all_networks=False, third_party_instance_id=None)

    def test_fallback(self) -> None:
        if False:
            return 10
        'Test that searching public rooms over federation falls back if it gets a 404'
        self.federation_client.get_public_rooms.side_effect = (HttpResponseException(HTTPStatus.NOT_FOUND, 'Not Found', b''), {})
        search_filter = {PublicRoomsFilterFields.GENERIC_SEARCH_TERM: 'foobar'}
        channel = self.make_request('POST', b'/_matrix/client/r0/publicRooms?server=testserv', content={'filter': search_filter}, access_token=self.token)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self.federation_client.get_public_rooms.assert_has_calls([call('testserv', limit=100, since_token=None, search_filter=search_filter, include_all_networks=False, third_party_instance_id=None), call('testserv', limit=None, since_token=None, search_filter=None, include_all_networks=False, third_party_instance_id=None)])

class PerRoomProfilesForbiddenTestCase(unittest.HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets_for_client_rest_resource, room.register_servlets, login.register_servlets, profile.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            return 10
        config = self.default_config()
        config['allow_per_room_profiles'] = False
        self.hs = self.setup_test_homeserver(config=config)
        return self.hs

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.user_id = self.register_user('test', 'test')
        self.tok = self.login('test', 'test')
        self.displayname = 'test user'
        request_data = {'displayname': self.displayname}
        channel = self.make_request('PUT', '/_matrix/client/r0/profile/%s/displayname' % (self.user_id,), request_data, access_token=self.tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self.room_id = self.helper.create_room_as(self.user_id, tok=self.tok)

    def test_per_room_profile_forbidden(self) -> None:
        if False:
            while True:
                i = 10
        request_data = {'membership': 'join', 'displayname': 'other test user'}
        channel = self.make_request('PUT', '/_matrix/client/r0/rooms/%s/state/m.room.member/%s' % (self.room_id, self.user_id), request_data, access_token=self.tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        event_id = channel.json_body['event_id']
        channel = self.make_request('GET', '/_matrix/client/r0/rooms/%s/event/%s' % (self.room_id, event_id), access_token=self.tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        res_displayname = channel.json_body['content']['displayname']
        self.assertEqual(res_displayname, self.displayname, channel.result)

class RoomMembershipReasonTestCase(unittest.HomeserverTestCase):
    """Tests that clients can add a "reason" field to membership events and
    that they get correctly added to the generated events and propagated.
    """
    servlets = [synapse.rest.admin.register_servlets_for_client_rest_resource, room.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            i = 10
            return i + 15
        self.creator = self.register_user('creator', 'test')
        self.creator_tok = self.login('creator', 'test')
        self.second_user_id = self.register_user('second', 'test')
        self.second_tok = self.login('second', 'test')
        self.room_id = self.helper.create_room_as(self.creator, tok=self.creator_tok)

    def test_join_reason(self) -> None:
        if False:
            i = 10
            return i + 15
        reason = 'hello'
        channel = self.make_request('POST', f'/_matrix/client/r0/rooms/{self.room_id}/join', content={'reason': reason}, access_token=self.second_tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self._check_for_reason(reason)

    def test_leave_reason(self) -> None:
        if False:
            i = 10
            return i + 15
        self.helper.join(self.room_id, user=self.second_user_id, tok=self.second_tok)
        reason = 'hello'
        channel = self.make_request('POST', f'/_matrix/client/r0/rooms/{self.room_id}/leave', content={'reason': reason}, access_token=self.second_tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self._check_for_reason(reason)

    def test_kick_reason(self) -> None:
        if False:
            return 10
        self.helper.join(self.room_id, user=self.second_user_id, tok=self.second_tok)
        reason = 'hello'
        channel = self.make_request('POST', f'/_matrix/client/r0/rooms/{self.room_id}/kick', content={'reason': reason, 'user_id': self.second_user_id}, access_token=self.second_tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self._check_for_reason(reason)

    def test_ban_reason(self) -> None:
        if False:
            i = 10
            return i + 15
        self.helper.join(self.room_id, user=self.second_user_id, tok=self.second_tok)
        reason = 'hello'
        channel = self.make_request('POST', f'/_matrix/client/r0/rooms/{self.room_id}/ban', content={'reason': reason, 'user_id': self.second_user_id}, access_token=self.creator_tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self._check_for_reason(reason)

    def test_unban_reason(self) -> None:
        if False:
            i = 10
            return i + 15
        reason = 'hello'
        channel = self.make_request('POST', f'/_matrix/client/r0/rooms/{self.room_id}/unban', content={'reason': reason, 'user_id': self.second_user_id}, access_token=self.creator_tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self._check_for_reason(reason)

    def test_invite_reason(self) -> None:
        if False:
            print('Hello World!')
        reason = 'hello'
        channel = self.make_request('POST', f'/_matrix/client/r0/rooms/{self.room_id}/invite', content={'reason': reason, 'user_id': self.second_user_id}, access_token=self.creator_tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self._check_for_reason(reason)

    def test_reject_invite_reason(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.helper.invite(self.room_id, src=self.creator, targ=self.second_user_id, tok=self.creator_tok)
        reason = 'hello'
        channel = self.make_request('POST', f'/_matrix/client/r0/rooms/{self.room_id}/leave', content={'reason': reason}, access_token=self.second_tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self._check_for_reason(reason)

    def _check_for_reason(self, reason: str) -> None:
        if False:
            return 10
        channel = self.make_request('GET', '/_matrix/client/r0/rooms/{}/state/m.room.member/{}'.format(self.room_id, self.second_user_id), access_token=self.creator_tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        event_content = channel.json_body
        self.assertEqual(event_content.get('reason'), reason, channel.result)

class LabelsTestCase(unittest.HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets_for_client_rest_resource, room.register_servlets, login.register_servlets, profile.register_servlets]
    FILTER_LABELS = {'types': [EventTypes.Message], 'org.matrix.labels': ['#fun']}
    FILTER_NOT_LABELS = {'types': [EventTypes.Message], 'org.matrix.not_labels': ['#fun']}
    FILTER_LABELS_NOT_LABELS = {'types': [EventTypes.Message], 'org.matrix.labels': ['#work'], 'org.matrix.not_labels': ['#notfun']}

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.user_id = self.register_user('test', 'test')
        self.tok = self.login('test', 'test')
        self.room_id = self.helper.create_room_as(self.user_id, tok=self.tok)

    def test_context_filter_labels(self) -> None:
        if False:
            while True:
                i = 10
        'Test that we can filter by a label on a /context request.'
        event_id = self._send_labelled_messages_in_room()
        channel = self.make_request('GET', '/rooms/%s/context/%s?filter=%s' % (self.room_id, event_id, json.dumps(self.FILTER_LABELS)), access_token=self.tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        events_before = channel.json_body['events_before']
        self.assertEqual(len(events_before), 1, [event['content'] for event in events_before])
        self.assertEqual(events_before[0]['content']['body'], 'with right label', events_before[0])
        events_after = channel.json_body['events_before']
        self.assertEqual(len(events_after), 1, [event['content'] for event in events_after])
        self.assertEqual(events_after[0]['content']['body'], 'with right label', events_after[0])

    def test_context_filter_not_labels(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that we can filter by the absence of a label on a /context request.'
        event_id = self._send_labelled_messages_in_room()
        channel = self.make_request('GET', '/rooms/%s/context/%s?filter=%s' % (self.room_id, event_id, json.dumps(self.FILTER_NOT_LABELS)), access_token=self.tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        events_before = channel.json_body['events_before']
        self.assertEqual(len(events_before), 1, [event['content'] for event in events_before])
        self.assertEqual(events_before[0]['content']['body'], 'without label', events_before[0])
        events_after = channel.json_body['events_after']
        self.assertEqual(len(events_after), 2, [event['content'] for event in events_after])
        self.assertEqual(events_after[0]['content']['body'], 'with wrong label', events_after[0])
        self.assertEqual(events_after[1]['content']['body'], 'with two wrong labels', events_after[1])

    def test_context_filter_labels_not_labels(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that we can filter by both a label and the absence of another label on a\n        /context request.\n        '
        event_id = self._send_labelled_messages_in_room()
        channel = self.make_request('GET', '/rooms/%s/context/%s?filter=%s' % (self.room_id, event_id, json.dumps(self.FILTER_LABELS_NOT_LABELS)), access_token=self.tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        events_before = channel.json_body['events_before']
        self.assertEqual(len(events_before), 0, [event['content'] for event in events_before])
        events_after = channel.json_body['events_after']
        self.assertEqual(len(events_after), 1, [event['content'] for event in events_after])
        self.assertEqual(events_after[0]['content']['body'], 'with wrong label', events_after[0])

    def test_messages_filter_labels(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that we can filter by a label on a /messages request.'
        self._send_labelled_messages_in_room()
        token = 's0_0_0_0_0_0_0_0_0_0'
        channel = self.make_request('GET', '/rooms/%s/messages?access_token=%s&from=%s&filter=%s' % (self.room_id, self.tok, token, json.dumps(self.FILTER_LABELS)))
        events = channel.json_body['chunk']
        self.assertEqual(len(events), 2, [event['content'] for event in events])
        self.assertEqual(events[0]['content']['body'], 'with right label', events[0])
        self.assertEqual(events[1]['content']['body'], 'with right label', events[1])

    def test_messages_filter_not_labels(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that we can filter by the absence of a label on a /messages request.'
        self._send_labelled_messages_in_room()
        token = 's0_0_0_0_0_0_0_0_0_0'
        channel = self.make_request('GET', '/rooms/%s/messages?access_token=%s&from=%s&filter=%s' % (self.room_id, self.tok, token, json.dumps(self.FILTER_NOT_LABELS)))
        events = channel.json_body['chunk']
        self.assertEqual(len(events), 4, [event['content'] for event in events])
        self.assertEqual(events[0]['content']['body'], 'without label', events[0])
        self.assertEqual(events[1]['content']['body'], 'without label', events[1])
        self.assertEqual(events[2]['content']['body'], 'with wrong label', events[2])
        self.assertEqual(events[3]['content']['body'], 'with two wrong labels', events[3])

    def test_messages_filter_labels_not_labels(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that we can filter by both a label and the absence of another label on a\n        /messages request.\n        '
        self._send_labelled_messages_in_room()
        token = 's0_0_0_0_0_0_0_0_0_0'
        channel = self.make_request('GET', '/rooms/%s/messages?access_token=%s&from=%s&filter=%s' % (self.room_id, self.tok, token, json.dumps(self.FILTER_LABELS_NOT_LABELS)))
        events = channel.json_body['chunk']
        self.assertEqual(len(events), 1, [event['content'] for event in events])
        self.assertEqual(events[0]['content']['body'], 'with wrong label', events[0])

    def test_search_filter_labels(self) -> None:
        if False:
            while True:
                i = 10
        'Test that we can filter by a label on a /search request.'
        request_data = {'search_categories': {'room_events': {'search_term': 'label', 'filter': self.FILTER_LABELS}}}
        self._send_labelled_messages_in_room()
        channel = self.make_request('POST', '/search?access_token=%s' % self.tok, request_data)
        results = channel.json_body['search_categories']['room_events']['results']
        self.assertEqual(len(results), 2, [result['result']['content'] for result in results])
        self.assertEqual(results[0]['result']['content']['body'], 'with right label', results[0]['result']['content']['body'])
        self.assertEqual(results[1]['result']['content']['body'], 'with right label', results[1]['result']['content']['body'])

    def test_search_filter_not_labels(self) -> None:
        if False:
            return 10
        'Test that we can filter by the absence of a label on a /search request.'
        request_data = {'search_categories': {'room_events': {'search_term': 'label', 'filter': self.FILTER_NOT_LABELS}}}
        self._send_labelled_messages_in_room()
        channel = self.make_request('POST', '/search?access_token=%s' % self.tok, request_data)
        results = channel.json_body['search_categories']['room_events']['results']
        self.assertEqual(len(results), 4, [result['result']['content'] for result in results])
        self.assertEqual(results[0]['result']['content']['body'], 'without label', results[0]['result']['content']['body'])
        self.assertEqual(results[1]['result']['content']['body'], 'without label', results[1]['result']['content']['body'])
        self.assertEqual(results[2]['result']['content']['body'], 'with wrong label', results[2]['result']['content']['body'])
        self.assertEqual(results[3]['result']['content']['body'], 'with two wrong labels', results[3]['result']['content']['body'])

    def test_search_filter_labels_not_labels(self) -> None:
        if False:
            return 10
        'Test that we can filter by both a label and the absence of another label on a\n        /search request.\n        '
        request_data = {'search_categories': {'room_events': {'search_term': 'label', 'filter': self.FILTER_LABELS_NOT_LABELS}}}
        self._send_labelled_messages_in_room()
        channel = self.make_request('POST', '/search?access_token=%s' % self.tok, request_data)
        results = channel.json_body['search_categories']['room_events']['results']
        self.assertEqual(len(results), 1, [result['result']['content'] for result in results])
        self.assertEqual(results[0]['result']['content']['body'], 'with wrong label', results[0]['result']['content']['body'])

    def _send_labelled_messages_in_room(self) -> str:
        if False:
            return 10
        "Sends several messages to a room with different labels (or without any) to test\n        filtering by label.\n        Returns:\n            The ID of the event to use if we're testing filtering on /context.\n        "
        self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'with right label', EventContentFields.LABELS: ['#fun']}, tok=self.tok)
        self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'without label'}, tok=self.tok)
        res = self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'without label'}, tok=self.tok)
        event_id = res['event_id']
        self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'with wrong label', EventContentFields.LABELS: ['#work']}, tok=self.tok)
        self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'with two wrong labels', EventContentFields.LABELS: ['#work', '#notfun']}, tok=self.tok)
        self.helper.send_event(room_id=self.room_id, type=EventTypes.Message, content={'msgtype': 'm.text', 'body': 'with right label', EventContentFields.LABELS: ['#fun']}, tok=self.tok)
        return event_id

class RelationsTestCase(PaginationTestCase):

    def _filter_messages(self, filter: JsonDict) -> List[str]:
        if False:
            print('Hello World!')
        'Make a request to /messages with a filter, returns the chunk of events.'
        from_token = self.get_success(self.from_token.to_string(self.hs.get_datastores().main))
        channel = self.make_request('GET', f'/rooms/{self.room_id}/messages?filter={json.dumps(filter)}&dir=f&from={from_token}', access_token=self.tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        return [ev['event_id'] for ev in channel.json_body['chunk']]

class ContextTestCase(unittest.HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets_for_client_rest_resource, room.register_servlets, login.register_servlets, account.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.user_id = self.register_user('user', 'password')
        self.tok = self.login('user', 'password')
        self.room_id = self.helper.create_room_as(self.user_id, tok=self.tok, is_public=False)
        self.other_user_id = self.register_user('user2', 'password')
        self.other_tok = self.login('user2', 'password')
        self.helper.invite(self.room_id, self.user_id, self.other_user_id, tok=self.tok)
        self.helper.join(self.room_id, self.other_user_id, tok=self.other_tok)

    def test_erased_sender(self) -> None:
        if False:
            while True:
                i = 10
        "Test that an erasure request results in the requester's events being hidden\n        from any new member of the room.\n        "
        self.helper.send(self.room_id, 'message 1', tok=self.tok)
        self.helper.send(self.room_id, 'message 2', tok=self.tok)
        event_id = self.helper.send(self.room_id, 'message 3', tok=self.tok)['event_id']
        self.helper.send(self.room_id, 'message 4', tok=self.tok)
        self.helper.send(self.room_id, 'message 5', tok=self.tok)
        channel = self.make_request('GET', '/rooms/%s/context/%s?filter={"types":["m.room.message"]}' % (self.room_id, event_id), access_token=self.tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        events_before = channel.json_body['events_before']
        self.assertEqual(len(events_before), 2, events_before)
        self.assertEqual(events_before[0].get('content', {}).get('body'), 'message 2', events_before[0])
        self.assertEqual(events_before[1].get('content', {}).get('body'), 'message 1', events_before[1])
        self.assertEqual(channel.json_body['event'].get('content', {}).get('body'), 'message 3', channel.json_body['event'])
        events_after = channel.json_body['events_after']
        self.assertEqual(len(events_after), 2, events_after)
        self.assertEqual(events_after[0].get('content', {}).get('body'), 'message 4', events_after[0])
        self.assertEqual(events_after[1].get('content', {}).get('body'), 'message 5', events_after[1])
        deactivate_account_handler = self.hs.get_deactivate_account_handler()
        self.get_success(deactivate_account_handler.deactivate_account(self.user_id, True, create_requester(self.user_id)))
        invited_user_id = self.register_user('user3', 'password')
        invited_tok = self.login('user3', 'password')
        self.helper.invite(self.room_id, self.other_user_id, invited_user_id, tok=self.other_tok)
        self.helper.join(self.room_id, invited_user_id, tok=invited_tok)
        channel = self.make_request('GET', '/rooms/%s/context/%s?filter={"types":["m.room.message"]}' % (self.room_id, event_id), access_token=invited_tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        events_before = channel.json_body['events_before']
        self.assertEqual(len(events_before), 2, events_before)
        self.assertDictEqual(events_before[0].get('content'), {}, events_before[0])
        self.assertDictEqual(events_before[1].get('content'), {}, events_before[1])
        self.assertDictEqual(channel.json_body['event'].get('content'), {}, channel.json_body['event'])
        events_after = channel.json_body['events_after']
        self.assertEqual(len(events_after), 2, events_after)
        self.assertDictEqual(events_after[0].get('content'), {}, events_after[0])
        self.assertEqual(events_after[1].get('content'), {}, events_after[1])

class RoomAliasListTestCase(unittest.HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets_for_client_rest_resource, directory.register_servlets, login.register_servlets, room.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.room_owner = self.register_user('room_owner', 'test')
        self.room_owner_tok = self.login('room_owner', 'test')
        self.room_id = self.helper.create_room_as(self.room_owner, tok=self.room_owner_tok)

    def test_no_aliases(self) -> None:
        if False:
            i = 10
            return i + 15
        res = self._get_aliases(self.room_owner_tok)
        self.assertEqual(res['aliases'], [])

    def test_not_in_room(self) -> None:
        if False:
            while True:
                i = 10
        self.register_user('user', 'test')
        user_tok = self.login('user', 'test')
        res = self._get_aliases(user_tok, expected_code=403)
        self.assertEqual(res['errcode'], 'M_FORBIDDEN')

    def test_admin_user(self) -> None:
        if False:
            i = 10
            return i + 15
        alias1 = self._random_alias()
        self._set_alias_via_directory(alias1)
        self.register_user('user', 'test', admin=True)
        user_tok = self.login('user', 'test')
        res = self._get_aliases(user_tok)
        self.assertEqual(res['aliases'], [alias1])

    def test_with_aliases(self) -> None:
        if False:
            while True:
                i = 10
        alias1 = self._random_alias()
        alias2 = self._random_alias()
        self._set_alias_via_directory(alias1)
        self._set_alias_via_directory(alias2)
        res = self._get_aliases(self.room_owner_tok)
        self.assertEqual(set(res['aliases']), {alias1, alias2})

    def test_peekable_room(self) -> None:
        if False:
            return 10
        alias1 = self._random_alias()
        self._set_alias_via_directory(alias1)
        self.helper.send_state(self.room_id, EventTypes.RoomHistoryVisibility, body={'history_visibility': 'world_readable'}, tok=self.room_owner_tok)
        self.register_user('user', 'test')
        user_tok = self.login('user', 'test')
        res = self._get_aliases(user_tok)
        self.assertEqual(res['aliases'], [alias1])

    def _get_aliases(self, access_token: str, expected_code: int=200) -> JsonDict:
        if False:
            for i in range(10):
                print('nop')
        'Calls the endpoint under test. returns the json response object.'
        channel = self.make_request('GET', '/_matrix/client/r0/rooms/%s/aliases' % (self.room_id,), access_token=access_token)
        self.assertEqual(channel.code, expected_code, channel.result)
        res = channel.json_body
        self.assertIsInstance(res, dict)
        if expected_code == 200:
            self.assertIsInstance(res['aliases'], list)
        return res

    def _random_alias(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return RoomAlias(random_string(5), self.hs.hostname).to_string()

    def _set_alias_via_directory(self, alias: str, expected_code: int=200) -> None:
        if False:
            print('Hello World!')
        url = '/_matrix/client/r0/directory/room/' + alias
        request_data = {'room_id': self.room_id}
        channel = self.make_request('PUT', url, request_data, access_token=self.room_owner_tok)
        self.assertEqual(channel.code, expected_code, channel.result)

class RoomCanonicalAliasTestCase(unittest.HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets_for_client_rest_resource, directory.register_servlets, login.register_servlets, room.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.room_owner = self.register_user('room_owner', 'test')
        self.room_owner_tok = self.login('room_owner', 'test')
        self.room_id = self.helper.create_room_as(self.room_owner, tok=self.room_owner_tok)
        self.alias = '#alias:test'
        self._set_alias_via_directory(self.alias)

    def _set_alias_via_directory(self, alias: str, expected_code: int=200) -> None:
        if False:
            while True:
                i = 10
        url = '/_matrix/client/r0/directory/room/' + alias
        request_data = {'room_id': self.room_id}
        channel = self.make_request('PUT', url, request_data, access_token=self.room_owner_tok)
        self.assertEqual(channel.code, expected_code, channel.result)

    def _get_canonical_alias(self, expected_code: int=200) -> JsonDict:
        if False:
            print('Hello World!')
        'Calls the endpoint under test. returns the json response object.'
        channel = self.make_request('GET', 'rooms/%s/state/m.room.canonical_alias' % (self.room_id,), access_token=self.room_owner_tok)
        self.assertEqual(channel.code, expected_code, channel.result)
        res = channel.json_body
        self.assertIsInstance(res, dict)
        return res

    def _set_canonical_alias(self, content: JsonDict, expected_code: int=200) -> JsonDict:
        if False:
            return 10
        'Calls the endpoint under test. returns the json response object.'
        channel = self.make_request('PUT', 'rooms/%s/state/m.room.canonical_alias' % (self.room_id,), content, access_token=self.room_owner_tok)
        self.assertEqual(channel.code, expected_code, channel.result)
        res = channel.json_body
        self.assertIsInstance(res, dict)
        return res

    def test_canonical_alias(self) -> None:
        if False:
            print('Hello World!')
        'Test a basic alias message.'
        self._get_canonical_alias(expected_code=404)
        self._set_canonical_alias({'alias': self.alias})
        res = self._get_canonical_alias()
        self.assertEqual(res, {'alias': self.alias})
        self._set_canonical_alias({})
        res = self._get_canonical_alias()
        self.assertEqual(res, {})

    def test_alt_aliases(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test a canonical alias message with alt_aliases.'
        self._set_canonical_alias({'alt_aliases': [self.alias]})
        res = self._get_canonical_alias()
        self.assertEqual(res, {'alt_aliases': [self.alias]})
        self._set_canonical_alias({})
        res = self._get_canonical_alias()
        self.assertEqual(res, {})

    def test_alias_alt_aliases(self) -> None:
        if False:
            print('Hello World!')
        'Test a canonical alias message with an alias and alt_aliases.'
        self._set_canonical_alias({'alias': self.alias, 'alt_aliases': [self.alias]})
        res = self._get_canonical_alias()
        self.assertEqual(res, {'alias': self.alias, 'alt_aliases': [self.alias]})
        self._set_canonical_alias({})
        res = self._get_canonical_alias()
        self.assertEqual(res, {})

    def test_partial_modify(self) -> None:
        if False:
            print('Hello World!')
        'Test removing only the alt_aliases.'
        self._set_canonical_alias({'alias': self.alias, 'alt_aliases': [self.alias]})
        res = self._get_canonical_alias()
        self.assertEqual(res, {'alias': self.alias, 'alt_aliases': [self.alias]})
        self._set_canonical_alias({'alias': self.alias})
        res = self._get_canonical_alias()
        self.assertEqual(res, {'alias': self.alias})

    def test_add_alias(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test removing only the alt_aliases.'
        second_alias = '#second:test'
        self._set_alias_via_directory(second_alias)
        self._set_canonical_alias({'alias': self.alias, 'alt_aliases': [self.alias]})
        self._set_canonical_alias({'alias': self.alias, 'alt_aliases': [self.alias, second_alias]})
        res = self._get_canonical_alias()
        self.assertEqual(res, {'alias': self.alias, 'alt_aliases': [self.alias, second_alias]})

    def test_bad_data(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Invalid data for alt_aliases should cause errors.'
        self._set_canonical_alias({'alt_aliases': '@bad:test'}, expected_code=400)
        self._set_canonical_alias({'alt_aliases': None}, expected_code=400)
        self._set_canonical_alias({'alt_aliases': 0}, expected_code=400)
        self._set_canonical_alias({'alt_aliases': 1}, expected_code=400)
        self._set_canonical_alias({'alt_aliases': False}, expected_code=400)
        self._set_canonical_alias({'alt_aliases': True}, expected_code=400)
        self._set_canonical_alias({'alt_aliases': {}}, expected_code=400)

    def test_bad_alias(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'An alias which does not point to the room raises a SynapseError.'
        self._set_canonical_alias({'alias': '@unknown:test'}, expected_code=400)
        self._set_canonical_alias({'alt_aliases': ['@unknown:test']}, expected_code=400)

class ThreepidInviteTestCase(unittest.HomeserverTestCase):
    servlets = [admin.register_servlets, login.register_servlets, room.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.user_id = self.register_user('thomas', 'hackme')
        self.tok = self.login('thomas', 'hackme')
        self.room_id = self.helper.create_room_as(self.user_id, tok=self.tok)

    def test_threepid_invite_spamcheck_deprecated(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test allowing/blocking threepid invites with a spam-check module.\n\n        In this test, we use the deprecated API in which callbacks return a bool.\n        '
        make_invite_mock = AsyncMock(return_value=(Mock(event_id='abc'), 0))
        self.hs.get_room_member_handler()._make_and_store_3pid_invite = make_invite_mock
        self.hs.get_identity_handler().lookup_3pid = AsyncMock(return_value=None)
        mock = AsyncMock(return_value=True, spec=lambda *x: None)
        self.hs.get_module_api_callbacks().spam_checker._user_may_send_3pid_invite_callbacks.append(mock)
        email_to_invite = 'teresa@example.com'
        channel = self.make_request(method='POST', path='/rooms/' + self.room_id + '/invite', content={'id_server': 'example.com', 'id_access_token': 'sometoken', 'medium': 'email', 'address': email_to_invite}, access_token=self.tok)
        self.assertEqual(channel.code, 200)
        mock.assert_called_with(self.user_id, 'email', email_to_invite, self.room_id)
        make_invite_mock.assert_called_once()
        mock.return_value = False
        channel = self.make_request(method='POST', path='/rooms/' + self.room_id + '/invite', content={'id_server': 'example.com', 'id_access_token': 'sometoken', 'medium': 'email', 'address': email_to_invite}, access_token=self.tok)
        self.assertEqual(channel.code, 403)
        make_invite_mock.assert_called_once()

    def test_threepid_invite_spamcheck(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test allowing/blocking threepid invites with a spam-check module.\n\n        In this test, we use the more recent API in which callbacks return a `Union[Codes, Literal["NOT_SPAM"]]`.\n        '
        make_invite_mock = AsyncMock(return_value=(Mock(event_id='abc'), 0))
        self.hs.get_room_member_handler()._make_and_store_3pid_invite = make_invite_mock
        self.hs.get_identity_handler().lookup_3pid = AsyncMock(return_value=None)
        mock = AsyncMock(return_value=synapse.module_api.NOT_SPAM, spec=lambda *x: None)
        self.hs.get_module_api_callbacks().spam_checker._user_may_send_3pid_invite_callbacks.append(mock)
        email_to_invite = 'teresa@example.com'
        channel = self.make_request(method='POST', path='/rooms/' + self.room_id + '/invite', content={'id_server': 'example.com', 'id_access_token': 'sometoken', 'medium': 'email', 'address': email_to_invite}, access_token=self.tok)
        self.assertEqual(channel.code, 200)
        mock.assert_called_with(self.user_id, 'email', email_to_invite, self.room_id)
        make_invite_mock.assert_called_once()
        mock.return_value = Codes.CONSENT_NOT_GIVEN
        channel = self.make_request(method='POST', path='/rooms/' + self.room_id + '/invite', content={'id_server': 'example.com', 'id_access_token': 'sometoken', 'medium': 'email', 'address': email_to_invite}, access_token=self.tok)
        self.assertEqual(channel.code, 403)
        self.assertEqual(channel.json_body['errcode'], Codes.CONSENT_NOT_GIVEN)
        make_invite_mock.assert_called_once()
        mock.return_value = (Codes.EXPIRED_ACCOUNT, {'field': 'value'})
        channel = self.make_request(method='POST', path='/rooms/' + self.room_id + '/invite', content={'id_server': 'example.com', 'id_access_token': 'sometoken', 'medium': 'email', 'address': email_to_invite}, access_token=self.tok)
        self.assertEqual(channel.code, 403)
        self.assertEqual(channel.json_body['errcode'], Codes.EXPIRED_ACCOUNT)
        self.assertEqual(channel.json_body['field'], 'value')
        make_invite_mock.assert_called_once()

    def test_400_missing_param_without_id_access_token(self) -> None:
        if False:
            return 10
        '\n        Test that a 3pid invite request returns 400 M_MISSING_PARAM\n        if we do not include id_access_token.\n        '
        channel = self.make_request(method='POST', path='/rooms/' + self.room_id + '/invite', content={'id_server': 'example.com', 'medium': 'email', 'address': 'teresa@example.com'}, access_token=self.tok)
        self.assertEqual(channel.code, 400)
        self.assertEqual(channel.json_body['errcode'], 'M_MISSING_PARAM')

class TimestampLookupTestCase(unittest.HomeserverTestCase):
    servlets = [admin.register_servlets, room.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self._storage_controllers = self.hs.get_storage_controllers()
        self.room_owner = self.register_user('room_owner', 'test')
        self.room_owner_tok = self.login('room_owner', 'test')

    def _inject_outlier(self, room_id: str) -> EventBase:
        if False:
            while True:
                i = 10
        (event, _context) = self.get_success(create_event(self.hs, room_id=room_id, type='m.test', sender='@test_remote_user:remote'))
        event.internal_metadata.outlier = True
        persistence = self._storage_controllers.persistence
        assert persistence is not None
        self.get_success(persistence.persist_event(event, EventContext.for_outlier(self._storage_controllers)))
        return event

    def test_no_outliers(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Test to make sure `/timestamp_to_event` does not return `outlier` events.\n        We're unable to determine whether an `outlier` is next to a gap so we\n        don't know whether it's actually the closest event. Instead, let's just\n        ignore `outliers` with this endpoint.\n\n        This test is really seeing that we choose the non-`outlier` event behind the\n        `outlier`. Since the gap checking logic considers the latest message in the room\n        as *not* next to a gap, asking over federation does not come into play here.\n        "
        room_id = self.helper.create_room_as(self.room_owner, tok=self.room_owner_tok)
        outlier_event = self._inject_outlier(room_id)
        channel = self.make_request('GET', f'/_matrix/client/v1/rooms/{room_id}/timestamp_to_event?dir=b&ts={outlier_event.origin_server_ts}', access_token=self.room_owner_tok)
        self.assertEqual(HTTPStatus.OK, channel.code, msg=channel.json_body)
        self.assertNotEqual(channel.json_body['event_id'], outlier_event.event_id)