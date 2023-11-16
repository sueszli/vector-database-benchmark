from typing import Tuple
from unittest.mock import AsyncMock, Mock
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import EventTypes, LimitBlockingTypes, ServerNoticeMsgType
from synapse.api.errors import ResourceLimitError
from synapse.rest import admin
from synapse.rest.client import login, room, sync
from synapse.server import HomeServer
from synapse.server_notices.resource_limits_server_notices import ResourceLimitsServerNotices
from synapse.server_notices.server_notices_sender import ServerNoticesSender
from synapse.types import JsonDict
from synapse.util import Clock
from tests import unittest
from tests.unittest import override_config
from tests.utils import default_config

class TestResourceLimitsServerNotices(unittest.HomeserverTestCase):

    def default_config(self) -> JsonDict:
        if False:
            i = 10
            return i + 15
        config = default_config('test')
        config.update({'admin_contact': 'mailto:user@test.com', 'limit_usage_by_mau': True, 'server_notices': {'system_mxid_localpart': 'server', 'system_mxid_display_name': 'test display name', 'system_mxid_avatar_url': None, 'room_name': 'Server Notices'}})
        if self._extra_config is not None:
            config.update(self._extra_config)
        return config

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            i = 10
            return i + 15
        server_notices_sender = self.hs.get_server_notices_sender()
        assert isinstance(server_notices_sender, ServerNoticesSender)
        rlsn = list(server_notices_sender._server_notices)[1]
        assert isinstance(rlsn, ResourceLimitsServerNotices)
        self._rlsn = rlsn
        self._rlsn._store.user_last_seen_monthly_active = AsyncMock(return_value=1000)
        self._rlsn._server_notices_manager.send_notice = AsyncMock(return_value=Mock())
        self._send_notice = self._rlsn._server_notices_manager.send_notice
        self.user_id = '@user_id:test'
        self._rlsn._server_notices_manager.get_or_create_notice_room_for_user = AsyncMock(return_value='!something:localhost')
        self._rlsn._server_notices_manager.maybe_get_notice_room_for_user = AsyncMock(return_value='!something:localhost')
        self._rlsn._store.add_tag_to_room = AsyncMock(return_value=None)
        self._rlsn._store.get_tags_for_room = AsyncMock(return_value={})

    @override_config({'hs_disabled': True})
    def test_maybe_send_server_notice_disabled_hs(self) -> None:
        if False:
            while True:
                i = 10
        'If the HS is disabled, we should not send notices'
        self.get_success(self._rlsn.maybe_send_server_notice_to_user(self.user_id))
        self._send_notice.assert_not_called()

    @override_config({'limit_usage_by_mau': False})
    def test_maybe_send_server_notice_to_user_flag_off(self) -> None:
        if False:
            i = 10
            return i + 15
        'If mau limiting is disabled, we should not send notices'
        self.get_success(self._rlsn.maybe_send_server_notice_to_user(self.user_id))
        self._send_notice.assert_not_called()

    def test_maybe_send_server_notice_to_user_remove_blocked_notice(self) -> None:
        if False:
            while True:
                i = 10
        'Test when user has blocked notice, but should have it removed'
        self._rlsn._auth_blocking.check_auth_blocking = AsyncMock(return_value=None)
        mock_event = Mock(type=EventTypes.Message, content={'msgtype': ServerNoticeMsgType})
        self._rlsn._store.get_events = AsyncMock(return_value={'123': mock_event})
        self.get_success(self._rlsn.maybe_send_server_notice_to_user(self.user_id))
        maybe_get_notice_room_for_user = self._rlsn._server_notices_manager.maybe_get_notice_room_for_user
        assert isinstance(maybe_get_notice_room_for_user, Mock)
        maybe_get_notice_room_for_user.assert_called_once()
        self._send_notice.assert_called_once()

    def test_maybe_send_server_notice_to_user_remove_blocked_notice_noop(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test when user has blocked notice, but notice ought to be there (NOOP)\n        '
        self._rlsn._auth_blocking.check_auth_blocking = AsyncMock(return_value=None, side_effect=ResourceLimitError(403, 'foo'))
        mock_event = Mock(type=EventTypes.Message, content={'msgtype': ServerNoticeMsgType})
        self._rlsn._store.get_events = AsyncMock(return_value={'123': mock_event})
        self.get_success(self._rlsn.maybe_send_server_notice_to_user(self.user_id))
        self._send_notice.assert_not_called()

    def test_maybe_send_server_notice_to_user_add_blocked_notice(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test when user does not have blocked notice, but should have one\n        '
        self._rlsn._auth_blocking.check_auth_blocking = AsyncMock(return_value=None, side_effect=ResourceLimitError(403, 'foo'))
        self.get_success(self._rlsn.maybe_send_server_notice_to_user(self.user_id))
        self.assertEqual(self._send_notice.call_count, 2)

    def test_maybe_send_server_notice_to_user_add_blocked_notice_noop(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test when user does not have blocked notice, nor should they (NOOP)\n        '
        self._rlsn._auth_blocking.check_auth_blocking = AsyncMock(return_value=None)
        self.get_success(self._rlsn.maybe_send_server_notice_to_user(self.user_id))
        self._send_notice.assert_not_called()

    def test_maybe_send_server_notice_to_user_not_in_mau_cohort(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test when user is not part of the MAU cohort - this should not ever\n        happen - but ...\n        '
        self._rlsn._auth_blocking.check_auth_blocking = AsyncMock(return_value=None)
        self._rlsn._store.user_last_seen_monthly_active = AsyncMock(return_value=None)
        self.get_success(self._rlsn.maybe_send_server_notice_to_user(self.user_id))
        self._send_notice.assert_not_called()

    @override_config({'mau_limit_alerting': False})
    def test_maybe_send_server_notice_when_alerting_suppressed_room_unblocked(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test that when server is over MAU limit and alerting is suppressed, then\n        an alert message is not sent into the room\n        '
        self._rlsn._auth_blocking.check_auth_blocking = AsyncMock(return_value=None, side_effect=ResourceLimitError(403, 'foo', limit_type=LimitBlockingTypes.MONTHLY_ACTIVE_USER))
        self.get_success(self._rlsn.maybe_send_server_notice_to_user(self.user_id))
        self.assertEqual(self._send_notice.call_count, 0)

    @override_config({'mau_limit_alerting': False})
    def test_check_hs_disabled_unaffected_by_mau_alert_suppression(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test that when a server is disabled, that MAU limit alerting is ignored.\n        '
        self._rlsn._auth_blocking.check_auth_blocking = AsyncMock(return_value=None, side_effect=ResourceLimitError(403, 'foo', limit_type=LimitBlockingTypes.HS_DISABLED))
        self.get_success(self._rlsn.maybe_send_server_notice_to_user(self.user_id))
        self.assertEqual(self._send_notice.call_count, 2)

    @override_config({'mau_limit_alerting': False})
    def test_maybe_send_server_notice_when_alerting_suppressed_room_blocked(self) -> None:
        if False:
            print('Hello World!')
        '\n        When the room is already in a blocked state, test that when alerting\n        is suppressed that the room is returned to an unblocked state.\n        '
        self._rlsn._auth_blocking.check_auth_blocking = AsyncMock(return_value=None, side_effect=ResourceLimitError(403, 'foo', limit_type=LimitBlockingTypes.MONTHLY_ACTIVE_USER))
        self._rlsn._is_room_currently_blocked = AsyncMock(return_value=(True, []))
        mock_event = Mock(type=EventTypes.Message, content={'msgtype': ServerNoticeMsgType})
        self._rlsn._store.get_events = AsyncMock(return_value={'123': mock_event})
        self.get_success(self._rlsn.maybe_send_server_notice_to_user(self.user_id))
        self._send_notice.assert_called_once()

class TestResourceLimitsServerNoticesWithRealRooms(unittest.HomeserverTestCase):
    servlets = [admin.register_servlets, login.register_servlets, room.register_servlets, sync.register_servlets]

    def default_config(self) -> JsonDict:
        if False:
            while True:
                i = 10
        c = super().default_config()
        c['server_notices'] = {'system_mxid_localpart': 'server', 'system_mxid_display_name': None, 'system_mxid_avatar_url': None, 'room_name': 'Test Server Notice Room'}
        c['limit_usage_by_mau'] = True
        c['max_mau_value'] = 5
        c['admin_contact'] = 'mailto:user@test.com'
        return c

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.store = self.hs.get_datastores().main
        self.server_notices_manager = self.hs.get_server_notices_manager()
        self.event_source = self.hs.get_event_sources()
        server_notices_sender = self.hs.get_server_notices_sender()
        assert isinstance(server_notices_sender, ServerNoticesSender)
        rlsn = list(server_notices_sender._server_notices)[1]
        assert isinstance(rlsn, ResourceLimitsServerNotices)
        self._rlsn = rlsn
        self.user_id = '@user_id:test'

    def test_server_notice_only_sent_once(self) -> None:
        if False:
            while True:
                i = 10
        self.store.get_monthly_active_count = AsyncMock(return_value=1000)
        self.store.user_last_seen_monthly_active = AsyncMock(return_value=1000)
        self.get_success(self._rlsn.maybe_send_server_notice_to_user(self.user_id))
        self.get_success(self._rlsn.maybe_send_server_notice_to_user(self.user_id))
        self.get_success(self._rlsn.maybe_send_server_notice_to_user(self.user_id))
        room_id = self.get_success(self.server_notices_manager.get_or_create_notice_room_for_user(self.user_id))
        token = self.event_source.get_current_token()
        (events, _) = self.get_success(self.store.get_recent_events_for_room(room_id, limit=100, end_token=token.room_key))
        count = 0
        for event in events:
            if event.type != EventTypes.Message:
                continue
            if event.content.get('msgtype') != ServerNoticeMsgType:
                continue
            count += 1
        self.assertEqual(count, 1)

    def test_no_invite_without_notice(self) -> None:
        if False:
            i = 10
            return i + 15
        "Tests that a user doesn't get invited to a server notices room without a\n        server notice being sent.\n\n        The scenario for this test is a single user on a server where the MAU limit\n        hasn't been reached (since it's the only user and the limit is 5), so users\n        shouldn't receive a server notice.\n        "
        m = AsyncMock(return_value=None)
        self._rlsn._server_notices_manager.maybe_get_notice_room_for_user = m
        user_id = self.register_user('user', 'password')
        tok = self.login('user', 'password')
        channel = self.make_request('GET', '/sync?timeout=0', access_token=tok)
        self.assertNotIn('rooms', channel.json_body, 'Got invites without server notice')
        m.assert_called_once_with(user_id)

    def test_invite_with_notice(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that, if the MAU limit is hit, the server notices user invites each user\n        to a room in which it has sent a notice.\n        '
        (user_id, tok, room_id) = self._trigger_notice_and_join()
        channel = self.make_request('GET', '/sync?timeout=0', access_token=tok)
        events = channel.json_body['rooms']['join'][room_id]['timeline']['events']
        notice_in_room = False
        for event in events:
            if event['type'] == EventTypes.Message and event['sender'] == self.hs.config.servernotices.server_notices_mxid:
                notice_in_room = True
        self.assertTrue(notice_in_room, 'No server notice in room')

    def _trigger_notice_and_join(self) -> Tuple[str, str, str]:
        if False:
            for i in range(10):
                print('nop')
        "Creates enough active users to hit the MAU limit and trigger a system notice\n        about it, then joins the system notices room with one of the users created.\n\n        Returns:\n            A tuple of:\n                user_id: The ID of the user that joined the room.\n                tok: The access token of the user that joined the room.\n                room_id: The ID of the room that's been joined.\n        "
        self.assertGreater(self.hs.config.server.max_mau_value, 0)
        invites = {}
        for i in range(self.hs.config.server.max_mau_value):
            localpart = 'user%d' % i
            user_id = self.register_user(localpart, 'password')
            tok = self.login(localpart, 'password')
            channel = self.make_request('GET', '/sync?timeout=0', access_token=tok)
            if 'rooms' in channel.json_body:
                invites = channel.json_body['rooms']['invite']
        self.assertEqual(len(invites), 1, invites)
        room_id = list(invites.keys())[0]
        self.helper.join(room=room_id, user=user_id, tok=tok)
        return (user_id, tok, room_id)