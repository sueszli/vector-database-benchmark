import threading
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
from unittest.mock import AsyncMock, Mock
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import EventTypes, LoginType, Membership
from synapse.api.errors import SynapseError
from synapse.api.room_versions import RoomVersion
from synapse.config.homeserver import HomeServerConfig
from synapse.events import EventBase
from synapse.module_api.callbacks.third_party_event_rules_callbacks import load_legacy_third_party_event_rules
from synapse.rest import admin
from synapse.rest.client import account, login, profile, room
from synapse.server import HomeServer
from synapse.types import JsonDict, Requester, StateMap
from synapse.util import Clock
from synapse.util.frozenutils import unfreeze
from tests import unittest
if TYPE_CHECKING:
    from synapse.module_api import ModuleApi
thread_local = threading.local()

class LegacyThirdPartyRulesTestModule:

    def __init__(self, config: Dict, module_api: 'ModuleApi') -> None:
        if False:
            i = 10
            return i + 15
        thread_local.rules_module = self
        self.module_api = module_api

    async def on_create_room(self, requester: Requester, config: dict, is_requester_admin: bool) -> bool:
        return True

    async def check_event_allowed(self, event: EventBase, state: StateMap[EventBase]) -> Union[bool, dict]:
        return True

    @staticmethod
    def parse_config(config: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        return config

class LegacyDenyNewRooms(LegacyThirdPartyRulesTestModule):

    def __init__(self, config: Dict, module_api: 'ModuleApi') -> None:
        if False:
            print('Hello World!')
        super().__init__(config, module_api)

    async def on_create_room(self, requester: Requester, config: dict, is_requester_admin: bool) -> bool:
        return False

class LegacyChangeEvents(LegacyThirdPartyRulesTestModule):

    def __init__(self, config: Dict, module_api: 'ModuleApi') -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config, module_api)

    async def check_event_allowed(self, event: EventBase, state: StateMap[EventBase]) -> JsonDict:
        d = event.get_dict()
        content = unfreeze(event.content)
        content['foo'] = 'bar'
        d['content'] = content
        return d

class ThirdPartyRulesTestCase(unittest.FederatingHomeserverTestCase):
    servlets = [admin.register_servlets, login.register_servlets, room.register_servlets, profile.register_servlets, account.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            for i in range(10):
                print('nop')
        hs = self.setup_test_homeserver()
        load_legacy_third_party_event_rules(hs)

        async def approve_all_signature_checking(_: RoomVersion, pdu: EventBase) -> EventBase:
            return pdu
        hs.get_federation_server()._check_sigs_and_hash = approve_all_signature_checking

        async def _check_event_auth(origin: Any, event: Any, context: Any) -> None:
            pass
        hs.get_federation_event_handler()._check_event_auth = _check_event_auth
        return hs

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            while True:
                i = 10
        super().prepare(reactor, clock, hs)
        self.user_id = self.register_user('kermit', 'monkey')
        self.invitee = self.register_user('invitee', 'hackme')
        self.tok = self.login('kermit', 'monkey')
        try:
            self.room_id = self.helper.create_room_as(self.user_id, tok=self.tok)
        except Exception:
            pass

    def test_third_party_rules(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that a forbidden event is forbidden from being sent, but an allowed one\n        can be sent.\n        '

        async def check(ev: EventBase, state: StateMap[EventBase]) -> Tuple[bool, Optional[JsonDict]]:
            return (ev.type != 'foo.bar.forbidden', None)
        callback = Mock(spec=[], side_effect=check)
        self.hs.get_module_api_callbacks().third_party_event_rules._check_event_allowed_callbacks = [callback]
        channel = self.make_request('PUT', '/_matrix/client/r0/rooms/%s/send/foo.bar.allowed/1' % self.room_id, {}, access_token=self.tok)
        self.assertEqual(channel.code, 200, channel.result)
        callback.assert_called_once()
        state_arg = callback.call_args[0][1]
        for k in (('m.room.create', ''), ('m.room.member', self.user_id)):
            self.assertIn(k, state_arg)
            ev = state_arg[k]
            self.assertEqual(ev.type, k[0])
            self.assertEqual(ev.state_key, k[1])
        channel = self.make_request('PUT', '/_matrix/client/r0/rooms/%s/send/foo.bar.forbidden/2' % self.room_id, {}, access_token=self.tok)
        self.assertEqual(channel.code, 403, channel.result)

    def test_third_party_rules_workaround_synapse_errors_pass_through(self) -> None:
        if False:
            return 10
        '\n        Tests that the workaround introduced by https://github.com/matrix-org/synapse/pull/11042\n        is functional: that SynapseErrors are passed through from check_event_allowed\n        and bubble up to the web resource.\n\n        NEW MODULES SHOULD NOT MAKE USE OF THIS WORKAROUND!\n        This is a temporary workaround!\n        '

        class NastyHackException(SynapseError):

            def error_dict(self, config: Optional[HomeServerConfig]) -> JsonDict:
                if False:
                    for i in range(10):
                        print('nop')
                "\n                This overrides SynapseError's `error_dict` to nastily inject\n                JSON into the error response.\n                "
                result = super().error_dict(config)
                result['nasty'] = 'very'
                return result

        async def check(ev: EventBase, state: StateMap[EventBase]) -> Tuple[bool, Optional[JsonDict]]:
            raise NastyHackException(429, 'message')
        self.hs.get_module_api_callbacks().third_party_event_rules._check_event_allowed_callbacks = [check]
        channel = self.make_request('PUT', '/_matrix/client/r0/rooms/%s/send/foo.bar.forbidden/2' % self.room_id, {}, access_token=self.tok)
        self.assertEqual(channel.code, 429, channel.result)
        self.assertEqual(channel.json_body, {'errcode': 'M_UNKNOWN', 'error': 'message', 'nasty': 'very'})

    def test_cannot_modify_event(self) -> None:
        if False:
            i = 10
            return i + 15
        'cannot accidentally modify an event before it is persisted'

        async def check(ev: EventBase, state: StateMap[EventBase]) -> Tuple[bool, Optional[JsonDict]]:
            ev.content = {'x': 'y'}
            return (True, None)
        self.hs.get_module_api_callbacks().third_party_event_rules._check_event_allowed_callbacks = [check]
        channel = self.make_request('PUT', '/_matrix/client/r0/rooms/%s/send/modifyme/1' % self.room_id, {'x': 'x'}, access_token=self.tok)
        self.assertEqual(channel.code, 500, channel.result)

    def test_modify_event(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'The module can return a modified version of the event'

        async def check(ev: EventBase, state: StateMap[EventBase]) -> Tuple[bool, Optional[JsonDict]]:
            d = ev.get_dict()
            d['content'] = {'x': 'y'}
            return (True, d)
        self.hs.get_module_api_callbacks().third_party_event_rules._check_event_allowed_callbacks = [check]
        channel = self.make_request('PUT', '/_matrix/client/r0/rooms/%s/send/modifyme/1' % self.room_id, {'x': 'x'}, access_token=self.tok)
        self.assertEqual(channel.code, 200, channel.result)
        event_id = channel.json_body['event_id']
        channel = self.make_request('GET', '/_matrix/client/r0/rooms/%s/event/%s' % (self.room_id, event_id), access_token=self.tok)
        self.assertEqual(channel.code, 200, channel.result)
        ev = channel.json_body
        self.assertEqual(ev['content']['x'], 'y')

    def test_message_edit(self) -> None:
        if False:
            i = 10
            return i + 15
        "Ensure that the module doesn't cause issues with edited messages."

        async def check(ev: EventBase, state: StateMap[EventBase]) -> Tuple[bool, Optional[JsonDict]]:
            d = ev.get_dict()
            d['content'] = {'msgtype': 'm.text', 'body': d['content']['body'].upper()}
            return (True, d)
        self.hs.get_module_api_callbacks().third_party_event_rules._check_event_allowed_callbacks = [check]
        channel = self.make_request('PUT', '/_matrix/client/r0/rooms/%s/send/modifyme/1' % self.room_id, {'msgtype': 'm.text', 'body': 'Original body'}, access_token=self.tok)
        self.assertEqual(channel.code, 200, channel.result)
        orig_event_id = channel.json_body['event_id']
        channel = self.make_request('PUT', '/_matrix/client/r0/rooms/%s/send/m.room.message/2' % self.room_id, {'m.new_content': {'msgtype': 'm.text', 'body': 'Edited body'}, 'm.relates_to': {'rel_type': 'm.replace', 'event_id': orig_event_id}, 'msgtype': 'm.text', 'body': 'Edited body'}, access_token=self.tok)
        self.assertEqual(channel.code, 200, channel.result)
        edited_event_id = channel.json_body['event_id']
        channel = self.make_request('GET', '/_matrix/client/r0/rooms/%s/event/%s' % (self.room_id, orig_event_id), access_token=self.tok)
        self.assertEqual(channel.code, 200, channel.result)
        ev = channel.json_body
        self.assertEqual(ev['content']['body'], 'ORIGINAL BODY')
        channel = self.make_request('GET', '/_matrix/client/r0/rooms/%s/event/%s' % (self.room_id, edited_event_id), access_token=self.tok)
        self.assertEqual(channel.code, 200, channel.result)
        ev = channel.json_body
        self.assertEqual(ev['content']['body'], 'EDITED BODY')

    def test_send_event(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that a module can send an event into a room via the module api'
        content = {'msgtype': 'm.text', 'body': 'Hello!'}
        event_dict = {'room_id': self.room_id, 'type': 'm.room.message', 'content': content, 'sender': self.user_id}
        event: EventBase = self.get_success(self.hs.get_module_api().create_and_send_event_into_room(event_dict))
        self.assertEqual(event.sender, self.user_id)
        self.assertEqual(event.room_id, self.room_id)
        self.assertEqual(event.type, 'm.room.message')
        self.assertEqual(event.content, content)

    @unittest.override_config({'third_party_event_rules': {'module': __name__ + '.LegacyChangeEvents', 'config': {}}})
    def test_legacy_check_event_allowed(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that the wrapper for legacy check_event_allowed callbacks works\n        correctly.\n        '
        channel = self.make_request('PUT', '/_matrix/client/r0/rooms/%s/send/m.room.message/1' % self.room_id, {'msgtype': 'm.text', 'body': 'Original body'}, access_token=self.tok)
        self.assertEqual(channel.code, 200, channel.result)
        event_id = channel.json_body['event_id']
        channel = self.make_request('GET', '/_matrix/client/r0/rooms/%s/event/%s' % (self.room_id, event_id), access_token=self.tok)
        self.assertEqual(channel.code, 200, channel.result)
        self.assertIn('foo', channel.json_body['content'].keys())
        self.assertEqual(channel.json_body['content']['foo'], 'bar')

    @unittest.override_config({'third_party_event_rules': {'module': __name__ + '.LegacyDenyNewRooms', 'config': {}}})
    def test_legacy_on_create_room(self) -> None:
        if False:
            print('Hello World!')
        'Tests that the wrapper for legacy on_create_room callbacks works\n        correctly.\n        '
        self.helper.create_room_as(self.user_id, tok=self.tok, expect_code=403)

    def test_sent_event_end_up_in_room_state(self) -> None:
        if False:
            i = 10
            return i + 15
        "Tests that a state event sent by a module while processing another state event\n        doesn't get dropped from the state of the room. This is to guard against a bug\n        where Synapse has been observed doing so, see https://github.com/matrix-org/synapse/issues/10830\n        "
        event_type = 'org.matrix.test_state'
        event_content = {'i': -1}
        api = self.hs.get_module_api()

        async def test_fn(event: EventBase, state_events: StateMap[EventBase]) -> Tuple[bool, Optional[JsonDict]]:
            if event.is_state() and event.type == EventTypes.PowerLevels:
                await api.create_and_send_event_into_room({'room_id': event.room_id, 'sender': event.sender, 'type': event_type, 'content': event_content, 'state_key': ''})
            return (True, None)
        self.hs.get_module_api_callbacks().third_party_event_rules._check_event_allowed_callbacks = [test_fn]
        for i in range(5):
            event_content['i'] = i
            self._update_power_levels(event_default=i)
            channel = self.make_request(method='GET', path='/rooms/' + self.room_id + '/state/' + event_type, access_token=self.tok)
            self.assertEqual(channel.code, 200)
            self.assertEqual(channel.json_body['i'], i)

    def test_on_new_event(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that the on_new_event callback is called on new events'
        on_new_event = AsyncMock(return_value=None)
        self.hs.get_module_api_callbacks().third_party_event_rules._on_new_event_callbacks.append(on_new_event)
        self.helper.send(room_id=self.room_id, tok=self.tok)
        self.assertEqual(on_new_event.call_count, 1)
        self.helper.invite(room=self.room_id, src=self.user_id, targ=self.invitee, tok=self.tok)
        self.assertEqual(on_new_event.call_count, 2)
        (args, _) = on_new_event.call_args
        self.assertEqual(args[0].membership, Membership.INVITE)
        self.assertEqual(args[0].state_key, self.invitee)
        self.assertEqual(args[1][EventTypes.Member, self.invitee].membership, Membership.INVITE)
        self._send_event_over_federation()
        self.assertEqual(on_new_event.call_count, 3)

    def _send_event_over_federation(self) -> None:
        if False:
            while True:
                i = 10
        'Send a dummy event over federation and check that the request succeeds.'
        body = {'pdus': [{'sender': self.user_id, 'type': EventTypes.Message, 'state_key': '', 'content': {'body': 'hello world', 'msgtype': 'm.text'}, 'room_id': self.room_id, 'depth': 0, 'origin_server_ts': self.clock.time_msec(), 'prev_events': [], 'auth_events': [], 'signatures': {}, 'unsigned': {}}]}
        channel = self.make_signed_federation_request(method='PUT', path='/_matrix/federation/v1/send/1', content=body)
        self.assertEqual(channel.code, 200, channel.result)

    def _update_power_levels(self, event_default: int=0) -> None:
        if False:
            return 10
        "Updates the room's power levels.\n\n        Args:\n            event_default: Value to use for 'events_default'.\n        "
        self.helper.send_state(room_id=self.room_id, event_type=EventTypes.PowerLevels, body={'ban': 50, 'events': {'m.room.avatar': 50, 'm.room.canonical_alias': 50, 'm.room.encryption': 100, 'm.room.history_visibility': 100, 'm.room.name': 50, 'm.room.power_levels': 100, 'm.room.server_acl': 100, 'm.room.tombstone': 100}, 'events_default': event_default, 'invite': 0, 'kick': 50, 'redact': 50, 'state_default': 50, 'users': {self.user_id: 100}, 'users_default': 0}, tok=self.tok)

    def test_on_profile_update(self) -> None:
        if False:
            print('Hello World!')
        'Tests that the on_profile_update module callback is correctly called on\n        profile updates.\n        '
        displayname = 'Foo'
        avatar_url = 'mxc://matrix.org/oWQDvfewxmlRaRCkVbfetyEo'
        m = AsyncMock(return_value=None)
        self.hs.get_module_api_callbacks().third_party_event_rules._on_profile_update_callbacks.append(m)
        channel = self.make_request('PUT', '/_matrix/client/v3/profile/%s/displayname' % self.user_id, {'displayname': displayname}, access_token=self.tok)
        self.assertEqual(channel.code, 200, channel.json_body)
        m.assert_called_once()
        args = m.call_args[0]
        self.assertEqual(args[0], self.user_id)
        self.assertFalse(args[2])
        self.assertFalse(args[3])
        profile_info = args[1]
        self.assertEqual(profile_info.display_name, displayname)
        self.assertIsNone(profile_info.avatar_url)
        channel = self.make_request('PUT', '/_matrix/client/v3/profile/%s/avatar_url' % self.user_id, {'avatar_url': avatar_url}, access_token=self.tok)
        self.assertEqual(channel.code, 200, channel.json_body)
        self.assertEqual(m.call_count, 2)
        args = m.call_args[0]
        self.assertEqual(args[0], self.user_id)
        self.assertFalse(args[2])
        self.assertFalse(args[3])
        profile_info = args[1]
        self.assertEqual(profile_info.display_name, displayname)
        self.assertEqual(profile_info.avatar_url, avatar_url)

    def test_on_profile_update_admin(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that the on_profile_update module callback is correctly called on\n        profile updates triggered by a server admin.\n        '
        displayname = 'Foo'
        avatar_url = 'mxc://matrix.org/oWQDvfewxmlRaRCkVbfetyEo'
        m = AsyncMock(return_value=None)
        self.hs.get_module_api_callbacks().third_party_event_rules._on_profile_update_callbacks.append(m)
        self.register_user('admin', 'password', admin=True)
        admin_tok = self.login('admin', 'password')
        channel = self.make_request('PUT', '/_synapse/admin/v2/users/%s' % self.user_id, {'displayname': displayname, 'avatar_url': avatar_url}, access_token=admin_tok)
        self.assertEqual(channel.code, 200, channel.json_body)
        self.assertEqual(m.call_count, 2)
        args = m.call_args[0]
        self.assertEqual(args[0], self.user_id)
        self.assertTrue(args[2])
        self.assertFalse(args[3])
        profile_info = args[1]
        self.assertEqual(profile_info.display_name, displayname)
        self.assertEqual(profile_info.avatar_url, avatar_url)

    def test_on_user_deactivation_status_changed(self) -> None:
        if False:
            print('Hello World!')
        "Tests that the on_user_deactivation_status_changed module callback is called\n        correctly when processing a user's deactivation.\n        "
        deactivation_mock = AsyncMock(return_value=None)
        third_party_rules = self.hs.get_module_api_callbacks().third_party_event_rules
        third_party_rules._on_user_deactivation_status_changed_callbacks.append(deactivation_mock)
        profile_mock = AsyncMock(return_value=None)
        self.hs.get_module_api_callbacks().third_party_event_rules._on_profile_update_callbacks.append(profile_mock)
        user_id = self.register_user('altan', 'password')
        tok = self.login('altan', 'password')
        channel = self.make_request('POST', '/_matrix/client/v3/account/deactivate', {'auth': {'type': LoginType.PASSWORD, 'password': 'password', 'identifier': {'type': 'm.id.user', 'user': user_id}}, 'erase': True}, access_token=tok)
        self.assertEqual(channel.code, 200, channel.json_body)
        deactivation_mock.assert_called_once()
        args = deactivation_mock.call_args[0]
        self.assertEqual(args[0], user_id)
        self.assertTrue(args[1])
        self.assertFalse(args[2])
        self.assertEqual(profile_mock.call_count, 2)
        args = profile_mock.call_args[0]
        self.assertTrue(args[3])

    def test_on_user_deactivation_status_changed_admin(self) -> None:
        if False:
            print('Hello World!')
        "Tests that the on_user_deactivation_status_changed module callback is called\n        correctly when processing a user's deactivation triggered by a server admin as\n        well as a reactivation.\n        "
        m = AsyncMock(return_value=None)
        third_party_rules = self.hs.get_module_api_callbacks().third_party_event_rules
        third_party_rules._on_user_deactivation_status_changed_callbacks.append(m)
        self.register_user('admin', 'password', admin=True)
        admin_tok = self.login('admin', 'password')
        user_id = self.register_user('altan', 'password')
        channel = self.make_request('PUT', '/_synapse/admin/v2/users/%s' % user_id, {'deactivated': True}, access_token=admin_tok)
        self.assertEqual(channel.code, 200, channel.json_body)
        m.assert_called_once()
        args = m.call_args[0]
        self.assertEqual(args[0], user_id)
        self.assertTrue(args[1])
        self.assertTrue(args[2])
        channel = self.make_request('PUT', '/_synapse/admin/v2/users/%s' % user_id, {'deactivated': False, 'password': 'hackme'}, access_token=admin_tok)
        self.assertEqual(channel.code, 200, channel.json_body)
        self.assertEqual(m.call_count, 2)
        args = m.call_args[0]
        self.assertEqual(args[0], user_id)
        self.assertFalse(args[1])
        self.assertTrue(args[2])

    def test_check_can_deactivate_user(self) -> None:
        if False:
            print('Hello World!')
        "Tests that the on_user_deactivation_status_changed module callback is called\n        correctly when processing a user's deactivation.\n        "
        deactivation_mock = AsyncMock(return_value=False)
        third_party_rules = self.hs.get_module_api_callbacks().third_party_event_rules
        third_party_rules._check_can_deactivate_user_callbacks.append(deactivation_mock)
        user_id = self.register_user('altan', 'password')
        tok = self.login('altan', 'password')
        channel = self.make_request('POST', '/_matrix/client/v3/account/deactivate', {'auth': {'type': LoginType.PASSWORD, 'password': 'password', 'identifier': {'type': 'm.id.user', 'user': user_id}}, 'erase': True}, access_token=tok)
        self.assertEqual(channel.code, 403, channel.json_body)
        deactivation_mock.assert_called_once()
        args = deactivation_mock.call_args[0]
        self.assertEqual(args[0], user_id)
        self.assertEqual(args[1], False)

    def test_check_can_deactivate_user_admin(self) -> None:
        if False:
            while True:
                i = 10
        "Tests that the on_user_deactivation_status_changed module callback is called\n        correctly when processing a user's deactivation triggered by a server admin.\n        "
        deactivation_mock = AsyncMock(return_value=False)
        third_party_rules = self.hs.get_module_api_callbacks().third_party_event_rules
        third_party_rules._check_can_deactivate_user_callbacks.append(deactivation_mock)
        self.register_user('admin', 'password', admin=True)
        admin_tok = self.login('admin', 'password')
        user_id = self.register_user('altan', 'password')
        channel = self.make_request('PUT', '/_synapse/admin/v2/users/%s' % user_id, {'deactivated': True}, access_token=admin_tok)
        self.assertEqual(channel.code, 403, channel.json_body)
        deactivation_mock.assert_called_once()
        args = deactivation_mock.call_args[0]
        self.assertEqual(args[0], user_id)
        self.assertEqual(args[1], True)

    def test_check_can_shutdown_room(self) -> None:
        if False:
            i = 10
            return i + 15
        "Tests that the check_can_shutdown_room module callback is called\n        correctly when processing an admin's shutdown room request.\n        "
        shutdown_mock = AsyncMock(return_value=False)
        third_party_rules = self.hs.get_module_api_callbacks().third_party_event_rules
        third_party_rules._check_can_shutdown_room_callbacks.append(shutdown_mock)
        admin_user_id = self.register_user('admin', 'password', admin=True)
        admin_tok = self.login('admin', 'password')
        channel = self.make_request('DELETE', '/_synapse/admin/v2/rooms/%s' % self.room_id, {}, access_token=admin_tok)
        self.assertEqual(channel.code, 403, channel.json_body)
        shutdown_mock.assert_called_once()
        args = shutdown_mock.call_args[0]
        self.assertEqual(args[0], admin_user_id)
        self.assertEqual(args[1], self.room_id)

    def test_on_threepid_bind(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests that the on_threepid_bind module callback is called correctly after\n        associating a 3PID to an account.\n        '
        threepid_bind_mock = AsyncMock(return_value=None)
        third_party_rules = self.hs.get_module_api_callbacks().third_party_event_rules
        third_party_rules._on_threepid_bind_callbacks.append(threepid_bind_mock)
        self.register_user('admin', 'password', admin=True)
        admin_tok = self.login('admin', 'password')
        user_id = self.register_user('user', 'password')
        channel = self.make_request('PUT', '/_synapse/admin/v2/users/%s' % user_id, {'threepids': [{'medium': 'email', 'address': 'foo@example.com'}]}, access_token=admin_tok)
        self.assertEqual(channel.code, 200, channel.json_body)
        threepid_bind_mock.assert_called_once()
        args = threepid_bind_mock.call_args[0]
        self.assertEqual(args, (user_id, 'email', 'foo@example.com'))

    def test_on_add_and_remove_user_third_party_identifier(self) -> None:
        if False:
            print('Hello World!')
        'Tests that the on_add_user_third_party_identifier and\n        on_remove_user_third_party_identifier module callbacks are called\n        just before associating and removing a 3PID to/from an account.\n        '
        on_add_user_third_party_identifier_callback_mock = AsyncMock(return_value=None)
        on_remove_user_third_party_identifier_callback_mock = AsyncMock(return_value=None)
        self.hs.get_module_api().register_third_party_rules_callbacks(on_add_user_third_party_identifier=on_add_user_third_party_identifier_callback_mock, on_remove_user_third_party_identifier=on_remove_user_third_party_identifier_callback_mock)
        self.register_user('admin', 'password', admin=True)
        admin_tok = self.login('admin', 'password')
        user_id = self.register_user('user', 'password')
        channel = self.make_request('PUT', '/_synapse/admin/v2/users/%s' % user_id, {'threepids': [{'medium': 'email', 'address': 'foo@example.com'}]}, access_token=admin_tok)
        self.assertEqual(channel.code, 200, channel.json_body)
        on_add_user_third_party_identifier_callback_mock.assert_called_once()
        args = on_add_user_third_party_identifier_callback_mock.call_args[0]
        self.assertEqual(args, (user_id, 'email', 'foo@example.com'))
        channel = self.make_request('PUT', '/_synapse/admin/v2/users/%s' % user_id, {'threepids': []}, access_token=admin_tok)
        self.assertEqual(channel.code, 200, channel.json_body)
        on_remove_user_third_party_identifier_callback_mock.assert_called_once()
        args = on_remove_user_third_party_identifier_callback_mock.call_args[0]
        self.assertEqual(args, (user_id, 'email', 'foo@example.com'))

    def test_on_remove_user_third_party_identifier_is_called_on_deactivate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests that the on_remove_user_third_party_identifier module callback is called\n        when a user is deactivated and their third-party ID associations are deleted.\n        '
        on_remove_user_third_party_identifier_callback_mock = AsyncMock(return_value=None)
        self.hs.get_module_api().register_third_party_rules_callbacks(on_remove_user_third_party_identifier=on_remove_user_third_party_identifier_callback_mock)
        self.register_user('admin', 'password', admin=True)
        admin_tok = self.login('admin', 'password')
        user_id = self.register_user('user', 'password')
        channel = self.make_request('PUT', '/_synapse/admin/v2/users/%s' % user_id, {'threepids': [{'medium': 'email', 'address': 'foo@example.com'}]}, access_token=admin_tok)
        self.assertEqual(channel.code, 200, channel.json_body)
        on_remove_user_third_party_identifier_callback_mock.assert_not_called()
        channel = self.make_request('PUT', '/_synapse/admin/v2/users/%s' % user_id, {'deactivated': True}, access_token=admin_tok)
        self.assertEqual(channel.code, 200, channel.json_body)
        on_remove_user_third_party_identifier_callback_mock.assert_called_once()
        args = on_remove_user_third_party_identifier_callback_mock.call_args[0]
        self.assertEqual(args, (user_id, 'email', 'foo@example.com'))