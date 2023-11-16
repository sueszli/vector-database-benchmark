from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
from unittest.mock import AsyncMock, Mock
import attr
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import EduTypes
from synapse.events.presence_router import PresenceRouter, load_legacy_presence_router
from synapse.federation.units import Transaction
from synapse.handlers.presence import UserPresenceState
from synapse.module_api import ModuleApi
from synapse.rest import admin
from synapse.rest.client import login, presence, room
from synapse.server import HomeServer
from synapse.types import JsonDict, StreamToken, create_requester
from synapse.util import Clock
from tests.handlers.test_sync import generate_sync_config
from tests.unittest import FederatingHomeserverTestCase, HomeserverTestCase, override_config

@attr.s
class PresenceRouterTestConfig:
    users_who_should_receive_all_presence = attr.ib(type=List[str], default=[])

class LegacyPresenceRouterTestModule:

    def __init__(self, config: PresenceRouterTestConfig, module_api: ModuleApi):
        if False:
            return 10
        self._config = config
        self._module_api = module_api

    async def get_users_for_states(self, state_updates: Iterable[UserPresenceState]) -> Dict[str, Set[UserPresenceState]]:
        users_to_state = {user_id: set(state_updates) for user_id in self._config.users_who_should_receive_all_presence}
        return users_to_state

    async def get_interested_users(self, user_id: str) -> Union[Set[str], str]:
        if user_id in self._config.users_who_should_receive_all_presence:
            return PresenceRouter.ALL_USERS
        return set()

    @staticmethod
    def parse_config(config_dict: dict) -> PresenceRouterTestConfig:
        if False:
            i = 10
            return i + 15
        'Parse a configuration dictionary from the homeserver config, do\n        some validation and return a typed PresenceRouterConfig.\n\n        Args:\n            config_dict: The configuration dictionary.\n\n        Returns:\n            A validated config object.\n        '
        config = PresenceRouterTestConfig()
        users_who_should_receive_all_presence = config_dict.get('users_who_should_receive_all_presence')
        assert isinstance(users_who_should_receive_all_presence, list)
        config.users_who_should_receive_all_presence = users_who_should_receive_all_presence
        return config

class PresenceRouterTestModule:

    def __init__(self, config: PresenceRouterTestConfig, api: ModuleApi):
        if False:
            i = 10
            return i + 15
        self._config = config
        self._module_api = api
        api.register_presence_router_callbacks(get_users_for_states=self.get_users_for_states, get_interested_users=self.get_interested_users)

    async def get_users_for_states(self, state_updates: Iterable[UserPresenceState]) -> Dict[str, Set[UserPresenceState]]:
        users_to_state = {user_id: set(state_updates) for user_id in self._config.users_who_should_receive_all_presence}
        return users_to_state

    async def get_interested_users(self, user_id: str) -> Union[Set[str], str]:
        if user_id in self._config.users_who_should_receive_all_presence:
            return PresenceRouter.ALL_USERS
        return set()

    @staticmethod
    def parse_config(config_dict: dict) -> PresenceRouterTestConfig:
        if False:
            while True:
                i = 10
        'Parse a configuration dictionary from the homeserver config, do\n        some validation and return a typed PresenceRouterConfig.\n\n        Args:\n            config_dict: The configuration dictionary.\n\n        Returns:\n            A validated config object.\n        '
        config = PresenceRouterTestConfig()
        users_who_should_receive_all_presence = config_dict.get('users_who_should_receive_all_presence')
        assert isinstance(users_who_should_receive_all_presence, list)
        config.users_who_should_receive_all_presence = users_who_should_receive_all_presence
        return config

class PresenceRouterTestCase(FederatingHomeserverTestCase):
    """
    Test cases using a custom PresenceRouter

    By default in test cases, federation sending is disabled. This class re-enables it
    for the main process by setting `federation_sender_instances` to None.
    """
    servlets = [admin.register_servlets, login.register_servlets, room.register_servlets, presence.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            print('Hello World!')
        self.fed_transport_client = Mock(spec=['send_transaction'])
        self.fed_transport_client.send_transaction = AsyncMock(return_value={})
        hs = self.setup_test_homeserver(federation_transport_client=self.fed_transport_client)
        load_legacy_presence_router(hs)
        return hs

    def prepare(self, reactor: MemoryReactor, clock: Clock, homeserver: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.sync_handler = self.hs.get_sync_handler()
        self.module_api = homeserver.get_module_api()

    def default_config(self) -> JsonDict:
        if False:
            return 10
        config = super().default_config()
        config['federation_sender_instances'] = None
        return config

    @override_config({'presence': {'presence_router': {'module': __name__ + '.LegacyPresenceRouterTestModule', 'config': {'users_who_should_receive_all_presence': ['@presence_gobbler:test']}}}})
    def test_receiving_all_presence_legacy(self) -> None:
        if False:
            return 10
        self.receiving_all_presence_test_body()

    @override_config({'modules': [{'module': __name__ + '.PresenceRouterTestModule', 'config': {'users_who_should_receive_all_presence': ['@presence_gobbler:test']}}]})
    def test_receiving_all_presence(self) -> None:
        if False:
            print('Hello World!')
        self.receiving_all_presence_test_body()

    def receiving_all_presence_test_body(self) -> None:
        if False:
            while True:
                i = 10
        'Test that a user that does not share a room with another other can receive\n        presence for them, due to presence routing.\n        '
        self.presence_receiving_user_id = self.register_user('presence_gobbler', 'monkey')
        self.presence_receiving_user_tok = self.login('presence_gobbler', 'monkey')
        self.other_user_one_id = self.register_user('other_user_one', 'monkey')
        self.other_user_one_tok = self.login('other_user_one', 'monkey')
        self.other_user_two_id = self.register_user('other_user_two', 'monkey')
        self.other_user_two_tok = self.login('other_user_two', 'monkey')
        room_id = self.helper.create_room_as(self.other_user_one_id, tok=self.other_user_one_tok)
        self.helper.invite(room_id, self.other_user_one_id, self.other_user_two_id, tok=self.other_user_one_tok)
        self.helper.join(room_id, self.other_user_two_id, tok=self.other_user_two_tok)
        send_presence_update(self, self.other_user_one_id, self.other_user_one_tok, 'online', 'boop')
        (presence_updates, sync_token) = sync_presence(self, self.presence_receiving_user_id)
        self.assertEqual(len(presence_updates), 1)
        presence_update: UserPresenceState = presence_updates[0]
        self.assertEqual(presence_update.user_id, self.other_user_one_id)
        self.assertEqual(presence_update.state, 'online')
        self.assertEqual(presence_update.status_msg, 'boop')
        send_presence_update(self, self.other_user_one_id, self.other_user_one_tok, 'online', 'user_one')
        send_presence_update(self, self.other_user_two_id, self.other_user_two_tok, 'online', 'user_two')
        send_presence_update(self, self.presence_receiving_user_id, self.presence_receiving_user_tok, 'online', 'presence_gobbler')
        (presence_updates, _) = sync_presence(self, self.presence_receiving_user_id, sync_token)
        self.assertEqual(len(presence_updates), 3)
        (presence_updates, _) = sync_presence(self, self.other_user_one_id)
        self.assertEqual(len(presence_updates), 2)
        found = False
        for update in presence_updates:
            if update.user_id == self.other_user_two_id:
                self.assertEqual(update.state, 'online')
                self.assertEqual(update.status_msg, 'user_two')
                found = True
        self.assertTrue(found)

    @override_config({'presence': {'presence_router': {'module': __name__ + '.LegacyPresenceRouterTestModule', 'config': {'users_who_should_receive_all_presence': ['@presence_gobbler1:test', '@presence_gobbler2:test', '@far_away_person:island']}}}})
    def test_send_local_online_presence_to_with_module_legacy(self) -> None:
        if False:
            i = 10
            return i + 15
        self.send_local_online_presence_to_with_module_test_body()

    @override_config({'modules': [{'module': __name__ + '.PresenceRouterTestModule', 'config': {'users_who_should_receive_all_presence': ['@presence_gobbler1:test', '@presence_gobbler2:test', '@far_away_person:island']}}]})
    def test_send_local_online_presence_to_with_module(self) -> None:
        if False:
            print('Hello World!')
        self.send_local_online_presence_to_with_module_test_body()

    def send_local_online_presence_to_with_module_test_body(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that send_local_presence_to_users sends local online presence to a set\n        of specified local and remote users, with a custom PresenceRouter module enabled.\n        '
        self.other_user_id = self.register_user('other_user', 'monkey')
        self.other_user_tok = self.login('other_user', 'monkey')
        self.presence_receiving_user_one_id = self.register_user('presence_gobbler1', 'monkey')
        self.presence_receiving_user_one_tok = self.login('presence_gobbler1', 'monkey')
        self.presence_receiving_user_two_id = self.register_user('presence_gobbler2', 'monkey')
        self.presence_receiving_user_two_tok = self.login('presence_gobbler2', 'monkey')
        send_presence_update(self, self.other_user_id, self.other_user_tok, 'online', "I'm online!")
        send_presence_update(self, self.presence_receiving_user_one_id, self.presence_receiving_user_one_tok, 'online', "I'm also online!")
        send_presence_update(self, self.presence_receiving_user_two_id, self.presence_receiving_user_two_tok, 'unavailable', "I'm in a meeting!")
        self.get_success(self.module_api.send_local_online_presence_to([self.presence_receiving_user_one_id, self.presence_receiving_user_two_id]))
        (presence_updates, _) = sync_presence(self, self.other_user_id)
        self.assertEqual(len(presence_updates), 1)
        presence_update: UserPresenceState = presence_updates[0]
        self.assertEqual(presence_update.user_id, self.other_user_id)
        self.assertEqual(presence_update.state, 'online')
        self.assertEqual(presence_update.status_msg, "I'm online!")
        (presence_updates, _) = sync_presence(self, self.presence_receiving_user_one_id)
        self.assertEqual(len(presence_updates), 3)
        (presence_updates, _) = sync_presence(self, self.presence_receiving_user_two_id)
        self.assertEqual(len(presence_updates), 3)
        self.reactor.advance(60)
        remote_user_id = '@far_away_person:island'
        self.fed_transport_client.send_transaction.reset_mock()
        self.get_success(self.module_api.send_local_online_presence_to([remote_user_id]))
        self.reactor.advance(60)
        expected_users = {self.other_user_id, self.presence_receiving_user_one_id, self.presence_receiving_user_two_id}
        found_users = set()
        calls = self.fed_transport_client.send_transaction.call_args_list
        for call in calls:
            call_args = call[0]
            federation_transaction: Transaction = call_args[0]
            edus = federation_transaction.get_dict()['edus']
            for edu in edus:
                if edu['edu_type'] != EduTypes.PRESENCE:
                    continue
                for presence_edu in edu['content']['push']:
                    found_users.add(presence_edu['user_id'])
                    self.assertNotEqual(presence_edu['presence'], 'offline')
        self.assertEqual(found_users, expected_users)

def send_presence_update(testcase: HomeserverTestCase, user_id: str, access_token: str, presence_state: str, status_message: Optional[str]=None) -> JsonDict:
    if False:
        i = 10
        return i + 15
    body = {'presence': presence_state}
    if status_message:
        body['status_msg'] = status_message
    channel = testcase.make_request('PUT', '/presence/%s/status' % (user_id,), body, access_token=access_token)
    testcase.assertEqual(channel.code, 200)
    return channel.json_body

def sync_presence(testcase: HomeserverTestCase, user_id: str, since_token: Optional[StreamToken]=None) -> Tuple[List[UserPresenceState], StreamToken]:
    if False:
        return 10
    "Perform a sync request for the given user and return the user presence updates\n    they've received, as well as the next_batch token.\n\n    This method assumes testcase.sync_handler points to the homeserver's sync handler.\n\n    Args:\n        testcase: The testcase that is currently being run.\n        user_id: The ID of the user to generate a sync response for.\n        since_token: An optional token to indicate from at what point to sync from.\n\n    Returns:\n        A tuple containing a list of presence updates, and the sync response's\n        next_batch token.\n    "
    requester = create_requester(user_id)
    sync_config = generate_sync_config(requester.user.to_string())
    sync_result = testcase.get_success(testcase.hs.get_sync_handler().wait_for_sync_for_user(requester, sync_config, since_token))
    return (sync_result.presence, sync_result.next_batch)