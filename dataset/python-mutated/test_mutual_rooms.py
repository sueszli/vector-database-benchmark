from urllib.parse import quote
from twisted.test.proto_helpers import MemoryReactor
import synapse.rest.admin
from synapse.rest.client import login, mutual_rooms, room
from synapse.server import HomeServer
from synapse.util import Clock
from tests import unittest
from tests.server import FakeChannel

class UserMutualRoomsTest(unittest.HomeserverTestCase):
    """
    Tests the UserMutualRoomsServlet.
    """
    servlets = [login.register_servlets, synapse.rest.admin.register_servlets_for_client_rest_resource, room.register_servlets, mutual_rooms.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            for i in range(10):
                print('nop')
        config = self.default_config()
        return self.setup_test_homeserver(config=config)

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            i = 10
            return i + 15
        self.store = hs.get_datastores().main

    def _get_mutual_rooms(self, token: str, other_user: str) -> FakeChannel:
        if False:
            i = 10
            return i + 15
        return self.make_request('GET', f'/_matrix/client/unstable/uk.half-shot.msc2666/user/mutual_rooms?user_id={quote(other_user)}', access_token=token)

    def test_shared_room_list_public(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        A room should show up in the shared list of rooms between two users\n        if it is public.\n        '
        self._check_mutual_rooms_with(room_one_is_public=True, room_two_is_public=True)

    def test_shared_room_list_private(self) -> None:
        if False:
            print('Hello World!')
        '\n        A room should show up in the shared list of rooms between two users\n        if it is private.\n        '
        self._check_mutual_rooms_with(room_one_is_public=False, room_two_is_public=False)

    def test_shared_room_list_mixed(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        The shared room list between two users should contain both public and private\n        rooms.\n        '
        self._check_mutual_rooms_with(room_one_is_public=True, room_two_is_public=False)

    def _check_mutual_rooms_with(self, room_one_is_public: bool, room_two_is_public: bool) -> None:
        if False:
            while True:
                i = 10
        'Checks that shared public or private rooms between two users appear in\n        their shared room lists\n        '
        u1 = self.register_user('user1', 'pass')
        u1_token = self.login(u1, 'pass')
        u2 = self.register_user('user2', 'pass')
        u2_token = self.login(u2, 'pass')
        room_id_one = self.helper.create_room_as(u1, is_public=room_one_is_public, tok=u1_token)
        self.helper.invite(room_id_one, src=u1, targ=u2, tok=u1_token)
        self.helper.join(room_id_one, user=u2, tok=u2_token)
        channel = self._get_mutual_rooms(u1_token, u2)
        self.assertEqual(200, channel.code, channel.result)
        self.assertEqual(len(channel.json_body['joined']), 1)
        self.assertEqual(channel.json_body['joined'][0], room_id_one)
        room_id_two = self.helper.create_room_as(u1, is_public=room_two_is_public, tok=u1_token)
        self.helper.invite(room_id_two, src=u1, targ=u2, tok=u1_token)
        self.helper.join(room_id_two, user=u2, tok=u2_token)
        channel = self._get_mutual_rooms(u1_token, u2)
        self.assertEqual(200, channel.code, channel.result)
        self.assertEqual(len(channel.json_body['joined']), 2)
        for room_id_id in channel.json_body['joined']:
            self.assertIn(room_id_id, [room_id_one, room_id_two])

    def test_shared_room_list_after_leave(self) -> None:
        if False:
            print('Hello World!')
        '\n        A room should no longer be considered shared if the other\n        user has left it.\n        '
        u1 = self.register_user('user1', 'pass')
        u1_token = self.login(u1, 'pass')
        u2 = self.register_user('user2', 'pass')
        u2_token = self.login(u2, 'pass')
        room = self.helper.create_room_as(u1, is_public=True, tok=u1_token)
        self.helper.invite(room, src=u1, targ=u2, tok=u1_token)
        self.helper.join(room, user=u2, tok=u2_token)
        channel = self._get_mutual_rooms(u1_token, u2)
        self.assertEqual(200, channel.code, channel.result)
        self.assertEqual(len(channel.json_body['joined']), 1)
        self.assertEqual(channel.json_body['joined'][0], room)
        self.helper.leave(room, user=u1, tok=u1_token)
        channel = self._get_mutual_rooms(u1_token, u2)
        self.assertEqual(200, channel.code, channel.result)
        self.assertEqual(len(channel.json_body['joined']), 0)
        channel = self._get_mutual_rooms(u2_token, u1)
        self.assertEqual(200, channel.code, channel.result)
        self.assertEqual(len(channel.json_body['joined']), 0)