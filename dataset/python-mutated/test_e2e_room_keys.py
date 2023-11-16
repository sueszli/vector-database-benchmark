from twisted.test.proto_helpers import MemoryReactor
from synapse.server import HomeServer
from synapse.storage.databases.main.e2e_room_keys import RoomKey
from synapse.util import Clock
from tests import unittest
room_key: RoomKey = {'first_message_index': 1, 'forwarded_count': 1, 'is_verified': False, 'session_data': 'SSBBTSBBIEZJU0gK'}

class E2eRoomKeysHandlerTestCase(unittest.HomeserverTestCase):

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            for i in range(10):
                print('nop')
        hs = self.setup_test_homeserver('server')
        self.store = hs.get_datastores().main
        return hs

    def test_room_keys_version_delete(self) -> None:
        if False:
            i = 10
            return i + 15
        version1 = self.get_success(self.store.create_e2e_room_keys_version('user_id', {'algorithm': 'rot13', 'auth_data': {}}))
        self.get_success(self.store.add_e2e_room_keys('user_id', version1, [('room', 'session', room_key)]))
        version2 = self.get_success(self.store.create_e2e_room_keys_version('user_id', {'algorithm': 'rot13', 'auth_data': {}}))
        self.get_success(self.store.add_e2e_room_keys('user_id', version2, [('room', 'session', room_key)]))
        keys = self.get_success(self.store.get_e2e_room_keys('user_id', version1))
        self.assertEqual(len(keys['rooms']), 1)
        keys = self.get_success(self.store.get_e2e_room_keys('user_id', version2))
        self.assertEqual(len(keys['rooms']), 1)
        self.get_success(self.store.delete_e2e_room_keys_version('user_id', version1))
        keys = self.get_success(self.store.get_e2e_room_keys('user_id', version1))
        self.assertEqual(len(keys['rooms']), 0)
        keys = self.get_success(self.store.get_e2e_room_keys('user_id', version2))
        self.assertEqual(len(keys['rooms']), 1)