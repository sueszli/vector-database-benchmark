from twisted.test.proto_helpers import MemoryReactor
from synapse.rest import admin
from synapse.rest.client import devices
from synapse.server import HomeServer
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class DeviceInboxBackgroundUpdateStoreTestCase(HomeserverTestCase):
    servlets = [admin.register_servlets, devices.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.store = hs.get_datastores().main
        self.user_id = self.register_user('foo', 'pass')

    def test_background_remove_deleted_devices_from_device_inbox(self) -> None:
        if False:
            while True:
                i = 10
        'Test that the background task to delete old device_inboxes works properly.'
        self.get_success(self.store.store_device(self.user_id, 'cur_device', 'display_name'))
        self.get_success(self.store.db_pool.simple_insert('device_inbox', {'user_id': self.user_id, 'device_id': 'cur_device', 'stream_id': 1, 'message_json': '{}'}))
        self.get_success(self.store.db_pool.simple_insert('device_inbox', {'user_id': self.user_id, 'device_id': 'old_device', 'stream_id': 2, 'message_json': '{}'}))
        self.get_success(self.store.db_pool.simple_insert('background_updates', {'update_name': 'remove_dead_devices_from_device_inbox', 'progress_json': '{}'}))
        self.store.db_pool.updates._all_done = False
        self.wait_for_background_updates()
        res = self.get_success(self.store.db_pool.simple_select_onecol(table='device_inbox', keyvalues={}, retcol='device_id', desc='get_device_id_from_device_inbox'))
        self.assertEqual(1, len(res))
        self.assertEqual(res[0], 'cur_device')

    def test_background_remove_hidden_devices_from_device_inbox(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that the background task to delete hidden devices\n        from device_inboxes works properly.'
        self.get_success(self.store.store_device(self.user_id, 'cur_device', 'display_name'))
        self.get_success(self.store.db_pool.simple_insert('devices', values={'user_id': self.user_id, 'device_id': 'hidden_device', 'display_name': 'hidden_display_name', 'hidden': True}))
        self.get_success(self.store.db_pool.simple_insert('device_inbox', {'user_id': self.user_id, 'device_id': 'cur_device', 'stream_id': 1, 'message_json': '{}'}))
        self.get_success(self.store.db_pool.simple_insert('device_inbox', {'user_id': self.user_id, 'device_id': 'hidden_device', 'stream_id': 2, 'message_json': '{}'}))
        self.get_success(self.store.db_pool.simple_insert('background_updates', {'update_name': 'remove_dead_devices_from_device_inbox', 'progress_json': '{}'}))
        self.store.db_pool.updates._all_done = False
        self.wait_for_background_updates()
        res = self.get_success(self.store.db_pool.simple_select_onecol(table='device_inbox', keyvalues={}, retcol='device_id', desc='get_device_id_from_device_inbox'))
        self.assertEqual(1, len(res))
        self.assertEqual(res[0], 'cur_device')