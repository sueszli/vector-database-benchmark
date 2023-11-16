from typing import Collection, List, Tuple
from twisted.test.proto_helpers import MemoryReactor
import synapse.api.errors
from synapse.api.constants import EduTypes
from synapse.server import HomeServer
from synapse.types import JsonDict
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class DeviceStoreTestCase(HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            while True:
                i = 10
        self.store = hs.get_datastores().main

    def add_device_change(self, user_id: str, device_ids: List[str], host: str) -> None:
        if False:
            i = 10
            return i + 15
        'Add a device list change for the given device to\n        `device_lists_outbound_pokes` table.\n        '
        for device_id in device_ids:
            self.get_success(self.store.add_device_change_to_streams(user_id, [device_id], ['!some:room']))
            self.get_success(self.store.add_device_list_outbound_pokes(user_id=user_id, device_id=device_id, room_id='!some:room', hosts=[host], context={}))

    def test_store_new_device(self) -> None:
        if False:
            i = 10
            return i + 15
        self.get_success(self.store.store_device('user_id', 'device_id', 'display_name'))
        res = self.get_success(self.store.get_device('user_id', 'device_id'))
        assert res is not None
        self.assertLessEqual({'user_id': 'user_id', 'device_id': 'device_id', 'display_name': 'display_name'}.items(), res.items())

    def test_get_devices_by_user(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.get_success(self.store.store_device('user_id', 'device1', 'display_name 1'))
        self.get_success(self.store.store_device('user_id', 'device2', 'display_name 2'))
        self.get_success(self.store.store_device('user_id2', 'device3', 'display_name 3'))
        res = self.get_success(self.store.get_devices_by_user('user_id'))
        self.assertEqual(2, len(res.keys()))
        self.assertLessEqual({'user_id': 'user_id', 'device_id': 'device1', 'display_name': 'display_name 1'}.items(), res['device1'].items())
        self.assertLessEqual({'user_id': 'user_id', 'device_id': 'device2', 'display_name': 'display_name 2'}.items(), res['device2'].items())

    def test_count_devices_by_users(self) -> None:
        if False:
            while True:
                i = 10
        self.get_success(self.store.store_device('user_id', 'device1', 'display_name 1'))
        self.get_success(self.store.store_device('user_id', 'device2', 'display_name 2'))
        self.get_success(self.store.store_device('user_id2', 'device3', 'display_name 3'))
        res = self.get_success(self.store.count_devices_by_users())
        self.assertEqual(0, res)
        res = self.get_success(self.store.count_devices_by_users(['unknown']))
        self.assertEqual(0, res)
        res = self.get_success(self.store.count_devices_by_users(['user_id']))
        self.assertEqual(2, res)
        res = self.get_success(self.store.count_devices_by_users(['user_id', 'user_id2']))
        self.assertEqual(3, res)

    def test_get_device_updates_by_remote(self) -> None:
        if False:
            return 10
        device_ids = ['device_id1', 'device_id2']
        self.add_device_change('@user_id:test', device_ids, 'somehost')
        (now_stream_id, device_updates) = self.get_success(self.store.get_device_updates_by_remote('somehost', -1, limit=100))
        self._check_devices_in_updates(device_ids, device_updates)

    def test_get_device_updates_by_remote_can_limit_properly(self) -> None:
        if False:
            print('Hello World!')
        '\n        Tests that `get_device_updates_by_remote` returns an appropriate\n        stream_id to resume fetching from (without skipping any results).\n        '
        device_ids = ['device_id1', 'device_id2', 'device_id3', 'device_id4', 'device_id5']
        self.add_device_change('@user_id:test', device_ids, 'somehost')
        (next_stream_id, device_updates) = self.get_success(self.store.get_device_updates_by_remote('somehost', -1, limit=3))
        self._check_devices_in_updates(device_ids[:3], device_updates)
        (next_stream_id, device_updates) = self.get_success(self.store.get_device_updates_by_remote('somehost', next_stream_id, limit=3))
        self._check_devices_in_updates(device_ids[3:], device_updates)
        device_ids = ['device_id6', 'device_id7']
        self.add_device_change('@user_id:test', device_ids, 'somehost')
        (next_stream_id, device_updates) = self.get_success(self.store.get_device_updates_by_remote('somehost', next_stream_id, limit=3))
        self._check_devices_in_updates(device_ids, device_updates)
        (_, device_updates) = self.get_success(self.store.get_device_updates_by_remote('somehost', next_stream_id, limit=3))
        self.assertEqual(device_updates, [])

    def test_get_device_updates_by_remote_cross_signing_key_updates(self) -> None:
        if False:
            print('Hello World!')
        '\n        Tests that `get_device_updates_by_remote` limits the length of the return value\n        properly when cross-signing key updates are present.\n        Current behaviour is that the cross-signing key updates will always come in pairs,\n        even if that means leaving an earlier batch one EDU short of the limit.\n        '
        assert self.hs.is_mine_id('@user_id:test'), 'Test not valid: this MXID should be considered local'
        self.get_success(self.store.set_e2e_cross_signing_key('@user_id:test', 'master', {'keys': {'ed25519:fakeMaster': 'aaafakefakefake1AAAAAAAAAAAAAAAAAAAAAAAAAAA='}, 'signatures': {'@user_id:test': {'ed25519:fake2': 'aaafakefakefake2AAAAAAAAAAAAAAAAAAAAAAAAAAA='}}}))
        self.get_success(self.store.set_e2e_cross_signing_key('@user_id:test', 'self_signing', {'keys': {'ed25519:fakeSelfSigning': 'aaafakefakefake3AAAAAAAAAAAAAAAAAAAAAAAAAAA='}, 'signatures': {'@user_id:test': {'ed25519:fake4': 'aaafakefakefake4AAAAAAAAAAAAAAAAAAAAAAAAAAA='}}}))
        device_ids = ['device_id1', 'device_id2', 'fakeMaster', 'fakeSelfSigning']
        self.add_device_change('@user_id:test', device_ids, 'somehost')
        (next_stream_id, device_updates) = self.get_success(self.store.get_device_updates_by_remote('somehost', -1, limit=3))
        self.assertEqual(len(device_updates), 2, device_updates)
        self._check_devices_in_updates(device_ids[:2], device_updates)
        (next_stream_id, device_updates) = self.get_success(self.store.get_device_updates_by_remote('somehost', next_stream_id, limit=3))
        self.assertEqual(len(device_updates), 2, device_updates)
        self.assertEqual(device_updates[0][0], EduTypes.SIGNING_KEY_UPDATE, device_updates[0])
        self.assertEqual(device_updates[1][0], EduTypes.UNSTABLE_SIGNING_KEY_UPDATE, device_updates[1])
        (_, device_updates) = self.get_success(self.store.get_device_updates_by_remote('somehost', next_stream_id, limit=3))
        self.assertEqual(device_updates, [])

    def _check_devices_in_updates(self, expected_device_ids: Collection[str], device_updates: List[Tuple[str, JsonDict]]) -> None:
        if False:
            while True:
                i = 10
        'Check that an specific device ids exist in a list of device update EDUs'
        self.assertEqual(len(device_updates), len(expected_device_ids))
        received_device_ids = {update['device_id'] for (edu_type, update) in device_updates}
        self.assertEqual(received_device_ids, set(expected_device_ids))

    def test_update_device(self) -> None:
        if False:
            print('Hello World!')
        self.get_success(self.store.store_device('user_id', 'device_id', 'display_name 1'))
        res = self.get_success(self.store.get_device('user_id', 'device_id'))
        assert res is not None
        self.assertEqual('display_name 1', res['display_name'])
        self.get_success(self.store.update_device('user_id', 'device_id'))
        res = self.get_success(self.store.get_device('user_id', 'device_id'))
        assert res is not None
        self.assertEqual('display_name 1', res['display_name'])
        self.get_success(self.store.update_device('user_id', 'device_id', new_display_name='display_name 2'))
        res = self.get_success(self.store.get_device('user_id', 'device_id'))
        assert res is not None
        self.assertEqual('display_name 2', res['display_name'])

    def test_update_unknown_device(self) -> None:
        if False:
            return 10
        exc = self.get_failure(self.store.update_device('user_id', 'unknown_device_id', new_display_name='display_name 2'), synapse.api.errors.StoreError)
        self.assertEqual(404, exc.value.code)