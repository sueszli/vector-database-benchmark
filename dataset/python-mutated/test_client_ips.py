from typing import Any, Dict, List, Optional, Tuple, cast
from unittest.mock import AsyncMock
from parameterized import parameterized
from twisted.test.proto_helpers import MemoryReactor
import synapse.rest.admin
from synapse.http.site import XForwardedForRequest
from synapse.rest.client import login
from synapse.server import HomeServer
from synapse.storage.databases.main.client_ips import LAST_SEEN_GRANULARITY, DeviceLastConnectionInfo
from synapse.types import UserID
from synapse.util import Clock
from tests import unittest
from tests.server import make_request
from tests.unittest import override_config

class ClientIpStoreTestCase(unittest.HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            while True:
                i = 10
        self.store = hs.get_datastores().main

    def test_insert_new_client_ip(self) -> None:
        if False:
            print('Hello World!')
        self.reactor.advance(12345678)
        user_id = '@user:id'
        device_id = 'MY_DEVICE'
        self.get_success(self.store.store_device(user_id, device_id, 'display name'))
        self.get_success(self.store.insert_client_ip(user_id, 'access_token', 'ip', 'user_agent', device_id))
        self.reactor.advance(10)
        result = self.get_success(self.store.get_last_client_ip_by_device(user_id, device_id))
        r = result[user_id, device_id]
        self.assertEqual(DeviceLastConnectionInfo(user_id=user_id, device_id=device_id, ip='ip', user_agent='user_agent', last_seen=12345678000), r)

    def test_insert_new_client_ip_none_device_id(self) -> None:
        if False:
            return 10
        '\n        An insert with a device ID of NULL will not create a new entry, but\n        update an existing entry in the user_ips table.\n        '
        self.reactor.advance(12345678)
        user_id = '@user:id'
        self.get_success(self.store.insert_client_ip(user_id, 'access_token', 'ip', 'user_agent', None))
        self.reactor.advance(200)
        self.pump(0)
        result = cast(List[Tuple[str, str, str, Optional[str], int]], self.get_success(self.store.db_pool.simple_select_list(table='user_ips', keyvalues={'user_id': user_id}, retcols=['access_token', 'ip', 'user_agent', 'device_id', 'last_seen'], desc='get_user_ip_and_agents')))
        self.assertEqual(result, [('access_token', 'ip', 'user_agent', None, 12345678000)])
        self.get_success(self.store.insert_client_ip(user_id, 'access_token', 'ip', 'user_agent', None))
        self.reactor.advance(10)
        self.pump(0)
        result = cast(List[Tuple[str, str, str, Optional[str], int]], self.get_success(self.store.db_pool.simple_select_list(table='user_ips', keyvalues={'user_id': user_id}, retcols=['access_token', 'ip', 'user_agent', 'device_id', 'last_seen'], desc='get_user_ip_and_agents')))
        self.assertEqual(result, [('access_token', 'ip', 'user_agent', None, 12345878000)])

    @parameterized.expand([(False,), (True,)])
    def test_get_last_client_ip_by_device(self, after_persisting: bool) -> None:
        if False:
            return 10
        'Test `get_last_client_ip_by_device` for persisted and unpersisted data'
        self.reactor.advance(12345678)
        user_id = '@user:id'
        device_id = 'MY_DEVICE'
        self.get_success(self.store.store_device(user_id, device_id, 'display name'))
        self.get_success(self.store.insert_client_ip(user_id, 'access_token', 'ip', 'user_agent', device_id))
        if after_persisting:
            self.reactor.advance(10)
        else:
            db_result = cast(List[Tuple[str, Optional[str], Optional[str], str, Optional[int]]], self.get_success(self.store.db_pool.simple_select_list(table='devices', keyvalues={}, retcols=('user_id', 'ip', 'user_agent', 'device_id', 'last_seen'))))
            self.assertEqual(db_result, [(user_id, None, None, device_id, None)])
        result = self.get_success(self.store.get_last_client_ip_by_device(user_id, device_id))
        self.assertEqual(result, {(user_id, device_id): DeviceLastConnectionInfo(user_id=user_id, device_id=device_id, ip='ip', user_agent='user_agent', last_seen=12345678000)})

    def test_get_last_client_ip_by_device_combined_data(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that `get_last_client_ip_by_device` combines persisted and unpersisted\n        data together correctly\n        '
        self.reactor.advance(12345678)
        user_id = '@user:id'
        device_id_1 = 'MY_DEVICE_1'
        device_id_2 = 'MY_DEVICE_2'
        self.get_success(self.store.store_device(user_id, device_id_1, 'display name'))
        self.get_success(self.store.store_device(user_id, device_id_2, 'display name'))
        self.get_success(self.store.insert_client_ip(user_id, 'access_token_1', 'ip_1', 'user_agent_1', device_id_1))
        self.get_success(self.store.insert_client_ip(user_id, 'access_token_2', 'ip_2', 'user_agent_2', device_id_2))
        self.reactor.advance(10 + LAST_SEEN_GRANULARITY / 1000)
        self.get_success(self.store.insert_client_ip(user_id, 'access_token_2', 'ip_2', 'user_agent_3', device_id_2))
        db_result = cast(List[Tuple[str, Optional[str], Optional[str], str, Optional[int]]], self.get_success(self.store.db_pool.simple_select_list(table='devices', keyvalues={}, retcols=('user_id', 'ip', 'user_agent', 'device_id', 'last_seen'))))
        self.assertCountEqual(db_result, [(user_id, 'ip_1', 'user_agent_1', device_id_1, 12345678000), (user_id, 'ip_2', 'user_agent_2', device_id_2, 12345678000)])
        result = self.get_success(self.store.get_last_client_ip_by_device(user_id, None))
        self.assertEqual(result, {(user_id, device_id_1): DeviceLastConnectionInfo(user_id=user_id, device_id=device_id_1, ip='ip_1', user_agent='user_agent_1', last_seen=12345678000), (user_id, device_id_2): DeviceLastConnectionInfo(user_id=user_id, device_id=device_id_2, ip='ip_2', user_agent='user_agent_3', last_seen=12345688000 + LAST_SEEN_GRANULARITY)})

    @parameterized.expand([(False,), (True,)])
    def test_get_user_ip_and_agents(self, after_persisting: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test `get_user_ip_and_agents` for persisted and unpersisted data'
        self.reactor.advance(12345678)
        user_id = '@user:id'
        user = UserID.from_string(user_id)
        self.get_success(self.store.insert_client_ip(user_id, 'access_token', 'ip', 'user_agent', 'MY_DEVICE'))
        if after_persisting:
            self.reactor.advance(10)
        else:
            db_result = self.get_success(self.store.db_pool.simple_select_list(table='user_ips', keyvalues={}, retcols=('access_token', 'ip', 'user_agent', 'last_seen')))
            self.assertEqual(db_result, [])
        self.assertEqual(self.get_success(self.store.get_user_ip_and_agents(user)), [{'access_token': 'access_token', 'ip': 'ip', 'user_agent': 'user_agent', 'last_seen': 12345678000}])

    def test_get_user_ip_and_agents_combined_data(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that `get_user_ip_and_agents` combines persisted and unpersisted data\n        together correctly\n        '
        self.reactor.advance(12345678)
        user_id = '@user:id'
        user = UserID.from_string(user_id)
        self.get_success(self.store.insert_client_ip(user_id, 'access_token', 'ip_1', 'user_agent_1', 'MY_DEVICE_1'))
        self.get_success(self.store.insert_client_ip(user_id, 'access_token', 'ip_2', 'user_agent_2', 'MY_DEVICE_2'))
        self.reactor.advance(10 + LAST_SEEN_GRANULARITY / 1000)
        self.get_success(self.store.insert_client_ip(user_id, 'access_token', 'ip_2', 'user_agent_3', 'MY_DEVICE_2'))
        db_result = cast(List[Tuple[str, str, str, int]], self.get_success(self.store.db_pool.simple_select_list(table='user_ips', keyvalues={}, retcols=('access_token', 'ip', 'user_agent', 'last_seen'))))
        self.assertEqual(db_result, [('access_token', 'ip_1', 'user_agent_1', 12345678000), ('access_token', 'ip_2', 'user_agent_2', 12345678000)])
        self.assertCountEqual(self.get_success(self.store.get_user_ip_and_agents(user)), [{'access_token': 'access_token', 'ip': 'ip_1', 'user_agent': 'user_agent_1', 'last_seen': 12345678000}, {'access_token': 'access_token', 'ip': 'ip_2', 'user_agent': 'user_agent_3', 'last_seen': 12345688000 + LAST_SEEN_GRANULARITY}])

    @override_config({'limit_usage_by_mau': False, 'max_mau_value': 50})
    def test_disabled_monthly_active_user(self) -> None:
        if False:
            return 10
        user_id = '@user:server'
        self.get_success(self.store.insert_client_ip(user_id, 'access_token', 'ip', 'user_agent', 'device_id'))
        active = self.get_success(self.store.user_last_seen_monthly_active(user_id))
        self.assertFalse(active)

    @override_config({'limit_usage_by_mau': True, 'max_mau_value': 50})
    def test_adding_monthly_active_user_when_full(self) -> None:
        if False:
            print('Hello World!')
        lots_of_users = 100
        user_id = '@user:server'
        self.store.get_monthly_active_count = AsyncMock(return_value=lots_of_users)
        self.get_success(self.store.insert_client_ip(user_id, 'access_token', 'ip', 'user_agent', 'device_id'))
        active = self.get_success(self.store.user_last_seen_monthly_active(user_id))
        self.assertFalse(active)

    @override_config({'limit_usage_by_mau': True, 'max_mau_value': 50})
    def test_adding_monthly_active_user_when_space(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        user_id = '@user:server'
        active = self.get_success(self.store.user_last_seen_monthly_active(user_id))
        self.assertFalse(active)
        self.reactor.advance(10)
        self.get_success(self.store.insert_client_ip(user_id, 'access_token', 'ip', 'user_agent', 'device_id'))
        active = self.get_success(self.store.user_last_seen_monthly_active(user_id))
        self.assertTrue(active)

    @override_config({'limit_usage_by_mau': True, 'max_mau_value': 50})
    def test_updating_monthly_active_user_when_space(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        user_id = '@user:server'
        self.get_success(self.store.register_user(user_id=user_id, password_hash=None))
        active = self.get_success(self.store.user_last_seen_monthly_active(user_id))
        self.assertFalse(active)
        self.reactor.advance(10)
        self.get_success(self.store.insert_client_ip(user_id, 'access_token', 'ip', 'user_agent', 'device_id'))
        active = self.get_success(self.store.user_last_seen_monthly_active(user_id))
        self.assertTrue(active)

    def test_devices_last_seen_bg_update(self) -> None:
        if False:
            return 10
        self.wait_for_background_updates()
        user_id = '@user:id'
        device_id = 'MY_DEVICE'
        self.get_success(self.store.store_device(user_id, device_id, 'display name'))
        self.get_success(self.store.insert_client_ip(user_id, 'access_token', 'ip', 'user_agent', device_id))
        self.reactor.advance(200)
        self.get_success(self.store.db_pool.simple_update(table='devices', keyvalues={'user_id': user_id, 'device_id': device_id}, updatevalues={'last_seen': None, 'ip': None, 'user_agent': None}, desc='test_devices_last_seen_bg_update'))
        result = self.get_success(self.store.get_last_client_ip_by_device(user_id, device_id))
        r = result[user_id, device_id]
        self.assertEqual(DeviceLastConnectionInfo(user_id=user_id, device_id=device_id, ip=None, user_agent=None, last_seen=None), r)
        self.get_success(self.store.db_pool.simple_insert(table='background_updates', values={'update_name': 'devices_last_seen', 'progress_json': '{}', 'depends_on': None}))
        self.store.db_pool.updates._all_done = False
        self.wait_for_background_updates()
        result = self.get_success(self.store.get_last_client_ip_by_device(user_id, device_id))
        r = result[user_id, device_id]
        self.assertEqual(DeviceLastConnectionInfo(user_id=user_id, device_id=device_id, ip='ip', user_agent='user_agent', last_seen=0), r)

    def test_old_user_ips_pruned(self) -> None:
        if False:
            i = 10
            return i + 15
        self.wait_for_background_updates()
        user_id = '@user:id'
        device_id = 'MY_DEVICE'
        self.get_success(self.store.store_device(user_id, device_id, 'display name'))
        self.get_success(self.store.insert_client_ip(user_id, 'access_token', 'ip', 'user_agent', device_id))
        self.reactor.advance(200)
        result = cast(List[Tuple[str, str, str, Optional[str], int]], self.get_success(self.store.db_pool.simple_select_list(table='user_ips', keyvalues={'user_id': user_id}, retcols=['access_token', 'ip', 'user_agent', 'device_id', 'last_seen'], desc='get_user_ip_and_agents')))
        self.assertEqual(result, [('access_token', 'ip', 'user_agent', device_id, 0)])
        self.reactor.advance(60 * 24 * 60 * 60)
        result = cast(List[Tuple[str, str, str, Optional[str], int]], self.get_success(self.store.db_pool.simple_select_list(table='user_ips', keyvalues={'user_id': user_id}, retcols=['access_token', 'ip', 'user_agent', 'device_id', 'last_seen'], desc='get_user_ip_and_agents')))
        self.assertEqual(result, [])
        result2 = self.get_success(self.store.get_last_client_ip_by_device(user_id, device_id))
        r = result2[user_id, device_id]
        self.assertEqual(DeviceLastConnectionInfo(user_id=user_id, device_id=device_id, ip='ip', user_agent='user_agent', last_seen=0), r)

    def test_invalid_user_agents_are_ignored(self) -> None:
        if False:
            return 10
        self.wait_for_background_updates()
        user_id1 = '@user1:id'
        user_id2 = '@user2:id'
        device_id1 = 'MY_DEVICE1'
        device_id2 = 'MY_DEVICE2'
        access_token1 = 'access_token1'
        access_token2 = 'access_token2'
        self.get_success(self.store.store_device(user_id1, device_id1, 'display name1'))
        self.get_success(self.store.store_device(user_id2, device_id2, 'display name2'))
        self.get_success(self.store.insert_client_ip(user_id1, access_token1, 'ip', 'sync-v3-proxy-', device_id1))
        self.get_success(self.store.insert_client_ip(user_id2, access_token2, 'ip', 'user_agent', device_id2))
        self.reactor.advance(200)
        result = cast(List[Tuple[str, str, str, Optional[str], int]], self.get_success(self.store.db_pool.simple_select_list(table='user_ips', keyvalues={}, retcols=['access_token', 'ip', 'user_agent', 'device_id', 'last_seen'], desc='get_user_ip_and_agents')))
        self.assertEqual(result, [(access_token2, 'ip', 'user_agent', device_id2, 0)])

class ClientIpAuthTestCase(unittest.HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.store = self.hs.get_datastores().main
        self.user_id = self.register_user('bob', 'abc123', True)

    def test_request_with_xforwarded(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        The IP in X-Forwarded-For is entered into the client IPs table.\n        '
        self._runtest({b'X-Forwarded-For': b'127.9.0.1'}, '127.9.0.1', {'request': XForwardedForRequest})

    def test_request_from_getPeer(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        The IP returned by getPeer is entered into the client IPs table, if\n        there's no X-Forwarded-For header.\n        "
        self._runtest({}, '127.0.0.1', {})

    def _runtest(self, headers: Dict[bytes, bytes], expected_ip: str, make_request_args: Dict[str, Any]) -> None:
        if False:
            return 10
        device_id = 'bleb'
        access_token = self.login('bob', 'abc123', device_id=device_id)
        self.reactor.advance(123456 - self.reactor.seconds())
        headers1 = {b'User-Agent': b'Mozzila pizza'}
        headers1.update(headers)
        make_request(self.reactor, self.site, 'GET', '/_synapse/admin/v2/users/' + self.user_id, access_token=access_token, custom_headers=headers1.items(), **make_request_args)
        self.reactor.advance(100)
        result = self.get_success(self.store.get_last_client_ip_by_device(self.user_id, device_id))
        r = result[self.user_id, device_id]
        self.assertEqual(DeviceLastConnectionInfo(user_id=self.user_id, device_id=device_id, ip=expected_ip, user_agent='Mozzila pizza', last_seen=123456100), r)