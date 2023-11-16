"""Tests REST events for /rooms paths."""
from typing import List, Optional
from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import APP_SERVICE_REGISTRATION_TYPE, LoginType
from synapse.api.errors import Codes, HttpResponseException, SynapseError
from synapse.appservice import ApplicationService
from synapse.rest.client import register, sync
from synapse.server import HomeServer
from synapse.types import JsonDict
from synapse.util import Clock
from tests import unittest
from tests.unittest import override_config
from tests.utils import default_config

class TestMauLimit(unittest.HomeserverTestCase):
    servlets = [register.register_servlets, sync.register_servlets]

    def default_config(self) -> JsonDict:
        if False:
            i = 10
            return i + 15
        config = default_config('test')
        config.update({'registrations_require_3pid': [], 'limit_usage_by_mau': True, 'max_mau_value': 2, 'mau_trial_days': 0, 'server_notices': {'system_mxid_localpart': 'server', 'room_name': 'Test Server Notice Room'}})
        if self._extra_config is not None:
            config.update(self._extra_config)
        return config

    def prepare(self, reactor: MemoryReactor, clock: Clock, homeserver: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.store = homeserver.get_datastores().main

    def test_simple_deny_mau(self) -> None:
        if False:
            while True:
                i = 10
        token1 = self.create_user('kermit1')
        self.do_sync_for_user(token1)
        token2 = self.create_user('kermit2')
        self.do_sync_for_user(token2)
        self.assertEqual(self.get_success(self.store.get_monthly_active_count()), 2)
        with self.assertRaises(SynapseError) as cm:
            self.create_user('kermit3')
        e = cm.exception
        self.assertEqual(e.code, 403)
        self.assertEqual(e.errcode, Codes.RESOURCE_LIMIT_EXCEEDED)

    def test_as_ignores_mau(self) -> None:
        if False:
            while True:
                i = 10
        'Test that application services can still create users when the MAU\n        limit has been reached. This only works when application service\n        user ip tracking is disabled.\n        '
        token1 = self.create_user('kermit1')
        self.do_sync_for_user(token1)
        token2 = self.create_user('kermit2')
        self.do_sync_for_user(token2)
        self.assertEqual(self.get_success(self.store.get_monthly_active_count()), 2)
        with self.assertRaises(SynapseError) as cm:
            self.create_user('kermit3')
        e = cm.exception
        self.assertEqual(e.code, 403)
        self.assertEqual(e.errcode, Codes.RESOURCE_LIMIT_EXCEEDED)
        as_token = 'foobartoken'
        self.store.services_cache.append(ApplicationService(token=as_token, id='SomeASID', sender='@as_sender:test', namespaces={'users': [{'regex': '@as_*', 'exclusive': True}]}))
        self.create_user('as_kermit4', token=as_token, appservice=True)

    def test_allowed_after_a_month_mau(self) -> None:
        if False:
            i = 10
            return i + 15
        token1 = self.create_user('kermit1')
        self.do_sync_for_user(token1)
        token2 = self.create_user('kermit2')
        self.do_sync_for_user(token2)
        self.reactor.advance(31 * 24 * 60 * 60)
        self.get_success(self.store.reap_monthly_active_users())
        self.reactor.advance(0)
        token3 = self.create_user('kermit3')
        self.do_sync_for_user(token3)

    @override_config({'mau_trial_days': 1})
    def test_trial_delay(self) -> None:
        if False:
            while True:
                i = 10
        token1 = self.create_user('kermit1')
        self.do_sync_for_user(token1)
        token2 = self.create_user('kermit2')
        self.do_sync_for_user(token2)
        token3 = self.create_user('kermit3')
        self.do_sync_for_user(token3)
        self.reactor.advance(2 * 24 * 60 * 60)
        self.do_sync_for_user(token1)
        self.do_sync_for_user(token2)
        with self.assertRaises(SynapseError) as cm:
            self.do_sync_for_user(token3)
        e = cm.exception
        self.assertEqual(e.code, 403)
        self.assertEqual(e.errcode, Codes.RESOURCE_LIMIT_EXCEEDED)
        with self.assertRaises(SynapseError) as cm:
            self.create_user('kermit4')
        e = cm.exception
        self.assertEqual(e.code, 403)
        self.assertEqual(e.errcode, Codes.RESOURCE_LIMIT_EXCEEDED)

    @override_config({'mau_trial_days': 1})
    def test_trial_users_cant_come_back(self) -> None:
        if False:
            i = 10
            return i + 15
        self.hs.config.server.mau_trial_days = 1
        token1 = self.create_user('kermit1')
        self.do_sync_for_user(token1)
        token2 = self.create_user('kermit2')
        self.do_sync_for_user(token2)
        token3 = self.create_user('kermit3')
        self.do_sync_for_user(token3)
        self.reactor.advance(2 * 24 * 60 * 60)
        self.do_sync_for_user(token1)
        self.do_sync_for_user(token2)
        self.reactor.advance(60 * 24 * 60 * 60)
        self.get_success(self.store.reap_monthly_active_users())
        token4 = self.create_user('kermit4')
        self.do_sync_for_user(token4)
        token5 = self.create_user('kermit5')
        self.do_sync_for_user(token5)
        token6 = self.create_user('kermit6')
        self.do_sync_for_user(token6)
        self.do_sync_for_user(token2)
        self.do_sync_for_user(token3)
        self.do_sync_for_user(token4)
        self.do_sync_for_user(token5)
        self.do_sync_for_user(token6)
        with self.assertRaises(SynapseError) as cm:
            self.do_sync_for_user(token1)
        e = cm.exception
        self.assertEqual(e.code, 403)
        self.assertEqual(e.errcode, Codes.RESOURCE_LIMIT_EXCEEDED)

    @override_config({'max_mau_value': 1, 'limit_usage_by_mau': False, 'mau_stats_only': True})
    def test_tracked_but_not_limited(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        token1 = self.create_user('kermit1')
        self.do_sync_for_user(token1)
        token2 = self.create_user('kermit2')
        self.do_sync_for_user(token2)
        count = self.store.get_monthly_active_count()
        self.reactor.advance(100)
        self.assertEqual(2, self.successResultOf(count))

    @override_config({'mau_trial_days': 3, 'mau_appservice_trial_days': {'SomeASID': 1, 'AnotherASID': 2}})
    def test_as_trial_days(self) -> None:
        if False:
            i = 10
            return i + 15
        user_tokens: List[str] = []

        def advance_time_and_sync() -> None:
            if False:
                i = 10
                return i + 15
            self.reactor.advance(24 * 60 * 61)
            for token in user_tokens:
                self.do_sync_for_user(token)
        as_token_1 = 'foobartoken1'
        self.store.services_cache.append(ApplicationService(token=as_token_1, id='SomeASID', sender='@as_sender_1:test', namespaces={'users': [{'regex': '@as_1.*', 'exclusive': True}]}))
        as_token_2 = 'foobartoken2'
        self.store.services_cache.append(ApplicationService(token=as_token_2, id='AnotherASID', sender='@as_sender_2:test', namespaces={'users': [{'regex': '@as_2.*', 'exclusive': True}]}))
        user_tokens.append(self.create_user('kermit1'))
        user_tokens.append(self.create_user('kermit2'))
        user_tokens.append(self.create_user('as_1kermit3', token=as_token_1, appservice=True))
        user_tokens.append(self.create_user('as_2kermit4', token=as_token_2, appservice=True))
        advance_time_and_sync()
        self.assertEqual(self.get_success(self.store.get_monthly_active_count_by_service()), {'SomeASID': 1})
        advance_time_and_sync()
        self.assertEqual(self.get_success(self.store.get_monthly_active_count_by_service()), {'SomeASID': 1, 'AnotherASID': 1})
        advance_time_and_sync()
        self.assertEqual(self.get_success(self.store.get_monthly_active_count_by_service()), {'SomeASID': 1, 'AnotherASID': 1, 'native': 2})

    def create_user(self, localpart: str, token: Optional[str]=None, appservice: bool=False) -> str:
        if False:
            for i in range(10):
                print('nop')
        request_data = {'username': localpart, 'password': 'monkey', 'auth': {'type': LoginType.DUMMY}}
        if appservice:
            request_data['type'] = APP_SERVICE_REGISTRATION_TYPE
        channel = self.make_request('POST', '/register', request_data, access_token=token)
        if channel.code != 200:
            raise HttpResponseException(channel.code, channel.result['reason'], channel.result['body']).to_synapse_error()
        access_token = channel.json_body['access_token']
        return access_token

    def do_sync_for_user(self, token: str) -> None:
        if False:
            while True:
                i = 10
        channel = self.make_request('GET', '/sync', access_token=token)
        if channel.code != 200:
            raise HttpResponseException(channel.code, channel.result['reason'], channel.result['body']).to_synapse_error()