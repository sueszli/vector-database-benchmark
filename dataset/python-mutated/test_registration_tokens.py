import random
import string
from typing import Optional
from twisted.test.proto_helpers import MemoryReactor
import synapse.rest.admin
from synapse.api.errors import Codes
from synapse.rest.client import login
from synapse.server import HomeServer
from synapse.util import Clock
from tests import unittest

class ManageRegistrationTokensTestCase(unittest.HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.store = hs.get_datastores().main
        self.admin_user = self.register_user('admin', 'pass', admin=True)
        self.admin_user_tok = self.login('admin', 'pass')
        self.other_user = self.register_user('user', 'pass')
        self.other_user_tok = self.login('user', 'pass')
        self.url = '/_synapse/admin/v1/registration_tokens'

    def _new_token(self, token: Optional[str]=None, uses_allowed: Optional[int]=None, pending: int=0, completed: int=0, expiry_time: Optional[int]=None) -> str:
        if False:
            print('Hello World!')
        'Helper function to create a token.'
        if token is None:
            token = ''.join(random.choices(string.ascii_letters, k=8))
        self.get_success(self.store.db_pool.simple_insert('registration_tokens', {'token': token, 'uses_allowed': uses_allowed, 'pending': pending, 'completed': completed, 'expiry_time': expiry_time}))
        return token

    def test_create_no_auth(self) -> None:
        if False:
            while True:
                i = 10
        'Try to create a token without authentication.'
        channel = self.make_request('POST', self.url + '/new', {})
        self.assertEqual(401, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.MISSING_TOKEN, channel.json_body['errcode'])

    def test_create_requester_not_admin(self) -> None:
        if False:
            print('Hello World!')
        'Try to create a token while not an admin.'
        channel = self.make_request('POST', self.url + '/new', {}, access_token=self.other_user_tok)
        self.assertEqual(403, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.FORBIDDEN, channel.json_body['errcode'])

    def test_create_using_defaults(self) -> None:
        if False:
            i = 10
            return i + 15
        'Create a token using all the defaults.'
        channel = self.make_request('POST', self.url + '/new', {}, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self.assertEqual(len(channel.json_body['token']), 16)
        self.assertIsNone(channel.json_body['uses_allowed'])
        self.assertIsNone(channel.json_body['expiry_time'])
        self.assertEqual(channel.json_body['pending'], 0)
        self.assertEqual(channel.json_body['completed'], 0)

    def test_create_specifying_fields(self) -> None:
        if False:
            print('Hello World!')
        'Create a token specifying the value of all fields.'
        token = 'adefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._~-'
        data = {'token': token, 'uses_allowed': 1, 'expiry_time': self.clock.time_msec() + 1000000}
        channel = self.make_request('POST', self.url + '/new', data, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['token'], token)
        self.assertEqual(channel.json_body['uses_allowed'], 1)
        self.assertEqual(channel.json_body['expiry_time'], data['expiry_time'])
        self.assertEqual(channel.json_body['pending'], 0)
        self.assertEqual(channel.json_body['completed'], 0)

    def test_create_with_null_value(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Create a token specifying unlimited uses and no expiry.'
        data = {'uses_allowed': None, 'expiry_time': None}
        channel = self.make_request('POST', self.url + '/new', data, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self.assertEqual(len(channel.json_body['token']), 16)
        self.assertIsNone(channel.json_body['uses_allowed'])
        self.assertIsNone(channel.json_body['expiry_time'])
        self.assertEqual(channel.json_body['pending'], 0)
        self.assertEqual(channel.json_body['completed'], 0)

    def test_create_token_too_long(self) -> None:
        if False:
            i = 10
            return i + 15
        'Check token longer than 64 chars is invalid.'
        data = {'token': 'a' * 65}
        channel = self.make_request('POST', self.url + '/new', data, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.INVALID_PARAM)

    def test_create_token_invalid_chars(self) -> None:
        if False:
            i = 10
            return i + 15
        "Check you can't create token with invalid characters."
        data = {'token': 'abc/def'}
        channel = self.make_request('POST', self.url + '/new', data, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.INVALID_PARAM)

    def test_create_token_already_exists(self) -> None:
        if False:
            print('Hello World!')
        "Check you can't create token that already exists."
        data = {'token': 'abcd'}
        channel1 = self.make_request('POST', self.url + '/new', data, access_token=self.admin_user_tok)
        self.assertEqual(200, channel1.code, msg=channel1.json_body)
        channel2 = self.make_request('POST', self.url + '/new', data, access_token=self.admin_user_tok)
        self.assertEqual(400, channel2.code, msg=channel2.json_body)
        self.assertEqual(channel2.json_body['errcode'], Codes.INVALID_PARAM)

    def test_create_unable_to_generate_token(self) -> None:
        if False:
            return 10
        "Check right error is raised when server can't generate unique token."
        tokens = []
        for c in string.ascii_letters + string.digits + '._~-':
            tokens.append((c, None, 0, 0, None))
        self.get_success(self.store.db_pool.simple_insert_many('registration_tokens', keys=('token', 'uses_allowed', 'pending', 'completed', 'expiry_time'), values=tokens, desc='create_all_registration_tokens'))
        channel = self.make_request('POST', self.url + '/new', {'length': 1}, access_token=self.admin_user_tok)
        self.assertEqual(500, channel.code, msg=channel.json_body)

    def test_create_uses_allowed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Check you can only create a token with good values for uses_allowed.'
        channel = self.make_request('POST', self.url + '/new', {'uses_allowed': 0}, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['uses_allowed'], 0)
        channel = self.make_request('POST', self.url + '/new', {'uses_allowed': -5}, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.INVALID_PARAM)
        channel = self.make_request('POST', self.url + '/new', {'uses_allowed': 1.5}, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.INVALID_PARAM)

    def test_create_expiry_time(self) -> None:
        if False:
            i = 10
            return i + 15
        "Check you can't create a token with an invalid expiry_time."
        channel = self.make_request('POST', self.url + '/new', {'expiry_time': self.clock.time_msec() - 10000}, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.INVALID_PARAM)
        channel = self.make_request('POST', self.url + '/new', {'expiry_time': self.clock.time_msec() + 1000000.5}, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.INVALID_PARAM)

    def test_create_length(self) -> None:
        if False:
            return 10
        'Check you can only generate a token with a valid length.'
        channel = self.make_request('POST', self.url + '/new', {'length': 64}, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self.assertEqual(len(channel.json_body['token']), 64)
        channel = self.make_request('POST', self.url + '/new', {'length': 0}, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.INVALID_PARAM)
        channel = self.make_request('POST', self.url + '/new', {'length': -5}, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.INVALID_PARAM)
        channel = self.make_request('POST', self.url + '/new', {'length': 8.5}, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.INVALID_PARAM)
        channel = self.make_request('POST', self.url + '/new', {'length': 65}, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.INVALID_PARAM)

    def test_update_no_auth(self) -> None:
        if False:
            while True:
                i = 10
        'Try to update a token without authentication.'
        channel = self.make_request('PUT', self.url + '/1234', {})
        self.assertEqual(401, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.MISSING_TOKEN, channel.json_body['errcode'])

    def test_update_requester_not_admin(self) -> None:
        if False:
            print('Hello World!')
        'Try to update a token while not an admin.'
        channel = self.make_request('PUT', self.url + '/1234', {}, access_token=self.other_user_tok)
        self.assertEqual(403, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.FORBIDDEN, channel.json_body['errcode'])

    def test_update_non_existent(self) -> None:
        if False:
            print('Hello World!')
        "Try to update a token that doesn't exist."
        channel = self.make_request('PUT', self.url + '/1234', {'uses_allowed': 1}, access_token=self.admin_user_tok)
        self.assertEqual(404, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.NOT_FOUND)

    def test_update_uses_allowed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test updating just uses_allowed.'
        token = self._new_token()
        channel = self.make_request('PUT', self.url + '/' + token, {'uses_allowed': 1}, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['uses_allowed'], 1)
        self.assertIsNone(channel.json_body['expiry_time'])
        channel = self.make_request('PUT', self.url + '/' + token, {'uses_allowed': 0}, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['uses_allowed'], 0)
        self.assertIsNone(channel.json_body['expiry_time'])
        channel = self.make_request('PUT', self.url + '/' + token, {'uses_allowed': None}, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self.assertIsNone(channel.json_body['uses_allowed'])
        self.assertIsNone(channel.json_body['expiry_time'])
        channel = self.make_request('PUT', self.url + '/' + token, {'uses_allowed': 1.5}, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.INVALID_PARAM)
        channel = self.make_request('PUT', self.url + '/' + token, {'uses_allowed': -5}, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.INVALID_PARAM)

    def test_update_expiry_time(self) -> None:
        if False:
            print('Hello World!')
        'Test updating just expiry_time.'
        token = self._new_token()
        new_expiry_time = self.clock.time_msec() + 1000000
        channel = self.make_request('PUT', self.url + '/' + token, {'expiry_time': new_expiry_time}, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['expiry_time'], new_expiry_time)
        self.assertIsNone(channel.json_body['uses_allowed'])
        channel = self.make_request('PUT', self.url + '/' + token, {'expiry_time': None}, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self.assertIsNone(channel.json_body['expiry_time'])
        self.assertIsNone(channel.json_body['uses_allowed'])
        past_time = self.clock.time_msec() - 10000
        channel = self.make_request('PUT', self.url + '/' + token, {'expiry_time': past_time}, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.INVALID_PARAM)
        channel = self.make_request('PUT', self.url + '/' + token, {'expiry_time': new_expiry_time + 0.5}, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.INVALID_PARAM)

    def test_update_both(self) -> None:
        if False:
            return 10
        'Test updating both uses_allowed and expiry_time.'
        token = self._new_token()
        new_expiry_time = self.clock.time_msec() + 1000000
        data = {'uses_allowed': 1, 'expiry_time': new_expiry_time}
        channel = self.make_request('PUT', self.url + '/' + token, data, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['uses_allowed'], 1)
        self.assertEqual(channel.json_body['expiry_time'], new_expiry_time)

    def test_update_invalid_type(self) -> None:
        if False:
            print('Hello World!')
        "Test using invalid types doesn't work."
        token = self._new_token()
        data = {'uses_allowed': False, 'expiry_time': '1626430124000'}
        channel = self.make_request('PUT', self.url + '/' + token, data, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.INVALID_PARAM)

    def test_delete_no_auth(self) -> None:
        if False:
            while True:
                i = 10
        'Try to delete a token without authentication.'
        channel = self.make_request('DELETE', self.url + '/1234', {})
        self.assertEqual(401, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.MISSING_TOKEN, channel.json_body['errcode'])

    def test_delete_requester_not_admin(self) -> None:
        if False:
            i = 10
            return i + 15
        'Try to delete a token while not an admin.'
        channel = self.make_request('DELETE', self.url + '/1234', {}, access_token=self.other_user_tok)
        self.assertEqual(403, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.FORBIDDEN, channel.json_body['errcode'])

    def test_delete_non_existent(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Try to delete a token that doesn't exist."
        channel = self.make_request('DELETE', self.url + '/1234', {}, access_token=self.admin_user_tok)
        self.assertEqual(404, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.NOT_FOUND)

    def test_delete(self) -> None:
        if False:
            return 10
        'Test deleting a token.'
        token = self._new_token()
        channel = self.make_request('DELETE', self.url + '/' + token, {}, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)

    def test_get_no_auth(self) -> None:
        if False:
            print('Hello World!')
        'Try to get a token without authentication.'
        channel = self.make_request('GET', self.url + '/1234', {})
        self.assertEqual(401, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.MISSING_TOKEN, channel.json_body['errcode'])

    def test_get_requester_not_admin(self) -> None:
        if False:
            while True:
                i = 10
        'Try to get a token while not an admin.'
        channel = self.make_request('GET', self.url + '/1234', {}, access_token=self.other_user_tok)
        self.assertEqual(403, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.FORBIDDEN, channel.json_body['errcode'])

    def test_get_non_existent(self) -> None:
        if False:
            while True:
                i = 10
        "Try to get a token that doesn't exist."
        channel = self.make_request('GET', self.url + '/1234', {}, access_token=self.admin_user_tok)
        self.assertEqual(404, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], Codes.NOT_FOUND)

    def test_get(self) -> None:
        if False:
            return 10
        'Test getting a token.'
        token = self._new_token()
        channel = self.make_request('GET', self.url + '/' + token, {}, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['token'], token)
        self.assertIsNone(channel.json_body['uses_allowed'])
        self.assertIsNone(channel.json_body['expiry_time'])
        self.assertEqual(channel.json_body['pending'], 0)
        self.assertEqual(channel.json_body['completed'], 0)

    def test_list_no_auth(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Try to list tokens without authentication.'
        channel = self.make_request('GET', self.url, {})
        self.assertEqual(401, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.MISSING_TOKEN, channel.json_body['errcode'])

    def test_list_requester_not_admin(self) -> None:
        if False:
            while True:
                i = 10
        'Try to list tokens while not an admin.'
        channel = self.make_request('GET', self.url, {}, access_token=self.other_user_tok)
        self.assertEqual(403, channel.code, msg=channel.json_body)
        self.assertEqual(Codes.FORBIDDEN, channel.json_body['errcode'])

    def test_list_all(self) -> None:
        if False:
            print('Hello World!')
        'Test listing all tokens.'
        token = self._new_token()
        channel = self.make_request('GET', self.url, {}, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self.assertEqual(len(channel.json_body['registration_tokens']), 1)
        token_info = channel.json_body['registration_tokens'][0]
        self.assertEqual(token_info['token'], token)
        self.assertIsNone(token_info['uses_allowed'])
        self.assertIsNone(token_info['expiry_time'])
        self.assertEqual(token_info['pending'], 0)
        self.assertEqual(token_info['completed'], 0)

    def test_list_invalid_query_parameter(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test with `valid` query parameter not `true` or `false`.'
        channel = self.make_request('GET', self.url + '?valid=x', {}, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)

    def _test_list_query_parameter(self, valid: str) -> None:
        if False:
            i = 10
            return i + 15
        'Helper used to test both valid=true and valid=false.'
        now = self.hs.get_clock().time_msec()
        valid1 = self._new_token()
        valid2 = self._new_token(uses_allowed=1)
        invalid1 = self._new_token(expiry_time=now - 10000)
        invalid2 = self._new_token(uses_allowed=2, pending=1, completed=1, expiry_time=now + 1000000)
        if valid == 'true':
            tokens = [valid1, valid2]
        else:
            tokens = [invalid1, invalid2]
        channel = self.make_request('GET', self.url + '?valid=' + valid, {}, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self.assertEqual(len(channel.json_body['registration_tokens']), 2)
        token_info_1 = channel.json_body['registration_tokens'][0]
        token_info_2 = channel.json_body['registration_tokens'][1]
        self.assertIn(token_info_1['token'], tokens)
        self.assertIn(token_info_2['token'], tokens)

    def test_list_valid(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test listing just valid tokens.'
        self._test_list_query_parameter(valid='true')

    def test_list_invalid(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test listing just invalid tokens.'
        self._test_list_query_parameter(valid='false')