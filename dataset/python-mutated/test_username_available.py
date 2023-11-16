from typing import Optional
from twisted.test.proto_helpers import MemoryReactor
import synapse.rest.admin
from synapse.api.errors import Codes, SynapseError
from synapse.rest.client import login
from synapse.server import HomeServer
from synapse.util import Clock
from tests import unittest

class UsernameAvailableTestCase(unittest.HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets, login.register_servlets]
    url = '/_synapse/admin/v1/username_available'

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.register_user('admin', 'pass', admin=True)
        self.admin_user_tok = self.login('admin', 'pass')

        async def check_username(localpart: str, guest_access_token: Optional[str]=None, assigned_user_id: Optional[str]=None, inhibit_user_in_use_error: bool=False) -> None:
            if localpart == 'allowed':
                return
            raise SynapseError(400, 'User ID already taken.', errcode=Codes.USER_IN_USE)
        handler = self.hs.get_registration_handler()
        handler.check_username = check_username

    def test_username_available(self) -> None:
        if False:
            while True:
                i = 10
        '\n        The endpoint should return a 200 response if the username does not exist\n        '
        url = '%s?username=%s' % (self.url, 'allowed')
        channel = self.make_request('GET', url, access_token=self.admin_user_tok)
        self.assertEqual(200, channel.code, msg=channel.json_body)
        self.assertTrue(channel.json_body['available'])

    def test_username_unavailable(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        The endpoint should return a 200 response if the username does not exist\n        '
        url = '%s?username=%s' % (self.url, 'disallowed')
        channel = self.make_request('GET', url, access_token=self.admin_user_tok)
        self.assertEqual(400, channel.code, msg=channel.json_body)
        self.assertEqual(channel.json_body['errcode'], 'M_USER_IN_USE')
        self.assertEqual(channel.json_body['error'], 'User ID already taken.')