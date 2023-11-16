from twisted.test.proto_helpers import MemoryReactor
from synapse.rest.client import rendezvous
from synapse.server import HomeServer
from synapse.util import Clock
from tests import unittest
from tests.unittest import override_config
endpoint = '/_matrix/client/unstable/org.matrix.msc3886/rendezvous'

class RendezvousServletTestCase(unittest.HomeserverTestCase):
    servlets = [rendezvous.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            i = 10
            return i + 15
        self.hs = self.setup_test_homeserver()
        return self.hs

    def test_disabled(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        channel = self.make_request('POST', endpoint, {}, access_token=None)
        self.assertEqual(channel.code, 404)

    @override_config({'experimental_features': {'msc3886_endpoint': '/asd'}})
    def test_redirect(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        channel = self.make_request('POST', endpoint, {}, access_token=None)
        self.assertEqual(channel.code, 307)
        self.assertEqual(channel.headers.getRawHeaders('Location'), ['/asd'])