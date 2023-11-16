from ipv8.keyvault.crypto import default_eccrypto
from ipv8.peer import Peer
from ipv8.peerdiscovery.network import Network
from tribler.core.components.gigachannel.community.sync_strategy import RemovePeers
from tribler.core.components.ipv8.adapters_tests import TriblerTestBase

class MockCommunity:

    def __init__(self):
        if False:
            print('Hello World!')
        self.fetch_next_called = False
        self.send_random_to_called = []
        self.get_peers_return = []
        self.network = Network()

    def send_random_to(self, peer):
        if False:
            for i in range(10):
                print('nop')
        self.send_random_to_called.append(peer)

    def fetch_next(self):
        if False:
            while True:
                i = 10
        self.fetch_next_called = True

    def get_peers(self):
        if False:
            return 10
        return self.get_peers_return

class TestRemovePeers(TriblerTestBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.community = MockCommunity()
        self.strategy = RemovePeers(self.community)
        return super().setUp()

    def test_strategy_no_peers(self):
        if False:
            while True:
                i = 10
        '\n        If we have no peers, nothing should happen.\n        '
        self.strategy.take_step()
        self.assertSetEqual(set(), self.community.network.verified_peers)

    def test_strategy_one_peer(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If we have one peer, it should not be removed.\n        '
        test_peer = Peer(default_eccrypto.generate_key('very-low'))
        self.community.network.add_verified_peer(test_peer)
        self.community.get_peers_return.append(test_peer)
        self.strategy.take_step()
        self.assertSetEqual({test_peer}, self.community.network.verified_peers)

    def test_strategy_multi_peer(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If we have over 20 peers, one should be removed.\n        '
        for _ in range(21):
            test_peer = Peer(default_eccrypto.generate_key('very-low'))
            self.community.network.add_verified_peer(test_peer)
            self.community.get_peers_return.append(test_peer)
        self.strategy.take_step()
        self.assertEqual(20, len(self.community.network.verified_peers))