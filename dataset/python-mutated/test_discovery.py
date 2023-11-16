from unittest.mock import Mock
from ipv8.keyvault.crypto import default_eccrypto
from ipv8.messaging.anonymization.tunnel import PEER_FLAG_EXIT_BT
from ipv8.peer import Peer
from ipv8.peerdiscovery.network import Network
from tribler.core.components.tunnel.community.discovery import GoldenRatioStrategy

class FakeOverlay:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.exit_candidates = []
        self.candidates = {}
        self.network = Network()
        self.send_introduction_request = Mock()

    def get_candidates(self, flag):
        if False:
            return 10
        return self.exit_candidates if flag == PEER_FLAG_EXIT_BT else []

    def get_peers(self):
        if False:
            print('Hello World!')
        return self.network.verified_peers

def generate_peer():
    if False:
        return 10
    return Peer(default_eccrypto.generate_key('very-low'))

def generate_overlay_and_peers():
    if False:
        while True:
            i = 10
    overlay = FakeOverlay()
    peer1 = generate_peer()
    peer2 = generate_peer()
    overlay.exit_candidates.append(peer2)
    overlay.network.add_verified_peer(peer1)
    overlay.network.add_verified_peer(peer2)
    return (overlay, peer1, peer2)

def test_invariant():
    if False:
        return 10
    "\n    If we are not at our target peer count, don't do anything.\n    "
    (overlay, peer1, peer2) = generate_overlay_and_peers()
    strategy = GoldenRatioStrategy(overlay, 0.0, 3)
    strategy.take_step()
    strategy.golden_ratio = 1.0
    strategy.take_step()
    assert len(overlay.network.verified_peers) == 2
    assert peer1 in overlay.network.verified_peers
    assert peer2 in overlay.network.verified_peers

def test_remove_normal():
    if False:
        i = 10
        return i + 15
    '\n    If we have a normal node and an exit node, check if enforcing a ratio of 0.0 removes the normal node.\n    '
    (overlay, _, peer2) = generate_overlay_and_peers()
    strategy = GoldenRatioStrategy(overlay, 0.0, 1)
    strategy.take_step()
    assert len(overlay.network.verified_peers) == 1
    assert peer2 in overlay.network.verified_peers

def test_remove_exit():
    if False:
        return 10
    '\n    If we have a normal node and an exit node, check if enforcing a ratio of 1.0 removes the exit node.\n    '
    (overlay, peer1, _) = generate_overlay_and_peers()
    strategy = GoldenRatioStrategy(overlay, 1.0, 1)
    strategy.take_step()
    assert len(overlay.network.verified_peers) == 1
    assert peer1 in overlay.network.verified_peers

def test_send_introduction_request():
    if False:
        print('Hello World!')
    '\n    If a node has sent us its peer_flag, check if an introduction_request is sent.\n    '
    (overlay, peer1, peer2) = generate_overlay_and_peers()
    overlay.candidates[peer2] = []
    strategy = GoldenRatioStrategy(overlay, 1.0, 1)
    strategy.take_step()
    overlay.send_introduction_request.assert_called_once_with(peer1)