import time
from random import choice
from ipv8.messaging.anonymization.tunnel import PEER_FLAG_EXIT_BT
from ipv8.peerdiscovery.discovery import DiscoveryStrategy

class GoldenRatioStrategy(DiscoveryStrategy):
    """
    Strategy for removing peers once we have too many in the TunnelCommunity.

    This strategy will remove a "normal" peer if the current ratio of "normal" peers to exit node peers is larger
    than the set golden ratio.
    This strategy will remove an exit peer if the current ratio of "normal" peers to exit node peers is smaller than
    the set golden ratio.
    """

    def __init__(self, overlay, golden_ratio=9 / 16, target_peers=23):
        if False:
            while True:
                i = 10
        '\n        Initialize the GoldenRatioStrategy.\n\n        :param overlay: the overlay instance to walk over\n        :type overlay: TriblerTunnelCommunity\n        :param golden_ratio: the ratio of normal/exit node peers to pursue (between 0.0 and 1.0)\n        :type golden_ratio: float\n        :param target_peers: the amount of peers at which to start removing (>0)\n        :type target_peers: int\n        :returns: None\n        '
        super().__init__(overlay)
        self.golden_ratio = golden_ratio
        self.target_peers = target_peers
        self.intro_sent = {}
        assert target_peers > 0
        assert 0.0 <= golden_ratio <= 1.0

    def take_step(self):
        if False:
            i = 10
            return i + 15
        '\n        We are asked to update, see if we have enough peers to start culling them.\n        If we do have enough peers, select a suitable peer to remove.\n\n        :returns: None\n        '
        with self.walk_lock:
            peers = self.overlay.get_peers()
            for peer in list(self.intro_sent.keys()):
                if peer not in peers:
                    self.intro_sent.pop(peer, None)
            now = time.time()
            for peer in peers:
                if peer not in self.overlay.candidates and now > self.intro_sent.get(peer, 0) + 300:
                    self.overlay.send_introduction_request(peer)
                    self.intro_sent[peer] = now
            peer_count = len(peers)
            if peer_count > self.target_peers:
                exit_peers = set(self.overlay.get_candidates(PEER_FLAG_EXIT_BT))
                exit_count = len(exit_peers)
                ratio = 1.0 - exit_count / peer_count
                if ratio < self.golden_ratio:
                    self.overlay.network.remove_peer(choice(list(exit_peers)))
                elif ratio > self.golden_ratio:
                    self.overlay.network.remove_peer(choice(list(set(self.overlay.get_peers()) - exit_peers)))