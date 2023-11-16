import struct
import asyncio
from lbry.utils import generate_id
from lbry.dht.protocol.routing_table import KBucket
from lbry.dht.peer import PeerManager, make_kademlia_peer
from lbry.dht import constants
from lbry.testcase import AsyncioTestCase

def address_generator(address=(1, 2, 3, 4)):
    if False:
        i = 10
        return i + 15

    def increment(addr):
        if False:
            return 10
        value = struct.unpack('I', ''.join([chr(x) for x in list(addr)[::-1]]).encode())[0] + 1
        new_addr = []
        for i in range(4):
            new_addr.append(value % 256)
            value >>= 8
        return tuple(new_addr[::-1])
    while True:
        yield '{}.{}.{}.{}'.format(*address)
        address = increment(address)

class TestKBucket(AsyncioTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.loop = asyncio.get_event_loop()
        self.address_generator = address_generator()
        self.peer_manager = PeerManager(self.loop)
        self.kbucket = KBucket(self.peer_manager, 0, 2 ** constants.HASH_BITS, generate_id())

    def test_add_peer(self):
        if False:
            for i in range(10):
                print('nop')
        peer = make_kademlia_peer(constants.generate_id(2), '1.2.3.4', udp_port=4444)
        peer_update2 = make_kademlia_peer(constants.generate_id(2), '1.2.3.4', udp_port=4445)
        self.assertListEqual([], self.kbucket.peers)
        self.kbucket.add_peer(peer)
        self.assertListEqual([peer], self.kbucket.peers)
        self.kbucket.add_peer(peer)
        self.assertListEqual([peer], self.kbucket.peers)
        self.assertEqual(self.kbucket.peers[0].udp_port, 4444)
        self.kbucket.add_peer(peer_update2)
        self.assertListEqual([peer_update2], self.kbucket.peers)
        self.assertEqual(self.kbucket.peers[0].udp_port, 4445)
        peer_update2.udp_port = 4444
        self.kbucket.add_peer(peer_update2)
        self.assertListEqual([peer_update2], self.kbucket.peers)
        self.assertEqual(self.kbucket.peers[0].udp_port, 4444)
        self.kbucket.peers.clear()
        for i in range(constants.K):
            peer = make_kademlia_peer(generate_id(), next(self.address_generator), 4444)
            self.assertTrue(self.kbucket.add_peer(peer))
            self.assertEqual(peer, self.kbucket.peers[i])
        peer = make_kademlia_peer(generate_id(), next(self.address_generator), 4444)
        self.assertFalse(self.kbucket.add_peer(peer))
        existing_peer = self.kbucket.peers[0]
        self.assertTrue(self.kbucket.add_peer(existing_peer))
        self.assertEqual(existing_peer, self.kbucket.peers[-1])

    def test_remove_peer(self):
        if False:
            return 10
        peer = make_kademlia_peer(generate_id(), next(self.address_generator), 4444)
        self.assertRaises(ValueError, self.kbucket.remove_peer, peer)
        added = []
        for i in range(constants.K - 2):
            peer = make_kademlia_peer(generate_id(), next(self.address_generator), 4444)
            self.assertTrue(self.kbucket.add_peer(peer))
            added.append(peer)
        while added:
            peer = added.pop()
            self.assertIn(peer, self.kbucket.peers)
            self.kbucket.remove_peer(peer)
            self.assertNotIn(peer, self.kbucket.peers)