import asyncio
import random
import logging
import typing
import itertools
from prometheus_client import Gauge
from lbry import utils
from lbry.dht import constants
from lbry.dht.error import RemoteException
from lbry.dht.protocol.distance import Distance
if typing.TYPE_CHECKING:
    from lbry.dht.peer import KademliaPeer, PeerManager
log = logging.getLogger(__name__)

class KBucket:
    """
    Kademlia K-bucket implementation.
    """
    peer_in_routing_table_metric = Gauge('peers_in_routing_table', 'Number of peers on routing table', namespace='dht_node', labelnames=('scope',))
    peer_with_x_bit_colliding_metric = Gauge('peer_x_bit_colliding', 'Number of peers with at least X bits colliding with this node id', namespace='dht_node', labelnames=('amount',))

    def __init__(self, peer_manager: 'PeerManager', range_min: int, range_max: int, node_id: bytes, capacity: int=constants.K):
        if False:
            i = 10
            return i + 15
        '\n        @param range_min: The lower boundary for the range in the n-bit ID\n                         space covered by this k-bucket\n        @param range_max: The upper boundary for the range in the ID space\n                         covered by this k-bucket\n        '
        self._peer_manager = peer_manager
        self.range_min = range_min
        self.range_max = range_max
        self.peers: typing.List['KademliaPeer'] = []
        self._node_id = node_id
        self._distance_to_self = Distance(node_id)
        self.capacity = capacity

    def add_peer(self, peer: 'KademliaPeer') -> bool:
        if False:
            print('Hello World!')
        " Add contact to _contact list in the right order. This will move the\n        contact to the end of the k-bucket if it is already present.\n\n        @raise kademlia.kbucket.BucketFull: Raised when the bucket is full and\n                                            the contact isn't in the bucket\n                                            already\n\n        @param peer: The contact to add\n        @type peer: dht.contact._Contact\n        "
        if peer in self.peers:
            self.peers.remove(peer)
            self.peers.append(peer)
            return True
        else:
            for (i, _) in enumerate(self.peers):
                local_peer = self.peers[i]
                if local_peer.node_id == peer.node_id:
                    self.peers.remove(local_peer)
                    self.peers.append(peer)
                    return True
        if len(self.peers) < self.capacity:
            self.peers.append(peer)
            self.peer_in_routing_table_metric.labels('global').inc()
            bits_colliding = utils.get_colliding_prefix_bits(peer.node_id, self._node_id)
            self.peer_with_x_bit_colliding_metric.labels(amount=bits_colliding).inc()
            return True
        else:
            return False

    def get_peer(self, node_id: bytes) -> 'KademliaPeer':
        if False:
            while True:
                i = 10
        for peer in self.peers:
            if peer.node_id == node_id:
                return peer

    def get_peers(self, count=-1, exclude_contact=None, sort_distance_to=None) -> typing.List['KademliaPeer']:
        if False:
            while True:
                i = 10
        " Returns a list containing up to the first count number of contacts\n\n        @param count: The amount of contacts to return (if 0 or less, return\n                      all contacts)\n        @type count: int\n        @param exclude_contact: A node node_id to exclude; if this contact is in\n                               the list of returned values, it will be\n                               discarded before returning. If a C{str} is\n                               passed as this argument, it must be the\n                               contact's ID.\n        @type exclude_contact: str\n\n        @param sort_distance_to: Sort distance to the node_id, defaulting to the parent node node_id. If False don't\n                                 sort the contacts\n\n        @raise IndexError: If the number of requested contacts is too large\n\n        @return: Return up to the first count number of contacts in a list\n                If no contacts are present an empty is returned\n        @rtype: list\n        "
        peers = [peer for peer in self.peers if peer.node_id != exclude_contact]
        if count <= 0:
            count = len(peers)
        current_len = len(peers)
        if count > constants.K:
            count = constants.K
        if not current_len:
            return peers
        if sort_distance_to is False:
            pass
        else:
            sort_distance_to = sort_distance_to or self._node_id
            peers.sort(key=lambda c: Distance(sort_distance_to)(c.node_id))
        return peers[:min(current_len, count)]

    def get_bad_or_unknown_peers(self) -> typing.List['KademliaPeer']:
        if False:
            i = 10
            return i + 15
        peer = self.get_peers(sort_distance_to=False)
        return [peer for peer in peer if self._peer_manager.contact_triple_is_good(peer.node_id, peer.address, peer.udp_port) is not True]

    def remove_peer(self, peer: 'KademliaPeer') -> None:
        if False:
            while True:
                i = 10
        self.peers.remove(peer)
        self.peer_in_routing_table_metric.labels('global').dec()
        bits_colliding = utils.get_colliding_prefix_bits(peer.node_id, self._node_id)
        self.peer_with_x_bit_colliding_metric.labels(amount=bits_colliding).dec()

    def key_in_range(self, key: bytes) -> bool:
        if False:
            for i in range(10):
                print('nop')
        " Tests whether the specified key (i.e. node ID) is in the range\n        of the n-bit ID space covered by this k-bucket (in otherwords, it\n        returns whether or not the specified key should be placed in this\n        k-bucket)\n\n        @param key: The key to test\n        @type key: str or int\n\n        @return: C{True} if the key is in this k-bucket's range, or C{False}\n                 if not.\n        @rtype: bool\n        "
        return self.range_min <= self._distance_to_self(key) < self.range_max

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self.peers)

    def __contains__(self, item) -> bool:
        if False:
            i = 10
            return i + 15
        return item in self.peers

class TreeRoutingTable:
    """ This class implements a routing table used by a Node class.

    The Kademlia routing table is a binary tree whose leaves are k-buckets,
    where each k-bucket contains nodes with some common prefix of their IDs.
    This prefix is the k-bucket's position in the binary tree; it therefore
    covers some range of ID values, and together all of the k-buckets cover
    the entire n-bit ID (or key) space (with no overlap).

    @note: In this implementation, nodes in the tree (the k-buckets) are
    added dynamically, as needed; this technique is described in the 13-page
    version of the Kademlia paper, in section 2.4. It does, however, use the
    ping RPC-based k-bucket eviction algorithm described in section 2.2 of
    that paper.

    BOOTSTRAP MODE: if set to True, we always add all peers. This is so a
    bootstrap node does not get a bias towards its own node id and replies are
    the best it can provide (joining peer knows its neighbors immediately).
    Over time, this will need to be optimized so we use the disk as holding
    everything in memory won't be feasible anymore.
    See: https://github.com/bittorrent/bootstrap-dht
    """
    bucket_in_routing_table_metric = Gauge('buckets_in_routing_table', 'Number of buckets on routing table', namespace='dht_node', labelnames=('scope',))

    def __init__(self, loop: asyncio.AbstractEventLoop, peer_manager: 'PeerManager', parent_node_id: bytes, split_buckets_under_index: int=constants.SPLIT_BUCKETS_UNDER_INDEX, is_bootstrap_node: bool=False):
        if False:
            return 10
        self._loop = loop
        self._peer_manager = peer_manager
        self._parent_node_id = parent_node_id
        self._split_buckets_under_index = split_buckets_under_index
        self.buckets: typing.List[KBucket] = [KBucket(self._peer_manager, range_min=0, range_max=2 ** constants.HASH_BITS, node_id=self._parent_node_id, capacity=1 << 32 if is_bootstrap_node else constants.K)]

    def get_peers(self) -> typing.List['KademliaPeer']:
        if False:
            while True:
                i = 10
        return list(itertools.chain.from_iterable(map(lambda bucket: bucket.peers, self.buckets)))

    def _should_split(self, bucket_index: int, to_add: bytes) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if bucket_index < self._split_buckets_under_index:
            return True
        contacts = self.get_peers()
        distance = Distance(self._parent_node_id)
        contacts.sort(key=lambda c: distance(c.node_id))
        kth_contact = contacts[-1] if len(contacts) < constants.K else contacts[constants.K - 1]
        return distance(to_add) < distance(kth_contact.node_id)

    def find_close_peers(self, key: bytes, count: typing.Optional[int]=None, sender_node_id: typing.Optional[bytes]=None) -> typing.List['KademliaPeer']:
        if False:
            return 10
        exclude = [self._parent_node_id]
        if sender_node_id:
            exclude.append(sender_node_id)
        count = count or constants.K
        distance = Distance(key)
        contacts = self.get_peers()
        contacts = [c for c in contacts if c.node_id not in exclude]
        if contacts:
            contacts.sort(key=lambda c: distance(c.node_id))
            return contacts[:min(count, len(contacts))]
        return []

    def get_peer(self, contact_id: bytes) -> 'KademliaPeer':
        if False:
            return 10
        return self.buckets[self._kbucket_index(contact_id)].get_peer(contact_id)

    def get_refresh_list(self, start_index: int=0, force: bool=False) -> typing.List[bytes]:
        if False:
            return 10
        refresh_ids = []
        for (offset, _) in enumerate(self.buckets[start_index:]):
            refresh_ids.append(self._midpoint_id_in_bucket_range(start_index + offset))
        buckets_with_contacts = self.buckets_with_contacts()
        if buckets_with_contacts <= 3:
            for i in range(buckets_with_contacts):
                refresh_ids.append(self._random_id_in_bucket_range(i))
                refresh_ids.append(self._random_id_in_bucket_range(i))
        return refresh_ids

    def remove_peer(self, peer: 'KademliaPeer') -> None:
        if False:
            while True:
                i = 10
        if not peer.node_id:
            return
        bucket_index = self._kbucket_index(peer.node_id)
        try:
            self.buckets[bucket_index].remove_peer(peer)
            self._join_buckets()
        except ValueError:
            return

    def _kbucket_index(self, key: bytes) -> int:
        if False:
            while True:
                i = 10
        i = 0
        for bucket in self.buckets:
            if bucket.key_in_range(key):
                return i
            else:
                i += 1
        return i

    def _random_id_in_bucket_range(self, bucket_index: int) -> bytes:
        if False:
            return 10
        random_id = int(random.randrange(self.buckets[bucket_index].range_min, self.buckets[bucket_index].range_max))
        return Distance(self._parent_node_id)(random_id.to_bytes(constants.HASH_LENGTH, 'big')).to_bytes(constants.HASH_LENGTH, 'big')

    def _midpoint_id_in_bucket_range(self, bucket_index: int) -> bytes:
        if False:
            while True:
                i = 10
        half = int((self.buckets[bucket_index].range_max - self.buckets[bucket_index].range_min) // 2)
        return Distance(self._parent_node_id)(int(self.buckets[bucket_index].range_min + half).to_bytes(constants.HASH_LENGTH, 'big')).to_bytes(constants.HASH_LENGTH, 'big')

    def _split_bucket(self, old_bucket_index: int) -> None:
        if False:
            print('Hello World!')
        " Splits the specified k-bucket into two new buckets which together\n        cover the same range in the key/ID space\n\n        @param old_bucket_index: The index of k-bucket to split (in this table's\n                                 list of k-buckets)\n        @type old_bucket_index: int\n        "
        old_bucket = self.buckets[old_bucket_index]
        split_point = old_bucket.range_max - (old_bucket.range_max - old_bucket.range_min) // 2
        new_bucket = KBucket(self._peer_manager, split_point, old_bucket.range_max, self._parent_node_id)
        old_bucket.range_max = split_point
        self.buckets.insert(old_bucket_index + 1, new_bucket)
        for contact in old_bucket.peers:
            if new_bucket.key_in_range(contact.node_id):
                new_bucket.add_peer(contact)
        for contact in new_bucket.peers:
            old_bucket.remove_peer(contact)
        self.bucket_in_routing_table_metric.labels('global').set(len(self.buckets))

    def _join_buckets(self):
        if False:
            print('Hello World!')
        if len(self.buckets) == 1:
            return
        to_pop = [i for (i, bucket) in enumerate(self.buckets) if len(bucket) == 0]
        if not to_pop:
            return
        log.info('join buckets %i', len(to_pop))
        bucket_index_to_pop = to_pop[0]
        assert len(self.buckets[bucket_index_to_pop]) == 0
        can_go_lower = bucket_index_to_pop - 1 >= 0
        can_go_higher = bucket_index_to_pop + 1 < len(self.buckets)
        assert can_go_higher or can_go_lower
        bucket = self.buckets[bucket_index_to_pop]
        if can_go_lower and can_go_higher:
            midpoint = (bucket.range_max - bucket.range_min) // 2 + bucket.range_min
            self.buckets[bucket_index_to_pop - 1].range_max = midpoint - 1
            self.buckets[bucket_index_to_pop + 1].range_min = midpoint
        elif can_go_lower:
            self.buckets[bucket_index_to_pop - 1].range_max = bucket.range_max
        elif can_go_higher:
            self.buckets[bucket_index_to_pop + 1].range_min = bucket.range_min
        self.buckets.remove(bucket)
        self.bucket_in_routing_table_metric.labels('global').set(len(self.buckets))
        return self._join_buckets()

    def buckets_with_contacts(self) -> int:
        if False:
            return 10
        count = 0
        for bucket in self.buckets:
            if len(bucket) > 0:
                count += 1
        return count

    async def add_peer(self, peer: 'KademliaPeer', probe: typing.Callable[['KademliaPeer'], typing.Awaitable]):
        if not peer.node_id:
            log.warning('Tried adding a peer with no node id!')
            return False
        for my_peer in self.get_peers():
            if (my_peer.address, my_peer.udp_port) == (peer.address, peer.udp_port) and my_peer.node_id != peer.node_id:
                self.remove_peer(my_peer)
                self._join_buckets()
        bucket_index = self._kbucket_index(peer.node_id)
        if self.buckets[bucket_index].add_peer(peer):
            return True
        if self._should_split(bucket_index, peer.node_id):
            self._split_bucket(bucket_index)
            result = await self.add_peer(peer, probe)
            self._join_buckets()
            return result
        else:
            not_good_contacts = self.buckets[bucket_index].get_bad_or_unknown_peers()
            not_recently_replied = []
            for my_peer in not_good_contacts:
                last_replied = self._peer_manager.get_last_replied(my_peer.address, my_peer.udp_port)
                if not last_replied or last_replied + 60 < self._loop.time():
                    not_recently_replied.append(my_peer)
            if not_recently_replied:
                to_replace = not_recently_replied[0]
            else:
                to_replace = self.buckets[bucket_index].peers[0]
                last_replied = self._peer_manager.get_last_replied(to_replace.address, to_replace.udp_port)
                if last_replied and last_replied + 60 > self._loop.time():
                    return False
            log.debug('pinging %s:%s', to_replace.address, to_replace.udp_port)
            try:
                await probe(to_replace)
                return False
            except (asyncio.TimeoutError, RemoteException):
                log.debug('Replacing dead contact in bucket %i: %s:%i with %s:%i ', bucket_index, to_replace.address, to_replace.udp_port, peer.address, peer.udp_port)
                if to_replace in self.buckets[bucket_index]:
                    self.buckets[bucket_index].remove_peer(to_replace)
                return await self.add_peer(peer, probe)