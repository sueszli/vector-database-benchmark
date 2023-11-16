import itertools
import logging
import math
import operator
import random
import time
from collections import deque, Counter
from statistics import median
logger = logging.getLogger('golem.network.p2p.peerkeeper')
K = 16
CONCURRENCY = 3
K_SIZE = 512
PONG_TIMEOUT = 5
REQUEST_TIMEOUT = 10
IDLE_REFRESH = 3

class PeerKeeper(object):
    """ Keeps information about peers in a network"""

    def __init__(self, key, k_size=K_SIZE):
        if False:
            print('Hello World!')
        '\n        Create new peer keeper instance\n        :param hex key: hexadecimal representation of a this peer key\n        :param int k_size: pubkey size\n        '
        self.key = key
        self.key_num = int(key, 16)
        self.k = K
        self.concurrency = CONCURRENCY
        self.k_size = k_size
        self.buckets = [KBucket(0, 2 ** k_size, self.k)]
        self.pong_timeout = PONG_TIMEOUT
        self.request_timeout = REQUEST_TIMEOUT
        self.idle_refresh = IDLE_REFRESH
        self.sessions_to_end = []
        self.expected_pongs = {}
        self.find_requests = {}

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '\n'.join([str(bucket) for bucket in self.buckets])

    def restart(self, key):
        if False:
            for i in range(10):
                print('nop')
        "\n        Restart peer keeper after peer key has changed.\n        Remove all buckets and empty all queues.\n        :param hex key: hexadecimal representation of a peer's public key\n        "
        self.key = key
        self.key_num = int(key, 16)
        self.buckets = [KBucket(0, 2 ** self.k_size, self.k)]
        self.expected_pongs = {}
        self.find_requests = {}
        self.sessions_to_end = []

    def add_peer(self, peer_info):
        if False:
            i = 10
            return i + 15
        "\n        Try to add information about new peer. If it's possible just add it to\n        a proper bucket. Otherwise try to find a candidate to replace.\n        :param Node peer_info: information about a new peer\n        :return None|Node: None if peer has been added to a bucket or\n         if there is no candidate for replacement, otherwise return a candidate\n         to replacement.\n        "
        if peer_info.key == self.key:
            logger.warning('Trying to add self to Routing table')
            return
        key_num = int(peer_info.key, 16)
        bucket = self.bucket_for_peer(key_num)
        peer_to_remove = bucket.add_peer(peer_info)
        if peer_to_remove:
            if bucket.start <= self.key_num < bucket.end:
                self.split_bucket(bucket)
                return self.add_peer(peer_info)
            self.expected_pongs[peer_to_remove.key] = (peer_info, time.time())
            return peer_to_remove
        for bucket in self.buckets:
            logger.debug(str(bucket))
        return None

    def set_last_message_time(self, key):
        if False:
            print('Hello World!')
        '\n        Set current time as a last message time for a bucket which range\n        contain given key.\n        :param hex key: some peer public key in hexadecimal format\n        '
        if not key:
            return
        if isinstance(key, str):
            key = key.encode()
        for (i, bucket) in enumerate(self.buckets):
            if bucket.start <= int(key.hex(), 16) < bucket.end:
                self.buckets[i].last_updated = time.time()
                break

    def get_random_known_peer(self):
        if False:
            print('Hello World!')
        ' Return random peer from any bucket\n        :return Node|None: information about random peer\n        '
        bucket = self.buckets[random.randint(0, len(self.buckets) - 1)]
        if bucket.peers:
            return bucket.peers[random.randint(0, len(bucket.peers) - 1)]
        return None

    def pong_received(self, key):
        if False:
            i = 10
            return i + 15
        '\n        React to the fact that pong message was received from peer\n        with given key\n        :param hex key: public key of a node that has send pong message\n        '
        if key in self.expected_pongs:
            del self.expected_pongs[key]

    def bucket_for_peer(self, key_num):
        if False:
            print('Hello World!')
        "\n        Find a bucket which contains given num in it's range\n        :param long key_num: key long representation for which a bucket\n         should be found\n        :return KBucket: bucket containing key in it's range\n        "
        for bucket in self.buckets:
            if bucket.start <= key_num < bucket.end:
                return bucket
        logger.error('Did not find a bucket for {}'.format(key_num))

    def split_bucket(self, bucket):
        if False:
            i = 10
            return i + 15
        ' Split given bucket into two buckets\n        :param KBucket bucket: bucket to be split\n        '
        logger.debug('Splitting bucket')
        (buck1, buck2) = bucket.split()
        idx = self.buckets.index(bucket)
        self.buckets[idx] = buck1
        self.buckets.insert(idx + 1, buck2)

    def cnt_distance(self, key):
        if False:
            return 10
        '\n        Return distance between this peer and peer with a given key.\n        Distance is a xor between keys.\n        :param hex key: other peer public key\n        :return long: distance to other peer\n        '
        return self.key_num ^ int(key, 16)

    def sync(self):
        if False:
            print('Hello World!')
        "\n        Sync peer keeper state. Remove old requests and expected pongs,\n        add new peers if old peers didn't answer to ping. Additionally prepare a\n        list of peers that should be found to correctly fill the buckets.\n        :return dict: information about peers that should be found (key and list\n          of closest known neighbours)\n        "
        self.__remove_old_expected_pongs()
        self.__remove_old_requests()
        peers_to_find = self.__send_new_requests()
        return peers_to_find

    def neighbours(self, key_num, alpha=None):
        if False:
            while True:
                i = 10
        '\n        Return alpha nearest known neighbours to a peer with given key\n        :param long key_num: given key in a long format\n        :param None|int alpha: *Default: None* number of neighbours to find.\n         If alpha is set to None then\n        default concurrency parameter will be used\n        :return list: list of nearest known neighbours\n        '
        if not alpha:
            alpha = self.concurrency

        def gen_neigh():
            if False:
                return 10
            for bucket in self.buckets_by_id_distance(key_num):
                for peer in bucket.peers_by_id_distance(key_num):
                    if int(peer.key, 16) != key_num:
                        yield peer
        return list(itertools.islice(gen_neigh(), alpha))

    def buckets_by_id_distance(self, key_num):
        if False:
            return 10
        '\n        Return list of buckets sorted by distance from given key.\n        Bucket middle range element will be taken into account\n        :param long key_num: given key in long format\n        :return list: sorted buckets list\n        '
        return sorted(self.buckets, key=operator.methodcaller('id_distance', key_num))

    def get_estimated_network_size(self) -> int:
        if False:
            print('Hello World!')
        '\n        Get estimated network size\n        Based on https://gnunet.org/bartmsthesis p. 55\n        '

        def depth(peer):
            if False:
                while True:
                    i = 10
            " Get peer 'depth' i.e. number of common leading digits in binary\n            representations of peer's key and own key which is equivalent to the\n            position of the first '1' in (peer_key XOR own_key)"
            return self.k_size - int(math.log2(node_id_distance(peer, self.key_num))) - 1

        def filter_outliers(data, m=2.0):
            if False:
                i = 10
                return i + 15
            ' Simple median-based outlier detection '
            med = median(data)
            distance = [abs(x - med) for x in data]
            med_dist = median(distance)
            norm_distance = [d / med_dist for d in distance] if med_dist else [0] * len(data)
            return (x for (x, d) in zip(data, norm_distance) if d < m)
        peers_depths = [depth(p) for b in self.buckets for p in b.peers]
        logical_buckets = Counter(peers_depths)
        if not logical_buckets:
            return 0
        data = [num_peers * 2 ** (depth + 1) for (depth, num_peers) in logical_buckets.items() if num_peers < self.k]
        if not data:
            return 0
        return median(filter_outliers(data, m=2))

    def __remove_old_expected_pongs(self):
        if False:
            print('Hello World!')
        cur_time = time.time()
        for (key, (replacement, time_)) in list(self.expected_pongs.items()):
            key_num = int(key, 16)
            if cur_time - time_ > self.pong_timeout:
                peer_info = self.bucket_for_peer(key_num).remove_peer(key_num)
                if peer_info:
                    self.sessions_to_end.append(peer_info)
                if replacement:
                    self.add_peer(replacement)
                del self.expected_pongs[key]

    def __send_new_requests(self):
        if False:
            print('Hello World!')
        peers_to_find = {}
        cur_time = time.time()
        for bucket in self.buckets:
            if cur_time - bucket.last_updated > self.idle_refresh:
                key_num = random.randint(bucket.start, bucket.end - 1)
                self.find_requests[key_num] = cur_time
                peers_to_find[key_num] = self.neighbours(key_num)
                bucket.last_updated = cur_time
        return peers_to_find

    def __remove_old_requests(self):
        if False:
            i = 10
            return i + 15
        cur_time = time.time()
        for (key_num, _) in list(self.find_requests.items()):
            if cur_time - time.time() > self.request_timeout:
                del self.find_requests[key_num]

def node_id_distance(node_info, key_num):
    if False:
        while True:
            i = 10
    '\n    Return distance in XOR metrics between two peers when we have full\n    information about one node and only public key of a second node\n    :param Node node_info: information about node (peer)\n    :param long key_num: other node public key in long format\n    :return long: distance between two peers\n    '
    return int(node_info.key, 16) ^ key_num

def key_distance(key, second_key):
    if False:
        while True:
            i = 10
    return int(key, 16) ^ int(second_key, 16)

class KBucket(object):
    """
    K-bucket for keeping information about peers from a given distance range
    """

    def __init__(self, start, end, k):
        if False:
            print('Hello World!')
        ' Create new bucket with range [start, end)\n        :param long start: bucket range start\n        :param long end: bucket range end\n        :param int k: bucket size\n        '
        self.start = start
        self.end = end
        self.k = k
        self.peers = deque()
        self.last_updated = time.time()

    def add_peer(self, peer):
        if False:
            for i in range(10):
                print('nop')
        "\n        Try to append peer to a bucket. If it's already in a bucket remove it\n        and append it at the end. If a bucket is full then return oldest peer in\n        a bucket as a candidate for replacement\n        :param Node peer: peer to add\n        :return Node|None: oldest peer in a bucket, if a new peer hasn't been\n         added or None otherwise\n        "
        logger.debug('KBucket adding peer {}'.format(peer))
        self.last_updated = time.time()
        old_peer = None
        for p in self.peers:
            if p.key == peer.key:
                old_peer = p
                break
        if old_peer:
            self.peers.remove(old_peer)
            self.peers.append(peer)
        elif len(self.peers) < self.k:
            self.peers.append(peer)
        else:
            return self.peers[0]
        return None

    def remove_peer(self, key_num):
        if False:
            i = 10
            return i + 15
        '\n        Remove peer with given key from this bucket\n        :param long key_num: public key of a node that should be removed from\n         this bucket in long format\n        :return Node|None: information about peer if it was in this bucket,\n         None otherwise\n        '
        for peer in self.peers:
            if int(peer.key, 16) == key_num:
                self.peers.remove(peer)
                return peer
        return None

    def id_distance(self, key_num):
        if False:
            return 10
        ' Return distance from a middle of a bucket range to a given key\n        :param long key_num:  other node public key in long format\n        :return long: distance from a middle of this bucket to a given key\n        '
        return math.floor((self.start + self.end) / 2) ^ key_num

    def peers_by_id_distance(self, key_num):
        if False:
            while True:
                i = 10
        return sorted(self.peers, key=lambda p: node_id_distance(p, key_num))

    def split(self):
        if False:
            while True:
                i = 10
        ' Split bucket into two buckets\n        :return (KBucket, KBucket): two buckets that were created from this\n         bucket\n        '
        midpoint = (self.start + self.end) / 2
        lower = KBucket(self.start, midpoint, self.k)
        upper = KBucket(midpoint, self.end, self.k)
        for peer in self.peers:
            if int(peer.key, 16) < midpoint:
                lower.add_peer(peer)
            else:
                upper.add_peer(peer)
        return (lower, upper)

    @property
    def num_peers(self) -> int:
        if False:
            print('Hello World!')
        return len(self.peers)

    def __str__(self):
        if False:
            print('Hello World!')
        return 'Bucket: {} - {} peers {}'.format(self.start, self.end, len(self.peers))