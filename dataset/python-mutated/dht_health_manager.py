import math
from asyncio import Future
from typing import Awaitable
from ipv8.taskmanager import TaskManager
from tribler.core.components.libtorrent.utils.libtorrent_helper import libtorrent as lt
from tribler.core.components.torrent_checker.torrent_checker.dataclasses import HealthInfo
from tribler.core.utilities.unicode import hexlify

class DHTHealthManager(TaskManager):
    """
    This class manages BEP33 health requests to the libtorrent DHT.
    """

    def __init__(self, lt_session):
        if False:
            print('Hello World!')
        '\n        Initialize the DHT health manager.\n        :param lt_session: The session used to perform health lookups.\n        '
        TaskManager.__init__(self)
        self.lookup_futures = {}
        self.bf_seeders = {}
        self.bf_peers = {}
        self.outstanding = {}
        self.lt_session = lt_session

    def get_health(self, infohash, timeout=15) -> Awaitable[HealthInfo]:
        if False:
            while True:
                i = 10
        '\n        Lookup the health of a given infohash.\n        :param infohash: The 20-byte infohash to lookup.\n        :param timeout: The timeout of the lookup.\n        '
        if infohash in self.lookup_futures:
            return self.lookup_futures[infohash]
        lookup_future = Future()
        self.lookup_futures[infohash] = lookup_future
        self.bf_seeders[infohash] = bytearray(256)
        self.bf_peers[infohash] = bytearray(256)
        self.lt_session.dht_get_peers(lt.sha1_hash(bytes(infohash)))
        self.register_task(f'lookup_{hexlify(infohash)}', self.finalize_lookup, infohash, delay=timeout)
        return lookup_future

    def finalize_lookup(self, infohash):
        if False:
            return 10
        '\n        Finalize the lookup of the provided infohash and invoke the appropriate deferred.\n        :param infohash: The infohash of the lookup we finialize.\n        '
        for transaction_id in [key for (key, value) in self.outstanding.items() if value == infohash]:
            self.outstanding.pop(transaction_id, None)
        if infohash not in self.lookup_futures:
            return
        bf_seeders = self.bf_seeders.pop(infohash)
        bf_peers = self.bf_peers.pop(infohash)
        seeders = DHTHealthManager.get_size_from_bloomfilter(bf_seeders)
        peers = DHTHealthManager.get_size_from_bloomfilter(bf_peers)
        if not self.lookup_futures[infohash].done():
            health = HealthInfo(infohash, seeders=seeders, leechers=peers)
            self.lookup_futures[infohash].set_result(health)
        self.lookup_futures.pop(infohash, None)

    @staticmethod
    def combine_bloomfilters(bf1, bf2):
        if False:
            return 10
        '\n        Combine two given bloom filters by ORing the bits.\n        :param bf1: The first bloom filter to combine.\n        :param bf2: The second bloom filter to combine.\n        :return: A bytearray with the combined bloomfilter.\n        '
        final_bf_len = min(len(bf1), len(bf2))
        final_bf = bytearray(final_bf_len)
        for bf_index in range(final_bf_len):
            final_bf[bf_index] = bf1[bf_index] | bf2[bf_index]
        return final_bf

    @staticmethod
    def get_size_from_bloomfilter(bf):
        if False:
            print('Hello World!')
        '\n        Return the estimated number of items in the bloom filter.\n        :param bf: The bloom filter of which we estimate the size.\n        :return: A rounded integer, approximating the number of items in the filter.\n        '

        def tobits(s):
            if False:
                return 10
            result = []
            for c in s:
                num = ord(c) if isinstance(c, str) else c
                bits = bin(num)[2:]
                bits = '00000000'[len(bits):] + bits
                result.extend([int(b) for b in bits])
            return result
        bits_array = tobits(bytes(bf))
        total_zeros = 0
        for bit in bits_array:
            if bit == 0:
                total_zeros += 1
        if total_zeros == 0:
            return 6000
        m = 256 * 8
        c = min(m - 1, total_zeros)
        return int(math.log(c / float(m)) / (2 * math.log(1 - 1 / float(m))))

    def requesting_bloomfilters(self, transaction_id, infohash):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tne libtorrent DHT has sent a get_peers query for an infohash we may be interested in.\n        If so, keep track of the transaction and node IDs.\n        :param transaction_id: The ID of the query\n        :param infohash: The infohash for which the query was sent.\n        '
        if infohash in self.lookup_futures:
            self.outstanding[transaction_id] = infohash
        elif transaction_id in self.outstanding:
            self.outstanding.pop(transaction_id, None)

    def received_bloomfilters(self, transaction_id, bf_seeds=bytearray(256), bf_peers=bytearray(256)):
        if False:
            return 10
        '\n        We have received bloom filters from the libtorrent DHT. Register the bloom filters and process them.\n        :param transaction_id: The ID of the query for which we are receiving the bloom filter.\n        :param bf_seeds: The bloom filter indicating the IP addresses of the seeders.\n        :param bf_peers: The bloom filter indicating the IP addresses of the peers (leechers).\n        '
        infohash = self.outstanding.get(transaction_id)
        if not infohash:
            self._logger.info('Could not find lookup infohash for incoming BEP33 bloomfilters')
            return
        self.bf_seeders[infohash] = DHTHealthManager.combine_bloomfilters(self.bf_seeders[infohash], bf_seeds)
        self.bf_peers[infohash] = DHTHealthManager.combine_bloomfilters(self.bf_peers[infohash], bf_peers)