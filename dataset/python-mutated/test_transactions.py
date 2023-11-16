from twisted.test.proto_helpers import MemoryReactor
from synapse.server import HomeServer
from synapse.storage.databases.main.transactions import DestinationRetryTimings
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class TransactionStoreTestCase(HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, homeserver: HomeServer) -> None:
        if False:
            return 10
        self.store = homeserver.get_datastores().main

    def test_get_set_transactions(self) -> None:
        if False:
            return 10
        'Tests that we can successfully get a non-existent entry for\n        destination retries, as well as testing tht we can set and get\n        correctly.\n        '
        r = self.get_success(self.store.get_destination_retry_timings('example.com'))
        self.assertIsNone(r)
        self.get_success(self.store.set_destination_retry_timings('example.com', 1000, 50, 100))
        r = self.get_success(self.store.get_destination_retry_timings('example.com'))
        self.assertEqual(DestinationRetryTimings(retry_last_ts=50, retry_interval=100, failure_ts=1000), r)

    def test_initial_set_transactions(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that we can successfully set the destination retries (there\n        was a bug around invalidating the cache that broke this)\n        '
        d = self.store.set_destination_retry_timings('example.com', 1000, 50, 100)
        self.get_success(d)

    def test_large_destination_retry(self) -> None:
        if False:
            while True:
                i = 10
        max_retry_interval_ms = self.hs.config.federation.destination_max_retry_interval_ms
        d = self.store.set_destination_retry_timings('example.com', max_retry_interval_ms, max_retry_interval_ms, max_retry_interval_ms)
        self.get_success(d)
        d2 = self.store.get_destination_retry_timings('example.com')
        self.get_success(d2)