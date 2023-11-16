from typing import List, Optional, Tuple
from twisted.test.proto_helpers import MemoryReactor
from synapse.server import HomeServer
from synapse.storage._base import db_to_json
from synapse.storage.database import LoggingTransaction
from synapse.types import JsonDict
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class EndToEndKeyWorkerStoreTestCase(HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            while True:
                i = 10
        self.store = hs.get_datastores().main

    def test_get_master_cross_signing_key_updatable_before(self) -> None:
        if False:
            i = 10
            return i + 15
        alice = '@alice:test'
        (exists, timestamp) = self.get_success(self.store.get_master_cross_signing_key_updatable_before(alice))
        self.assertIs(exists, False)
        self.assertIsNone(timestamp)
        dummy_key = {'keys': {'a': 'b'}}
        self.get_success(self.store.set_e2e_cross_signing_key(alice, 'master', dummy_key))
        (exists, timestamp) = self.get_success(self.store.get_master_cross_signing_key_updatable_before(alice))
        self.assertIs(exists, True)
        self.assertIsNone(timestamp)
        written_timestamp = self.get_success(self.store.allow_master_cross_signing_key_replacement_without_uia(alice, 1000))
        (exists, timestamp) = self.get_success(self.store.get_master_cross_signing_key_updatable_before(alice))
        self.assertIs(exists, True)
        self.assertEqual(timestamp, written_timestamp)

    def test_master_replacement_only_applies_to_latest_master_key(self) -> None:
        if False:
            return 10
        "We shouldn't allow updates w/o UIA to old master keys or other key types."
        alice = '@alice:test'
        key1 = {'keys': {'a': 'b'}}
        key2 = {'keys': {'c': 'd'}}
        key3 = {'keys': {'e': 'f'}}
        self.get_success(self.store.set_e2e_cross_signing_key(alice, 'master', key1))
        self.get_success(self.store.set_e2e_cross_signing_key(alice, 'other', key2))
        self.get_success(self.store.set_e2e_cross_signing_key(alice, 'master', key3))
        key = self.get_success(self.store.get_e2e_cross_signing_key(alice, 'master', alice))
        self.assertEqual(key, key3)
        timestamp = self.get_success(self.store.allow_master_cross_signing_key_replacement_without_uia(alice, 1000))
        assert timestamp is not None

        def check_timestamp_column(txn: LoggingTransaction) -> List[Tuple[JsonDict, Optional[int]]]:
            if False:
                return 10
            "Fetch all rows for Alice's keys."
            txn.execute('\n                SELECT keydata, updatable_without_uia_before_ms\n                FROM e2e_cross_signing_keys\n                WHERE user_id = ?\n                ORDER BY stream_id ASC;\n            ', (alice,))
            return [(db_to_json(keydata), ts) for (keydata, ts) in txn.fetchall()]
        values = self.get_success(self.store.db_pool.runInteraction('check_timestamp_column', check_timestamp_column))
        self.assertEqual(values, [(key1, None), (key2, None), (key3, timestamp)])