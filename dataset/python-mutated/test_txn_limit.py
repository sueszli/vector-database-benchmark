from twisted.test.proto_helpers import MemoryReactor
from synapse.server import HomeServer
from synapse.storage.types import Cursor
from synapse.util import Clock
from tests import unittest

class SQLTransactionLimitTestCase(unittest.HomeserverTestCase):
    """Test SQL transaction limit doesn't break transactions."""

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            while True:
                i = 10
        return self.setup_test_homeserver(db_txn_limit=1000)

    def test_config(self) -> None:
        if False:
            print('Hello World!')
        db_config = self.hs.config.database.get_single_database()
        self.assertEqual(db_config.config['txn_limit'], 1000)

    def test_select(self) -> None:
        if False:
            return 10

        def do_select(txn: Cursor) -> None:
            if False:
                for i in range(10):
                    print('nop')
            txn.execute('SELECT 1')
        db_pool = self.hs.get_datastores().databases[0]
        for _ in range(1001):
            self.get_success_or_raise(db_pool.runInteraction('test_select', do_select))