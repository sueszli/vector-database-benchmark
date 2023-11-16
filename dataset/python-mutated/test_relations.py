from twisted.test.proto_helpers import MemoryReactor
from synapse.api.constants import MAIN_TIMELINE
from synapse.server import HomeServer
from synapse.util import Clock
from tests import unittest

class RelationsStoreTestCase(unittest.HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        '\n        Creates a DAG:\n\n            A <---[m.thread]-- B <--[m.annotation]-- C\n            ^\n            |--[m.reference]-- D <--[m.annotation]-- E\n\n            F <--[m.annotation]-- G\n\n        '
        self._main_store = self.hs.get_datastores().main
        self._create_relation('A', 'B', 'm.thread')
        self._create_relation('B', 'C', 'm.annotation')
        self._create_relation('A', 'D', 'm.reference')
        self._create_relation('D', 'E', 'm.annotation')
        self._create_relation('F', 'G', 'm.annotation')

    def _create_relation(self, parent_id: str, event_id: str, rel_type: str) -> None:
        if False:
            print('Hello World!')
        self.get_success(self._main_store.db_pool.simple_insert(table='event_relations', values={'event_id': event_id, 'relates_to_id': parent_id, 'relation_type': rel_type}))

    def test_get_thread_id(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Ensure that get_thread_id only searches up the tree for threads.\n        '
        thread_id = self.get_success(self._main_store.get_thread_id('B'))
        self.assertEqual('A', thread_id)
        thread_id = self.get_success(self._main_store.get_thread_id('C'))
        self.assertEqual('A', thread_id)
        thread_id = self.get_success(self._main_store.get_thread_id('A'))
        self.assertEqual(MAIN_TIMELINE, thread_id)
        thread_id = self.get_success(self._main_store.get_thread_id('D'))
        self.assertEqual(MAIN_TIMELINE, thread_id)
        thread_id = self.get_success(self._main_store.get_thread_id('E'))
        self.assertEqual(MAIN_TIMELINE, thread_id)
        thread_id = self.get_success(self._main_store.get_thread_id('F'))
        self.assertEqual(MAIN_TIMELINE, thread_id)
        thread_id = self.get_success(self._main_store.get_thread_id('G'))
        self.assertEqual(MAIN_TIMELINE, thread_id)

    def test_get_thread_id_for_receipts(self) -> None:
        if False:
            print('Hello World!')
        '\n        Ensure that get_thread_id_for_receipts searches up and down the tree for a thread.\n        '
        thread_id = self.get_success(self._main_store.get_thread_id_for_receipts('A'))
        self.assertEqual('A', thread_id)
        thread_id = self.get_success(self._main_store.get_thread_id_for_receipts('B'))
        self.assertEqual('A', thread_id)
        thread_id = self.get_success(self._main_store.get_thread_id_for_receipts('C'))
        self.assertEqual('A', thread_id)
        thread_id = self.get_success(self._main_store.get_thread_id_for_receipts('D'))
        self.assertEqual('A', thread_id)
        thread_id = self.get_success(self._main_store.get_thread_id_for_receipts('E'))
        self.assertEqual('A', thread_id)
        thread_id = self.get_success(self._main_store.get_thread_id('F'))
        self.assertEqual(MAIN_TIMELINE, thread_id)
        thread_id = self.get_success(self._main_store.get_thread_id('G'))
        self.assertEqual(MAIN_TIMELINE, thread_id)