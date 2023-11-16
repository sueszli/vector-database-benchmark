from typing import Any, Dict, Optional, Sequence, Tuple
from twisted.test.proto_helpers import MemoryReactor
from synapse.rest import admin
from synapse.rest.client import login, room
from synapse.server import HomeServer
from synapse.storage.database import LoggingTransaction
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

class ReceiptsBackgroundUpdateStoreTestCase(HomeserverTestCase):
    servlets = [admin.register_servlets, room.register_servlets, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            while True:
                i = 10
        self.store = hs.get_datastores().main
        self.user_id = self.register_user('foo', 'pass')
        self.token = self.login('foo', 'pass')
        self.room_id = self.helper.create_room_as(self.user_id, tok=self.token)
        self.other_room_id = self.helper.create_room_as(self.user_id, tok=self.token)

    def _test_background_receipts_unique_index(self, update_name: str, index_name: str, table: str, receipts: Dict[Tuple[str, str, str], Sequence[Dict[str, Any]]], expected_unique_receipts: Dict[Tuple[str, str, str], Optional[Dict[str, Any]]]) -> None:
        if False:
            print('Hello World!')
        'Test that the background update to uniqueify non-thread receipts in\n        the given receipts table works properly.\n\n        Args:\n            update_name: The name of the background update to test.\n            index_name: The name of the index that the background update creates.\n            table: The table of receipts that the background update fixes.\n            receipts: The test data containing duplicate receipts.\n                A list of receipt rows to insert, grouped by\n                `(room_id, receipt_type, user_id)`.\n            expected_unique_receipts: A dictionary of `(room_id, receipt_type, user_id)`\n                keys and expected receipt key-values after duplicate receipts have been\n                removed.\n        '

        def drop_receipts_unique_index(txn: LoggingTransaction) -> None:
            if False:
                while True:
                    i = 10
            txn.execute(f'DROP INDEX IF EXISTS {index_name}')
        self.get_success(self.store.db_pool.runInteraction('drop_receipts_unique_index', drop_receipts_unique_index))
        for ((room_id, receipt_type, user_id), rows) in receipts.items():
            for row in rows:
                self.get_success(self.store.db_pool.simple_insert(table, {'room_id': room_id, 'receipt_type': receipt_type, 'user_id': user_id, 'thread_id': None, 'data': '{}', **row}))
        self.get_success(self.store.db_pool.simple_insert('background_updates', {'update_name': update_name, 'progress_json': '{}'}))
        self.store.db_pool.updates._all_done = False
        self.wait_for_background_updates()
        for ((room_id, receipt_type, user_id), expected_row) in expected_unique_receipts.items():
            columns = ['room_id', 'receipt_type', 'user_id']
            if expected_row is not None:
                columns += expected_row.keys()
            row_tuples = self.get_success(self.store.db_pool.simple_select_list(table=table, keyvalues={'room_id': room_id, 'receipt_type': receipt_type, 'user_id': user_id}, retcols=columns, desc='get_receipt'))
            if expected_row is not None:
                self.assertEqual(len(row_tuples), 1, f'Background update did not leave behind latest receipt in {table}')
                self.assertEqual(row_tuples[0], (room_id, receipt_type, user_id, *expected_row.values()))
            else:
                self.assertEqual(len(row_tuples), 0, f'Background update did not remove all duplicate receipts from {table}')

    def test_background_receipts_linearized_unique_index(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that the background update to uniqueify non-thread receipts in\n        `receipts_linearized` works properly.\n        '
        self._test_background_receipts_unique_index('receipts_linearized_unique_index', 'receipts_linearized_unique_index', 'receipts_linearized', receipts={(self.room_id, 'm.read', self.user_id): [{'stream_id': 5, 'event_id': '$some_event'}, {'stream_id': 6, 'event_id': '$some_event'}], (self.other_room_id, 'm.read', self.user_id): [{'stream_id': 7, 'event_id': '$some_event'}, {'stream_id': 7, 'event_id': '$some_event'}]}, expected_unique_receipts={(self.room_id, 'm.read', self.user_id): {'stream_id': 6}, (self.other_room_id, 'm.read', self.user_id): {'stream_id': 7}})

    def test_background_receipts_graph_unique_index(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that the background update to uniqueify non-thread receipts in\n        `receipts_graph` works properly.\n        '
        self._test_background_receipts_unique_index('receipts_graph_unique_index', 'receipts_graph_unique_index', 'receipts_graph', receipts={(self.room_id, 'm.read', self.user_id): [{'event_ids': '["$some_event"]'}, {'event_ids': '["$some_event"]'}], (self.other_room_id, 'm.read', self.user_id): [{'event_ids': '["$some_event"]'}]}, expected_unique_receipts={(self.room_id, 'm.read', self.user_id): None, (self.other_room_id, 'm.read', self.user_id): {'event_ids': '["$some_event"]'}})