import secrets
from typing import Generator, List, Tuple, cast
from twisted.test.proto_helpers import MemoryReactor
from synapse.server import HomeServer
from synapse.util import Clock
from tests import unittest

class UpdateUpsertManyTests(unittest.HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            while True:
                i = 10
        self.storage = hs.get_datastores().main
        self.table_name = 'table_' + secrets.token_hex(6)
        self.get_success(self.storage.db_pool.runInteraction('create', lambda x, *a: x.execute(*a), 'CREATE TABLE %s (id INTEGER, username TEXT, value TEXT)' % (self.table_name,)))
        self.get_success(self.storage.db_pool.runInteraction('index', lambda x, *a: x.execute(*a), 'CREATE UNIQUE INDEX %sindex ON %s(id, username)' % (self.table_name, self.table_name)))

    def _dump_table_to_tuple(self) -> Generator[Tuple[int, str, str], None, None]:
        if False:
            for i in range(10):
                print('nop')
        yield from cast(List[Tuple[int, str, str]], self.get_success(self.storage.db_pool.simple_select_list(self.table_name, None, ['id, username, value'])))

    def test_upsert_many(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Upsert_many will perform the upsert operation across a batch of data.\n        '
        key_names = ['id', 'username']
        value_names = ['value']
        key_values = [[1, 'user1'], [2, 'user2']]
        value_values = [['hello'], ['there']]
        self.get_success(self.storage.db_pool.runInteraction('test', self.storage.db_pool.simple_upsert_many_txn, self.table_name, key_names, key_values, value_names, value_values))
        self.assertEqual(set(self._dump_table_to_tuple()), {(1, 'user1', 'hello'), (2, 'user2', 'there')})
        key_values = [[2, 'user2']]
        value_values = [['bleb']]
        self.get_success(self.storage.db_pool.runInteraction('test', self.storage.db_pool.simple_upsert_many_txn, self.table_name, key_names, key_values, value_names, value_values))
        self.assertEqual(set(self._dump_table_to_tuple()), {(1, 'user1', 'hello'), (2, 'user2', 'bleb')})

    def test_simple_update_many(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        simple_update_many performs many updates at once.\n        '
        self.get_success(self.storage.db_pool.simple_insert_many(table=self.table_name, keys=('id', 'username', 'value'), values=[(1, 'alice', 'A'), (2, 'bob', 'B'), (3, 'charlie', 'C')], desc='insert'))
        self.assertEqual(set(self._dump_table_to_tuple()), {(1, 'alice', 'A'), (2, 'bob', 'B'), (3, 'charlie', 'C')})
        self.get_success(self.storage.db_pool.simple_update_many(table=self.table_name, key_names=('username',), key_values=(('alice',), ('bob',), ('stranger',)), value_names=('value',), value_values=(('aaa!',), ('bbb!',), ('???',)), desc='update_many1'))
        self.assertEqual(set(self._dump_table_to_tuple()), {(1, 'alice', 'aaa!'), (2, 'bob', 'bbb!'), (3, 'charlie', 'C')})