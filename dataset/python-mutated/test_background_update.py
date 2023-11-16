import logging
from typing import List, Tuple, cast
from unittest.mock import AsyncMock, Mock
import yaml
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.test.proto_helpers import MemoryReactor
from synapse.server import HomeServer
from synapse.storage.background_updates import BackgroundUpdater, ForeignKeyConstraint, NotNullConstraint, run_validate_constraint_and_delete_rows_schema_delta
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import PostgresEngine, Sqlite3Engine
from synapse.types import JsonDict
from synapse.util import Clock
from tests import unittest
from tests.unittest import override_config

class BackgroundUpdateTestCase(unittest.HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.updates: BackgroundUpdater = self.hs.get_datastores().main.db_pool.updates
        self.assertTrue(self.get_success(self.updates.has_completed_background_updates()))
        self.update_handler = Mock()
        self.updates.register_background_update_handler('test_update', self.update_handler)
        self.store = self.hs.get_datastores().main

    async def update(self, progress: JsonDict, count: int) -> int:
        duration_ms = 10
        await self.clock.sleep(count * duration_ms / 1000)
        progress = {'my_key': progress['my_key'] + 1}
        await self.store.db_pool.runInteraction('update_progress', self.updates._background_update_progress_txn, 'test_update', progress)
        return count

    def test_do_background_update(self) -> None:
        if False:
            print('Hello World!')
        duration_ms = 10
        target_background_update_duration_ms = 100
        self.get_success(self.store.db_pool.simple_insert('background_updates', values={'update_name': 'test_update', 'progress_json': '{"my_key": 1}'}))
        self.update_handler.side_effect = self.update
        self.update_handler.reset_mock()
        res = self.get_success(self.updates.do_next_background_update(False), by=0.02)
        self.assertFalse(res)
        self.update_handler.assert_called_once_with({'my_key': 1}, self.updates.default_background_batch_size)

        async def update(progress: JsonDict, count: int) -> int:
            self.assertEqual(progress, {'my_key': 2})
            self.assertAlmostEqual(count, target_background_update_duration_ms / duration_ms, places=0)
            await self.updates._end_background_update('test_update')
            return count
        self.update_handler.side_effect = update
        self.update_handler.reset_mock()
        result = self.get_success(self.updates.do_next_background_update(False))
        self.assertFalse(result)
        self.update_handler.assert_called_once()
        self.update_handler.reset_mock()
        result = self.get_success(self.updates.do_next_background_update(False))
        self.assertTrue(result)
        self.assertFalse(self.update_handler.called)

    @override_config(yaml.safe_load('\n            background_updates:\n                default_batch_size: 20\n            '))
    def test_background_update_default_batch_set_by_config(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test that the background update is run with the default_batch_size set by the config\n        '
        self.get_success(self.store.db_pool.simple_insert('background_updates', values={'update_name': 'test_update', 'progress_json': '{"my_key": 1}'}))
        self.update_handler.side_effect = self.update
        self.update_handler.reset_mock()
        res = self.get_success(self.updates.do_next_background_update(False), by=0.01)
        self.assertFalse(res)
        self.update_handler.assert_called_once_with({'my_key': 1}, 20)

    def test_background_update_default_sleep_behavior(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test default background update behavior, which is to sleep\n        '
        self.get_success(self.store.db_pool.simple_insert('background_updates', values={'update_name': 'test_update', 'progress_json': '{"my_key": 1}'}))
        self.update_handler.side_effect = self.update
        self.update_handler.reset_mock()
        self.updates.start_doing_background_updates()
        self.reactor.pump([0.5])
        self.update_handler.assert_not_called()
        self.reactor.pump([1])
        self.update_handler.assert_called()

    @override_config(yaml.safe_load('\n            background_updates:\n                sleep_duration_ms: 500\n            '))
    def test_background_update_sleep_set_in_config(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test that changing the sleep time in the config changes how long it sleeps\n        '
        self.get_success(self.store.db_pool.simple_insert('background_updates', values={'update_name': 'test_update', 'progress_json': '{"my_key": 1}'}))
        self.update_handler.side_effect = self.update
        self.update_handler.reset_mock()
        self.updates.start_doing_background_updates()
        self.reactor.pump([0.45])
        self.update_handler.assert_not_called()
        self.reactor.pump([0.75])
        self.update_handler.assert_called()

    @override_config(yaml.safe_load('\n            background_updates:\n                sleep_enabled: false\n            '))
    def test_disabling_background_update_sleep(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test that disabling sleep in the config results in bg update not sleeping\n        '
        self.get_success(self.store.db_pool.simple_insert('background_updates', values={'update_name': 'test_update', 'progress_json': '{"my_key": 1}'}))
        self.update_handler.side_effect = self.update
        self.update_handler.reset_mock()
        self.updates.start_doing_background_updates()
        self.reactor.pump([0.025])
        self.update_handler.assert_called()

    @override_config(yaml.safe_load('\n            background_updates:\n                background_update_duration_ms: 500\n            '))
    def test_background_update_duration_set_in_config(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test that the desired duration set in the config is used in determining batch size\n        '
        duration_ms = 10
        self.get_success(self.store.db_pool.simple_insert('background_updates', values={'update_name': 'test_update', 'progress_json': '{"my_key": 1}'}))
        self.update_handler.side_effect = self.update
        self.update_handler.reset_mock()
        res = self.get_success(self.updates.do_next_background_update(False), by=0.02)
        self.assertFalse(res)

        async def update(progress: JsonDict, count: int) -> int:
            self.assertEqual(progress, {'my_key': 2})
            self.assertAlmostEqual(count, 500 / duration_ms, places=0)
            await self.updates._end_background_update('test_update')
            return count
        self.update_handler.side_effect = update
        self.get_success(self.updates.do_next_background_update(False))

    @override_config(yaml.safe_load('\n            background_updates:\n                min_batch_size: 5\n            '))
    def test_background_update_min_batch_set_in_config(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test that the minimum batch size set in the config is used\n        '
        duration_ms = 50
        self.get_success(self.store.db_pool.simple_insert('background_updates', values={'update_name': 'test_update', 'progress_json': '{"my_key": 1}'}))

        async def update_long(progress: JsonDict, count: int) -> int:
            await self.clock.sleep(count * duration_ms / 1000)
            progress = {'my_key': progress['my_key'] + 1}
            await self.store.db_pool.runInteraction('update_progress', self.updates._background_update_progress_txn, 'test_update', progress)
            return count
        self.update_handler.side_effect = update_long
        self.update_handler.reset_mock()
        res = self.get_success(self.updates.do_next_background_update(False), by=1)
        self.assertFalse(res)

        async def update_short(progress: JsonDict, count: int) -> int:
            self.assertEqual(progress, {'my_key': 2})
            self.assertEqual(count, 5)
            await self.updates._end_background_update('test_update')
            return count
        self.update_handler.side_effect = update_short
        self.get_success(self.updates.do_next_background_update(False))

    def test_failed_update_logs_exception_details(self) -> None:
        if False:
            print('Hello World!')
        needle = 'RUH ROH RAGGY'

        def failing_update(progress: JsonDict, count: int) -> int:
            if False:
                return 10
            raise Exception(needle)
        self.update_handler.side_effect = failing_update
        self.update_handler.reset_mock()
        self.get_success(self.store.db_pool.simple_insert('background_updates', values={'update_name': 'test_update', 'progress_json': '{}'}))
        with self.assertLogs(level=logging.ERROR) as logs:
            self.get_failure(self.updates.run_background_updates(False), RuntimeError)
        self.assertTrue(any((needle in log for log in logs.output)), logs.output)

class BackgroundUpdateControllerTestCase(unittest.HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            while True:
                i = 10
        self.updates: BackgroundUpdater = self.hs.get_datastores().main.db_pool.updates
        self.assertTrue(self.get_success(self.updates.has_completed_background_updates()))
        self.update_deferred: Deferred[int] = Deferred()
        self.update_handler = Mock(return_value=self.update_deferred)
        self.updates.register_background_update_handler('test_update', self.update_handler)

        class MockCM:
            __aenter__ = AsyncMock(return_value=None)
            __aexit__ = AsyncMock(return_value=None)
        self._update_ctx_manager = MockCM
        self._on_update = Mock(return_value=self._update_ctx_manager())
        self._default_batch_size = 500
        self.hs.get_module_api().register_background_update_controller_callbacks(on_update=self._on_update, min_batch_size=AsyncMock(return_value=self._default_batch_size), default_batch_size=AsyncMock(return_value=self._default_batch_size))

    def test_controller(self) -> None:
        if False:
            while True:
                i = 10
        store = self.hs.get_datastores().main
        self.get_success(store.db_pool.simple_insert('background_updates', values={'update_name': 'test_update', 'progress_json': '{}'}))
        enter_defer: Deferred[int] = Deferred()
        self._update_ctx_manager.__aenter__ = Mock(return_value=enter_defer)
        do_update_d = ensureDeferred(self.updates.do_next_background_update(True))
        self.pump()
        self._on_update.assert_called_once_with('test_update', 'master', False)
        self.assertFalse(do_update_d.called)
        self.assertFalse(self.update_deferred.called)
        enter_defer.callback(100)
        self.pump()
        self.update_handler.assert_called_once_with({}, self._default_batch_size)
        self.assertFalse(self.update_deferred.called)
        self._update_ctx_manager.__aexit__.assert_not_called()
        self.update_deferred.callback(100)
        self.pump()
        self._update_ctx_manager.__aexit__.assert_called()
        self.get_success(do_update_d)

class BackgroundUpdateValidateConstraintTestCase(unittest.HomeserverTestCase):
    """Tests the validate contraint and delete background handlers."""

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.updates: BackgroundUpdater = self.hs.get_datastores().main.db_pool.updates
        self.assertTrue(self.get_success(self.updates.has_completed_background_updates()))
        self.store = self.hs.get_datastores().main

    def test_not_null_constraint(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests adding a not null constraint.'
        table_sql = '\n            CREATE TABLE test_constraint(\n                a INT PRIMARY KEY,\n                b INT\n            );\n        '
        self.get_success(self.store.db_pool.runInteraction('test_not_null_constraint', lambda txn: txn.execute(table_sql)))
        index_sql = 'CREATE INDEX test_index ON test_constraint(a)'
        self.get_success(self.store.db_pool.runInteraction('test_not_null_constraint', lambda txn: txn.execute(index_sql)))
        self.get_success(self.store.db_pool.simple_insert('test_constraint', {'a': 1, 'b': 1}))
        self.get_success(self.store.db_pool.simple_insert('test_constraint', {'a': 2, 'b': None}))
        self.get_success(self.store.db_pool.simple_insert('test_constraint', {'a': 3, 'b': 3}))
        table2_sqlite = '\n            CREATE TABLE test_constraint2(\n                a INT PRIMARY KEY,\n                b INT,\n                CONSTRAINT test_constraint_name CHECK (b is NOT NULL)\n            );\n        '

        def delta(txn: LoggingTransaction) -> None:
            if False:
                i = 10
                return i + 15
            run_validate_constraint_and_delete_rows_schema_delta(txn, ordering=1000, update_name='test_bg_update', table='test_constraint', constraint_name='test_constraint_name', constraint=NotNullConstraint('b'), sqlite_table_name='test_constraint2', sqlite_table_schema=table2_sqlite)
        self.get_success(self.store.db_pool.runInteraction('test_not_null_constraint', delta))
        if isinstance(self.store.database_engine, PostgresEngine):
            self.updates.register_background_validate_constraint_and_delete_rows('test_bg_update', table='test_constraint', constraint_name='test_constraint_name', constraint=NotNullConstraint('b'), unique_columns=['a'])
            self.store.db_pool.updates._all_done = False
            self.wait_for_background_updates()
        rows = cast(List[Tuple[int, int]], self.get_success(self.store.db_pool.simple_select_list(table='test_constraint', keyvalues={}, retcols=('a', 'b'))))
        self.assertCountEqual(rows, [(1, 1), (3, 3)])
        self.get_failure(self.store.db_pool.simple_insert('test_constraint', {'a': 2, 'b': None}), exc=self.store.database_engine.module.IntegrityError)
        if isinstance(self.store.database_engine, Sqlite3Engine):
            self.get_success(self.store.db_pool.simple_select_one_onecol(table='sqlite_master', keyvalues={'tbl_name': 'test_constraint'}, retcol='name'))

    def test_foreign_constraint(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests adding a not foreign key constraint.'
        base_sql = '\n            CREATE TABLE base_table(\n                b INT PRIMARY KEY\n            );\n        '
        table_sql = '\n            CREATE TABLE test_constraint(\n                a INT PRIMARY KEY,\n                b INT NOT NULL\n            );\n        '
        self.get_success(self.store.db_pool.runInteraction('test_foreign_key_constraint', lambda txn: txn.execute(base_sql)))
        self.get_success(self.store.db_pool.runInteraction('test_foreign_key_constraint', lambda txn: txn.execute(table_sql)))
        self.get_success(self.store.db_pool.simple_insert('base_table', {'b': 1}))
        self.get_success(self.store.db_pool.simple_insert('test_constraint', {'a': 1, 'b': 1}))
        self.get_success(self.store.db_pool.simple_insert('test_constraint', {'a': 2, 'b': 2}))
        self.get_success(self.store.db_pool.simple_insert('base_table', {'b': 3}))
        self.get_success(self.store.db_pool.simple_insert('test_constraint', {'a': 3, 'b': 3}))
        table2_sqlite = '\n            CREATE TABLE test_constraint2(\n                a INT PRIMARY KEY,\n                b INT NOT NULL,\n                CONSTRAINT test_constraint_name FOREIGN KEY (b) REFERENCES base_table (b)\n            );\n        '

        def delta(txn: LoggingTransaction) -> None:
            if False:
                print('Hello World!')
            run_validate_constraint_and_delete_rows_schema_delta(txn, ordering=1000, update_name='test_bg_update', table='test_constraint', constraint_name='test_constraint_name', constraint=ForeignKeyConstraint('base_table', [('b', 'b')], deferred=False), sqlite_table_name='test_constraint2', sqlite_table_schema=table2_sqlite)
        self.get_success(self.store.db_pool.runInteraction('test_foreign_key_constraint', delta))
        if isinstance(self.store.database_engine, PostgresEngine):
            self.updates.register_background_validate_constraint_and_delete_rows('test_bg_update', table='test_constraint', constraint_name='test_constraint_name', constraint=ForeignKeyConstraint('base_table', [('b', 'b')], deferred=False), unique_columns=['a'])
            self.store.db_pool.updates._all_done = False
            self.wait_for_background_updates()
        rows = cast(List[Tuple[int, int]], self.get_success(self.store.db_pool.simple_select_list(table='test_constraint', keyvalues={}, retcols=('a', 'b'))))
        self.assertCountEqual(rows, [(1, 1), (3, 3)])
        self.get_failure(self.store.db_pool.simple_insert('test_constraint', {'a': 2, 'b': 2}), exc=self.store.database_engine.module.IntegrityError)