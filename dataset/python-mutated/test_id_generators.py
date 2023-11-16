from typing import List, Optional
from twisted.test.proto_helpers import MemoryReactor
from synapse.server import HomeServer
from synapse.storage.database import DatabasePool, LoggingDatabaseConnection, LoggingTransaction
from synapse.storage.engines import IncorrectDatabaseSetup
from synapse.storage.types import Cursor
from synapse.storage.util.id_generators import MultiWriterIdGenerator, StreamIdGenerator
from synapse.util import Clock
from tests.unittest import HomeserverTestCase
from tests.utils import USE_POSTGRES_FOR_TESTS

class StreamIdGeneratorTestCase(HomeserverTestCase):

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            return 10
        self.store = hs.get_datastores().main
        self.db_pool: DatabasePool = self.store.db_pool
        self.get_success(self.db_pool.runInteraction('_setup_db', self._setup_db))

    def _setup_db(self, txn: LoggingTransaction) -> None:
        if False:
            i = 10
            return i + 15
        txn.execute('\n            CREATE TABLE foobar (\n                stream_id BIGINT NOT NULL,\n                data TEXT\n            );\n            ')
        txn.execute("INSERT INTO foobar VALUES (123, 'hello world');")

    def _create_id_generator(self) -> StreamIdGenerator:
        if False:
            print('Hello World!')

        def _create(conn: LoggingDatabaseConnection) -> StreamIdGenerator:
            if False:
                for i in range(10):
                    print('nop')
            return StreamIdGenerator(db_conn=conn, notifier=self.hs.get_replication_notifier(), table='foobar', column='stream_id')
        return self.get_success_or_raise(self.db_pool.runWithConnection(_create))

    def test_initial_value(self) -> None:
        if False:
            return 10
        'Check that we read the current token from the DB.'
        id_gen = self._create_id_generator()
        self.assertEqual(id_gen.get_current_token(), 123)

    def test_single_gen_next(self) -> None:
        if False:
            while True:
                i = 10
        'Check that we correctly increment the current token from the DB.'
        id_gen = self._create_id_generator()

        async def test_gen_next() -> None:
            async with id_gen.get_next() as next_id:
                self.assertEqual(id_gen.get_current_token(), 123)
                self.assertEqual(next_id, 124)
            self.assertEqual(id_gen.get_current_token(), 124)
        self.get_success(test_gen_next())

    def test_multiple_gen_nexts(self) -> None:
        if False:
            print('Hello World!')
        'Check that we handle overlapping calls to gen_next sensibly.'
        id_gen = self._create_id_generator()

        async def test_gen_next() -> None:
            ctx1 = id_gen.get_next()
            ctx2 = id_gen.get_next()
            ctx3 = id_gen.get_next()
            self.assertEqual(await ctx1.__aenter__(), 124)
            self.assertEqual(await ctx2.__aenter__(), 125)
            self.assertEqual(await ctx3.__aenter__(), 126)
            self.assertEqual(id_gen.get_current_token(), 123)
            await ctx1.__aexit__(None, None, None)
            self.assertEqual(id_gen.get_current_token(), 124)
            await ctx2.__aexit__(None, None, None)
            self.assertEqual(id_gen.get_current_token(), 125)
            await ctx3.__aexit__(None, None, None)
            self.assertEqual(id_gen.get_current_token(), 126)
        self.get_success(test_gen_next())

    def test_multiple_gen_nexts_closed_in_different_order(self) -> None:
        if False:
            i = 10
            return i + 15
        'Check that we handle overlapping calls to gen_next, even when their IDs\n        created and persisted in different orders.'
        id_gen = self._create_id_generator()

        async def test_gen_next() -> None:
            ctx1 = id_gen.get_next()
            ctx2 = id_gen.get_next()
            ctx3 = id_gen.get_next()
            self.assertEqual(await ctx1.__aenter__(), 124)
            self.assertEqual(await ctx2.__aenter__(), 125)
            self.assertEqual(await ctx3.__aenter__(), 126)
            self.assertEqual(id_gen.get_current_token(), 123)
            await ctx3.__aexit__(None, None, None)
            self.assertEqual(id_gen.get_current_token(), 123)
            await ctx1.__aexit__(None, None, None)
            self.assertEqual(id_gen.get_current_token(), 124)
            await ctx2.__aexit__(None, None, None)
            self.assertEqual(id_gen.get_current_token(), 126)
        self.get_success(test_gen_next())

    def test_gen_next_while_still_waiting_for_persistence(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Check that we handle overlapping calls to gen_next.'
        id_gen = self._create_id_generator()

        async def test_gen_next() -> None:
            ctx1 = id_gen.get_next()
            ctx2 = id_gen.get_next()
            ctx3 = id_gen.get_next()
            self.assertEqual(await ctx1.__aenter__(), 124)
            self.assertEqual(await ctx2.__aenter__(), 125)
            await ctx2.__aexit__(None, None, None)
            self.assertEqual(id_gen.get_current_token(), 123)
            self.assertEqual(await ctx3.__aenter__(), 126)
        self.get_success(test_gen_next())

class MultiWriterIdGeneratorTestCase(HomeserverTestCase):
    if not USE_POSTGRES_FOR_TESTS:
        skip = 'Requires Postgres'

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            i = 10
            return i + 15
        self.store = hs.get_datastores().main
        self.db_pool: DatabasePool = self.store.db_pool
        self.get_success(self.db_pool.runInteraction('_setup_db', self._setup_db))

    def _setup_db(self, txn: LoggingTransaction) -> None:
        if False:
            i = 10
            return i + 15
        txn.execute('CREATE SEQUENCE foobar_seq')
        txn.execute('\n            CREATE TABLE foobar (\n                stream_id BIGINT NOT NULL,\n                instance_name TEXT NOT NULL,\n                data TEXT\n            );\n            ')

    def _create_id_generator(self, instance_name: str='master', writers: Optional[List[str]]=None) -> MultiWriterIdGenerator:
        if False:
            for i in range(10):
                print('nop')

        def _create(conn: LoggingDatabaseConnection) -> MultiWriterIdGenerator:
            if False:
                while True:
                    i = 10
            return MultiWriterIdGenerator(conn, self.db_pool, notifier=self.hs.get_replication_notifier(), stream_name='test_stream', instance_name=instance_name, tables=[('foobar', 'instance_name', 'stream_id')], sequence_name='foobar_seq', writers=writers or ['master'])
        return self.get_success_or_raise(self.db_pool.runWithConnection(_create))

    def _insert_rows(self, instance_name: str, number: int) -> None:
        if False:
            print('Hello World!')
        'Insert N rows as the given instance, inserting with stream IDs pulled\n        from the postgres sequence.\n        '

        def _insert(txn: LoggingTransaction) -> None:
            if False:
                while True:
                    i = 10
            for _ in range(number):
                txn.execute("INSERT INTO foobar VALUES (nextval('foobar_seq'), ?)", (instance_name,))
                txn.execute("\n                    INSERT INTO stream_positions VALUES ('test_stream', ?,  lastval())\n                    ON CONFLICT (stream_name, instance_name) DO UPDATE SET stream_id = lastval()\n                    ", (instance_name,))
        self.get_success(self.db_pool.runInteraction('_insert_rows', _insert))

    def _insert_row_with_id(self, instance_name: str, stream_id: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Insert one row as the given instance with given stream_id, updating\n        the postgres sequence position to match.\n        '

        def _insert(txn: LoggingTransaction) -> None:
            if False:
                i = 10
                return i + 15
            txn.execute('INSERT INTO foobar VALUES (?, ?)', (stream_id, instance_name))
            txn.execute("SELECT setval('foobar_seq', ?)", (stream_id,))
            txn.execute("\n                INSERT INTO stream_positions VALUES ('test_stream', ?, ?)\n                ON CONFLICT (stream_name, instance_name) DO UPDATE SET stream_id = ?\n                ", (instance_name, stream_id, stream_id))
        self.get_success(self.db_pool.runInteraction('_insert_row_with_id', _insert))

    def test_empty(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test an ID generator against an empty database gives sensible\n        current positions.\n        '
        id_gen = self._create_id_generator()
        self.assertEqual(id_gen.get_positions(), {'master': 1})

    def test_single_instance(self) -> None:
        if False:
            print('Hello World!')
        'Test that reads and writes from a single process are handled\n        correctly.\n        '
        self._insert_rows('master', 7)
        id_gen = self._create_id_generator()
        self.assertEqual(id_gen.get_positions(), {'master': 7})
        self.assertEqual(id_gen.get_current_token_for_writer('master'), 7)

        async def _get_next_async() -> None:
            async with id_gen.get_next() as stream_id:
                self.assertEqual(stream_id, 8)
                self.assertEqual(id_gen.get_positions(), {'master': 7})
                self.assertEqual(id_gen.get_current_token_for_writer('master'), 7)
        self.get_success(_get_next_async())
        self.assertEqual(id_gen.get_positions(), {'master': 8})
        self.assertEqual(id_gen.get_current_token_for_writer('master'), 8)

    def test_out_of_order_finish(self) -> None:
        if False:
            print('Hello World!')
        'Test that IDs persisted out of order are correctly handled'
        self._insert_rows('master', 7)
        id_gen = self._create_id_generator()
        self.assertEqual(id_gen.get_positions(), {'master': 7})
        self.assertEqual(id_gen.get_current_token_for_writer('master'), 7)
        ctx1 = id_gen.get_next()
        ctx2 = id_gen.get_next()
        ctx3 = id_gen.get_next()
        ctx4 = id_gen.get_next()
        s1 = self.get_success(ctx1.__aenter__())
        s2 = self.get_success(ctx2.__aenter__())
        s3 = self.get_success(ctx3.__aenter__())
        s4 = self.get_success(ctx4.__aenter__())
        self.assertEqual(s1, 8)
        self.assertEqual(s2, 9)
        self.assertEqual(s3, 10)
        self.assertEqual(s4, 11)
        self.assertEqual(id_gen.get_positions(), {'master': 7})
        self.assertEqual(id_gen.get_current_token_for_writer('master'), 7)
        self.get_success(ctx2.__aexit__(None, None, None))
        self.assertEqual(id_gen.get_positions(), {'master': 7})
        self.assertEqual(id_gen.get_current_token_for_writer('master'), 7)
        self.get_success(ctx1.__aexit__(None, None, None))
        self.assertEqual(id_gen.get_positions(), {'master': 9})
        self.assertEqual(id_gen.get_current_token_for_writer('master'), 9)
        self.get_success(ctx4.__aexit__(None, None, None))
        self.assertEqual(id_gen.get_positions(), {'master': 9})
        self.assertEqual(id_gen.get_current_token_for_writer('master'), 9)
        self.get_success(ctx3.__aexit__(None, None, None))
        self.assertEqual(id_gen.get_positions(), {'master': 11})
        self.assertEqual(id_gen.get_current_token_for_writer('master'), 11)

    def test_multi_instance(self) -> None:
        if False:
            return 10
        'Test that reads and writes from multiple processes are handled\n        correctly.\n        '
        self._insert_rows('first', 3)
        self._insert_rows('second', 4)
        first_id_gen = self._create_id_generator('first', writers=['first', 'second'])
        second_id_gen = self._create_id_generator('second', writers=['first', 'second'])
        self.assertEqual(first_id_gen.get_positions(), {'first': 3, 'second': 7})
        self.assertEqual(first_id_gen.get_current_token_for_writer('first'), 7)
        self.assertEqual(first_id_gen.get_current_token_for_writer('second'), 7)
        self.assertEqual(second_id_gen.get_positions(), {'first': 3, 'second': 7})
        self.assertEqual(second_id_gen.get_current_token_for_writer('first'), 7)
        self.assertEqual(second_id_gen.get_current_token_for_writer('second'), 7)

        async def _get_next_async() -> None:
            async with first_id_gen.get_next() as stream_id:
                self.assertEqual(stream_id, 8)
                self.assertEqual(first_id_gen.get_positions(), {'first': 3, 'second': 7})
                self.assertEqual(first_id_gen.get_persisted_upto_position(), 7)
        self.get_success(_get_next_async())
        self.assertEqual(first_id_gen.get_positions(), {'first': 8, 'second': 7})
        self.assertEqual(second_id_gen.get_positions(), {'first': 3, 'second': 7})

        async def _get_next_async2() -> None:
            async with second_id_gen.get_next() as stream_id:
                self.assertEqual(stream_id, 9)
                self.assertEqual(second_id_gen.get_positions(), {'first': 3, 'second': 7})
        self.get_success(_get_next_async2())
        self.assertEqual(second_id_gen.get_positions(), {'first': 3, 'second': 9})
        second_id_gen.advance('first', 8)
        self.assertEqual(second_id_gen.get_positions(), {'first': 8, 'second': 9})

    def test_multi_instance_empty_row(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that reads and writes from multiple processes are handled\n        correctly, when one of the writers starts without any rows.\n        '
        self._insert_rows('first', 3)
        self._insert_rows('second', 4)
        first_id_gen = self._create_id_generator('first', writers=['first', 'second', 'third'])
        second_id_gen = self._create_id_generator('second', writers=['first', 'second', 'third'])
        third_id_gen = self._create_id_generator('third', writers=['first', 'second', 'third'])
        self.assertEqual(first_id_gen.get_positions(), {'first': 3, 'second': 7, 'third': 7})
        self.assertEqual(first_id_gen.get_current_token_for_writer('first'), 7)
        self.assertEqual(first_id_gen.get_current_token_for_writer('second'), 7)
        self.assertEqual(first_id_gen.get_current_token_for_writer('third'), 7)
        self.assertEqual(second_id_gen.get_positions(), {'first': 3, 'second': 7, 'third': 7})
        self.assertEqual(second_id_gen.get_current_token_for_writer('first'), 7)
        self.assertEqual(second_id_gen.get_current_token_for_writer('second'), 7)
        self.assertEqual(second_id_gen.get_current_token_for_writer('third'), 7)

        async def _get_next_async() -> None:
            async with third_id_gen.get_next() as stream_id:
                self.assertEqual(stream_id, 8)
                self.assertEqual(third_id_gen.get_positions(), {'first': 3, 'second': 7, 'third': 7})
                self.assertEqual(third_id_gen.get_persisted_upto_position(), 7)
        self.get_success(_get_next_async())
        self.assertEqual(third_id_gen.get_positions(), {'first': 3, 'second': 7, 'third': 8})

    def test_get_next_txn(self) -> None:
        if False:
            print('Hello World!')
        'Test that the `get_next_txn` function works correctly.'
        self._insert_rows('master', 7)
        id_gen = self._create_id_generator()
        self.assertEqual(id_gen.get_positions(), {'master': 7})
        self.assertEqual(id_gen.get_current_token_for_writer('master'), 7)

        def _get_next_txn(txn: LoggingTransaction) -> None:
            if False:
                print('Hello World!')
            stream_id = id_gen.get_next_txn(txn)
            self.assertEqual(stream_id, 8)
            self.assertEqual(id_gen.get_positions(), {'master': 7})
            self.assertEqual(id_gen.get_current_token_for_writer('master'), 7)
        self.get_success(self.db_pool.runInteraction('test', _get_next_txn))
        self.assertEqual(id_gen.get_positions(), {'master': 8})
        self.assertEqual(id_gen.get_current_token_for_writer('master'), 8)

    def test_get_persisted_upto_position(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that `get_persisted_upto_position` correctly tracks updates to\n        positions.\n        '
        self._insert_row_with_id('first', 3)
        self._insert_row_with_id('second', 5)
        id_gen = self._create_id_generator('worker', writers=['first', 'second'])
        self.assertEqual(id_gen.get_positions(), {'first': 3, 'second': 5})
        self.assertEqual(id_gen.get_persisted_upto_position(), 3)
        id_gen.advance('first', 6)
        self.assertEqual(id_gen.get_persisted_upto_position(), 6)
        id_gen.advance('second', 7)
        self.assertEqual(id_gen.get_persisted_upto_position(), 7)
        id_gen.advance('second', 9)
        self.assertEqual(id_gen.get_persisted_upto_position(), 7)
        id_gen.advance('first', 8)
        self.assertEqual(id_gen.get_persisted_upto_position(), 9)
        id_gen.advance('first', 11)
        id_gen.advance('second', 15)
        self.assertEqual(id_gen.get_persisted_upto_position(), 11)

    def test_get_persisted_upto_position_get_next(self) -> None:
        if False:
            return 10
        'Test that `get_persisted_upto_position` correctly tracks updates to\n        positions when `get_next` is called.\n        '
        self._insert_row_with_id('first', 3)
        self._insert_row_with_id('second', 5)
        id_gen = self._create_id_generator('first', writers=['first', 'second'])
        self.assertEqual(id_gen.get_positions(), {'first': 3, 'second': 5})
        self.assertEqual(id_gen.get_persisted_upto_position(), 5)

        async def _get_next_async() -> None:
            async with id_gen.get_next() as stream_id:
                self.assertEqual(stream_id, 6)
                self.assertEqual(id_gen.get_persisted_upto_position(), 5)
        self.get_success(_get_next_async())
        self.assertEqual(id_gen.get_persisted_upto_position(), 6)

    def test_restart_during_out_of_order_persistence(self) -> None:
        if False:
            print('Hello World!')
        'Test that restarting a process while another process is writing out\n        of order updates are handled correctly.\n        '
        self._insert_rows('master', 7)
        id_gen = self._create_id_generator()
        self.assertEqual(id_gen.get_positions(), {'master': 7})
        self.assertEqual(id_gen.get_current_token_for_writer('master'), 7)
        ctx1 = id_gen.get_next()
        ctx2 = id_gen.get_next()
        s1 = self.get_success(ctx1.__aenter__())
        s2 = self.get_success(ctx2.__aenter__())
        self.assertEqual(s1, 8)
        self.assertEqual(s2, 9)
        self.assertEqual(id_gen.get_positions(), {'master': 7})
        self.assertEqual(id_gen.get_current_token_for_writer('master'), 7)
        self.get_success(ctx2.__aexit__(None, None, None))
        id_gen_worker = self._create_id_generator('worker')
        self.assertEqual(id_gen_worker.get_positions(), {'master': 7})
        self.assertEqual(id_gen_worker.get_current_token_for_writer('master'), 7)
        self.get_success(ctx1.__aexit__(None, None, None))
        self.assertEqual(id_gen.get_positions(), {'master': 9})
        id_gen_worker.advance('master', 9)
        self.assertEqual(id_gen_worker.get_positions(), {'master': 9})

    def test_writer_config_change(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that changing the writer config correctly works.'
        self._insert_row_with_id('first', 3)
        self._insert_row_with_id('second', 5)
        id_gen = self._create_id_generator('worker', writers=['first', 'second'])
        self.assertEqual(id_gen.get_persisted_upto_position(), 3)
        self.assertEqual(id_gen.get_current_token_for_writer('first'), 3)
        self.assertEqual(id_gen.get_current_token_for_writer('second'), 5)
        id_gen_2 = self._create_id_generator('second', writers=['second'])
        self.assertEqual(id_gen_2.get_persisted_upto_position(), 5)
        self.assertEqual(id_gen_2.get_current_token_for_writer('second'), 5)
        id_gen_3 = self._create_id_generator('third', writers=['third'])
        self.assertEqual(id_gen_3.get_persisted_upto_position(), 5)
        self.assertEqual(id_gen_3.get_current_token_for_writer('third'), 5)
        id_gen_4 = self._create_id_generator('fourth', writers=['third'])
        self.assertEqual(id_gen_4.get_current_token_for_writer('third'), 5)

        async def _get_next_async() -> None:
            async with id_gen_3.get_next() as stream_id:
                self.assertEqual(stream_id, 6)
        self.get_success(_get_next_async())
        self.assertEqual(id_gen_3.get_persisted_upto_position(), 6)
        id_gen_5 = self._create_id_generator('five', writers=['first', 'third'])
        self.assertEqual(id_gen_5.get_persisted_upto_position(), 6)
        self.assertEqual(id_gen_5.get_current_token_for_writer('first'), 6)
        self.assertEqual(id_gen_5.get_current_token_for_writer('third'), 6)

    def test_sequence_consistency(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that we error out if the table and sequence diverges.'
        self._insert_row_with_id('master', 3)

        def _insert(txn: Cursor) -> None:
            if False:
                return 10
            txn.execute("INSERT INTO foobar VALUES (26, 'master')")
        self.get_success(self.db_pool.runInteraction('_insert', _insert))
        with self.assertRaises(IncorrectDatabaseSetup):
            self._create_id_generator('first')

    def test_minimal_local_token(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._insert_rows('first', 3)
        self._insert_rows('second', 4)
        first_id_gen = self._create_id_generator('first', writers=['first', 'second'])
        second_id_gen = self._create_id_generator('second', writers=['first', 'second'])
        self.assertEqual(first_id_gen.get_positions(), {'first': 3, 'second': 7})
        self.assertEqual(first_id_gen.get_minimal_local_current_token(), 3)
        self.assertEqual(second_id_gen.get_positions(), {'first': 3, 'second': 7})
        self.assertEqual(second_id_gen.get_minimal_local_current_token(), 7)

    def test_current_token_gap(self) -> None:
        if False:
            while True:
                i = 10
        'Test that getting the current token for a writer returns the maximal\n        token when there are no writes.\n        '
        self._insert_rows('first', 3)
        self._insert_rows('second', 4)
        first_id_gen = self._create_id_generator('first', writers=['first', 'second', 'third'])
        second_id_gen = self._create_id_generator('second', writers=['first', 'second', 'third'])
        self.assertEqual(second_id_gen.get_current_token_for_writer('first'), 7)
        self.assertEqual(second_id_gen.get_current_token_for_writer('second'), 7)
        self.assertEqual(second_id_gen.get_current_token(), 7)

        async def _get_next_async() -> None:
            async with first_id_gen.get_next_mult(2):
                pass
        self.get_success(_get_next_async())
        second_id_gen.advance('first', 9)
        self.assertEqual(second_id_gen.get_current_token_for_writer('first'), 9)
        self.assertEqual(second_id_gen.get_current_token_for_writer('second'), 9)
        self.assertEqual(second_id_gen.get_current_token(), 7)
        self.get_success(_get_next_async())
        ctxmgr = second_id_gen.get_next()
        self.get_success(ctxmgr.__aenter__())
        second_id_gen.advance('first', 11)
        self.assertEqual(second_id_gen.get_current_token_for_writer('first'), 11)
        self.assertEqual(second_id_gen.get_current_token_for_writer('second'), 9)
        self.assertEqual(second_id_gen.get_current_token(), 7)
        self.get_success(ctxmgr.__aexit__(None, None, None))
        self.assertEqual(second_id_gen.get_current_token_for_writer('first'), 11)
        self.assertEqual(second_id_gen.get_current_token_for_writer('second'), 12)
        self.assertEqual(second_id_gen.get_current_token(), 7)

class BackwardsMultiWriterIdGeneratorTestCase(HomeserverTestCase):
    """Tests MultiWriterIdGenerator that produce *negative* stream IDs."""
    if not USE_POSTGRES_FOR_TESTS:
        skip = 'Requires Postgres'

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.store = hs.get_datastores().main
        self.db_pool: DatabasePool = self.store.db_pool
        self.get_success(self.db_pool.runInteraction('_setup_db', self._setup_db))

    def _setup_db(self, txn: LoggingTransaction) -> None:
        if False:
            print('Hello World!')
        txn.execute('CREATE SEQUENCE foobar_seq')
        txn.execute('\n            CREATE TABLE foobar (\n                stream_id BIGINT NOT NULL,\n                instance_name TEXT NOT NULL,\n                data TEXT\n            );\n            ')

    def _create_id_generator(self, instance_name: str='master', writers: Optional[List[str]]=None) -> MultiWriterIdGenerator:
        if False:
            while True:
                i = 10

        def _create(conn: LoggingDatabaseConnection) -> MultiWriterIdGenerator:
            if False:
                for i in range(10):
                    print('nop')
            return MultiWriterIdGenerator(conn, self.db_pool, notifier=self.hs.get_replication_notifier(), stream_name='test_stream', instance_name=instance_name, tables=[('foobar', 'instance_name', 'stream_id')], sequence_name='foobar_seq', writers=writers or ['master'], positive=False)
        return self.get_success(self.db_pool.runWithConnection(_create))

    def _insert_row(self, instance_name: str, stream_id: int) -> None:
        if False:
            i = 10
            return i + 15
        'Insert one row as the given instance with given stream_id.'

        def _insert(txn: LoggingTransaction) -> None:
            if False:
                i = 10
                return i + 15
            txn.execute('INSERT INTO foobar VALUES (?, ?)', (stream_id, instance_name))
            txn.execute("\n                INSERT INTO stream_positions VALUES ('test_stream', ?, ?)\n                ON CONFLICT (stream_name, instance_name) DO UPDATE SET stream_id = ?\n                ", (instance_name, -stream_id, -stream_id))
        self.get_success(self.db_pool.runInteraction('_insert_row', _insert))

    def test_single_instance(self) -> None:
        if False:
            return 10
        'Test that reads and writes from a single process are handled\n        correctly.\n        '
        id_gen = self._create_id_generator()

        async def _get_next_async() -> None:
            async with id_gen.get_next() as stream_id:
                self._insert_row('master', stream_id)
        self.get_success(_get_next_async())
        self.assertEqual(id_gen.get_positions(), {'master': -1})
        self.assertEqual(id_gen.get_current_token_for_writer('master'), -1)
        self.assertEqual(id_gen.get_persisted_upto_position(), -1)

        async def _get_next_async2() -> None:
            async with id_gen.get_next_mult(3) as stream_ids:
                for stream_id in stream_ids:
                    self._insert_row('master', stream_id)
        self.get_success(_get_next_async2())
        self.assertEqual(id_gen.get_positions(), {'master': -4})
        self.assertEqual(id_gen.get_current_token_for_writer('master'), -4)
        self.assertEqual(id_gen.get_persisted_upto_position(), -4)
        second_id_gen = self._create_id_generator()
        self.assertEqual(second_id_gen.get_positions(), {'master': -4})
        self.assertEqual(second_id_gen.get_current_token_for_writer('master'), -4)
        self.assertEqual(second_id_gen.get_persisted_upto_position(), -4)

    def test_multiple_instance(self) -> None:
        if False:
            while True:
                i = 10
        'Tests that having multiple instances that get advanced over\n        federation works corretly.\n        '
        id_gen_1 = self._create_id_generator('first', writers=['first', 'second'])
        id_gen_2 = self._create_id_generator('second', writers=['first', 'second'])

        async def _get_next_async() -> None:
            async with id_gen_1.get_next() as stream_id:
                self._insert_row('first', stream_id)
                id_gen_2.advance('first', stream_id)
        self.get_success(_get_next_async())
        self.assertEqual(id_gen_1.get_positions(), {'first': -1, 'second': -1})
        self.assertEqual(id_gen_2.get_positions(), {'first': -1, 'second': -1})
        self.assertEqual(id_gen_1.get_persisted_upto_position(), -1)
        self.assertEqual(id_gen_2.get_persisted_upto_position(), -1)

        async def _get_next_async2() -> None:
            async with id_gen_2.get_next() as stream_id:
                self._insert_row('second', stream_id)
                id_gen_1.advance('second', stream_id)
        self.get_success(_get_next_async2())
        self.assertEqual(id_gen_1.get_positions(), {'first': -1, 'second': -2})
        self.assertEqual(id_gen_2.get_positions(), {'first': -1, 'second': -2})
        self.assertEqual(id_gen_1.get_persisted_upto_position(), -2)
        self.assertEqual(id_gen_2.get_persisted_upto_position(), -2)

class MultiTableMultiWriterIdGeneratorTestCase(HomeserverTestCase):
    if not USE_POSTGRES_FOR_TESTS:
        skip = 'Requires Postgres'

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.store = hs.get_datastores().main
        self.db_pool: DatabasePool = self.store.db_pool
        self.get_success(self.db_pool.runInteraction('_setup_db', self._setup_db))

    def _setup_db(self, txn: LoggingTransaction) -> None:
        if False:
            print('Hello World!')
        txn.execute('CREATE SEQUENCE foobar_seq')
        txn.execute('\n            CREATE TABLE foobar1 (\n                stream_id BIGINT NOT NULL,\n                instance_name TEXT NOT NULL,\n                data TEXT\n            );\n            ')
        txn.execute('\n            CREATE TABLE foobar2 (\n                stream_id BIGINT NOT NULL,\n                instance_name TEXT NOT NULL,\n                data TEXT\n            );\n            ')

    def _create_id_generator(self, instance_name: str='master', writers: Optional[List[str]]=None) -> MultiWriterIdGenerator:
        if False:
            for i in range(10):
                print('nop')

        def _create(conn: LoggingDatabaseConnection) -> MultiWriterIdGenerator:
            if False:
                print('Hello World!')
            return MultiWriterIdGenerator(conn, self.db_pool, notifier=self.hs.get_replication_notifier(), stream_name='test_stream', instance_name=instance_name, tables=[('foobar1', 'instance_name', 'stream_id'), ('foobar2', 'instance_name', 'stream_id')], sequence_name='foobar_seq', writers=writers or ['master'])
        return self.get_success_or_raise(self.db_pool.runWithConnection(_create))

    def _insert_rows(self, table: str, instance_name: str, number: int, update_stream_table: bool=True) -> None:
        if False:
            return 10
        'Insert N rows as the given instance, inserting with stream IDs pulled\n        from the postgres sequence.\n        '

        def _insert(txn: LoggingTransaction) -> None:
            if False:
                i = 10
                return i + 15
            for _ in range(number):
                txn.execute("INSERT INTO %s VALUES (nextval('foobar_seq'), ?)" % (table,), (instance_name,))
                if update_stream_table:
                    txn.execute("\n                        INSERT INTO stream_positions VALUES ('test_stream', ?,  lastval())\n                        ON CONFLICT (stream_name, instance_name) DO UPDATE SET stream_id = lastval()\n                        ", (instance_name,))
        self.get_success(self.db_pool.runInteraction('_insert_rows', _insert))

    def test_load_existing_stream(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test creating ID gens with multiple tables that have rows from after\n        the position in `stream_positions` table.\n        '
        self._insert_rows('foobar1', 'first', 3)
        self._insert_rows('foobar2', 'second', 3)
        self._insert_rows('foobar2', 'second', 1, update_stream_table=False)
        first_id_gen = self._create_id_generator('first', writers=['first', 'second'])
        second_id_gen = self._create_id_generator('second', writers=['first', 'second'])
        self.assertEqual(first_id_gen.get_positions(), {'first': 3, 'second': 6})
        self.assertEqual(first_id_gen.get_current_token_for_writer('first'), 7)
        self.assertEqual(first_id_gen.get_current_token_for_writer('second'), 7)
        self.assertEqual(first_id_gen.get_persisted_upto_position(), 7)
        self.assertEqual(second_id_gen.get_positions(), {'first': 3, 'second': 7})
        self.assertEqual(second_id_gen.get_current_token_for_writer('first'), 7)
        self.assertEqual(second_id_gen.get_current_token_for_writer('second'), 7)
        self.assertEqual(second_id_gen.get_persisted_upto_position(), 7)