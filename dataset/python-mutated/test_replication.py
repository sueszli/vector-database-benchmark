import time
from select import select
import psycopg2
from psycopg2 import sql
from psycopg2.extras import PhysicalReplicationConnection, LogicalReplicationConnection, StopReplication
from . import testconfig
import unittest
from .testutils import ConnectingTestCase
from .testutils import skip_before_postgres, skip_if_green
skip_repl_if_green = skip_if_green('replication not supported in green mode')

class ReplicationTestCase(ConnectingTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.slot = testconfig.repl_slot
        self._slots = []

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
        time.sleep(0.025)
        if self._slots:
            kill_conn = self.connect()
            if kill_conn:
                kill_cur = kill_conn.cursor()
                for slot in self._slots:
                    kill_cur.execute('SELECT pg_drop_replication_slot(%s)', (slot,))
                kill_conn.commit()
                kill_conn.close()

    def create_replication_slot(self, cur, slot_name=testconfig.repl_slot, **kwargs):
        if False:
            while True:
                i = 10
        cur.create_replication_slot(slot_name, **kwargs)
        self._slots.append(slot_name)

    def drop_replication_slot(self, cur, slot_name=testconfig.repl_slot):
        if False:
            for i in range(10):
                print('nop')
        cur.drop_replication_slot(slot_name)
        self._slots.remove(slot_name)

    def make_replication_events(self):
        if False:
            print('Hello World!')
        conn = self.connect()
        if conn is None:
            return
        cur = conn.cursor()
        try:
            cur.execute('DROP TABLE dummy1')
        except psycopg2.ProgrammingError:
            conn.rollback()
        cur.execute('CREATE TABLE dummy1 AS SELECT * FROM generate_series(1, 5) AS id')
        conn.commit()

class ReplicationTest(ReplicationTestCase):

    @skip_before_postgres(9, 0)
    def test_physical_replication_connection(self):
        if False:
            while True:
                i = 10
        conn = self.repl_connect(connection_factory=PhysicalReplicationConnection)
        if conn is None:
            return
        cur = conn.cursor()
        cur.execute('IDENTIFY_SYSTEM')
        cur.fetchall()

    @skip_before_postgres(9, 0)
    def test_datestyle(self):
        if False:
            while True:
                i = 10
        if testconfig.repl_dsn is None:
            return self.skipTest('replication tests disabled by default')
        conn = self.repl_connect(dsn=testconfig.repl_dsn, options='-cdatestyle=german', connection_factory=PhysicalReplicationConnection)
        if conn is None:
            return
        cur = conn.cursor()
        cur.execute('IDENTIFY_SYSTEM')
        cur.fetchall()

    @skip_before_postgres(9, 4)
    def test_logical_replication_connection(self):
        if False:
            return 10
        conn = self.repl_connect(connection_factory=LogicalReplicationConnection)
        if conn is None:
            return
        cur = conn.cursor()
        cur.execute('IDENTIFY_SYSTEM')
        cur.fetchall()

    @skip_before_postgres(9, 4)
    def test_create_replication_slot(self):
        if False:
            for i in range(10):
                print('nop')
        conn = self.repl_connect(connection_factory=PhysicalReplicationConnection)
        if conn is None:
            return
        cur = conn.cursor()
        self.create_replication_slot(cur)
        self.assertRaises(psycopg2.ProgrammingError, self.create_replication_slot, cur)

    @skip_before_postgres(9, 4)
    @skip_repl_if_green
    def test_start_on_missing_replication_slot(self):
        if False:
            return 10
        conn = self.repl_connect(connection_factory=PhysicalReplicationConnection)
        if conn is None:
            return
        cur = conn.cursor()
        self.assertRaises(psycopg2.ProgrammingError, cur.start_replication, self.slot)
        self.create_replication_slot(cur)
        cur.start_replication(self.slot)

    @skip_before_postgres(9, 4)
    @skip_repl_if_green
    def test_start_replication_expert_sql(self):
        if False:
            i = 10
            return i + 15
        conn = self.repl_connect(connection_factory=LogicalReplicationConnection)
        if conn is None:
            return
        cur = conn.cursor()
        self.create_replication_slot(cur, output_plugin='test_decoding')
        cur.start_replication_expert(sql.SQL('START_REPLICATION SLOT {slot} LOGICAL 0/00000000').format(slot=sql.Identifier(self.slot)))

    @skip_before_postgres(9, 4)
    @skip_repl_if_green
    def test_start_and_recover_from_error(self):
        if False:
            print('Hello World!')
        conn = self.repl_connect(connection_factory=LogicalReplicationConnection)
        if conn is None:
            return
        cur = conn.cursor()
        self.create_replication_slot(cur, output_plugin='test_decoding')
        self.make_replication_events()

        def consume(msg):
            if False:
                print('Hello World!')
            raise StopReplication()
        with self.assertRaises(psycopg2.DataError):
            cur.start_replication(slot_name=self.slot, options={'invalid_param': 'value'})
            cur.consume_stream(consume)
        cur.start_replication(slot_name=self.slot)
        self.assertRaises(StopReplication, cur.consume_stream, consume)

    @skip_before_postgres(9, 4)
    @skip_repl_if_green
    def test_keepalive(self):
        if False:
            print('Hello World!')
        conn = self.repl_connect(connection_factory=LogicalReplicationConnection)
        if conn is None:
            return
        cur = conn.cursor()
        self.create_replication_slot(cur, output_plugin='test_decoding')
        self.make_replication_events()
        cur.start_replication(self.slot)

        def consume(msg):
            if False:
                return 10
            raise StopReplication()
        self.assertRaises(StopReplication, cur.consume_stream, consume, keepalive_interval=2)
        conn.close()

    @skip_before_postgres(9, 4)
    @skip_repl_if_green
    def test_stop_replication(self):
        if False:
            while True:
                i = 10
        conn = self.repl_connect(connection_factory=LogicalReplicationConnection)
        if conn is None:
            return
        cur = conn.cursor()
        self.create_replication_slot(cur, output_plugin='test_decoding')
        self.make_replication_events()
        cur.start_replication(self.slot)

        def consume(msg):
            if False:
                i = 10
                return i + 15
            raise StopReplication()
        self.assertRaises(StopReplication, cur.consume_stream, consume)

class AsyncReplicationTest(ReplicationTestCase):

    @skip_before_postgres(9, 4)
    @skip_repl_if_green
    def test_async_replication(self):
        if False:
            return 10
        conn = self.repl_connect(connection_factory=LogicalReplicationConnection, async_=1)
        if conn is None:
            return
        cur = conn.cursor()
        self.create_replication_slot(cur, output_plugin='test_decoding')
        self.wait(cur)
        cur.start_replication(self.slot)
        self.wait(cur)
        self.make_replication_events()
        self.msg_count = 0

        def consume(msg):
            if False:
                print('Hello World!')
            f'{cur.io_timestamp}: {repr(msg)}'
            f'{cur.feedback_timestamp}: {repr(msg)}'
            f'{cur.wal_end}: {repr(msg)}'
            self.msg_count += 1
            if self.msg_count > 3:
                cur.send_feedback(reply=True)
                raise StopReplication()
            cur.send_feedback(flush_lsn=msg.data_start)
        self.assertRaises(psycopg2.ProgrammingError, cur.consume_stream, consume)

        def process_stream():
            if False:
                while True:
                    i = 10
            while True:
                msg = cur.read_message()
                if msg:
                    consume(msg)
                else:
                    select([cur], [], [])
        self.assertRaises(StopReplication, process_stream)

def test_suite():
    if False:
        return 10
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main()