import select
import unittest
import warnings
import psycopg2
import psycopg2.extensions
import psycopg2.extras
from psycopg2.extensions import POLL_OK, POLL_READ, POLL_WRITE
from .testutils import ConnectingTestCase, skip_before_postgres, slow
from .testutils import skip_if_crdb

class ConnectionStub:
    """A `connection` wrapper allowing analysis of the `poll()` calls."""

    def __init__(self, conn):
        if False:
            for i in range(10):
                print('nop')
        self.conn = conn
        self.polls = []

    def fileno(self):
        if False:
            i = 10
            return i + 15
        return self.conn.fileno()

    def poll(self):
        if False:
            i = 10
            return i + 15
        rv = self.conn.poll()
        self.polls.append(rv)
        return rv

class GreenTestCase(ConnectingTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._cb = psycopg2.extensions.get_wait_callback()
        psycopg2.extensions.set_wait_callback(psycopg2.extras.wait_select)
        ConnectingTestCase.setUp(self)

    def tearDown(self):
        if False:
            print('Hello World!')
        ConnectingTestCase.tearDown(self)
        psycopg2.extensions.set_wait_callback(self._cb)

    def set_stub_wait_callback(self, conn, cb=None):
        if False:
            while True:
                i = 10
        stub = ConnectionStub(conn)
        psycopg2.extensions.set_wait_callback(lambda conn: (cb or psycopg2.extras.wait_select)(stub))
        return stub

    @slow
    @skip_if_crdb('flush on write flakey')
    def test_flush_on_write(self):
        if False:
            return 10
        conn = self.conn
        stub = self.set_stub_wait_callback(conn)
        curs = conn.cursor()
        for mb in (1, 5, 10, 20, 50):
            size = mb * 1024 * 1024
            del stub.polls[:]
            curs.execute('select %s;', ('x' * size,))
            self.assertEqual(size, len(curs.fetchone()[0]))
            if stub.polls.count(psycopg2.extensions.POLL_WRITE) > 1:
                return
        warnings.warn("sending a large query didn't trigger block on write.")

    def test_error_in_callback(self):
        if False:
            while True:
                i = 10
        conn = self.conn
        curs = conn.cursor()
        curs.execute('select 1')
        curs.fetchone()
        psycopg2.extensions.set_wait_callback(lambda conn: 1 // 0)
        self.assertRaises(ZeroDivisionError, curs.execute, 'select 2')
        self.assert_(conn.closed)

    def test_dont_freak_out(self):
        if False:
            i = 10
            return i + 15
        conn = self.conn
        curs = conn.cursor()
        self.assertRaises(psycopg2.ProgrammingError, curs.execute, 'select the unselectable')
        self.assert_(not conn.closed)
        conn.rollback()
        curs.execute('select 1')
        self.assertEqual(curs.fetchone()[0], 1)

    @skip_before_postgres(8, 2)
    def test_copy_no_hang(self):
        if False:
            for i in range(10):
                print('nop')
        cur = self.conn.cursor()
        self.assertRaises(psycopg2.ProgrammingError, cur.execute, 'copy (select 1) to stdout')

    @slow
    @skip_if_crdb('notice')
    @skip_before_postgres(9, 0)
    def test_non_block_after_notice(self):
        if False:
            while True:
                i = 10

        def wait(conn):
            if False:
                print('Hello World!')
            while 1:
                state = conn.poll()
                if state == POLL_OK:
                    break
                elif state == POLL_READ:
                    select.select([conn.fileno()], [], [], 0.1)
                elif state == POLL_WRITE:
                    select.select([], [conn.fileno()], [], 0.1)
                else:
                    raise conn.OperationalError(f'bad state from poll: {state}')
        stub = self.set_stub_wait_callback(self.conn, wait)
        cur = self.conn.cursor()
        cur.execute("\n            select 1;\n            do $$\n                begin\n                    raise notice 'hello';\n                end\n            $$ language plpgsql;\n            select pg_sleep(1);\n            ")
        polls = stub.polls.count(POLL_READ)
        self.assert_(polls > 8, polls)

class CallbackErrorTestCase(ConnectingTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._cb = psycopg2.extensions.get_wait_callback()
        psycopg2.extensions.set_wait_callback(self.crappy_callback)
        ConnectingTestCase.setUp(self)
        self.to_error = None

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        ConnectingTestCase.tearDown(self)
        psycopg2.extensions.set_wait_callback(self._cb)

    def crappy_callback(self, conn):
        if False:
            for i in range(10):
                print('nop')
        'green callback failing after `self.to_error` time it is called'
        while True:
            if self.to_error is not None:
                self.to_error -= 1
                if self.to_error <= 0:
                    raise ZeroDivisionError('I accidentally the connection')
            try:
                state = conn.poll()
                if state == POLL_OK:
                    break
                elif state == POLL_READ:
                    select.select([conn.fileno()], [], [])
                elif state == POLL_WRITE:
                    select.select([], [conn.fileno()], [])
                else:
                    raise conn.OperationalError(f'bad state from poll: {state}')
            except KeyboardInterrupt:
                conn.cancel()
                continue

    def test_errors_on_connection(self):
        if False:
            print('Hello World!')
        for i in range(100):
            self.to_error = i
            try:
                self.connect()
            except ZeroDivisionError:
                pass
            else:
                return
        self.fail('you should have had a success or an error by now')

    def test_errors_on_query(self):
        if False:
            i = 10
            return i + 15
        for i in range(100):
            self.to_error = None
            cnn = self.connect()
            cur = cnn.cursor()
            self.to_error = i
            try:
                cur.execute('select 1')
                cur.fetchone()
            except ZeroDivisionError:
                pass
            else:
                return
        self.fail('you should have had a success or an error by now')

    @skip_if_crdb('named cursor', version='< 22.1')
    def test_errors_named_cursor(self):
        if False:
            i = 10
            return i + 15
        for i in range(100):
            self.to_error = None
            cnn = self.connect()
            cur = cnn.cursor('foo')
            self.to_error = i
            try:
                cur.execute('select 1')
                cur.fetchone()
            except ZeroDivisionError:
                pass
            else:
                return
        self.fail('you should have had a success or an error by now')

def test_suite():
    if False:
        i = 10
        return i + 15
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main()