import itertools
import time
from unittest.mock import call
from unittest.mock import Mock
import sqlalchemy as tsa
from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy import exc
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import pool
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import testing
from sqlalchemy import util
from sqlalchemy.engine import default
from sqlalchemy.testing import assert_raises
from sqlalchemy.testing import assert_raises_message
from sqlalchemy.testing import assert_raises_message_context_ok
from sqlalchemy.testing import assert_warns_message
from sqlalchemy.testing import engines
from sqlalchemy.testing import eq_
from sqlalchemy.testing import expect_raises
from sqlalchemy.testing import expect_raises_message
from sqlalchemy.testing import fixtures
from sqlalchemy.testing import is_
from sqlalchemy.testing import is_false
from sqlalchemy.testing import is_true
from sqlalchemy.testing import mock
from sqlalchemy.testing import ne_
from sqlalchemy.testing.engines import DBAPIProxyConnection
from sqlalchemy.testing.engines import DBAPIProxyCursor
from sqlalchemy.testing.engines import testing_engine
from sqlalchemy.testing.schema import Column
from sqlalchemy.testing.schema import Table
from sqlalchemy.testing.util import gc_collect

class MockError(Exception):
    pass

class MockDisconnect(MockError):
    pass

class MockExitIsh(BaseException):
    pass

def mock_connection():
    if False:
        print('Hello World!')

    def mock_cursor():
        if False:
            i = 10
            return i + 15

        def execute(*args, **kwargs):
            if False:
                return 10
            if conn.explode == 'execute':
                raise MockDisconnect('Lost the DB connection on execute')
            elif conn.explode == 'interrupt':
                conn.explode = 'explode_no_disconnect'
                raise MockExitIsh('Keyboard / greenlet / etc interruption')
            elif conn.explode == 'interrupt_dont_break':
                conn.explode = None
                raise MockExitIsh('Keyboard / greenlet / etc interruption')
            elif conn.explode in ('execute_no_disconnect', 'explode_no_disconnect'):
                raise MockError("something broke on execute but we didn't lose the connection")
            elif conn.explode in ('rollback', 'rollback_no_disconnect', 'explode_no_disconnect'):
                raise MockError("something broke on execute but we didn't lose the connection")
            elif args and 'SELECT' in args[0]:
                cursor.description = [('foo', None, None, None, None, None)]
            else:
                return

        def close():
            if False:
                print('Hello World!')
            cursor.fetchall = cursor.fetchone = Mock(side_effect=MockError('cursor closed'))
        cursor = Mock(execute=Mock(side_effect=execute), close=Mock(side_effect=close))
        return cursor

    def cursor():
        if False:
            for i in range(10):
                print('nop')
        while True:
            yield mock_cursor()

    def rollback():
        if False:
            return 10
        if conn.explode == 'rollback':
            raise MockDisconnect('Lost the DB connection on rollback')
        if conn.explode == 'rollback_no_disconnect':
            raise MockError("something broke on rollback but we didn't lose the connection")
        else:
            return

    def commit():
        if False:
            for i in range(10):
                print('nop')
        if conn.explode == 'commit':
            raise MockDisconnect('Lost the DB connection on commit')
        elif conn.explode == 'commit_no_disconnect':
            raise MockError("something broke on commit but we didn't lose the connection")
        else:
            return
    conn = Mock(rollback=Mock(side_effect=rollback), commit=Mock(side_effect=commit), cursor=Mock(side_effect=cursor()))
    return conn

def MockDBAPI():
    if False:
        return 10
    connections = []
    stopped = [False]

    def connect():
        if False:
            i = 10
            return i + 15
        while True:
            if stopped[0]:
                raise MockDisconnect('database is stopped')
            conn = mock_connection()
            connections.append(conn)
            yield conn

    def shutdown(explode='execute', stop=False):
        if False:
            print('Hello World!')
        stopped[0] = stop
        for c in connections:
            c.explode = explode

    def restart():
        if False:
            return 10
        stopped[0] = False
        connections[:] = []

    def dispose():
        if False:
            for i in range(10):
                print('nop')
        stopped[0] = False
        for c in connections:
            c.explode = None
        connections[:] = []
    return Mock(connect=Mock(side_effect=connect()), shutdown=Mock(side_effect=shutdown), dispose=Mock(side_effect=dispose), restart=Mock(side_effect=restart), paramstyle='named', connections=connections, Error=MockError)

class PrePingMockTest(fixtures.TestBase):

    def setup_test(self):
        if False:
            i = 10
            return i + 15
        self.dbapi = MockDBAPI()

    def _pool_fixture(self, pre_ping, setup_disconnect=True, pool_kw=None):
        if False:
            for i in range(10):
                print('nop')
        dialect = default.DefaultDialect()
        dialect.dbapi = self.dbapi
        _pool = pool.QueuePool(creator=lambda : self.dbapi.connect('foo.db'), pre_ping=pre_ping, dialect=dialect, **pool_kw if pool_kw else {})
        if setup_disconnect:
            dialect.is_disconnect = lambda e, conn, cursor: isinstance(e, MockDisconnect)
        return _pool

    def teardown_test(self):
        if False:
            while True:
                i = 10
        self.dbapi.dispose()

    def test_ping_not_on_first_connect(self):
        if False:
            return 10
        pool = self._pool_fixture(pre_ping=True, pool_kw=dict(pool_size=1, max_overflow=0))
        conn = pool.connect()
        dbapi_conn = conn.dbapi_connection
        eq_(dbapi_conn.mock_calls, [])
        conn.close()
        eq_(dbapi_conn.mock_calls, [call.rollback()])
        conn = pool.connect()
        is_(conn.dbapi_connection, dbapi_conn)
        eq_(dbapi_conn.mock_calls, [call.rollback(), call.cursor()])
        conn.close()
        conn = pool.connect()
        is_(conn.dbapi_connection, dbapi_conn)
        eq_(dbapi_conn.mock_calls, [call.rollback(), call.cursor(), call.rollback(), call.cursor()])
        conn.close()

    def test_ping_not_on_reconnect(self):
        if False:
            while True:
                i = 10
        pool = self._pool_fixture(pre_ping=True, pool_kw=dict(pool_size=1, max_overflow=0))
        conn = pool.connect()
        dbapi_conn = conn.dbapi_connection
        conn_rec = conn._connection_record
        eq_(dbapi_conn.mock_calls, [])
        conn.close()
        conn = pool.connect()
        is_(conn.dbapi_connection, dbapi_conn)
        eq_(dbapi_conn.mock_calls, [call.rollback(), call.cursor()])
        conn.invalidate()
        is_(conn.dbapi_connection, None)
        conn = pool.connect()
        is_(conn._connection_record, conn_rec)
        dbapi_conn = conn.dbapi_connection
        eq_(dbapi_conn.mock_calls, [])

    def test_connect_across_restart(self):
        if False:
            print('Hello World!')
        pool = self._pool_fixture(pre_ping=True)
        conn = pool.connect()
        stale_connection = conn.dbapi_connection
        conn.close()
        self.dbapi.shutdown('execute')
        self.dbapi.restart()
        conn = pool.connect()
        cursor = conn.cursor()
        cursor.execute('hi')
        stale_cursor = stale_connection.cursor()
        assert_raises(MockDisconnect, stale_cursor.execute, 'hi')

    def test_handle_error_sets_disconnect(self):
        if False:
            for i in range(10):
                print('nop')
        pool = self._pool_fixture(pre_ping=True, setup_disconnect=False)

        @event.listens_for(pool._dialect, 'handle_error')
        def setup_disconnect(ctx):
            if False:
                i = 10
                return i + 15
            assert isinstance(ctx.sqlalchemy_exception, exc.DBAPIError)
            assert isinstance(ctx.original_exception, MockDisconnect)
            ctx.is_disconnect = True
        conn = pool.connect()
        stale_connection = conn.dbapi_connection
        conn.close()
        self.dbapi.shutdown('execute')
        self.dbapi.restart()
        conn = pool.connect()
        cursor = conn.cursor()
        cursor.execute('hi')
        stale_cursor = stale_connection.cursor()
        assert_raises(MockDisconnect, stale_cursor.execute, 'hi')

    def test_raise_db_is_stopped(self):
        if False:
            return 10
        pool = self._pool_fixture(pre_ping=True)
        conn = pool.connect()
        conn.close()
        self.dbapi.shutdown('execute', stop=True)
        assert_raises_message_context_ok(MockDisconnect, 'database is stopped', pool.connect)

    def test_waits_til_exec_wo_ping_db_is_stopped(self):
        if False:
            print('Hello World!')
        pool = self._pool_fixture(pre_ping=False)
        conn = pool.connect()
        conn.close()
        self.dbapi.shutdown('execute', stop=True)
        conn = pool.connect()
        cursor = conn.cursor()
        assert_raises_message(MockDisconnect, 'Lost the DB connection on execute', cursor.execute, 'foo')

    def test_waits_til_exec_wo_ping_db_is_restarted(self):
        if False:
            return 10
        pool = self._pool_fixture(pre_ping=False)
        conn = pool.connect()
        conn.close()
        self.dbapi.shutdown('execute', stop=True)
        self.dbapi.restart()
        conn = pool.connect()
        cursor = conn.cursor()
        assert_raises_message(MockDisconnect, 'Lost the DB connection on execute', cursor.execute, 'foo')

    @testing.requires.predictable_gc
    def test_pre_ping_weakref_finalizer(self):
        if False:
            for i in range(10):
                print('nop')
        pool = self._pool_fixture(pre_ping=True)
        conn = pool.connect()
        old_dbapi_conn = conn.dbapi_connection
        conn.close()
        eq_(old_dbapi_conn.mock_calls, [call.rollback()])
        conn = pool.connect()
        conn.close()
        eq_(old_dbapi_conn.mock_calls, [call.rollback(), call.cursor(), call.rollback()])
        self.dbapi.shutdown('execute', stop=True)
        self.dbapi.restart()
        conn = pool.connect()
        dbapi_conn = conn.dbapi_connection
        del conn
        gc_collect()
        eq_(dbapi_conn.mock_calls, [call.rollback()])
        eq_(old_dbapi_conn.mock_calls, [call.rollback(), call.cursor(), call.rollback(), call.cursor(), call.close()])

class MockReconnectTest(fixtures.TestBase):

    def setup_test(self):
        if False:
            print('Hello World!')
        self.dbapi = MockDBAPI()
        self.db = testing_engine('postgresql+psycopg2://foo:bar@localhost/test', options=dict(module=self.dbapi, _initialize=False))
        self.mock_connect = call(host='localhost', password='bar', user='foo', dbname='test')
        self.db.dialect.is_disconnect = lambda e, conn, cursor: isinstance(e, MockDisconnect)

    def teardown_test(self):
        if False:
            print('Hello World!')
        self.dbapi.dispose()

    def test_reconnect(self):
        if False:
            print('Hello World!')
        "test that an 'is_disconnect' condition will invalidate the\n        connection, and additionally dispose the previous connection\n        pool and recreate."
        conn = self.db.connect()
        conn.execute(select(1))
        conn2 = self.db.connect()
        conn2.close()
        assert len(self.dbapi.connections) == 2
        self.dbapi.shutdown()
        time.sleep(0.5)
        assert_raises(tsa.exc.DBAPIError, conn.execute, select(1))
        assert not conn.closed
        assert conn.invalidated
        conn.close()
        eq_([c.close.mock_calls for c in self.dbapi.connections], [[call()], []])
        conn = self.db.connect()
        eq_([c.close.mock_calls for c in self.dbapi.connections], [[call()], [call()], []])
        conn.execute(select(1))
        conn.close()
        eq_([c.close.mock_calls for c in self.dbapi.connections], [[call()], [call()], []])

    def test_invalidate_on_execute_trans(self):
        if False:
            print('Hello World!')
        conn = self.db.connect()
        trans = conn.begin()
        self.dbapi.shutdown()
        assert_raises(tsa.exc.DBAPIError, conn.execute, select(1))
        eq_([c.close.mock_calls for c in self.dbapi.connections], [[call()]])
        assert not conn.closed
        assert conn.invalidated
        assert trans.is_active
        assert_raises_message(tsa.exc.PendingRollbackError, "Can't reconnect until invalid transaction is rolled back", conn.execute, select(1))
        assert trans.is_active
        assert_raises_message(tsa.exc.PendingRollbackError, "Can't reconnect until invalid transaction is rolled back", trans.commit)
        assert not trans.is_active
        assert_raises_message(tsa.exc.PendingRollbackError, "Can't reconnect until invalid transaction is rolled back", conn.execute, select(1))
        assert not trans.is_active
        assert_raises_message(tsa.exc.PendingRollbackError, "Can't reconnect until invalid transaction is rolled back", trans.commit)
        trans.rollback()
        assert not trans.is_active
        conn.execute(select(1))
        assert not conn.invalidated
        eq_([c.close.mock_calls for c in self.dbapi.connections], [[call()], []])

    def test_invalidate_on_commit_trans(self):
        if False:
            return 10
        conn = self.db.connect()
        trans = conn.begin()
        self.dbapi.shutdown('commit')
        assert_raises(tsa.exc.DBAPIError, trans.commit)
        assert not conn.closed
        assert conn.invalidated
        assert not trans.is_active
        assert_raises_message(tsa.exc.PendingRollbackError, "Can't reconnect until invalid transaction is rolled back", conn.execute, select(1))
        assert not trans.is_active
        assert_raises_message(tsa.exc.PendingRollbackError, "Can't reconnect until invalid transaction is rolled back", trans.commit)
        assert not trans.is_active
        assert_raises_message(tsa.exc.PendingRollbackError, "Can't reconnect until invalid transaction is rolled back", conn.execute, select(1))
        assert not trans.is_active
        trans.rollback()
        assert not trans.is_active
        conn.execute(select(1))
        assert not conn.invalidated

    def test_commit_fails_contextmanager(self):
        if False:
            while True:
                i = 10
        conn = self.db.connect()

        def go():
            if False:
                print('Hello World!')
            with conn.begin():
                self.dbapi.shutdown('commit_no_disconnect')
        assert_raises(tsa.exc.DBAPIError, go)
        assert not conn.in_transaction()

    def test_commit_fails_trans(self):
        if False:
            while True:
                i = 10
        conn = self.db.connect()
        trans = conn.begin()
        self.dbapi.shutdown('commit_no_disconnect')
        assert_raises(tsa.exc.DBAPIError, trans.commit)
        assert not conn.closed
        assert not conn.invalidated
        assert not trans.is_active
        assert_raises_message(tsa.exc.PendingRollbackError, "Can't reconnect until invalid transaction is rolled back.  Please rollback\\(\\) fully before proceeding", conn.execute, select(1))
        assert not trans.is_active
        assert_raises_message(tsa.exc.PendingRollbackError, "Can't reconnect until invalid transaction is rolled back.  Please rollback\\(\\) fully before proceeding", trans.commit)
        assert not trans.is_active
        assert_raises_message(tsa.exc.PendingRollbackError, "Can't reconnect until invalid transaction is rolled back.  Please rollback\\(\\) fully before proceeding", conn.execute, select(1))
        assert not trans.is_active
        trans.rollback()
        assert not trans.is_active
        conn.execute(select(1))
        assert not conn.invalidated

    def test_invalidate_dont_call_finalizer(self):
        if False:
            i = 10
            return i + 15
        conn = self.db.connect()
        finalizer = mock.Mock()
        conn.connection._connection_record.finalize_callback.append(finalizer)
        conn.invalidate()
        assert conn.invalidated
        eq_(finalizer.call_count, 0)

    def test_conn_reusable(self):
        if False:
            return 10
        conn = self.db.connect()
        conn.execute(select(1))
        eq_(self.dbapi.connect.mock_calls, [self.mock_connect])
        self.dbapi.shutdown()
        with expect_raises(tsa.exc.DBAPIError):
            conn.execute(select(1))
        assert not conn.closed
        assert conn.invalidated
        eq_([c.close.mock_calls for c in self.dbapi.connections], [[call()]])
        with expect_raises(tsa.exc.PendingRollbackError):
            conn.execute(select(1))
        conn.rollback()
        conn.execute(select(1))
        assert not conn.invalidated
        eq_([c.close.mock_calls for c in self.dbapi.connections], [[call()], []])

    def test_invalidated_close(self):
        if False:
            i = 10
            return i + 15
        conn = self.db.connect()
        self.dbapi.shutdown()
        assert_raises(tsa.exc.DBAPIError, conn.execute, select(1))
        conn.close()
        assert conn.closed
        assert not conn.invalidated
        assert_raises_message(tsa.exc.ResourceClosedError, 'This Connection is closed', conn.execute, select(1))

    def test_noreconnect_execute(self):
        if False:
            while True:
                i = 10
        conn = self.db.connect()
        self.dbapi.shutdown('execute_no_disconnect')
        assert_raises_message(tsa.exc.DBAPIError, "something broke on execute but we didn't lose the connection", conn.execute, select(1))
        assert not conn.closed
        assert not conn.invalidated
        conn.close()

    def test_noreconnect_rollback(self):
        if False:
            while True:
                i = 10
        conn = self.db.connect()
        conn.execute(select(1))
        self.dbapi.shutdown('rollback_no_disconnect')
        with expect_raises_message(tsa.exc.DBAPIError, "something broke on rollback but we didn't lose the connection"):
            conn.rollback()
        assert not conn.closed
        assert not conn.invalidated
        conn.close()
        assert_raises_message(tsa.exc.ResourceClosedError, 'This Connection is closed', conn.execute, select(1))

    def test_reconnect_on_reentrant(self):
        if False:
            for i in range(10):
                print('nop')
        conn = self.db.connect()
        conn.execute(select(1))
        assert len(self.dbapi.connections) == 1
        self.dbapi.shutdown('rollback')
        assert_raises_message(tsa.exc.DBAPIError, 'Lost the DB connection on rollback', conn.rollback)
        assert not conn.closed
        assert conn.invalidated

    def test_check_disconnect_no_cursor(self):
        if False:
            while True:
                i = 10
        conn = self.db.connect()
        result = conn.execute(select(1))
        result.cursor.close()
        conn.close()
        assert_raises_message(tsa.exc.DBAPIError, 'cursor closed', list, result)

    def test_dialect_initialize_once(self):
        if False:
            i = 10
            return i + 15
        from sqlalchemy.engine.url import URL
        from sqlalchemy.engine.default import DefaultDialect
        dbapi = self.dbapi

        class MyURL(URL):

            def _get_entrypoint(self):
                if False:
                    print('Hello World!')
                return Dialect

            def get_dialect(self):
                if False:
                    for i in range(10):
                        print('nop')
                return Dialect

        class Dialect(DefaultDialect):
            initialize = Mock()
        engine = create_engine(MyURL.create('foo://'), module=dbapi)
        engine.connect()
        engine.dispose()
        engine.connect()
        eq_(Dialect.initialize.call_count, 1)

    def test_dialect_initialize_retry_if_exception(self):
        if False:
            for i in range(10):
                print('nop')
        from sqlalchemy.engine.url import URL
        from sqlalchemy.engine.default import DefaultDialect
        dbapi = self.dbapi

        class MyURL(URL):

            def _get_entrypoint(self):
                if False:
                    while True:
                        i = 10
                return Dialect

            def get_dialect(self):
                if False:
                    return 10
                return Dialect

        class Dialect(DefaultDialect):
            initialize = Mock()
        Dialect.initialize.side_effect = TypeError
        engine = create_engine(MyURL.create('foo://'), module=dbapi)
        assert_raises(TypeError, engine.connect)
        eq_(Dialect.initialize.call_count, 1)
        is_true(engine.pool._pool.empty())
        assert_raises(TypeError, engine.connect)
        eq_(Dialect.initialize.call_count, 2)
        is_true(engine.pool._pool.empty())
        engine.dispose()
        assert_raises(TypeError, engine.connect)
        eq_(Dialect.initialize.call_count, 3)
        is_true(engine.pool._pool.empty())
        Dialect.initialize.side_effect = None
        conn = engine.connect()
        eq_(Dialect.initialize.call_count, 4)
        conn.close()
        is_false(engine.pool._pool.empty())
        conn = engine.connect()
        eq_(Dialect.initialize.call_count, 4)
        conn.close()
        is_false(engine.pool._pool.empty())
        engine.dispose()
        conn = engine.connect()
        eq_(Dialect.initialize.call_count, 4)
        conn.close()
        is_false(engine.pool._pool.empty())

    def test_invalidate_conn_w_contextmanager_interrupt(self):
        if False:
            i = 10
            return i + 15
        pool = self.db.pool
        conn = self.db.connect()
        self.dbapi.shutdown('interrupt')

        def go():
            if False:
                for i in range(10):
                    print('nop')
            with conn.begin():
                conn.execute(select(1))
        assert_raises(MockExitIsh, go)
        assert conn.invalidated
        eq_(pool._invalidate_time, 0)
        conn.execute(select(1))
        assert not conn.invalidated

    def test_invalidate_conn_interrupt_nodisconnect_workaround(self):
        if False:
            while True:
                i = 10

        @event.listens_for(self.db, 'handle_error')
        def cancel_disconnect(ctx):
            if False:
                while True:
                    i = 10
            ctx.is_disconnect = False
        pool = self.db.pool
        conn = self.db.connect()
        self.dbapi.shutdown('interrupt_dont_break')

        def go():
            if False:
                for i in range(10):
                    print('nop')
            with conn.begin():
                conn.execute(select(1))
        assert_raises(MockExitIsh, go)
        assert not conn.invalidated
        eq_(pool._invalidate_time, 0)
        conn.execute(select(1))
        assert not conn.invalidated

    def test_invalidate_conn_w_contextmanager_disconnect(self):
        if False:
            for i in range(10):
                print('nop')
        pool = self.db.pool
        conn = self.db.connect()
        self.dbapi.shutdown('execute')

        def go():
            if False:
                while True:
                    i = 10
            with conn.begin():
                conn.execute(select(1))
        assert_raises(exc.DBAPIError, go)
        assert conn.invalidated
        ne_(pool._invalidate_time, 0)
        conn.execute(select(1))
        assert not conn.invalidated

class CursorErrTest(fixtures.TestBase):

    def _fixture(self, explode_on_exec, initialize):
        if False:
            for i in range(10):
                print('nop')

        class DBAPIError(Exception):
            pass

        def MockDBAPI():
            if False:
                return 10

            def cursor():
                if False:
                    print('Hello World!')
                while True:
                    if explode_on_exec:
                        yield Mock(description=[], close=Mock(side_effect=DBAPIError('explode')), execute=Mock(side_effect=DBAPIError('explode')))
                    else:
                        yield Mock(description=[], close=Mock(side_effect=Exception('explode')))

            def connect():
                if False:
                    return 10
                while True:
                    yield Mock(spec=['cursor', 'commit', 'rollback', 'close'], cursor=Mock(side_effect=cursor()))
            return Mock(Error=DBAPIError, paramstyle='qmark', connect=Mock(side_effect=connect()))
        dbapi = MockDBAPI()
        from sqlalchemy.engine import default
        url = Mock(get_dialect=lambda : default.DefaultDialect, _get_entrypoint=lambda : default.DefaultDialect, _instantiate_plugins=lambda kwargs: (url, [], kwargs), translate_connect_args=lambda : {}, query={})
        eng = testing_engine(url, options=dict(module=dbapi, _initialize=initialize))
        eng.pool.logger = Mock()

        def get_default_schema_name(connection):
            if False:
                print('Hello World!')
            try:
                cursor = connection.connection.cursor()
                connection._cursor_execute(cursor, 'statement', {})
                cursor.close()
            except exc.DBAPIError:
                util.warn('Exception attempting to detect')
        eng.dialect._get_default_schema_name = get_default_schema_name
        return eng

    def test_cursor_explode(self):
        if False:
            i = 10
            return i + 15
        db = self._fixture(False, False)
        conn = db.connect()
        result = conn.exec_driver_sql('select foo')
        result.close()
        conn.close()
        eq_(db.pool.logger.error.mock_calls, [call('Error closing cursor', exc_info=True)])

    def test_cursor_shutdown_in_initialize(self):
        if False:
            i = 10
            return i + 15
        db = self._fixture(True, True)
        assert_warns_message(exc.SAWarning, 'Exception attempting to detect', db.connect)
        eq_(db.pool.logger.error.mock_calls, [call('Error closing cursor', exc_info=True)])

def _assert_invalidated(fn, *args):
    if False:
        while True:
            i = 10
    try:
        fn(*args)
        assert False
    except tsa.exc.DBAPIError as e:
        if not e.connection_invalidated:
            raise

class RealPrePingEventHandlerTest(fixtures.TestBase):
    """real test for issue #5648, which had to be revisited for 2.0 as the
    initial version was not adequately tested and non-implementation for
    mysql, postgresql was not caught

    """
    __backend__ = True
    __requires__ = ('graceful_disconnects', 'ad_hoc_engines')

    @testing.fixture
    def ping_fixture(self, testing_engine):
        if False:
            while True:
                i = 10
        engine = testing_engine(options={'pool_pre_ping': True, '_initialize': False})
        existing_connect = engine.dialect.dbapi.connect
        fail = False
        fail_count = itertools.count()
        DBAPIError = engine.dialect.dbapi.Error

        class ExplodeConnection(DBAPIProxyConnection):

            def ping(self, *arg, **kw):
                if False:
                    print('Hello World!')
                if fail and next(fail_count) < 1:
                    raise DBAPIError('unhandled disconnect situation')
                else:
                    return True

        class ExplodeCursor(DBAPIProxyCursor):

            def execute(self, stmt, parameters=None, **kw):
                if False:
                    i = 10
                    return i + 15
                if fail and next(fail_count) < 1:
                    raise DBAPIError('unhandled disconnect situation')
                else:
                    return super().execute(stmt, parameters=parameters, **kw)

        def mock_connect(*arg, **kw):
            if False:
                for i in range(10):
                    print('nop')
            real_connection = existing_connect(*arg, **kw)
            return ExplodeConnection(engine, real_connection, ExplodeCursor)
        with mock.patch.object(engine.dialect.loaded_dbapi, 'connect', mock_connect):
            engine.connect().close()
            fail = True
            yield engine

    @testing.fixture
    def ping_fixture_all_errs_disconnect(self, ping_fixture):
        if False:
            print('Hello World!')
        engine = ping_fixture
        with mock.patch.object(engine.dialect, 'is_disconnect', lambda *arg, **kw: True):
            yield engine

    def test_control(self, ping_fixture):
        if False:
            return 10
        'test the fixture raises on connect'
        engine = ping_fixture
        with expect_raises_message(exc.DBAPIError, 'unhandled disconnect situation'):
            engine.connect()

    def test_downgrade_control(self, ping_fixture_all_errs_disconnect):
        if False:
            i = 10
            return i + 15
        "test the disconnect fixture doesn't raise, since it considers\n        all errors to be disconnect errors.\n\n        "
        engine = ping_fixture_all_errs_disconnect
        conn = engine.connect()
        conn.close()

    def test_event_handler_didnt_upgrade_disconnect(self, ping_fixture):
        if False:
            for i in range(10):
                print('nop')
        "test that having an event handler that doesn't do anything\n        keeps the behavior in place for a fatal error.\n\n        "
        engine = ping_fixture

        @event.listens_for(engine, 'handle_error')
        def setup_disconnect(ctx):
            if False:
                return 10
            assert not ctx.is_disconnect
        with expect_raises_message(exc.DBAPIError, 'unhandled disconnect situation'):
            engine.connect()

    def test_event_handler_didnt_downgrade_disconnect(self, ping_fixture_all_errs_disconnect):
        if False:
            for i in range(10):
                print('nop')
        "test that having an event handler that doesn't do anything\n        keeps the behavior in place for a disconnect error.\n\n        "
        engine = ping_fixture_all_errs_disconnect

        @event.listens_for(engine, 'handle_error')
        def setup_disconnect(ctx):
            if False:
                for i in range(10):
                    print('nop')
            assert ctx.is_pre_ping
            assert ctx.is_disconnect
        conn = engine.connect()
        conn.close()

    def test_event_handler_can_upgrade_disconnect(self, ping_fixture):
        if False:
            return 10
        'test that an event hook can receive a fatal error and convert\n        it to be a disconnect error during pre-ping'
        engine = ping_fixture

        @event.listens_for(engine, 'handle_error')
        def setup_disconnect(ctx):
            if False:
                for i in range(10):
                    print('nop')
            assert ctx.is_pre_ping
            ctx.is_disconnect = True
        conn = engine.connect()
        conn.close()

    def test_event_handler_can_downgrade_disconnect(self, ping_fixture_all_errs_disconnect):
        if False:
            for i in range(10):
                print('nop')
        'test that an event hook can receive a disconnect error and convert\n        it to be a fatal error during pre-ping'
        engine = ping_fixture_all_errs_disconnect

        @event.listens_for(engine, 'handle_error')
        def setup_disconnect(ctx):
            if False:
                while True:
                    i = 10
            assert ctx.is_disconnect
            if ctx.is_pre_ping:
                ctx.is_disconnect = False
        with expect_raises_message(exc.DBAPIError, 'unhandled disconnect situation'):
            engine.connect()

class RealReconnectTest(fixtures.TestBase):
    __backend__ = True
    __requires__ = ('graceful_disconnects', 'ad_hoc_engines')

    def setup_test(self):
        if False:
            while True:
                i = 10
        self.engine = engines.reconnecting_engine()

    def teardown_test(self):
        if False:
            print('Hello World!')
        self.engine.dispose()

    def test_reconnect(self):
        if False:
            i = 10
            return i + 15
        with self.engine.connect() as conn:
            eq_(conn.execute(select(1)).scalar(), 1)
            assert not conn.closed
            self.engine.test_shutdown()
            _assert_invalidated(conn.execute, select(1))
            assert not conn.closed
            assert conn.invalidated
            assert conn.invalidated
            with expect_raises(tsa.exc.PendingRollbackError):
                conn.execute(select(1))
            conn.rollback()
            eq_(conn.execute(select(1)).scalar(), 1)
            assert not conn.invalidated
            self.engine.test_shutdown()
            _assert_invalidated(conn.execute, select(1))
            assert conn.invalidated
            conn.rollback()
            eq_(conn.execute(select(1)).scalar(), 1)
            assert not conn.invalidated

    def test_detach_invalidated(self):
        if False:
            while True:
                i = 10
        with self.engine.connect() as conn:
            conn.invalidate()
            with expect_raises_message(exc.InvalidRequestError, "Can't detach an invalidated Connection"):
                conn.detach()

    def test_detach_closed(self):
        if False:
            return 10
        with self.engine.connect() as conn:
            pass
        with expect_raises_message(exc.ResourceClosedError, 'This Connection is closed'):
            conn.detach()

    @testing.requires.independent_connections
    def test_multiple_invalidate(self):
        if False:
            print('Hello World!')
        c1 = self.engine.connect()
        c2 = self.engine.connect()
        eq_(c1.execute(select(1)).scalar(), 1)
        self.engine.test_shutdown()
        _assert_invalidated(c1.execute, select(1))
        p2 = self.engine.pool
        _assert_invalidated(c2.execute, select(1))
        assert self.engine.pool is p2

    def test_ensure_is_disconnect_gets_connection(self):
        if False:
            i = 10
            return i + 15

        def is_disconnect(e, conn, cursor):
            if False:
                while True:
                    i = 10
            assert conn.dbapi_connection is not None
        self.engine.dialect.is_disconnect = is_disconnect
        with self.engine.connect() as conn:
            self.engine.test_shutdown()
            assert_raises(tsa.exc.DBAPIError, conn.execute, select(1))
            conn.invalidate()

    def test_rollback_on_invalid_plain(self):
        if False:
            for i in range(10):
                print('nop')
        with self.engine.connect() as conn:
            trans = conn.begin()
            conn.invalidate()
            trans.rollback()

    @testing.requires.two_phase_transactions
    def test_rollback_on_invalid_twophase(self):
        if False:
            print('Hello World!')
        with self.engine.connect() as conn:
            trans = conn.begin_twophase()
            conn.invalidate()
            trans.rollback()

    @testing.requires.savepoints
    def test_rollback_on_invalid_savepoint(self):
        if False:
            return 10
        with self.engine.connect() as conn:
            conn.begin()
            trans2 = conn.begin_nested()
            conn.invalidate()
            trans2.rollback()
            with expect_raises(exc.PendingRollbackError):
                conn.begin_nested()

    def test_no_begin_on_invalid(self):
        if False:
            i = 10
            return i + 15
        with self.engine.connect() as conn:
            conn.begin()
            conn.invalidate()
            with expect_raises(exc.PendingRollbackError):
                conn.commit()

    def test_invalidate_twice(self):
        if False:
            for i in range(10):
                print('nop')
        with self.engine.connect() as conn:
            conn.invalidate()
            conn.invalidate()

    def test_explode_in_initializer(self):
        if False:
            i = 10
            return i + 15
        engine = engines.testing_engine()

        def broken_initialize(connection):
            if False:
                for i in range(10):
                    print('nop')
            connection.exec_driver_sql('select fake_stuff from _fake_table')
        engine.dialect.initialize = broken_initialize
        assert_raises(exc.DBAPIError, engine.connect)

    def test_explode_in_initializer_disconnect(self):
        if False:
            return 10
        engine = engines.testing_engine()

        def broken_initialize(connection):
            if False:
                while True:
                    i = 10
            connection.exec_driver_sql('select fake_stuff from _fake_table')
        engine.dialect.initialize = broken_initialize

        def is_disconnect(e, conn, cursor):
            if False:
                while True:
                    i = 10
            return True
        engine.dialect.is_disconnect = is_disconnect
        assert_raises(exc.DBAPIError, engine.connect)

    def test_null_pool(self):
        if False:
            while True:
                i = 10
        engine = engines.reconnecting_engine(options=dict(poolclass=pool.NullPool))
        with engine.connect() as conn:
            eq_(conn.execute(select(1)).scalar(), 1)
            assert not conn.closed
            engine.test_shutdown()
            _assert_invalidated(conn.execute, select(1))
            assert not conn.closed
            assert conn.invalidated
            conn.rollback()
            eq_(conn.execute(select(1)).scalar(), 1)
            assert not conn.invalidated

    def test_close(self):
        if False:
            return 10
        with self.engine.connect() as conn:
            eq_(conn.execute(select(1)).scalar(), 1)
            assert not conn.closed
            self.engine.test_shutdown()
            _assert_invalidated(conn.execute, select(1))
        with self.engine.connect() as conn:
            eq_(conn.execute(select(1)).scalar(), 1)

    def test_with_transaction(self):
        if False:
            for i in range(10):
                print('nop')
        with self.engine.connect() as conn:
            trans = conn.begin()
            assert trans.is_valid
            eq_(conn.execute(select(1)).scalar(), 1)
            assert not conn.closed
            self.engine.test_shutdown()
            _assert_invalidated(conn.execute, select(1))
            assert not conn.closed
            assert conn.invalidated
            assert trans.is_active
            assert not trans.is_valid
            assert_raises_message(tsa.exc.PendingRollbackError, "Can't reconnect until invalid transaction is rolled back", conn.execute, select(1))
            assert trans.is_active
            assert not trans.is_valid
            assert_raises_message(tsa.exc.PendingRollbackError, "Can't reconnect until invalid transaction is rolled back", trans.commit)
            assert not trans.is_active
            assert not trans.is_valid
            assert_raises_message(tsa.exc.PendingRollbackError, "Can't reconnect until invalid transaction is rolled back", conn.execute, select(1))
            assert_raises_message(tsa.exc.PendingRollbackError, "Can't reconnect until invalid transaction is rolled back", trans.commit)
            assert_raises_message(tsa.exc.PendingRollbackError, "Can't reconnect until invalid transaction is rolled back", conn.execute, select(1))
            trans.rollback()
            assert not trans.is_active
            assert not trans.is_valid
            assert conn.invalidated
            eq_(conn.execute(select(1)).scalar(), 1)
            assert not conn.invalidated

class RecycleTest(fixtures.TestBase):
    __backend__ = True

    def test_basic(self):
        if False:
            for i in range(10):
                print('nop')
        engine = engines.reconnecting_engine()
        conn = engine.connect()
        eq_(conn.execute(select(1)).scalar(), 1)
        conn.close()
        engine.pool._recycle = 1
        engine.test_shutdown()
        time.sleep(2)
        conn = engine.connect()
        eq_(conn.execute(select(1)).scalar(), 1)
        conn.close()

class PrePingRealTest(fixtures.TestBase):
    __backend__ = True

    def test_pre_ping_db_is_restarted(self):
        if False:
            print('Hello World!')
        engine = engines.reconnecting_engine(options={'pool_pre_ping': True})
        conn = engine.connect()
        eq_(conn.execute(select(1)).scalar(), 1)
        stale_connection = conn.connection.dbapi_connection
        conn.close()
        engine.test_shutdown()
        engine.test_restart()
        conn = engine.connect()
        eq_(conn.execute(select(1)).scalar(), 1)
        conn.close()
        with expect_raises(engine.dialect.dbapi.Error, check_context=False):
            curs = stale_connection.cursor()
            curs.execute('select 1')

    def test_pre_ping_db_stays_shutdown(self):
        if False:
            while True:
                i = 10
        engine = engines.reconnecting_engine(options={'pool_pre_ping': True})
        if isinstance(engine.pool, pool.QueuePool):
            eq_(engine.pool.checkedin(), 0)
            eq_(engine.pool._overflow, -5)
        conn = engine.connect()
        eq_(conn.execute(select(1)).scalar(), 1)
        conn.close()
        if isinstance(engine.pool, pool.QueuePool):
            eq_(engine.pool.checkedin(), 1)
            eq_(engine.pool._overflow, -4)
        engine.test_shutdown(stop=True)
        assert_raises(exc.DBAPIError, engine.connect)
        if isinstance(engine.pool, pool.QueuePool):
            eq_(engine.pool.checkedin(), 1)
            eq_(engine.pool._overflow, -4)

class InvalidateDuringResultTest(fixtures.TestBase):
    __backend__ = True
    __requires__ = ('ad_hoc_engines',)

    def setup_test(self):
        if False:
            i = 10
            return i + 15
        self.engine = engines.reconnecting_engine()
        self.meta = MetaData()
        table = Table('sometable', self.meta, Column('id', Integer, primary_key=True), Column('name', String(50)))
        with self.engine.begin() as conn:
            self.meta.create_all(conn)
            conn.execute(table.insert(), [{'id': i, 'name': 'row %d' % i} for i in range(1, 100)])

    def teardown_test(self):
        if False:
            while True:
                i = 10
        with self.engine.begin() as conn:
            self.meta.drop_all(conn)
        self.engine.dispose()

    def test_invalidate_on_results(self):
        if False:
            for i in range(10):
                print('nop')
        conn = self.engine.connect()
        result = conn.exec_driver_sql('select * from sometable')
        for x in range(20):
            result.fetchone()
        real_cursor = result.cursor
        self.engine.test_shutdown()

        def produce_side_effect():
            if False:
                for i in range(10):
                    print('nop')
            real_cursor.execute('select * from sometable')
        result.cursor = Mock(fetchone=mock.Mock(side_effect=produce_side_effect))
        try:
            _assert_invalidated(result.fetchone)
            assert conn.invalidated
        finally:
            conn.invalidate()

class ReconnectRecipeTest(fixtures.TestBase):
    """Test for the reconnect recipe given at doc/build/faq/connections.rst.

    Make sure the above document is updated if changes are made here.

    """
    __only_on__ = ('+mysqldb', '+pymysql')

    def make_engine(self, engine):
        if False:
            i = 10
            return i + 15
        num_retries = 3
        retry_interval = 0.5

        def _run_with_retries(fn, context, cursor, statement, *arg, **kw):
            if False:
                return 10
            for retry in range(num_retries + 1):
                try:
                    fn(cursor, statement, *arg, context=context)
                except engine.dialect.dbapi.Error as raw_dbapi_err:
                    connection = context.root_connection
                    if engine.dialect.is_disconnect(raw_dbapi_err, connection, cursor):
                        if retry > num_retries:
                            raise
                        engine.logger.error('disconnection error, retrying operation', exc_info=True)
                        connection.invalidate()
                        connection.rollback()
                        time.sleep(retry_interval)
                        context.cursor = cursor = connection.connection.cursor()
                    else:
                        raise
                else:
                    return True
        e = engine.execution_options(isolation_level='AUTOCOMMIT')

        @event.listens_for(e, 'do_execute_no_params')
        def do_execute_no_params(cursor, statement, context):
            if False:
                return 10
            return _run_with_retries(context.dialect.do_execute_no_params, context, cursor, statement)

        @event.listens_for(e, 'do_execute')
        def do_execute(cursor, statement, parameters, context):
            if False:
                i = 10
                return i + 15
            return _run_with_retries(context.dialect.do_execute, context, cursor, statement, parameters)
        return e
    __backend__ = True

    def setup_test(self):
        if False:
            for i in range(10):
                print('nop')
        self.engine = engines.reconnecting_engine()
        self.meta = MetaData()
        self.table = Table('sometable', self.meta, Column('id', Integer, primary_key=True), Column('name', String(50)))
        self.meta.create_all(self.engine)

    def teardown_test(self):
        if False:
            while True:
                i = 10
        self.meta.drop_all(self.engine)
        self.engine.dispose()

    def test_restart_on_execute_no_txn(self):
        if False:
            return 10
        engine = self.make_engine(self.engine)
        with engine.connect() as conn:
            eq_(conn.execute(select(1)).scalar(), 1)
            self.engine.test_shutdown()
            self.engine.test_restart()
            eq_(conn.execute(select(1)).scalar(), 1)

    def test_restart_on_execute_txn(self):
        if False:
            return 10
        engine = self.make_engine(self.engine)
        with engine.begin() as conn:
            eq_(conn.execute(select(1)).scalar(), 1)
            self.engine.test_shutdown()
            self.engine.test_restart()
            eq_(conn.execute(select(1)).scalar(), 1)

    def test_autocommits_txn(self):
        if False:
            while True:
                i = 10
        engine = self.make_engine(self.engine)
        with engine.begin() as conn:
            conn.execute(self.table.insert(), [{'id': 1, 'name': 'some name 1'}, {'id': 2, 'name': 'some name 2'}, {'id': 3, 'name': 'some name 3'}])
            self.engine.test_shutdown()
            self.engine.test_restart()
            eq_(conn.execute(select(self.table).order_by(self.table.c.id)).fetchall(), [(1, 'some name 1'), (2, 'some name 2'), (3, 'some name 3')])

    def test_fail_on_executemany_txn(self):
        if False:
            print('Hello World!')
        engine = self.make_engine(self.engine)
        with engine.begin() as conn:
            conn.execute(self.table.insert(), [{'id': 1, 'name': 'some name 1'}, {'id': 2, 'name': 'some name 2'}, {'id': 3, 'name': 'some name 3'}])
            self.engine.test_shutdown()
            self.engine.test_restart()
            assert_raises(exc.DBAPIError, conn.execute, self.table.insert(), [{'id': 4, 'name': 'some name 4'}, {'id': 5, 'name': 'some name 5'}, {'id': 6, 'name': 'some name 6'}])
            conn.rollback()