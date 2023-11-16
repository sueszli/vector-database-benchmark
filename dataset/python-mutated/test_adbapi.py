"""
Tests for twisted.enterprise.adbapi.
"""
import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import Connection, ConnectionLost, ConnectionPool, Transaction
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
simple_table_schema = '\nCREATE TABLE simple (\n  x integer\n)\n'

class ADBAPITestBase:
    """
    Test the asynchronous DB-API code.
    """
    openfun_called: Dict[object, bool] = {}
    if interfaces.IReactorThreads(reactor, None) is None:
        skip = 'ADB-API requires threads, no way to test without them'

    def extraSetUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set up the database and create a connection pool pointing at it.\n        '
        self.startDB()
        self.dbpool = self.makePool(cp_openfun=self.openfun)
        self.dbpool.start()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        d = self.dbpool.runOperation('DROP TABLE simple')
        d.addCallback(lambda res: self.dbpool.close())
        d.addCallback(lambda res: self.stopDB())
        return d

    def openfun(self, conn):
        if False:
            while True:
                i = 10
        self.openfun_called[conn] = True

    def checkOpenfunCalled(self, conn=None):
        if False:
            print('Hello World!')
        if not conn:
            self.assertTrue(self.openfun_called)
        else:
            self.assertIn(conn, self.openfun_called)

    def test_pool(self):
        if False:
            while True:
                i = 10
        d = self.dbpool.runOperation(simple_table_schema)
        if self.test_failures:
            d.addCallback(self._testPool_1_1)
            d.addCallback(self._testPool_1_2)
            d.addCallback(self._testPool_1_3)
            d.addCallback(self._testPool_1_4)
            d.addCallback(lambda res: self.flushLoggedErrors())
        d.addCallback(self._testPool_2)
        d.addCallback(self._testPool_3)
        d.addCallback(self._testPool_4)
        d.addCallback(self._testPool_5)
        d.addCallback(self._testPool_6)
        d.addCallback(self._testPool_7)
        d.addCallback(self._testPool_8)
        d.addCallback(self._testPool_9)
        return d

    def _testPool_1_1(self, res):
        if False:
            i = 10
            return i + 15
        d = defer.maybeDeferred(self.dbpool.runQuery, 'select * from NOTABLE')
        d.addCallbacks(lambda res: self.fail('no exception'), lambda f: None)
        return d

    def _testPool_1_2(self, res):
        if False:
            for i in range(10):
                print('nop')
        d = defer.maybeDeferred(self.dbpool.runOperation, 'deletexxx from NOTABLE')
        d.addCallbacks(lambda res: self.fail('no exception'), lambda f: None)
        return d

    def _testPool_1_3(self, res):
        if False:
            while True:
                i = 10
        d = defer.maybeDeferred(self.dbpool.runInteraction, self.bad_interaction)
        d.addCallbacks(lambda res: self.fail('no exception'), lambda f: None)
        return d

    def _testPool_1_4(self, res):
        if False:
            i = 10
            return i + 15
        d = defer.maybeDeferred(self.dbpool.runWithConnection, self.bad_withConnection)
        d.addCallbacks(lambda res: self.fail('no exception'), lambda f: None)
        return d

    def _testPool_2(self, res):
        if False:
            print('Hello World!')
        sql = 'select count(1) from simple'
        d = self.dbpool.runQuery(sql)

        def _check(row):
            if False:
                i = 10
                return i + 15
            self.assertTrue(int(row[0][0]) == 0, 'Interaction not rolled back')
            self.checkOpenfunCalled()
        d.addCallback(_check)
        return d

    def _testPool_3(self, res):
        if False:
            i = 10
            return i + 15
        sql = 'select count(1) from simple'
        inserts = []
        for i in range(self.num_iterations):
            sql = 'insert into simple(x) values(%d)' % i
            inserts.append(self.dbpool.runOperation(sql))
        d = defer.gatherResults(inserts)

        def _select(res):
            if False:
                for i in range(10):
                    print('nop')
            sql = 'select x from simple order by x'
            d = self.dbpool.runQuery(sql)
            return d
        d.addCallback(_select)

        def _check(rows):
            if False:
                while True:
                    i = 10
            self.assertTrue(len(rows) == self.num_iterations, 'Wrong number of rows')
            for i in range(self.num_iterations):
                self.assertTrue(len(rows[i]) == 1, 'Wrong size row')
                self.assertTrue(rows[i][0] == i, 'Values not returned.')
        d.addCallback(_check)
        return d

    def _testPool_4(self, res):
        if False:
            while True:
                i = 10
        d = self.dbpool.runInteraction(self.interaction)
        d.addCallback(lambda res: self.assertEqual(res, 'done'))
        return d

    def _testPool_5(self, res):
        if False:
            return 10
        d = self.dbpool.runWithConnection(self.withConnection)
        d.addCallback(lambda res: self.assertEqual(res, 'done'))
        return d

    def _testPool_6(self, res):
        if False:
            while True:
                i = 10
        d = self.dbpool.runWithConnection(self.close_withConnection)
        return d

    def _testPool_7(self, res):
        if False:
            for i in range(10):
                print('nop')
        ds = []
        for i in range(self.num_iterations):
            sql = 'select x from simple where x = %d' % i
            ds.append(self.dbpool.runQuery(sql))
        dlist = defer.DeferredList(ds, fireOnOneErrback=True)

        def _check(result):
            if False:
                i = 10
                return i + 15
            for i in range(self.num_iterations):
                self.assertTrue(result[i][1][0][0] == i, 'Value not returned')
        dlist.addCallback(_check)
        return dlist

    def _testPool_8(self, res):
        if False:
            return 10
        ds = []
        for i in range(self.num_iterations):
            sql = 'delete from simple where x = %d' % i
            ds.append(self.dbpool.runOperation(sql))
        dlist = defer.DeferredList(ds, fireOnOneErrback=True)
        return dlist

    def _testPool_9(self, res):
        if False:
            print('Hello World!')
        sql = 'select count(1) from simple'
        d = self.dbpool.runQuery(sql)

        def _check(row):
            if False:
                i = 10
                return i + 15
            self.assertTrue(int(row[0][0]) == 0, "Didn't successfully delete table contents")
            self.checkConnect()
        d.addCallback(_check)
        return d

    def checkConnect(self):
        if False:
            for i in range(10):
                print('nop')
        'Check the connect/disconnect synchronous calls.'
        conn = self.dbpool.connect()
        self.checkOpenfunCalled(conn)
        curs = conn.cursor()
        curs.execute('insert into simple(x) values(1)')
        curs.execute('select x from simple')
        res = curs.fetchall()
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 1)
        self.assertEqual(res[0][0], 1)
        curs.execute('delete from simple')
        curs.execute('select x from simple')
        self.assertEqual(len(curs.fetchall()), 0)
        curs.close()
        self.dbpool.disconnect(conn)

    def interaction(self, transaction):
        if False:
            print('Hello World!')
        transaction.execute('select x from simple order by x')
        for i in range(self.num_iterations):
            row = transaction.fetchone()
            self.assertTrue(len(row) == 1, 'Wrong size row')
            self.assertTrue(row[0] == i, 'Value not returned.')
        self.assertIsNone(transaction.fetchone(), 'Too many rows')
        return 'done'

    def bad_interaction(self, transaction):
        if False:
            while True:
                i = 10
        if self.can_rollback:
            transaction.execute('insert into simple(x) values(0)')
        transaction.execute('select * from NOTABLE')

    def withConnection(self, conn):
        if False:
            while True:
                i = 10
        curs = conn.cursor()
        try:
            curs.execute('select x from simple order by x')
            for i in range(self.num_iterations):
                row = curs.fetchone()
                self.assertTrue(len(row) == 1, 'Wrong size row')
                self.assertTrue(row[0] == i, 'Value not returned.')
        finally:
            curs.close()
        return 'done'

    def close_withConnection(self, conn):
        if False:
            print('Hello World!')
        conn.close()

    def bad_withConnection(self, conn):
        if False:
            while True:
                i = 10
        curs = conn.cursor()
        try:
            curs.execute('select * from NOTABLE')
        finally:
            curs.close()

class ReconnectTestBase:
    """
    Test the asynchronous DB-API code with reconnect.
    """
    if interfaces.IReactorThreads(reactor, None) is None:
        skip = 'ADB-API requires threads, no way to test without them'

    def extraSetUp(self):
        if False:
            while True:
                i = 10
        '\n        Skip the test if C{good_sql} is unavailable.  Otherwise, set up the\n        database, create a connection pool pointed at it, and set up a simple\n        schema in it.\n        '
        if self.good_sql is None:
            raise unittest.SkipTest('no good sql for reconnect test')
        self.startDB()
        self.dbpool = self.makePool(cp_max=1, cp_reconnect=True, cp_good_sql=self.good_sql)
        self.dbpool.start()
        return self.dbpool.runOperation(simple_table_schema)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        d = self.dbpool.runOperation('DROP TABLE simple')
        d.addCallback(lambda res: self.dbpool.close())
        d.addCallback(lambda res: self.stopDB())
        return d

    def test_pool(self):
        if False:
            print('Hello World!')
        d = defer.succeed(None)
        d.addCallback(self._testPool_1)
        d.addCallback(self._testPool_2)
        if not self.early_reconnect:
            d.addCallback(self._testPool_3)
        d.addCallback(self._testPool_4)
        d.addCallback(self._testPool_5)
        return d

    def _testPool_1(self, res):
        if False:
            print('Hello World!')
        sql = 'select count(1) from simple'
        d = self.dbpool.runQuery(sql)

        def _check(row):
            if False:
                i = 10
                return i + 15
            self.assertTrue(int(row[0][0]) == 0, 'Table not empty')
        d.addCallback(_check)
        return d

    def _testPool_2(self, res):
        if False:
            print('Hello World!')
        list(self.dbpool.connections.values())[0].close()

    def _testPool_3(self, res):
        if False:
            while True:
                i = 10
        sql = 'select count(1) from simple'
        d = defer.maybeDeferred(self.dbpool.runQuery, sql)
        d.addCallbacks(lambda res: self.fail('no exception'), lambda f: None)
        return d

    def _testPool_4(self, res):
        if False:
            while True:
                i = 10
        sql = 'select count(1) from simple'
        d = self.dbpool.runQuery(sql)

        def _check(row):
            if False:
                return 10
            self.assertTrue(int(row[0][0]) == 0, 'Table not empty')
        d.addCallback(_check)
        return d

    def _testPool_5(self, res):
        if False:
            i = 10
            return i + 15
        self.flushLoggedErrors()
        sql = 'select * from NOTABLE'
        d = defer.maybeDeferred(self.dbpool.runQuery, sql)
        d.addCallbacks(lambda res: self.fail('no exception'), lambda f: self.assertFalse(f.check(ConnectionLost)))
        return d

class DBTestConnector:
    """
    A class which knows how to test for the presence of
    and establish a connection to a relational database.

    To enable test cases  which use a central, system database,
    you must create a database named DB_NAME with a user DB_USER
    and password DB_PASS with full access rights to database DB_NAME.
    """
    TEST_PREFIX: Optional[str] = None
    DB_NAME = 'twisted_test'
    DB_USER = 'twisted_test'
    DB_PASS = 'twisted_test'
    DB_DIR = None
    nulls_ok = True
    trailing_spaces_ok = True
    can_rollback = True
    test_failures = True
    escape_slashes = True
    good_sql: Optional[str] = ConnectionPool.good_sql
    early_reconnect = True
    can_clear = True
    num_iterations = 50

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.DB_DIR = self.mktemp()
        os.mkdir(self.DB_DIR)
        if not self.can_connect():
            raise unittest.SkipTest('%s: Cannot access db' % self.TEST_PREFIX)
        return self.extraSetUp()

    def can_connect(self):
        if False:
            for i in range(10):
                print('nop')
        'Return true if this database is present on the system\n        and can be used in a test.'
        raise NotImplementedError()

    def startDB(self):
        if False:
            for i in range(10):
                print('nop')
        'Take any steps needed to bring database up.'
        pass

    def stopDB(self):
        if False:
            return 10
        'Bring database down, if needed.'
        pass

    def makePool(self, **newkw):
        if False:
            while True:
                i = 10
        'Create a connection pool with additional keyword arguments.'
        (args, kw) = self.getPoolArgs()
        kw = kw.copy()
        kw.update(newkw)
        return ConnectionPool(*args, **kw)

    def getPoolArgs(self):
        if False:
            i = 10
            return i + 15
        'Return a tuple (args, kw) of list and keyword arguments\n        that need to be passed to ConnectionPool to create a connection\n        to this database.'
        raise NotImplementedError()

class SQLite3Connector(DBTestConnector):
    """
    Connector that uses the stdlib SQLite3 database support.
    """
    TEST_PREFIX = 'SQLite3'
    escape_slashes = False
    num_iterations = 1

    def can_connect(self):
        if False:
            for i in range(10):
                print('nop')
        if requireModule('sqlite3') is None:
            return False
        else:
            return True

    def startDB(self):
        if False:
            return 10
        self.database = os.path.join(self.DB_DIR, self.DB_NAME)
        if os.path.exists(self.database):
            os.unlink(self.database)

    def getPoolArgs(self):
        if False:
            return 10
        args = ('sqlite3',)
        kw = {'database': self.database, 'cp_max': 1, 'check_same_thread': False}
        return (args, kw)

class PySQLite2Connector(DBTestConnector):
    """
    Connector that uses pysqlite's SQLite database support.
    """
    TEST_PREFIX = 'pysqlite2'
    escape_slashes = False
    num_iterations = 1

    def can_connect(self):
        if False:
            for i in range(10):
                print('nop')
        if requireModule('pysqlite2.dbapi2') is None:
            return False
        else:
            return True

    def startDB(self):
        if False:
            for i in range(10):
                print('nop')
        self.database = os.path.join(self.DB_DIR, self.DB_NAME)
        if os.path.exists(self.database):
            os.unlink(self.database)

    def getPoolArgs(self):
        if False:
            while True:
                i = 10
        args = ('pysqlite2.dbapi2',)
        kw = {'database': self.database, 'cp_max': 1, 'check_same_thread': False}
        return (args, kw)

class PyPgSQLConnector(DBTestConnector):
    TEST_PREFIX = 'PyPgSQL'

    def can_connect(self):
        if False:
            print('Hello World!')
        try:
            from pyPgSQL import PgSQL
        except BaseException:
            return False
        try:
            conn = PgSQL.connect(database=self.DB_NAME, user=self.DB_USER, password=self.DB_PASS)
            conn.close()
            return True
        except BaseException:
            return False

    def getPoolArgs(self):
        if False:
            while True:
                i = 10
        args = ('pyPgSQL.PgSQL',)
        kw = {'database': self.DB_NAME, 'user': self.DB_USER, 'password': self.DB_PASS, 'cp_min': 0}
        return (args, kw)

class PsycopgConnector(DBTestConnector):
    TEST_PREFIX = 'Psycopg'

    def can_connect(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            import psycopg
        except BaseException:
            return False
        try:
            conn = psycopg.connect(database=self.DB_NAME, user=self.DB_USER, password=self.DB_PASS)
            conn.close()
            return True
        except BaseException:
            return False

    def getPoolArgs(self):
        if False:
            return 10
        args = ('psycopg',)
        kw = {'database': self.DB_NAME, 'user': self.DB_USER, 'password': self.DB_PASS, 'cp_min': 0}
        return (args, kw)

class MySQLConnector(DBTestConnector):
    TEST_PREFIX = 'MySQL'
    trailing_spaces_ok = False
    can_rollback = False
    early_reconnect = False

    def can_connect(self):
        if False:
            i = 10
            return i + 15
        try:
            import MySQLdb
        except BaseException:
            return False
        try:
            conn = MySQLdb.connect(db=self.DB_NAME, user=self.DB_USER, passwd=self.DB_PASS)
            conn.close()
            return True
        except BaseException:
            return False

    def getPoolArgs(self):
        if False:
            print('Hello World!')
        args = ('MySQLdb',)
        kw = {'db': self.DB_NAME, 'user': self.DB_USER, 'passwd': self.DB_PASS}
        return (args, kw)

class FirebirdConnector(DBTestConnector):
    TEST_PREFIX = 'Firebird'
    test_failures = False
    escape_slashes = False
    good_sql = None
    can_clear = False
    num_iterations = 5

    def can_connect(self):
        if False:
            while True:
                i = 10
        if requireModule('kinterbasdb') is None:
            return False
        try:
            self.startDB()
            self.stopDB()
            return True
        except BaseException:
            return False

    def startDB(self):
        if False:
            print('Hello World!')
        import kinterbasdb
        self.DB_NAME = os.path.join(self.DB_DIR, DBTestConnector.DB_NAME)
        os.chmod(self.DB_DIR, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
        sql = 'create database "%s" user "%s" password "%s"'
        sql %= (self.DB_NAME, self.DB_USER, self.DB_PASS)
        conn = kinterbasdb.create_database(sql)
        conn.close()

    def getPoolArgs(self):
        if False:
            i = 10
            return i + 15
        args = ('kinterbasdb',)
        kw = {'database': self.DB_NAME, 'host': '127.0.0.1', 'user': self.DB_USER, 'password': self.DB_PASS}
        return (args, kw)

    def stopDB(self):
        if False:
            i = 10
            return i + 15
        import kinterbasdb
        conn = kinterbasdb.connect(database=self.DB_NAME, host='127.0.0.1', user=self.DB_USER, password=self.DB_PASS)
        conn.drop_database()

def makeSQLTests(base, suffix, globals):
    if False:
        while True:
            i = 10
    '\n    Make a test case for every db connector which can connect.\n\n    @param base: Base class for test case. Additional base classes\n                 will be a DBConnector subclass and unittest.TestCase\n    @param suffix: A suffix used to create test case names. Prefixes\n                   are defined in the DBConnector subclasses.\n    '
    connectors = [PySQLite2Connector, SQLite3Connector, PyPgSQLConnector, PsycopgConnector, MySQLConnector, FirebirdConnector]
    tests = {}
    for connclass in connectors:
        name = connclass.TEST_PREFIX + suffix

        class testcase(connclass, base, unittest.TestCase):
            __module__ = connclass.__module__
        testcase.__name__ = name
        if hasattr(connclass, '__qualname__'):
            testcase.__qualname__ = '.'.join(connclass.__qualname__.split()[0:-1] + [name])
        tests[name] = testcase
    globals.update(tests)
makeSQLTests(ADBAPITestBase, 'ADBAPITests', globals())
makeSQLTests(ReconnectTestBase, 'ReconnectTests', globals())

class FakePool:
    """
    A fake L{ConnectionPool} for tests.

    @ivar connectionFactory: factory for making connections returned by the
        C{connect} method.
    @type connectionFactory: any callable
    """
    reconnect = True
    noisy = True

    def __init__(self, connectionFactory):
        if False:
            return 10
        self.connectionFactory = connectionFactory

    def connect(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an instance of C{self.connectionFactory}.\n        '
        return self.connectionFactory()

    def disconnect(self, connection):
        if False:
            while True:
                i = 10
        '\n        Do nothing.\n        '

class ConnectionTests(unittest.TestCase):
    """
    Tests for the L{Connection} class.
    """

    def test_rollbackErrorLogged(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If an error happens during rollback, L{ConnectionLost} is raised but\n        the original error is logged.\n        '

        class ConnectionRollbackRaise:

            def rollback(self):
                if False:
                    print('Hello World!')
                raise RuntimeError('problem!')
        pool = FakePool(ConnectionRollbackRaise)
        connection = Connection(pool)
        self.assertRaises(ConnectionLost, connection.rollback)
        errors = self.flushLoggedErrors(RuntimeError)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].value.args[0], 'problem!')

class TransactionTests(unittest.TestCase):
    """
    Tests for the L{Transaction} class.
    """

    def test_reopenLogErrorIfReconnect(self):
        if False:
            print('Hello World!')
        '\n        If the cursor creation raises an error in L{Transaction.reopen}, it\n        reconnects but log the error occurred.\n        '

        class ConnectionCursorRaise:
            count = 0

            def reconnect(self):
                if False:
                    return 10
                pass

            def cursor(self):
                if False:
                    return 10
                if self.count == 0:
                    self.count += 1
                    raise RuntimeError('problem!')
        pool = FakePool(None)
        transaction = Transaction(pool, ConnectionCursorRaise())
        transaction.reopen()
        errors = self.flushLoggedErrors(RuntimeError)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].value.args[0], 'problem!')

class NonThreadPool:

    def callInThreadWithCallback(self, onResult, f, *a, **kw):
        if False:
            for i in range(10):
                print('nop')
        success = True
        try:
            result = f(*a, **kw)
        except Exception:
            success = False
            result = Failure()
        onResult(success, result)

class DummyConnectionPool(ConnectionPool):
    """
    A testable L{ConnectionPool};
    """
    threadpool = NonThreadPool()

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Don't forward init call.\n        "
        self._reactor = reactor

class EventReactor:
    """
    Partial L{IReactorCore} implementation with simple event-related
    methods.

    @ivar _running: A C{bool} indicating whether the reactor is pretending
        to have been started already or not.

    @ivar triggers: A C{list} of pending system event triggers.
    """

    def __init__(self, running):
        if False:
            for i in range(10):
                print('nop')
        self._running = running
        self.triggers = []

    def callWhenRunning(self, function):
        if False:
            print('Hello World!')
        if self._running:
            function()
        else:
            return self.addSystemEventTrigger('after', 'startup', function)

    def addSystemEventTrigger(self, phase, event, trigger):
        if False:
            i = 10
            return i + 15
        handle = (phase, event, trigger)
        self.triggers.append(handle)
        return handle

    def removeSystemEventTrigger(self, handle):
        if False:
            for i in range(10):
                print('nop')
        self.triggers.remove(handle)

class ConnectionPoolTests(unittest.TestCase):
    """
    Unit tests for L{ConnectionPool}.
    """

    def test_runWithConnectionRaiseOriginalError(self):
        if False:
            i = 10
            return i + 15
        '\n        If rollback fails, L{ConnectionPool.runWithConnection} raises the\n        original exception and log the error of the rollback.\n        '

        class ConnectionRollbackRaise:

            def __init__(self, pool):
                if False:
                    print('Hello World!')
                pass

            def rollback(self):
                if False:
                    print('Hello World!')
                raise RuntimeError('problem!')

        def raisingFunction(connection):
            if False:
                for i in range(10):
                    print('nop')
            raise ValueError('foo')
        pool = DummyConnectionPool()
        pool.connectionFactory = ConnectionRollbackRaise
        d = pool.runWithConnection(raisingFunction)
        d = self.assertFailure(d, ValueError)

        def cbFailed(ignored):
            if False:
                i = 10
                return i + 15
            errors = self.flushLoggedErrors(RuntimeError)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0].value.args[0], 'problem!')
        d.addCallback(cbFailed)
        return d

    def test_closeLogError(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ConnectionPool._close} logs exceptions.\n        '

        class ConnectionCloseRaise:

            def close(self):
                if False:
                    i = 10
                    return i + 15
                raise RuntimeError('problem!')
        pool = DummyConnectionPool()
        pool._close(ConnectionCloseRaise())
        errors = self.flushLoggedErrors(RuntimeError)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].value.args[0], 'problem!')

    def test_runWithInteractionRaiseOriginalError(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If rollback fails, L{ConnectionPool.runInteraction} raises the\n        original exception and log the error of the rollback.\n        '

        class ConnectionRollbackRaise:

            def __init__(self, pool):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def rollback(self):
                if False:
                    return 10
                raise RuntimeError('problem!')

        class DummyTransaction:

            def __init__(self, pool, connection):
                if False:
                    while True:
                        i = 10
                pass

        def raisingFunction(transaction):
            if False:
                for i in range(10):
                    print('nop')
            raise ValueError('foo')
        pool = DummyConnectionPool()
        pool.connectionFactory = ConnectionRollbackRaise
        pool.transactionFactory = DummyTransaction
        d = pool.runInteraction(raisingFunction)
        d = self.assertFailure(d, ValueError)

        def cbFailed(ignored):
            if False:
                i = 10
                return i + 15
            errors = self.flushLoggedErrors(RuntimeError)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0].value.args[0], 'problem!')
        d.addCallback(cbFailed)
        return d

    def test_unstartedClose(self):
        if False:
            print('Hello World!')
        "\n        If L{ConnectionPool.close} is called without L{ConnectionPool.start}\n        having been called, the pool's startup event is cancelled.\n        "
        reactor = EventReactor(False)
        pool = ConnectionPool('twisted.test.test_adbapi', cp_reactor=reactor)
        self.assertEqual(reactor.triggers, [('after', 'startup', pool._start)])
        pool.close()
        self.assertFalse(reactor.triggers)

    def test_startedClose(self):
        if False:
            i = 10
            return i + 15
        '\n        If L{ConnectionPool.close} is called after it has been started, but\n        not by its shutdown trigger, the shutdown trigger is cancelled.\n        '
        reactor = EventReactor(True)
        pool = ConnectionPool('twisted.test.test_adbapi', cp_reactor=reactor)
        self.assertEqual(reactor.triggers, [('during', 'shutdown', pool.finalClose)])
        pool.close()
        self.assertFalse(reactor.triggers)