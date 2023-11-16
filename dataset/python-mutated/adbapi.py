"""
An asynchronous mapping to U{DB-API
2.0<http://www.python.org/topics/database/DatabaseAPI-2.0.html>}.
"""
from twisted.internet import threads
from twisted.python import log, reflect

class ConnectionLost(Exception):
    """
    This exception means that a db connection has been lost.  Client code may
    try again.
    """

class Connection:
    """
    A wrapper for a DB-API connection instance.

    The wrapper passes almost everything to the wrapped connection and so has
    the same API. However, the L{Connection} knows about its pool and also
    handle reconnecting should when the real connection dies.
    """

    def __init__(self, pool):
        if False:
            return 10
        self._pool = pool
        self._connection = None
        self.reconnect()

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def rollback(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._pool.reconnect:
            self._connection.rollback()
            return
        try:
            self._connection.rollback()
            curs = self._connection.cursor()
            curs.execute(self._pool.good_sql)
            curs.close()
            self._connection.commit()
            return
        except BaseException:
            log.err(None, 'Rollback failed')
        self._pool.disconnect(self._connection)
        if self._pool.noisy:
            log.msg('Connection lost.')
        raise ConnectionLost()

    def reconnect(self):
        if False:
            print('Hello World!')
        if self._connection is not None:
            self._pool.disconnect(self._connection)
        self._connection = self._pool.connect()

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        return getattr(self._connection, name)

class Transaction:
    """
    A lightweight wrapper for a DB-API 'cursor' object.

    Relays attribute access to the DB cursor. That is, you can call
    C{execute()}, C{fetchall()}, etc., and they will be called on the
    underlying DB-API cursor object. Attributes will also be retrieved from
    there.
    """
    _cursor = None

    def __init__(self, pool, connection):
        if False:
            i = 10
            return i + 15
        self._pool = pool
        self._connection = connection
        self.reopen()

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        _cursor = self._cursor
        self._cursor = None
        _cursor.close()

    def reopen(self):
        if False:
            print('Hello World!')
        if self._cursor is not None:
            self.close()
        try:
            self._cursor = self._connection.cursor()
            return
        except BaseException:
            if not self._pool.reconnect:
                raise
            else:
                log.err(None, 'Cursor creation failed')
        if self._pool.noisy:
            log.msg('Connection lost, reconnecting')
        self.reconnect()
        self._cursor = self._connection.cursor()

    def reconnect(self):
        if False:
            while True:
                i = 10
        self._connection.reconnect()
        self._cursor = None

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        return getattr(self._cursor, name)

class ConnectionPool:
    """
    Represent a pool of connections to a DB-API 2.0 compliant database.

    @ivar connectionFactory: factory for connections, default to L{Connection}.
    @type connectionFactory: any callable.

    @ivar transactionFactory: factory for transactions, default to
        L{Transaction}.
    @type transactionFactory: any callable

    @ivar shutdownID: L{None} or a handle on the shutdown event trigger which
        will be used to stop the connection pool workers when the reactor
        stops.

    @ivar _reactor: The reactor which will be used to schedule startup and
        shutdown events.
    @type _reactor: L{IReactorCore} provider
    """
    CP_ARGS = 'min max name noisy openfun reconnect good_sql'.split()
    noisy = False
    min = 3
    max = 5
    name = None
    openfun = None
    reconnect = False
    good_sql = 'select 1'
    running = False
    connectionFactory = Connection
    transactionFactory = Transaction
    shutdownID = None

    def __init__(self, dbapiName, *connargs, **connkw):
        if False:
            print('Hello World!')
        "\n        Create a new L{ConnectionPool}.\n\n        Any positional or keyword arguments other than those documented here\n        are passed to the DB-API object when connecting. Use these arguments to\n        pass database names, usernames, passwords, etc.\n\n        @param dbapiName: an import string to use to obtain a DB-API compatible\n            module (e.g. C{'pyPgSQL.PgSQL'})\n\n        @keyword cp_min: the minimum number of connections in pool (default 3)\n\n        @keyword cp_max: the maximum number of connections in pool (default 5)\n\n        @keyword cp_noisy: generate informational log messages during operation\n            (default C{False})\n\n        @keyword cp_openfun: a callback invoked after every C{connect()} on the\n            underlying DB-API object. The callback is passed a new DB-API\n            connection object. This callback can setup per-connection state\n            such as charset, timezone, etc.\n\n        @keyword cp_reconnect: detect connections which have failed and reconnect\n            (default C{False}). Failed connections may result in\n            L{ConnectionLost} exceptions, which indicate the query may need to\n            be re-sent.\n\n        @keyword cp_good_sql: an sql query which should always succeed and change\n            no state (default C{'select 1'})\n\n        @keyword cp_reactor: use this reactor instead of the global reactor\n            (added in Twisted 10.2).\n        @type cp_reactor: L{IReactorCore} provider\n        "
        self.dbapiName = dbapiName
        self.dbapi = reflect.namedModule(dbapiName)
        if getattr(self.dbapi, 'apilevel', None) != '2.0':
            log.msg('DB API module not DB API 2.0 compliant.')
        if getattr(self.dbapi, 'threadsafety', 0) < 1:
            log.msg('DB API module not sufficiently thread-safe.')
        reactor = connkw.pop('cp_reactor', None)
        if reactor is None:
            from twisted.internet import reactor
        self._reactor = reactor
        self.connargs = connargs
        self.connkw = connkw
        for arg in self.CP_ARGS:
            cpArg = f'cp_{arg}'
            if cpArg in connkw:
                setattr(self, arg, connkw[cpArg])
                del connkw[cpArg]
        self.min = min(self.min, self.max)
        self.max = max(self.min, self.max)
        self.connections = {}
        from twisted.python import threadable, threadpool
        self.threadID = threadable.getThreadID
        self.threadpool = threadpool.ThreadPool(self.min, self.max)
        self.startID = self._reactor.callWhenRunning(self._start)

    def _start(self):
        if False:
            i = 10
            return i + 15
        self.startID = None
        return self.start()

    def start(self):
        if False:
            i = 10
            return i + 15
        '\n        Start the connection pool.\n\n        If you are using the reactor normally, this function does *not*\n        need to be called.\n        '
        if not self.running:
            self.threadpool.start()
            self.shutdownID = self._reactor.addSystemEventTrigger('during', 'shutdown', self.finalClose)
            self.running = True

    def runWithConnection(self, func, *args, **kw):
        if False:
            i = 10
            return i + 15
        '\n        Execute a function with a database connection and return the result.\n\n        @param func: A callable object of one argument which will be executed\n            in a thread with a connection from the pool. It will be passed as\n            its first argument a L{Connection} instance (whose interface is\n            mostly identical to that of a connection object for your DB-API\n            module of choice), and its results will be returned as a\n            L{Deferred}. If the method raises an exception the transaction will\n            be rolled back. Otherwise, the transaction will be committed.\n            B{Note} that this function is B{not} run in the main thread: it\n            must be threadsafe.\n\n        @param args: positional arguments to be passed to func\n\n        @param kw: keyword arguments to be passed to func\n\n        @return: a L{Deferred} which will fire the return value of\n            C{func(Transaction(...), *args, **kw)}, or a\n            L{twisted.python.failure.Failure}.\n        '
        return threads.deferToThreadPool(self._reactor, self.threadpool, self._runWithConnection, func, *args, **kw)

    def _runWithConnection(self, func, *args, **kw):
        if False:
            return 10
        conn = self.connectionFactory(self)
        try:
            result = func(conn, *args, **kw)
            conn.commit()
            return result
        except BaseException:
            try:
                conn.rollback()
            except BaseException:
                log.err(None, 'Rollback failed')
            raise

    def runInteraction(self, interaction, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        "\n        Interact with the database and return the result.\n\n        The 'interaction' is a callable object which will be executed in a\n        thread using a pooled connection. It will be passed an L{Transaction}\n        object as an argument (whose interface is identical to that of the\n        database cursor for your DB-API module of choice), and its results will\n        be returned as a L{Deferred}. If running the method raises an\n        exception, the transaction will be rolled back. If the method returns a\n        value, the transaction will be committed.\n\n        NOTE that the function you pass is *not* run in the main thread: you\n        may have to worry about thread-safety in the function you pass to this\n        if it tries to use non-local objects.\n\n        @param interaction: a callable object whose first argument is an\n            L{adbapi.Transaction}.\n\n        @param args: additional positional arguments to be passed to\n            interaction\n\n        @param kw: keyword arguments to be passed to interaction\n\n        @return: a Deferred which will fire the return value of\n            C{interaction(Transaction(...), *args, **kw)}, or a\n            L{twisted.python.failure.Failure}.\n        "
        return threads.deferToThreadPool(self._reactor, self.threadpool, self._runInteraction, interaction, *args, **kw)

    def runQuery(self, *args, **kw):
        if False:
            print('Hello World!')
        "\n        Execute an SQL query and return the result.\n\n        A DB-API cursor which will be invoked with C{cursor.execute(*args,\n        **kw)}. The exact nature of the arguments will depend on the specific\n        flavor of DB-API being used, but the first argument in C{*args} be an\n        SQL statement. The result of a subsequent C{cursor.fetchall()} will be\n        fired to the L{Deferred} which is returned. If either the 'execute' or\n        'fetchall' methods raise an exception, the transaction will be rolled\n        back and a L{twisted.python.failure.Failure} returned.\n\n        The C{*args} and C{**kw} arguments will be passed to the DB-API\n        cursor's 'execute' method.\n\n        @return: a L{Deferred} which will fire the return value of a DB-API\n            cursor's 'fetchall' method, or a L{twisted.python.failure.Failure}.\n        "
        return self.runInteraction(self._runQuery, *args, **kw)

    def runOperation(self, *args, **kw):
        if False:
            while True:
                i = 10
        "\n        Execute an SQL query and return L{None}.\n\n        A DB-API cursor which will be invoked with C{cursor.execute(*args,\n        **kw)}. The exact nature of the arguments will depend on the specific\n        flavor of DB-API being used, but the first argument in C{*args} will be\n        an SQL statement. This method will not attempt to fetch any results\n        from the query and is thus suitable for C{INSERT}, C{DELETE}, and other\n        SQL statements which do not return values. If the 'execute' method\n        raises an exception, the transaction will be rolled back and a\n        L{Failure} returned.\n\n        The C{*args} and C{*kw} arguments will be passed to the DB-API cursor's\n        'execute' method.\n\n        @return: a L{Deferred} which will fire with L{None} or a\n            L{twisted.python.failure.Failure}.\n        "
        return self.runInteraction(self._runOperation, *args, **kw)

    def close(self):
        if False:
            while True:
                i = 10
        '\n        Close all pool connections and shutdown the pool.\n        '
        if self.shutdownID:
            self._reactor.removeSystemEventTrigger(self.shutdownID)
            self.shutdownID = None
        if self.startID:
            self._reactor.removeSystemEventTrigger(self.startID)
            self.startID = None
        self.finalClose()

    def finalClose(self):
        if False:
            i = 10
            return i + 15
        '\n        This should only be called by the shutdown trigger.\n        '
        self.shutdownID = None
        self.threadpool.stop()
        self.running = False
        for conn in self.connections.values():
            self._close(conn)
        self.connections.clear()

    def connect(self):
        if False:
            print('Hello World!')
        "\n        Return a database connection when one becomes available.\n\n        This method blocks and should be run in a thread from the internal\n        threadpool. Don't call this method directly from non-threaded code.\n        Using this method outside the external threadpool may exceed the\n        maximum number of connections in the pool.\n\n        @return: a database connection from the pool.\n        "
        tid = self.threadID()
        conn = self.connections.get(tid)
        if conn is None:
            if self.noisy:
                log.msg(f'adbapi connecting: {self.dbapiName}')
            conn = self.dbapi.connect(*self.connargs, **self.connkw)
            if self.openfun is not None:
                self.openfun(conn)
            self.connections[tid] = conn
        return conn

    def disconnect(self, conn):
        if False:
            i = 10
            return i + 15
        '\n        Disconnect a database connection associated with this pool.\n\n        Note: This function should only be used by the same thread which called\n        L{ConnectionPool.connect}. As with C{connect}, this function is not\n        used in normal non-threaded Twisted code.\n        '
        tid = self.threadID()
        if conn is not self.connections.get(tid):
            raise Exception('wrong connection for thread')
        if conn is not None:
            self._close(conn)
            del self.connections[tid]

    def _close(self, conn):
        if False:
            while True:
                i = 10
        if self.noisy:
            log.msg(f'adbapi closing: {self.dbapiName}')
        try:
            conn.close()
        except BaseException:
            log.err(None, 'Connection close failed')

    def _runInteraction(self, interaction, *args, **kw):
        if False:
            i = 10
            return i + 15
        conn = self.connectionFactory(self)
        trans = self.transactionFactory(self, conn)
        try:
            result = interaction(trans, *args, **kw)
            trans.close()
            conn.commit()
            return result
        except BaseException:
            try:
                conn.rollback()
            except BaseException:
                log.err(None, 'Rollback failed')
            raise

    def _runQuery(self, trans, *args, **kw):
        if False:
            print('Hello World!')
        trans.execute(*args, **kw)
        return trans.fetchall()

    def _runOperation(self, trans, *args, **kw):
        if False:
            while True:
                i = 10
        trans.execute(*args, **kw)

    def __getstate__(self):
        if False:
            while True:
                i = 10
        return {'dbapiName': self.dbapiName, 'min': self.min, 'max': self.max, 'noisy': self.noisy, 'reconnect': self.reconnect, 'good_sql': self.good_sql, 'connargs': self.connargs, 'connkw': self.connkw}

    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        self.__dict__ = state
        self.__init__(self.dbapiName, *self.connargs, **self.connkw)
__all__ = ['Transaction', 'ConnectionPool']