"""
.. dialect:: mysql+asyncmy
    :name: asyncmy
    :dbapi: asyncmy
    :connectstring: mysql+asyncmy://user:password@host:port/dbname[?key=value&key=value...]
    :url: https://github.com/long2ice/asyncmy

Using a special asyncio mediation layer, the asyncmy dialect is usable
as the backend for the :ref:`SQLAlchemy asyncio <asyncio_toplevel>`
extension package.

This dialect should normally be used only with the
:func:`_asyncio.create_async_engine` engine creation function::

    from sqlalchemy.ext.asyncio import create_async_engine
    engine = create_async_engine("mysql+asyncmy://user:pass@hostname/dbname?charset=utf8mb4")


"""
from contextlib import asynccontextmanager
from .pymysql import MySQLDialect_pymysql
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only

class AsyncAdapt_asyncmy_cursor:
    server_side = False
    __slots__ = ('_adapt_connection', '_connection', 'await_', '_cursor', '_rows')

    def __init__(self, adapt_connection):
        if False:
            for i in range(10):
                print('nop')
        self._adapt_connection = adapt_connection
        self._connection = adapt_connection._connection
        self.await_ = adapt_connection.await_
        cursor = self._connection.cursor()
        self._cursor = self.await_(cursor.__aenter__())
        self._rows = []

    @property
    def description(self):
        if False:
            print('Hello World!')
        return self._cursor.description

    @property
    def rowcount(self):
        if False:
            while True:
                i = 10
        return self._cursor.rowcount

    @property
    def arraysize(self):
        if False:
            i = 10
            return i + 15
        return self._cursor.arraysize

    @arraysize.setter
    def arraysize(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._cursor.arraysize = value

    @property
    def lastrowid(self):
        if False:
            print('Hello World!')
        return self._cursor.lastrowid

    def close(self):
        if False:
            print('Hello World!')
        self._rows[:] = []

    def execute(self, operation, parameters=None):
        if False:
            print('Hello World!')
        return self.await_(self._execute_async(operation, parameters))

    def executemany(self, operation, seq_of_parameters):
        if False:
            for i in range(10):
                print('nop')
        return self.await_(self._executemany_async(operation, seq_of_parameters))

    async def _execute_async(self, operation, parameters):
        async with self._adapt_connection._mutex_and_adapt_errors():
            if parameters is None:
                result = await self._cursor.execute(operation)
            else:
                result = await self._cursor.execute(operation, parameters)
            if not self.server_side:
                self._rows = list(await self._cursor.fetchall())
            return result

    async def _executemany_async(self, operation, seq_of_parameters):
        async with self._adapt_connection._mutex_and_adapt_errors():
            return await self._cursor.executemany(operation, seq_of_parameters)

    def setinputsizes(self, *inputsizes):
        if False:
            return 10
        pass

    def __iter__(self):
        if False:
            print('Hello World!')
        while self._rows:
            yield self._rows.pop(0)

    def fetchone(self):
        if False:
            while True:
                i = 10
        if self._rows:
            return self._rows.pop(0)
        else:
            return None

    def fetchmany(self, size=None):
        if False:
            print('Hello World!')
        if size is None:
            size = self.arraysize
        retval = self._rows[0:size]
        self._rows[:] = self._rows[size:]
        return retval

    def fetchall(self):
        if False:
            i = 10
            return i + 15
        retval = self._rows[:]
        self._rows[:] = []
        return retval

class AsyncAdapt_asyncmy_ss_cursor(AsyncAdapt_asyncmy_cursor):
    __slots__ = ()
    server_side = True

    def __init__(self, adapt_connection):
        if False:
            return 10
        self._adapt_connection = adapt_connection
        self._connection = adapt_connection._connection
        self.await_ = adapt_connection.await_
        cursor = self._connection.cursor(adapt_connection.dbapi.asyncmy.cursors.SSCursor)
        self._cursor = self.await_(cursor.__aenter__())

    def close(self):
        if False:
            i = 10
            return i + 15
        if self._cursor is not None:
            self.await_(self._cursor.close())
            self._cursor = None

    def fetchone(self):
        if False:
            while True:
                i = 10
        return self.await_(self._cursor.fetchone())

    def fetchmany(self, size=None):
        if False:
            return 10
        return self.await_(self._cursor.fetchmany(size=size))

    def fetchall(self):
        if False:
            for i in range(10):
                print('nop')
        return self.await_(self._cursor.fetchall())

class AsyncAdapt_asyncmy_connection(AdaptedConnection):
    await_ = staticmethod(await_only)
    __slots__ = ('dbapi', '_execute_mutex')

    def __init__(self, dbapi, connection):
        if False:
            print('Hello World!')
        self.dbapi = dbapi
        self._connection = connection
        self._execute_mutex = asyncio.Lock()

    @asynccontextmanager
    async def _mutex_and_adapt_errors(self):
        async with self._execute_mutex:
            try:
                yield
            except AttributeError:
                raise self.dbapi.InternalError('network operation failed due to asyncmy attribute error')

    def ping(self, reconnect):
        if False:
            while True:
                i = 10
        assert not reconnect
        return self.await_(self._do_ping())

    async def _do_ping(self):
        async with self._mutex_and_adapt_errors():
            return await self._connection.ping(False)

    def character_set_name(self):
        if False:
            i = 10
            return i + 15
        return self._connection.character_set_name()

    def autocommit(self, value):
        if False:
            while True:
                i = 10
        self.await_(self._connection.autocommit(value))

    def cursor(self, server_side=False):
        if False:
            i = 10
            return i + 15
        if server_side:
            return AsyncAdapt_asyncmy_ss_cursor(self)
        else:
            return AsyncAdapt_asyncmy_cursor(self)

    def rollback(self):
        if False:
            return 10
        self.await_(self._connection.rollback())

    def commit(self):
        if False:
            return 10
        self.await_(self._connection.commit())

    def close(self):
        if False:
            print('Hello World!')
        self._connection.close()

class AsyncAdaptFallback_asyncmy_connection(AsyncAdapt_asyncmy_connection):
    __slots__ = ()
    await_ = staticmethod(await_fallback)

def _Binary(x):
    if False:
        while True:
            i = 10
    'Return x as a binary type.'
    return bytes(x)

class AsyncAdapt_asyncmy_dbapi:

    def __init__(self, asyncmy):
        if False:
            while True:
                i = 10
        self.asyncmy = asyncmy
        self.paramstyle = 'format'
        self._init_dbapi_attributes()

    def _init_dbapi_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        for name in ('Warning', 'Error', 'InterfaceError', 'DataError', 'DatabaseError', 'OperationalError', 'InterfaceError', 'IntegrityError', 'ProgrammingError', 'InternalError', 'NotSupportedError'):
            setattr(self, name, getattr(self.asyncmy.errors, name))
    STRING = util.symbol('STRING')
    NUMBER = util.symbol('NUMBER')
    BINARY = util.symbol('BINARY')
    DATETIME = util.symbol('DATETIME')
    TIMESTAMP = util.symbol('TIMESTAMP')
    Binary = staticmethod(_Binary)

    def connect(self, *arg, **kw):
        if False:
            print('Hello World!')
        async_fallback = kw.pop('async_fallback', False)
        creator_fn = kw.pop('async_creator_fn', self.asyncmy.connect)
        if util.asbool(async_fallback):
            return AsyncAdaptFallback_asyncmy_connection(self, await_fallback(creator_fn(*arg, **kw)))
        else:
            return AsyncAdapt_asyncmy_connection(self, await_only(creator_fn(*arg, **kw)))

class MySQLDialect_asyncmy(MySQLDialect_pymysql):
    driver = 'asyncmy'
    supports_statement_cache = True
    supports_server_side_cursors = True
    _sscursor = AsyncAdapt_asyncmy_ss_cursor
    is_async = True

    @classmethod
    def import_dbapi(cls):
        if False:
            while True:
                i = 10
        return AsyncAdapt_asyncmy_dbapi(__import__('asyncmy'))

    @classmethod
    def get_pool_class(cls, url):
        if False:
            i = 10
            return i + 15
        async_fallback = url.query.get('async_fallback', False)
        if util.asbool(async_fallback):
            return pool.FallbackAsyncAdaptedQueuePool
        else:
            return pool.AsyncAdaptedQueuePool

    def create_connect_args(self, url):
        if False:
            return 10
        return super().create_connect_args(url, _translate_args=dict(username='user', database='db'))

    def is_disconnect(self, e, connection, cursor):
        if False:
            return 10
        if super().is_disconnect(e, connection, cursor):
            return True
        else:
            str_e = str(e).lower()
            return 'not connected' in str_e or 'network operation failed' in str_e

    def _found_rows_client_flag(self):
        if False:
            i = 10
            return i + 15
        from asyncmy.constants import CLIENT
        return CLIENT.FOUND_ROWS

    def get_driver_connection(self, connection):
        if False:
            i = 10
            return i + 15
        return connection._connection
dialect = MySQLDialect_asyncmy