"""
The PostgreSQL connector is a connectivity layer between the OpenERP code and
the database, *not* a database abstraction toolkit. Database abstraction is what
the ORM does, in fact.
"""
from contextlib import contextmanager
from functools import wraps
import logging
import time
import urlparse
import uuid
import psycopg2
import psycopg2.extras
import psycopg2.extensions
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, ISOLATION_LEVEL_READ_COMMITTED, ISOLATION_LEVEL_REPEATABLE_READ
from psycopg2.pool import PoolError
psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)
_logger = logging.getLogger(__name__)
types_mapping = {'date': (1082,), 'time': (1083,), 'datetime': (1114,)}

def unbuffer(symb, cr):
    if False:
        while True:
            i = 10
    if symb is None:
        return None
    return str(symb)

def undecimalize(symb, cr):
    if False:
        i = 10
        return i + 15
    if symb is None:
        return None
    return float(symb)
for (name, typeoid) in types_mapping.items():
    psycopg2.extensions.register_type(psycopg2.extensions.new_type(typeoid, name, lambda x, cr: x))
psycopg2.extensions.register_type(psycopg2.extensions.new_type((700, 701, 1700), 'float', undecimalize))
import tools
from tools import parse_version as pv
if pv(psycopg2.__version__) < pv('2.7'):
    from psycopg2._psycopg import QuotedString

    def adapt_string(adapted):
        if False:
            i = 10
            return i + 15
        'Python implementation of psycopg/psycopg2#459 from v2.7'
        if '\x00' in adapted:
            raise ValueError('A string literal cannot contain NUL (0x00) characters.')
        return QuotedString(adapted)
    psycopg2.extensions.register_adapter(str, adapt_string)
    psycopg2.extensions.register_adapter(unicode, adapt_string)
from tools.func import frame_codeinfo
from datetime import timedelta
import threading
from inspect import currentframe
import re
re_from = re.compile('.* from "?([a-zA-Z_0-9]+)"? .*$')
re_into = re.compile('.* into "?([a-zA-Z_0-9]+)"? .*$')
sql_counter = 0

class Cursor(object):
    """Represents an open transaction to the PostgreSQL DB backend,
       acting as a lightweight wrapper around psycopg2's
       ``cursor`` objects.

        ``Cursor`` is the object behind the ``cr`` variable used all
        over the OpenERP code.

        .. rubric:: Transaction Isolation

        One very important property of database transactions is the
        level of isolation between concurrent transactions.
        The SQL standard defines four levels of transaction isolation,
        ranging from the most strict *Serializable* level, to the least
        strict *Read Uncommitted* level. These levels are defined in
        terms of the phenomena that must not occur between concurrent
        transactions, such as *dirty read*, etc.
        In the context of a generic business data management software
        such as OpenERP, we need the best guarantees that no data
        corruption can ever be cause by simply running multiple
        transactions in parallel. Therefore, the preferred level would
        be the *serializable* level, which ensures that a set of
        transactions is guaranteed to produce the same effect as
        running them one at a time in some order.

        However, most database management systems implement a limited
        serializable isolation in the form of
        `snapshot isolation <http://en.wikipedia.org/wiki/Snapshot_isolation>`_,
        providing most of the same advantages as True Serializability,
        with a fraction of the performance cost.
        With PostgreSQL up to version 9.0, this snapshot isolation was
        the implementation of both the ``REPEATABLE READ`` and
        ``SERIALIZABLE`` levels of the SQL standard.
        As of PostgreSQL 9.1, the previous snapshot isolation implementation
        was kept for ``REPEATABLE READ``, while a new ``SERIALIZABLE``
        level was introduced, providing some additional heuristics to
        detect a concurrent update by parallel transactions, and forcing
        one of them to rollback.

        OpenERP implements its own level of locking protection
        for transactions that are highly likely to provoke concurrent
        updates, such as stock reservations or document sequences updates.
        Therefore we mostly care about the properties of snapshot isolation,
        but we don't really need additional heuristics to trigger transaction
        rollbacks, as we are taking care of triggering instant rollbacks
        ourselves when it matters (and we can save the additional performance
        hit of these heuristics).

        As a result of the above, we have selected ``REPEATABLE READ`` as
        the default transaction isolation level for OpenERP cursors, as
        it will be mapped to the desired ``snapshot isolation`` level for
        all supported PostgreSQL version (8.3 - 9.x).

        Note: up to psycopg2 v.2.4.2, psycopg2 itself remapped the repeatable
        read level to serializable before sending it to the database, so it would
        actually select the new serializable mode on PostgreSQL 9.1. Make
        sure you use psycopg2 v2.4.2 or newer if you use PostgreSQL 9.1 and
        the performance hit is a concern for you.

        .. attribute:: cache

            Cache dictionary with a "request" (-ish) lifecycle, only lives as
            long as the cursor itself does and proactively cleared when the
            cursor is closed.

            This cache should *only* be used to store repeatable reads as it
            ignores rollbacks and savepoints, it should not be used to store
            *any* data which may be modified during the life of the cursor.

    """
    IN_MAX = 1000

    def check(f):
        if False:
            print('Hello World!')

        @wraps(f)
        def wrapper(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if self._closed:
                msg = 'Unable to use a closed cursor.'
                if self.__closer:
                    msg += ' It was closed at %s, line %s' % self.__closer
                raise psycopg2.OperationalError(msg)
            return f(self, *args, **kwargs)
        return wrapper

    def __init__(self, pool, dbname, dsn, serialized=True):
        if False:
            return 10
        self.sql_from_log = {}
        self.sql_into_log = {}
        self.sql_log = _logger.isEnabledFor(logging.DEBUG)
        self.sql_log_count = 0
        self._closed = True
        self.__pool = pool
        self.dbname = dbname
        self._serialized = serialized
        self._cnx = pool.borrow(dsn)
        self._obj = self._cnx.cursor()
        if self.sql_log:
            self.__caller = frame_codeinfo(currentframe(), 2)
        else:
            self.__caller = False
        self._closed = False
        self.autocommit(False)
        self.__closer = False
        self._default_log_exceptions = True
        self.cache = {}
        self._event_handlers = {'commit': [], 'rollback': []}

    def __build_dict(self, row):
        if False:
            return 10
        return {d.name: row[i] for (i, d) in enumerate(self._obj.description)}

    def dictfetchone(self):
        if False:
            while True:
                i = 10
        row = self._obj.fetchone()
        return row and self.__build_dict(row)

    def dictfetchmany(self, size):
        if False:
            print('Hello World!')
        return map(self.__build_dict, self._obj.fetchmany(size))

    def dictfetchall(self):
        if False:
            while True:
                i = 10
        return map(self.__build_dict, self._obj.fetchall())

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._closed and (not self._cnx.closed):
            msg = 'Cursor not closed explicitly\n'
            if self.__caller:
                msg += 'Cursor was created at %s:%s' % self.__caller
            else:
                msg += 'Please enable sql debugging to trace the caller.'
            _logger.warning(msg)
            self._close(True)

    @check
    def execute(self, query, params=None, log_exceptions=None):
        if False:
            print('Hello World!')
        if params and (not isinstance(params, (tuple, list, dict))):
            raise ValueError('SQL query parameters should be a tuple, list or dict; got %r' % (params,))
        if self.sql_log:
            now = time.time()
            _logger.debug('query: %s', query)
        try:
            params = params or None
            res = self._obj.execute(query, params)
        except Exception:
            if self._default_log_exceptions if log_exceptions is None else log_exceptions:
                _logger.info('bad query: %s', self._obj.query or query)
            raise
        self.sql_log_count += 1
        if self.sql_log:
            delay = (time.time() - now) * 1000000.0
            res_from = re_from.match(query.lower())
            if res_from:
                self.sql_from_log.setdefault(res_from.group(1), [0, 0])
                self.sql_from_log[res_from.group(1)][0] += 1
                self.sql_from_log[res_from.group(1)][1] += delay
            res_into = re_into.match(query.lower())
            if res_into:
                self.sql_into_log.setdefault(res_into.group(1), [0, 0])
                self.sql_into_log[res_into.group(1)][0] += 1
                self.sql_into_log[res_into.group(1)][1] += delay
        return res

    def split_for_in_conditions(self, ids, size=None):
        if False:
            print('Hello World!')
        'Split a list of identifiers into one or more smaller tuples\n           safe for IN conditions, after uniquifying them.'
        return tools.misc.split_every(size or self.IN_MAX, ids)

    def print_log(self):
        if False:
            return 10
        global sql_counter
        if not self.sql_log:
            return

        def process(type):
            if False:
                print('Hello World!')
            sqllogs = {'from': self.sql_from_log, 'into': self.sql_into_log}
            sum = 0
            if sqllogs[type]:
                sqllogitems = sqllogs[type].items()
                sqllogitems.sort(key=lambda k: k[1][1])
                _logger.debug('SQL LOG %s:', type)
                sqllogitems.sort(lambda x, y: cmp(x[1][0], y[1][0]))
                for r in sqllogitems:
                    delay = timedelta(microseconds=r[1][1])
                    _logger.debug('table: %s: %s/%s', r[0], delay, r[1][0])
                    sum += r[1][1]
                sqllogs[type].clear()
            sum = timedelta(microseconds=sum)
            _logger.debug('SUM %s:%s/%d [%d]', type, sum, self.sql_log_count, sql_counter)
            sqllogs[type].clear()
        process('from')
        process('into')
        self.sql_log_count = 0
        self.sql_log = False

    @check
    def close(self):
        if False:
            i = 10
            return i + 15
        return self._close(False)

    def _close(self, leak=False):
        if False:
            return 10
        global sql_counter
        if not self._obj:
            return
        del self.cache
        if self.sql_log:
            self.__closer = frame_codeinfo(currentframe(), 3)
        sql_counter += self.sql_log_count
        self.print_log()
        self._obj.close()
        del self._obj
        self._closed = True
        self._cnx.rollback()
        if leak:
            self._cnx.leaked = True
        else:
            chosen_template = tools.config['db_template']
            templates_list = tuple(set(['template0', 'template1', 'postgres', chosen_template]))
            keep_in_pool = self.dbname not in templates_list
            self.__pool.give_back(self._cnx, keep_in_pool=keep_in_pool)

    @check
    def autocommit(self, on):
        if False:
            return 10
        if on:
            isolation_level = ISOLATION_LEVEL_AUTOCOMMIT
        else:
            isolation_level = ISOLATION_LEVEL_REPEATABLE_READ if self._serialized else ISOLATION_LEVEL_READ_COMMITTED
        self._cnx.set_isolation_level(isolation_level)

    @check
    def after(self, event, func):
        if False:
            while True:
                i = 10
        " Register an event handler.\n\n            :param event: the event, either `'commit'` or `'rollback'`\n            :param func: a callable object, called with no argument after the\n                event occurs\n\n            Be careful when coding an event handler, since any operation on the\n            cursor that was just committed/rolled back will take place in the\n            next transaction that has already begun, and may still be rolled\n            back or committed independently. You may consider the use of a\n            dedicated temporary cursor to do some database operation.\n        "
        self._event_handlers[event].append(func)

    def _pop_event_handlers(self):
        if False:
            while True:
                i = 10
        result = self._event_handlers
        self._event_handlers = {'commit': [], 'rollback': []}
        return result

    @check
    def commit(self):
        if False:
            i = 10
            return i + 15
        ' Perform an SQL `COMMIT`\n        '
        result = self._cnx.commit()
        for func in self._pop_event_handlers()['commit']:
            func()
        return result

    @check
    def rollback(self):
        if False:
            i = 10
            return i + 15
        ' Perform an SQL `ROLLBACK`\n        '
        result = self._cnx.rollback()
        for func in self._pop_event_handlers()['rollback']:
            func()
        return result

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        ' Using the cursor as a contextmanager automatically commits and\n            closes it::\n\n                with cr:\n                    cr.execute(...)\n\n                # cr is committed if no failure occurred\n                # cr is closed in any case\n        '
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            return 10
        if exc_type is None:
            self.commit()
        self.close()

    @contextmanager
    @check
    def savepoint(self):
        if False:
            while True:
                i = 10
        'context manager entering in a new savepoint'
        name = uuid.uuid1().hex
        self.execute('SAVEPOINT "%s"' % name)
        try:
            yield
        except Exception:
            self.execute('ROLLBACK TO SAVEPOINT "%s"' % name)
            raise
        else:
            self.execute('RELEASE SAVEPOINT "%s"' % name)

    @check
    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._obj, name)

    @property
    def closed(self):
        if False:
            print('Hello World!')
        return self._closed

class TestCursor(Cursor):
    """ A cursor to be used for tests. It keeps the transaction open across
        several requests, and simulates committing, rolling back, and closing.
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(TestCursor, self).__init__(*args, **kwargs)
        self.execute('SAVEPOINT test_cursor')
        self._lock = threading.RLock()

    def acquire(self):
        if False:
            for i in range(10):
                print('nop')
        self._lock.acquire()

    def release(self):
        if False:
            while True:
                i = 10
        self._lock.release()

    def force_close(self):
        if False:
            print('Hello World!')
        super(TestCursor, self).close()

    def close(self):
        if False:
            print('Hello World!')
        if not self._closed:
            self.rollback()
        self.release()

    def autocommit(self, on):
        if False:
            while True:
                i = 10
        _logger.debug('TestCursor.autocommit(%r) does nothing', on)

    def commit(self):
        if False:
            print('Hello World!')
        self.execute('RELEASE SAVEPOINT test_cursor')
        self.execute('SAVEPOINT test_cursor')

    def rollback(self):
        if False:
            return 10
        self.execute('ROLLBACK TO SAVEPOINT test_cursor')
        self.execute('SAVEPOINT test_cursor')

class LazyCursor(object):
    """ A proxy object to a cursor. The cursor itself is allocated only if it is
        needed. This class is useful for cached methods, that use the cursor
        only in the case of a cache miss.
    """

    def __init__(self, dbname=None):
        if False:
            i = 10
            return i + 15
        self._dbname = dbname
        self._cursor = None
        self._depth = 0

    @property
    def dbname(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dbname or threading.currentThread().dbname

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        cr = self._cursor
        if cr is None:
            from odoo import registry
            cr = self._cursor = registry(self.dbname).cursor()
            for _ in xrange(self._depth):
                cr.__enter__()
        return getattr(cr, name)

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self._depth += 1
        if self._cursor is not None:
            self._cursor.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            print('Hello World!')
        self._depth -= 1
        if self._cursor is not None:
            self._cursor.__exit__(exc_type, exc_value, traceback)

class PsycoConnection(psycopg2.extensions.connection):
    pass

class ConnectionPool(object):
    """ The pool of connections to database(s)

        Keep a set of connections to pg databases open, and reuse them
        to open cursors for all transactions.

        The connections are *not* automatically closed. Only a close_db()
        can trigger that.
    """

    def locked(fun):
        if False:
            return 10

        @wraps(fun)
        def _locked(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            self._lock.acquire()
            try:
                return fun(self, *args, **kwargs)
            finally:
                self._lock.release()
        return _locked

    def __init__(self, maxconn=64):
        if False:
            i = 10
            return i + 15
        self._connections = []
        self._maxconn = max(maxconn, 1)
        self._lock = threading.Lock()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        used = len([1 for (c, u) in self._connections[:] if u])
        count = len(self._connections)
        return 'ConnectionPool(used=%d/count=%d/max=%d)' % (used, count, self._maxconn)

    def _debug(self, msg, *args):
        if False:
            i = 10
            return i + 15
        _logger.debug('%r ' + msg, self, *args)

    @locked
    def borrow(self, connection_info):
        if False:
            i = 10
            return i + 15
        '\n        :param dict connection_info: dict of psql connection keywords\n        :rtype: PsycoConnection\n        '
        for (i, (cnx, _)) in tools.reverse_enumerate(self._connections):
            if cnx.closed:
                self._connections.pop(i)
                self._debug('Removing closed connection at index %d: %r', i, cnx.dsn)
                continue
            if getattr(cnx, 'leaked', False):
                delattr(cnx, 'leaked')
                self._connections.pop(i)
                self._connections.append((cnx, False))
                _logger.info('%r: Free leaked connection to %r', self, cnx.dsn)
        for (i, (cnx, used)) in enumerate(self._connections):
            if not used and cnx._original_dsn == connection_info:
                try:
                    cnx.reset()
                except psycopg2.OperationalError:
                    self._debug('Cannot reset connection at index %d: %r', i, cnx.dsn)
                    if not cnx.closed:
                        cnx.close()
                    continue
                self._connections.pop(i)
                self._connections.append((cnx, True))
                self._debug('Borrow existing connection to %r at index %d', cnx.dsn, i)
                return cnx
        if len(self._connections) >= self._maxconn:
            for (i, (cnx, used)) in enumerate(self._connections):
                if not used:
                    self._connections.pop(i)
                    if not cnx.closed:
                        cnx.close()
                    self._debug('Removing old connection at index %d: %r', i, cnx.dsn)
                    break
            else:
                raise PoolError('The Connection Pool Is Full')
        try:
            result = psycopg2.connect(connection_factory=PsycoConnection, **connection_info)
        except psycopg2.Error:
            _logger.info('Connection to the database failed')
            raise
        result._original_dsn = connection_info
        self._connections.append((result, True))
        self._debug('Create new connection')
        return result

    @locked
    def give_back(self, connection, keep_in_pool=True):
        if False:
            for i in range(10):
                print('nop')
        self._debug('Give back connection to %r', connection.dsn)
        for (i, (cnx, used)) in enumerate(self._connections):
            if cnx is connection:
                self._connections.pop(i)
                if keep_in_pool:
                    self._connections.append((cnx, False))
                    self._debug('Put connection to %r in pool', cnx.dsn)
                else:
                    self._debug('Forgot connection to %r', cnx.dsn)
                    cnx.close()
                break
        else:
            raise PoolError('This connection does not below to the pool')

    @locked
    def close_all(self, dsn=None):
        if False:
            for i in range(10):
                print('nop')
        count = 0
        last = None
        for (i, (cnx, used)) in tools.reverse_enumerate(self._connections):
            if dsn is None or cnx._original_dsn == dsn:
                cnx.close()
                last = self._connections.pop(i)[0]
                count += 1
        _logger.info('%r: Closed %d connections %s', self, count, dsn and last and 'to %r' % last.dsn or '')

class Connection(object):
    """ A lightweight instance of a connection to postgres
    """

    def __init__(self, pool, dbname, dsn):
        if False:
            print('Hello World!')
        self.dbname = dbname
        self.dsn = dsn
        self.__pool = pool

    def cursor(self, serialized=True):
        if False:
            i = 10
            return i + 15
        cursor_type = serialized and 'serialized ' or ''
        _logger.debug('create %scursor to %r', cursor_type, self.dsn)
        return Cursor(self.__pool, self.dbname, self.dsn, serialized=serialized)

    def test_cursor(self, serialized=True):
        if False:
            i = 10
            return i + 15
        cursor_type = serialized and 'serialized ' or ''
        _logger.debug('create test %scursor to %r', cursor_type, self.dsn)
        return TestCursor(self.__pool, self.dbname, self.dsn, serialized=serialized)
    serialized_cursor = cursor

    def __nonzero__(self):
        if False:
            return 10
        'Check if connection is possible'
        try:
            _logger.info('__nonzero__() is deprecated. (It is too expensive to test a connection.)')
            cr = self.cursor()
            cr.close()
            return True
        except Exception:
            return False

def connection_info_for(db_or_uri):
    if False:
        while True:
            i = 10
    ' parse the given `db_or_uri` and return a 2-tuple (dbname, connection_params)\n\n    Connection params are either a dictionary with a single key ``dsn``\n    containing a connection URI, or a dictionary containing connection\n    parameter keywords which psycopg2 can build a key/value connection string\n    (dsn) from\n\n    :param str db_or_uri: database name or postgres dsn\n    :rtype: (str, dict)\n    '
    if db_or_uri.startswith(('postgresql://', 'postgres://')):
        us = urlparse.urlsplit(db_or_uri)
        if len(us.path) > 1:
            db_name = us.path[1:]
        elif us.username:
            db_name = us.username
        else:
            db_name = us.hostname
        return (db_name, {'dsn': db_or_uri})
    connection_info = {'database': db_or_uri}
    for p in ('host', 'port', 'user', 'password'):
        cfg = tools.config['db_' + p]
        if cfg:
            connection_info[p] = cfg
    return (db_or_uri, connection_info)
_Pool = None

def db_connect(to, allow_uri=False):
    if False:
        while True:
            i = 10
    global _Pool
    if _Pool is None:
        _Pool = ConnectionPool(int(tools.config['db_maxconn']))
    (db, info) = connection_info_for(to)
    if not allow_uri and db != to:
        raise ValueError('URI connections not allowed')
    return Connection(_Pool, db, info)

def close_db(db_name):
    if False:
        while True:
            i = 10
    ' You might want to call odoo.modules.registry.Registry.delete(db_name) along this function.'
    global _Pool
    if _Pool:
        _Pool.close_all(connection_info_for(db_name)[1])

def close_all():
    if False:
        i = 10
        return i + 15
    global _Pool
    if _Pool:
        _Pool.close_all()