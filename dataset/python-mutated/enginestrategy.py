"""
A wrapper around `sqlalchemy.create_engine` that handles all of the
special cases that Buildbot needs.  Those include:

 - pool_recycle for MySQL
 - %(basedir) substitution
 - optimal thread pool size calculation

"""
import os
import sqlalchemy as sa
from sqlalchemy.engine import url
from sqlalchemy.pool import NullPool
from twisted.python import log

class ReconnectingListener:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.retried = False

class Strategy:

    def set_up(self, u, engine):
        if False:
            print('Hello World!')
        pass

    def should_retry(self, operational_error):
        if False:
            return 10
        try:
            text = operational_error.args[0]
            return 'Lost connection' in text or 'database is locked' in text
        except Exception:
            return False

class SqlLiteStrategy(Strategy):

    def set_up(self, u, engine):
        if False:
            while True:
                i = 10
        'Special setup for sqlite engines'

        def connect_listener_enable_fk(connection, record):
            if False:
                i = 10
                return i + 15
            if not getattr(engine, 'fk_disabled', False):
                return
        sa.event.listen(engine.pool, 'connect', connect_listener_enable_fk)
        if u.database:

            def connect_listener(connection, record):
                if False:
                    while True:
                        i = 10
                connection.execute('pragma checkpoint_fullfsync = off')
            sa.event.listen(engine.pool, 'connect', connect_listener)
            log.msg("setting database journal mode to 'wal'")
            try:
                engine.execute('pragma journal_mode = wal')
            except Exception:
                log.msg('failed to set journal mode - database may fail')

class MySQLStrategy(Strategy):
    disconnect_error_codes = (2006, 2013, 2014, 2045, 2055)
    deadlock_error_codes = (1213,)

    def in_error_codes(self, args, error_codes):
        if False:
            i = 10
            return i + 15
        if args:
            return args[0] in error_codes
        return False

    def is_disconnect(self, args):
        if False:
            while True:
                i = 10
        return self.in_error_codes(args, self.disconnect_error_codes)

    def is_deadlock(self, args):
        if False:
            print('Hello World!')
        return self.in_error_codes(args, self.deadlock_error_codes)

    def set_up(self, u, engine):
        if False:
            while True:
                i = 10
        'Special setup for mysql engines'

        def checkout_listener(dbapi_con, con_record, con_proxy):
            if False:
                return 10
            try:
                cursor = dbapi_con.cursor()
                cursor.execute('SELECT 1')
            except dbapi_con.OperationalError as ex:
                if self.is_disconnect(ex.args):
                    log.msg('connection will be removed')
                    raise sa.exc.DisconnectionError()
                log.msg(f'exception happened {ex}')
                raise
        sa.event.listen(engine.pool, 'checkout', checkout_listener)

    def should_retry(self, ex):
        if False:
            return 10
        return any([self.is_disconnect(ex.orig.args), self.is_deadlock(ex.orig.args), super().should_retry(ex)])

def sa_url_set_attr(u, attr, value):
    if False:
        for i in range(10):
            print('nop')
    if hasattr(u, 'set'):
        return u.set(**{attr: value})
    setattr(u, attr, value)
    return u

def special_case_sqlite(u, kwargs):
    if False:
        for i in range(10):
            print('nop')
    'For sqlite, percent-substitute %(basedir)s and use a full\n    path to the basedir.  If using a memory database, force the\n    pool size to be 1.'
    max_conns = 1
    if u.database:
        kwargs.setdefault('poolclass', NullPool)
        database = u.database
        database = database % {'basedir': kwargs['basedir']}
        if not os.path.isabs(database[0]):
            database = os.path.join(kwargs['basedir'], database)
        u = sa_url_set_attr(u, 'database', database)
    else:
        kwargs.setdefault('connect_args', {})['check_same_thread'] = False
    if 'serialize_access' in u.query:
        query = dict(u.query)
        query.pop('serialize_access')
        u = sa_url_set_attr(u, 'query', query)
    return (u, kwargs, max_conns)

def special_case_mysql(u, kwargs):
    if False:
        for i in range(10):
            print('nop')
    "For mysql, take max_idle out of the query arguments, and\n    use its value for pool_recycle.  Also, force use_unicode and\n    charset to be True and 'utf8', failing if they were set to\n    anything else."
    query = dict(u.query)
    kwargs['pool_recycle'] = int(query.pop('max_idle', 3600))
    storage_engine = query.pop('storage_engine', 'MyISAM')
    kwargs['connect_args'] = {'init_command': f'SET default_storage_engine={storage_engine}'}
    if 'use_unicode' in query:
        if query['use_unicode'] != 'True':
            raise TypeError('Buildbot requires use_unicode=True ' + '(and adds it automatically)')
    else:
        query['use_unicode'] = 'True'
    if 'charset' in query:
        if query['charset'] != 'utf8':
            raise TypeError('Buildbot requires charset=utf8 ' + '(and adds it automatically)')
    else:
        query['charset'] = 'utf8'
    u = sa_url_set_attr(u, 'query', query)
    return (u, kwargs, None)

def get_drivers_strategy(drivername):
    if False:
        for i in range(10):
            print('nop')
    if drivername.startswith('sqlite'):
        return SqlLiteStrategy()
    elif drivername.startswith('mysql'):
        return MySQLStrategy()
    return Strategy()

def create_engine(name_or_url, **kwargs):
    if False:
        return 10
    if 'basedir' not in kwargs:
        raise TypeError('no basedir supplied to create_engine')
    max_conns = None
    u = url.make_url(name_or_url)
    if u.drivername.startswith('sqlite'):
        (u, kwargs, max_conns) = special_case_sqlite(u, kwargs)
    elif u.drivername.startswith('mysql'):
        (u, kwargs, max_conns) = special_case_mysql(u, kwargs)
    kwargs.pop('basedir')
    if max_conns is None:
        max_conns = kwargs.get('pool_size', 5) + kwargs.get('max_overflow', 10)
    driver_strategy = get_drivers_strategy(u.drivername)
    engine = sa.create_engine(u, **kwargs)
    driver_strategy.set_up(u, engine)
    engine.should_retry = driver_strategy.should_retry
    engine.optimal_thread_pool_size = max_conns
    return engine