"""Apache Cassandra result store backend using the DataStax driver."""
import threading
from celery import states
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import BaseBackend
try:
    import cassandra
    import cassandra.auth
    import cassandra.cluster
    import cassandra.query
except ImportError:
    cassandra = None
__all__ = ('CassandraBackend',)
logger = get_logger(__name__)
E_NO_CASSANDRA = '\nYou need to install the cassandra-driver library to\nuse the Cassandra backend.  See https://github.com/datastax/python-driver\n'
E_NO_SUCH_CASSANDRA_AUTH_PROVIDER = '\nCASSANDRA_AUTH_PROVIDER you provided is not a valid auth_provider class.\nSee https://datastax.github.io/python-driver/api/cassandra/auth.html.\n'
E_CASSANDRA_MISCONFIGURED = 'Cassandra backend improperly configured.'
E_CASSANDRA_NOT_CONFIGURED = 'Cassandra backend not configured.'
Q_INSERT_RESULT = '\nINSERT INTO {table} (\n    task_id, status, result, date_done, traceback, children) VALUES (\n        %s, %s, %s, %s, %s, %s) {expires};\n'
Q_SELECT_RESULT = '\nSELECT status, result, date_done, traceback, children\nFROM {table}\nWHERE task_id=%s\nLIMIT 1\n'
Q_CREATE_RESULT_TABLE = '\nCREATE TABLE {table} (\n    task_id text,\n    status text,\n    result blob,\n    date_done timestamp,\n    traceback blob,\n    children blob,\n    PRIMARY KEY ((task_id), date_done)\n) WITH CLUSTERING ORDER BY (date_done DESC);\n'
Q_EXPIRES = '\n    USING TTL {0}\n'

def buf_t(x):
    if False:
        for i in range(10):
            print('nop')
    return bytes(x, 'utf8')

class CassandraBackend(BaseBackend):
    """Cassandra/AstraDB backend utilizing DataStax driver.

    Raises:
        celery.exceptions.ImproperlyConfigured:
            if module :pypi:`cassandra-driver` is not available,
            or not-exactly-one of the :setting:`cassandra_servers` and
            the :setting:`cassandra_secure_bundle_path` settings is set.
    """
    servers = None
    bundle_path = None
    supports_autoexpire = True

    def __init__(self, servers=None, keyspace=None, table=None, entry_ttl=None, port=9042, bundle_path=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        if not cassandra:
            raise ImproperlyConfigured(E_NO_CASSANDRA)
        conf = self.app.conf
        self.servers = servers or conf.get('cassandra_servers', None)
        self.bundle_path = bundle_path or conf.get('cassandra_secure_bundle_path', None)
        self.port = port or conf.get('cassandra_port', None)
        self.keyspace = keyspace or conf.get('cassandra_keyspace', None)
        self.table = table or conf.get('cassandra_table', None)
        self.cassandra_options = conf.get('cassandra_options', {})
        db_directions = self.servers or self.bundle_path
        if not db_directions or not self.keyspace or (not self.table):
            raise ImproperlyConfigured(E_CASSANDRA_NOT_CONFIGURED)
        if self.servers and self.bundle_path:
            raise ImproperlyConfigured(E_CASSANDRA_MISCONFIGURED)
        expires = entry_ttl or conf.get('cassandra_entry_ttl', None)
        self.cqlexpires = Q_EXPIRES.format(expires) if expires is not None else ''
        read_cons = conf.get('cassandra_read_consistency') or 'LOCAL_QUORUM'
        write_cons = conf.get('cassandra_write_consistency') or 'LOCAL_QUORUM'
        self.read_consistency = getattr(cassandra.ConsistencyLevel, read_cons, cassandra.ConsistencyLevel.LOCAL_QUORUM)
        self.write_consistency = getattr(cassandra.ConsistencyLevel, write_cons, cassandra.ConsistencyLevel.LOCAL_QUORUM)
        self.auth_provider = None
        auth_provider = conf.get('cassandra_auth_provider', None)
        auth_kwargs = conf.get('cassandra_auth_kwargs', None)
        if auth_provider and auth_kwargs:
            auth_provider_class = getattr(cassandra.auth, auth_provider, None)
            if not auth_provider_class:
                raise ImproperlyConfigured(E_NO_SUCH_CASSANDRA_AUTH_PROVIDER)
            self.auth_provider = auth_provider_class(**auth_kwargs)
        self._cluster = None
        self._session = None
        self._write_stmt = None
        self._read_stmt = None
        self._lock = threading.RLock()

    def _get_connection(self, write=False):
        if False:
            i = 10
            return i + 15
        'Prepare the connection for action.\n\n        Arguments:\n            write (bool): are we a writer?\n        '
        if self._session is not None:
            return
        self._lock.acquire()
        try:
            if self._session is not None:
                return
            if self.servers:
                self._cluster = cassandra.cluster.Cluster(self.servers, port=self.port, auth_provider=self.auth_provider, **self.cassandra_options)
            else:
                self._cluster = cassandra.cluster.Cluster(cloud={'secure_connect_bundle': self.bundle_path}, auth_provider=self.auth_provider, **self.cassandra_options)
            self._session = self._cluster.connect(self.keyspace)
            self._write_stmt = cassandra.query.SimpleStatement(Q_INSERT_RESULT.format(table=self.table, expires=self.cqlexpires))
            self._write_stmt.consistency_level = self.write_consistency
            self._read_stmt = cassandra.query.SimpleStatement(Q_SELECT_RESULT.format(table=self.table))
            self._read_stmt.consistency_level = self.read_consistency
            if write:
                make_stmt = cassandra.query.SimpleStatement(Q_CREATE_RESULT_TABLE.format(table=self.table))
                make_stmt.consistency_level = self.write_consistency
                try:
                    self._session.execute(make_stmt)
                except cassandra.AlreadyExists:
                    pass
        except cassandra.OperationTimedOut:
            if self._cluster is not None:
                self._cluster.shutdown()
            self._cluster = None
            self._session = None
            raise
        finally:
            self._lock.release()

    def _store_result(self, task_id, result, state, traceback=None, request=None, **kwargs):
        if False:
            while True:
                i = 10
        'Store return value and state of an executed task.'
        self._get_connection(write=True)
        self._session.execute(self._write_stmt, (task_id, state, buf_t(self.encode(result)), self.app.now(), buf_t(self.encode(traceback)), buf_t(self.encode(self.current_task_children(request)))))

    def as_uri(self, include_password=True):
        if False:
            i = 10
            return i + 15
        return 'cassandra://'

    def _get_task_meta_for(self, task_id):
        if False:
            while True:
                i = 10
        'Get task meta-data for a task by id.'
        self._get_connection()
        res = self._session.execute(self._read_stmt, (task_id,)).one()
        if not res:
            return {'status': states.PENDING, 'result': None}
        (status, result, date_done, traceback, children) = res
        return self.meta_from_decoded({'task_id': task_id, 'status': status, 'result': self.decode(result), 'date_done': date_done, 'traceback': self.decode(traceback), 'children': self.decode(children)})

    def __reduce__(self, args=(), kwargs=None):
        if False:
            i = 10
            return i + 15
        kwargs = {} if not kwargs else kwargs
        kwargs.update({'servers': self.servers, 'keyspace': self.keyspace, 'table': self.table})
        return super().__reduce__(args, kwargs)