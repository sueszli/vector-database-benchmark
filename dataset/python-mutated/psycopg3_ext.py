from peewee import *
try:
    import psycopg
except ImportError:
    psycopg = None

class Psycopg3Database(PostgresqlDatabase):

    def _connect(self):
        if False:
            while True:
                i = 10
        if psycopg is None:
            raise ImproperlyConfigured('psycopg3 is not installed!')
        conn = psycopg.connect(dbname=self.database, **self.connect_params)
        if self._isolation_level is not None:
            conn.isolation_level = self._isolation_level
        conn.autocommit = True
        return conn

    def get_binary_type(self):
        if False:
            return 10
        return psycopg.Binary

    def _set_server_version(self, conn):
        if False:
            print('Hello World!')
        self.server_version = conn.pgconn.server_version
        if self.server_version >= 90600:
            self.safe_create_index = True

    def is_connection_usable(self):
        if False:
            while True:
                i = 10
        if self._state.closed:
            return False
        conn = self._state.conn
        return conn.pgconn.transaction_status < conn.TransactionStatus.INERROR