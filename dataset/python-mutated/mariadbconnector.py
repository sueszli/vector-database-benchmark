"""

.. dialect:: mysql+mariadbconnector
    :name: MariaDB Connector/Python
    :dbapi: mariadb
    :connectstring: mariadb+mariadbconnector://<user>:<password>@<host>[:<port>]/<dbname>
    :url: https://pypi.org/project/mariadb/

Driver Status
-------------

MariaDB Connector/Python enables Python programs to access MariaDB and MySQL
databases using an API which is compliant with the Python DB API 2.0 (PEP-249).
It is written in C and uses MariaDB Connector/C client library for client server
communication.

Note that the default driver for a ``mariadb://`` connection URI continues to
be ``mysqldb``. ``mariadb+mariadbconnector://`` is required to use this driver.

.. mariadb: https://github.com/mariadb-corporation/mariadb-connector-python

"""
import re
from uuid import UUID as _python_UUID
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLExecutionContext
from ... import sql
from ... import util
from ...sql import sqltypes
mariadb_cpy_minimum_version = (1, 0, 1)

class _MariaDBUUID(sqltypes.UUID[sqltypes._UUID_RETURN]):

    def result_processor(self, dialect, coltype):
        if False:
            return 10
        if self.as_uuid:

            def process(value):
                if False:
                    return 10
                if value is not None:
                    if hasattr(value, 'decode'):
                        value = value.decode('ascii')
                    value = _python_UUID(value)
                return value
            return process
        else:

            def process(value):
                if False:
                    i = 10
                    return i + 15
                if value is not None:
                    if hasattr(value, 'decode'):
                        value = value.decode('ascii')
                    value = str(_python_UUID(value))
                return value
            return process

class MySQLExecutionContext_mariadbconnector(MySQLExecutionContext):
    _lastrowid = None

    def create_server_side_cursor(self):
        if False:
            while True:
                i = 10
        return self._dbapi_connection.cursor(buffered=False)

    def create_default_cursor(self):
        if False:
            return 10
        return self._dbapi_connection.cursor(buffered=True)

    def post_exec(self):
        if False:
            for i in range(10):
                print('nop')
        super().post_exec()
        self._rowcount = self.cursor.rowcount
        if self.isinsert and self.compiled.postfetch_lastrowid:
            self._lastrowid = self.cursor.lastrowid

    @property
    def rowcount(self):
        if False:
            while True:
                i = 10
        if self._rowcount is not None:
            return self._rowcount
        else:
            return self.cursor.rowcount

    def get_lastrowid(self):
        if False:
            while True:
                i = 10
        return self._lastrowid

class MySQLCompiler_mariadbconnector(MySQLCompiler):
    pass

class MySQLDialect_mariadbconnector(MySQLDialect):
    driver = 'mariadbconnector'
    supports_statement_cache = True
    supports_unicode_statements = True
    encoding = 'utf8mb4'
    convert_unicode = True
    supports_sane_rowcount = True
    supports_sane_multi_rowcount = True
    supports_native_decimal = True
    default_paramstyle = 'qmark'
    execution_ctx_cls = MySQLExecutionContext_mariadbconnector
    statement_compiler = MySQLCompiler_mariadbconnector
    supports_server_side_cursors = True
    colspecs = util.update_copy(MySQLDialect.colspecs, {sqltypes.Uuid: _MariaDBUUID})

    @util.memoized_property
    def _dbapi_version(self):
        if False:
            return 10
        if self.dbapi and hasattr(self.dbapi, '__version__'):
            return tuple([int(x) for x in re.findall('(\\d+)(?:[-\\.]?|$)', self.dbapi.__version__)])
        else:
            return (99, 99, 99)

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.paramstyle = 'qmark'
        if self.dbapi is not None:
            if self._dbapi_version < mariadb_cpy_minimum_version:
                raise NotImplementedError('The minimum required version for MariaDB Connector/Python is %s' % '.'.join((str(x) for x in mariadb_cpy_minimum_version)))

    @classmethod
    def import_dbapi(cls):
        if False:
            for i in range(10):
                print('nop')
        return __import__('mariadb')

    def is_disconnect(self, e, connection, cursor):
        if False:
            for i in range(10):
                print('nop')
        if super().is_disconnect(e, connection, cursor):
            return True
        elif isinstance(e, self.dbapi.Error):
            str_e = str(e).lower()
            return 'not connected' in str_e or "isn't valid" in str_e
        else:
            return False

    def create_connect_args(self, url):
        if False:
            return 10
        opts = url.translate_connect_args()
        int_params = ['connect_timeout', 'read_timeout', 'write_timeout', 'client_flag', 'port', 'pool_size']
        bool_params = ['local_infile', 'ssl_verify_cert', 'ssl', 'pool_reset_connection']
        for key in int_params:
            util.coerce_kw_type(opts, key, int)
        for key in bool_params:
            util.coerce_kw_type(opts, key, bool)
        client_flag = opts.get('client_flag', 0)
        if self.dbapi is not None:
            try:
                CLIENT_FLAGS = __import__(self.dbapi.__name__ + '.constants.CLIENT').constants.CLIENT
                client_flag |= CLIENT_FLAGS.FOUND_ROWS
            except (AttributeError, ImportError):
                self.supports_sane_rowcount = False
            opts['client_flag'] = client_flag
        return [[], opts]

    def _extract_error_code(self, exception):
        if False:
            for i in range(10):
                print('nop')
        try:
            rc = exception.errno
        except:
            rc = -1
        return rc

    def _detect_charset(self, connection):
        if False:
            print('Hello World!')
        return 'utf8mb4'

    def get_isolation_level_values(self, dbapi_connection):
        if False:
            return 10
        return ('SERIALIZABLE', 'READ UNCOMMITTED', 'READ COMMITTED', 'REPEATABLE READ', 'AUTOCOMMIT')

    def set_isolation_level(self, connection, level):
        if False:
            i = 10
            return i + 15
        if level == 'AUTOCOMMIT':
            connection.autocommit = True
        else:
            connection.autocommit = False
            super().set_isolation_level(connection, level)

    def do_begin_twophase(self, connection, xid):
        if False:
            i = 10
            return i + 15
        connection.execute(sql.text('XA BEGIN :xid').bindparams(sql.bindparam('xid', xid, literal_execute=True)))

    def do_prepare_twophase(self, connection, xid):
        if False:
            for i in range(10):
                print('nop')
        connection.execute(sql.text('XA END :xid').bindparams(sql.bindparam('xid', xid, literal_execute=True)))
        connection.execute(sql.text('XA PREPARE :xid').bindparams(sql.bindparam('xid', xid, literal_execute=True)))

    def do_rollback_twophase(self, connection, xid, is_prepared=True, recover=False):
        if False:
            return 10
        if not is_prepared:
            connection.execute(sql.text('XA END :xid').bindparams(sql.bindparam('xid', xid, literal_execute=True)))
        connection.execute(sql.text('XA ROLLBACK :xid').bindparams(sql.bindparam('xid', xid, literal_execute=True)))

    def do_commit_twophase(self, connection, xid, is_prepared=True, recover=False):
        if False:
            print('Hello World!')
        if not is_prepared:
            self.do_prepare_twophase(connection, xid)
        connection.execute(sql.text('XA COMMIT :xid').bindparams(sql.bindparam('xid', xid, literal_execute=True)))
dialect = MySQLDialect_mariadbconnector