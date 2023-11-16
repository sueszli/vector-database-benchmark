"""
.. dialect:: mssql+pymssql
    :name: pymssql
    :dbapi: pymssql
    :connectstring: mssql+pymssql://<username>:<password>@<freetds_name>/?charset=utf8

pymssql is a Python module that provides a Python DBAPI interface around
`FreeTDS <https://www.freetds.org/>`_.

.. versionchanged:: 2.0.5

    pymssql was restored to SQLAlchemy's continuous integration testing


"""
import re
from .base import MSDialect
from .base import MSIdentifierPreparer
from ... import types as sqltypes
from ... import util
from ...engine import processors

class _MSNumeric_pymssql(sqltypes.Numeric):

    def result_processor(self, dialect, type_):
        if False:
            return 10
        if not self.asdecimal:
            return processors.to_float
        else:
            return sqltypes.Numeric.result_processor(self, dialect, type_)

class MSIdentifierPreparer_pymssql(MSIdentifierPreparer):

    def __init__(self, dialect):
        if False:
            print('Hello World!')
        super().__init__(dialect)
        self._double_percents = False

class MSDialect_pymssql(MSDialect):
    supports_statement_cache = True
    supports_native_decimal = True
    supports_native_uuid = True
    driver = 'pymssql'
    preparer = MSIdentifierPreparer_pymssql
    colspecs = util.update_copy(MSDialect.colspecs, {sqltypes.Numeric: _MSNumeric_pymssql, sqltypes.Float: sqltypes.Float})

    @classmethod
    def import_dbapi(cls):
        if False:
            return 10
        module = __import__('pymssql')
        client_ver = tuple((int(x) for x in module.__version__.split('.')))
        if client_ver < (2, 1, 1):
            module.Binary = lambda x: x if hasattr(x, 'decode') else str(x)
        if client_ver < (1,):
            util.warn('The pymssql dialect expects at least the 1.0 series of the pymssql DBAPI.')
        return module

    def _get_server_version_info(self, connection):
        if False:
            print('Hello World!')
        vers = connection.exec_driver_sql('select @@version').scalar()
        m = re.match('Microsoft .*? - (\\d+)\\.(\\d+)\\.(\\d+)\\.(\\d+)', vers)
        if m:
            return tuple((int(x) for x in m.group(1, 2, 3, 4)))
        else:
            return None

    def create_connect_args(self, url):
        if False:
            for i in range(10):
                print('nop')
        opts = url.translate_connect_args(username='user')
        opts.update(url.query)
        port = opts.pop('port', None)
        if port and 'host' in opts:
            opts['host'] = '%s:%s' % (opts['host'], port)
        return ([], opts)

    def is_disconnect(self, e, connection, cursor):
        if False:
            while True:
                i = 10
        for msg in ('Adaptive Server connection timed out', 'Net-Lib error during Connection reset by peer', 'message 20003', 'Error 10054', 'Not connected to any MS SQL server', 'Connection is closed', 'message 20006', 'message 20017', 'message 20047'):
            if msg in str(e):
                return True
        else:
            return False

    def get_isolation_level_values(self, dbapi_connection):
        if False:
            return 10
        return super().get_isolation_level_values(dbapi_connection) + ['AUTOCOMMIT']

    def set_isolation_level(self, dbapi_connection, level):
        if False:
            for i in range(10):
                print('nop')
        if level == 'AUTOCOMMIT':
            dbapi_connection.autocommit(True)
        else:
            dbapi_connection.autocommit(False)
            super().set_isolation_level(dbapi_connection, level)
dialect = MSDialect_pymssql