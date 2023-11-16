from __future__ import annotations
import decimal
from .array import ARRAY as PGARRAY
from .base import _DECIMAL_TYPES
from .base import _FLOAT_TYPES
from .base import _INT_TYPES
from .base import PGDialect
from .base import PGExecutionContext
from .hstore import HSTORE
from .pg_catalog import _SpaceVector
from .pg_catalog import INT2VECTOR
from .pg_catalog import OIDVECTOR
from ... import exc
from ... import types as sqltypes
from ... import util
from ...engine import processors
_server_side_id = util.counter()

class _PsycopgNumeric(sqltypes.Numeric):

    def bind_processor(self, dialect):
        if False:
            return 10
        return None

    def result_processor(self, dialect, coltype):
        if False:
            i = 10
            return i + 15
        if self.asdecimal:
            if coltype in _FLOAT_TYPES:
                return processors.to_decimal_processor_factory(decimal.Decimal, self._effective_decimal_return_scale)
            elif coltype in _DECIMAL_TYPES or coltype in _INT_TYPES:
                return None
            else:
                raise exc.InvalidRequestError('Unknown PG numeric type: %d' % coltype)
        elif coltype in _FLOAT_TYPES:
            return None
        elif coltype in _DECIMAL_TYPES or coltype in _INT_TYPES:
            return processors.to_float
        else:
            raise exc.InvalidRequestError('Unknown PG numeric type: %d' % coltype)

class _PsycopgFloat(_PsycopgNumeric):
    __visit_name__ = 'float'

class _PsycopgHStore(HSTORE):

    def bind_processor(self, dialect):
        if False:
            print('Hello World!')
        if dialect._has_native_hstore:
            return None
        else:
            return super().bind_processor(dialect)

    def result_processor(self, dialect, coltype):
        if False:
            return 10
        if dialect._has_native_hstore:
            return None
        else:
            return super().result_processor(dialect, coltype)

class _PsycopgARRAY(PGARRAY):
    render_bind_cast = True

class _PsycopgINT2VECTOR(_SpaceVector, INT2VECTOR):
    pass

class _PsycopgOIDVECTOR(_SpaceVector, OIDVECTOR):
    pass

class _PGExecutionContext_common_psycopg(PGExecutionContext):

    def create_server_side_cursor(self):
        if False:
            print('Hello World!')
        ident = 'c_%s_%s' % (hex(id(self))[2:], hex(_server_side_id())[2:])
        return self._dbapi_connection.cursor(ident)

class _PGDialect_common_psycopg(PGDialect):
    supports_statement_cache = True
    supports_server_side_cursors = True
    default_paramstyle = 'pyformat'
    _has_native_hstore = True
    colspecs = util.update_copy(PGDialect.colspecs, {sqltypes.Numeric: _PsycopgNumeric, sqltypes.Float: _PsycopgFloat, HSTORE: _PsycopgHStore, sqltypes.ARRAY: _PsycopgARRAY, INT2VECTOR: _PsycopgINT2VECTOR, OIDVECTOR: _PsycopgOIDVECTOR})

    def __init__(self, client_encoding=None, use_native_hstore=True, **kwargs):
        if False:
            i = 10
            return i + 15
        PGDialect.__init__(self, **kwargs)
        if not use_native_hstore:
            self._has_native_hstore = False
        self.use_native_hstore = use_native_hstore
        self.client_encoding = client_encoding

    def create_connect_args(self, url):
        if False:
            i = 10
            return i + 15
        opts = url.translate_connect_args(username='user', database='dbname')
        (multihosts, multiports) = self._split_multihost_from_url(url)
        if opts or url.query:
            if not opts:
                opts = {}
            if 'port' in opts:
                opts['port'] = int(opts['port'])
            opts.update(url.query)
            if multihosts:
                opts['host'] = ','.join(multihosts)
                comma_ports = ','.join((str(p) if p else '' for p in multiports))
                if comma_ports:
                    opts['port'] = comma_ports
            return ([], opts)
        else:
            return ([''], opts)

    def get_isolation_level_values(self, dbapi_connection):
        if False:
            for i in range(10):
                print('nop')
        return ('AUTOCOMMIT', 'READ COMMITTED', 'READ UNCOMMITTED', 'REPEATABLE READ', 'SERIALIZABLE')

    def set_deferrable(self, connection, value):
        if False:
            while True:
                i = 10
        connection.deferrable = value

    def get_deferrable(self, connection):
        if False:
            while True:
                i = 10
        return connection.deferrable

    def _do_autocommit(self, connection, value):
        if False:
            for i in range(10):
                print('nop')
        connection.autocommit = value

    def do_ping(self, dbapi_connection):
        if False:
            for i in range(10):
                print('nop')
        cursor = None
        before_autocommit = dbapi_connection.autocommit
        if not before_autocommit:
            dbapi_connection.autocommit = True
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(self._dialect_specific_select_one)
        finally:
            cursor.close()
            if not before_autocommit and (not dbapi_connection.closed):
                dbapi_connection.autocommit = before_autocommit
        return True