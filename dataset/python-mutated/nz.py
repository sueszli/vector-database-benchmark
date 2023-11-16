import json
import logging
import traceback
from redash.query_runner import TYPE_BOOLEAN, TYPE_DATE, TYPE_DATETIME, TYPE_FLOAT, TYPE_INTEGER, TYPE_STRING, BaseSQLQueryRunner, register
logger = logging.getLogger(__name__)
try:
    import nzpy
    import nzpy.core
    _enabled = True
    _nztypes = {nzpy.core.NzTypeInt1: TYPE_INTEGER, nzpy.core.NzTypeInt2: TYPE_INTEGER, nzpy.core.NzTypeInt: TYPE_INTEGER, nzpy.core.NzTypeInt8: TYPE_INTEGER, nzpy.core.NzTypeBool: TYPE_BOOLEAN, nzpy.core.NzTypeDate: TYPE_DATE, nzpy.core.NzTypeTimestamp: TYPE_DATETIME, nzpy.core.NzTypeDouble: TYPE_FLOAT, nzpy.core.NzTypeFloat: TYPE_FLOAT, nzpy.core.NzTypeChar: TYPE_STRING, nzpy.core.NzTypeNChar: TYPE_STRING, nzpy.core.NzTypeNVarChar: TYPE_STRING, nzpy.core.NzTypeVarChar: TYPE_STRING, nzpy.core.NzTypeVarFixedChar: TYPE_STRING, nzpy.core.NzTypeNumeric: TYPE_FLOAT}
    _cat_types = {16: TYPE_BOOLEAN, 17: TYPE_STRING, 19: TYPE_STRING, 20: TYPE_INTEGER, 21: TYPE_INTEGER, 23: TYPE_INTEGER, 25: TYPE_STRING, 26: TYPE_INTEGER, 28: TYPE_INTEGER, 700: TYPE_FLOAT, 701: TYPE_FLOAT, 705: TYPE_STRING, 829: TYPE_STRING, 1042: TYPE_STRING, 1043: TYPE_STRING, 1082: TYPE_DATE, 1083: TYPE_DATETIME, 1114: TYPE_DATETIME, 1184: TYPE_DATETIME, 1700: TYPE_FLOAT, 2275: TYPE_STRING, 2950: TYPE_STRING}
except ImportError:
    _enabled = False
    _nztypes = {}
    _cat_types = {}

class Netezza(BaseSQLQueryRunner):
    noop_query = 'SELECT 1'

    @classmethod
    def configuration_schema(cls):
        if False:
            i = 10
            return i + 15
        return {'type': 'object', 'properties': {'user': {'type': 'string'}, 'password': {'type': 'string'}, 'host': {'type': 'string', 'default': '127.0.0.1'}, 'port': {'type': 'number', 'default': 5480}, 'database': {'type': 'string', 'title': 'Database Name', 'default': 'system'}}, 'order': ['host', 'port', 'user', 'password', 'database'], 'required': ['user', 'password', 'database'], 'secret': ['password']}

    @classmethod
    def type(cls):
        if False:
            while True:
                i = 10
        return 'nz'

    def __init__(self, configuration):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(configuration)
        self._conn = None

    @property
    def connection(self):
        if False:
            i = 10
            return i + 15
        if self._conn is None:
            self._conn = nzpy.connect(host=self.configuration.get('host'), user=self.configuration.get('user'), password=self.configuration.get('password'), port=self.configuration.get('port'), database=self.configuration.get('database'))
        return self._conn

    def get_schema(self, get_stats=False):
        if False:
            for i in range(10):
                print('nop')
        qry = "\n        select\n            table_schema || '.' || table_name as table_name,\n            column_name,\n            data_type\n        from\n            columns\n        where\n            table_schema not in (^information_schema^, ^definition_schema^) and\n            table_catalog = current_catalog;\n        "
        schema = {}
        with self.connection.cursor() as cursor:
            cursor.execute(qry)
            for (table_name, column_name, data_type) in cursor:
                if table_name not in schema:
                    schema[table_name] = {'name': table_name, 'columns': []}
                schema[table_name]['columns'].append({'name': column_name, 'type': data_type})
            return list(schema.values())

    @classmethod
    def enabled(cls):
        if False:
            return 10
        global _enabled
        return _enabled

    def type_map(self, typid, func):
        if False:
            while True:
                i = 10
        global _nztypes, _cat_types
        typ = _nztypes.get(typid)
        if typ is None:
            return _cat_types.get(typid)
        if typid == nzpy.core.NzTypeVarChar:
            return TYPE_BOOLEAN if 'bool' in func.__name__ else typ
        if typid == nzpy.core.NzTypeInt2:
            return TYPE_STRING if 'text' in func.__name__ else typ
        if typid in (nzpy.core.NzTypeVarFixedChar, nzpy.core.NzTypeVarBinary, nzpy.core.NzTypeNVarChar):
            return TYPE_INTEGER if 'int' in func.__name__ else typ
        return typ

    def run_query(self, query, user):
        if False:
            while True:
                i = 10
        (json_data, error) = (None, None)
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                if cursor.description is None:
                    columns = {'columns': [], 'rows': []}
                else:
                    columns = self.fetch_columns([(val[0], self.type_map(val[1], cursor.ps['row_desc'][i]['func'])) for (i, val) in enumerate(cursor.description)])
                rows = [dict(zip((column['name'] for column in columns), row)) for row in cursor]
                json_data = json.dumps({'columns': columns, 'rows': rows})
        except Exception:
            error = traceback.format_exc()
        return (json_data, error)
register(Netezza)