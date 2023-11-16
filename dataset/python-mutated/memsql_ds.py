import logging
from redash.query_runner import TYPE_BOOLEAN, TYPE_DATETIME, TYPE_FLOAT, TYPE_INTEGER, TYPE_STRING, BaseSQLQueryRunner, JobTimeoutException, register
from redash.utils import json_dumps
logger = logging.getLogger(__name__)
try:
    from memsql.common import database
    enabled = True
except ImportError:
    enabled = False
COLUMN_NAME = 0
COLUMN_TYPE = 1
types_map = {'BIGINT': TYPE_INTEGER, 'TINYINT': TYPE_INTEGER, 'SMALLINT': TYPE_INTEGER, 'MEDIUMINT': TYPE_INTEGER, 'INT': TYPE_INTEGER, 'DOUBLE': TYPE_FLOAT, 'DECIMAL': TYPE_FLOAT, 'FLOAT': TYPE_FLOAT, 'REAL': TYPE_FLOAT, 'BOOL': TYPE_BOOLEAN, 'BOOLEAN': TYPE_BOOLEAN, 'TIMESTAMP': TYPE_DATETIME, 'DATETIME': TYPE_DATETIME, 'DATE': TYPE_DATETIME, 'JSON': TYPE_STRING, 'CHAR': TYPE_STRING, 'VARCHAR': TYPE_STRING}

class MemSQL(BaseSQLQueryRunner):
    should_annotate_query = False
    noop_query = 'SELECT 1'

    @classmethod
    def configuration_schema(cls):
        if False:
            while True:
                i = 10
        return {'type': 'object', 'properties': {'host': {'type': 'string'}, 'port': {'type': 'number'}, 'user': {'type': 'string'}, 'password': {'type': 'string'}}, 'required': ['host', 'port'], 'secret': ['password']}

    @classmethod
    def type(cls):
        if False:
            for i in range(10):
                print('nop')
        return 'memsql'

    @classmethod
    def enabled(cls):
        if False:
            for i in range(10):
                print('nop')
        return enabled

    def _get_tables(self, schema):
        if False:
            print('Hello World!')
        schemas_query = 'show schemas'
        tables_query = 'show tables in %s'
        columns_query = 'show columns in %s'
        for schema_name in [a for a in [str(a['Database']) for a in self._run_query_internal(schemas_query)] if len(a) > 0]:
            for table_name in [a for a in [str(a['Tables_in_%s' % schema_name]) for a in self._run_query_internal(tables_query % schema_name)] if len(a) > 0]:
                table_name = '.'.join((schema_name, table_name))
                columns = [a for a in [str(a['Field']) for a in self._run_query_internal(columns_query % table_name)] if len(a) > 0]
                schema[table_name] = {'name': table_name, 'columns': columns}
        return list(schema.values())

    def run_query(self, query, user):
        if False:
            for i in range(10):
                print('nop')
        cursor = None
        try:
            cursor = database.connect(**self.configuration.to_dict())
            res = cursor.query(query)
            rows = [dict(zip(row.keys(), row.values())) for row in res]
            columns = []
            column_names = rows[0].keys() if rows else None
            if column_names:
                for column in column_names:
                    columns.append({'name': column, 'friendly_name': column, 'type': TYPE_STRING})
            data = {'columns': columns, 'rows': rows}
            json_data = json_dumps(data)
            error = None
        except (KeyboardInterrupt, JobTimeoutException):
            cursor.close()
            raise
        finally:
            if cursor:
                cursor.close()
        return (json_data, error)
register(MemSQL)