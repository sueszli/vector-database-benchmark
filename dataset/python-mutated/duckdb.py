"""
DuckDB module
"""
import os
import re
from tempfile import TemporaryDirectory
try:
    import duckdb
    DUCKDB = True
except ImportError:
    DUCKDB = False
from .embedded import Embedded
from .schema import Statement

class DuckDB(Embedded):
    """
    Database instance backed by DuckDB.
    """
    DELETE_DOCUMENT = 'DELETE FROM documents WHERE id = ?'
    DELETE_OBJECT = 'DELETE FROM objects WHERE id = ?'

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        if not DUCKDB:
            raise ImportError('DuckDB is not available - install "database" extra to enable')

    def execute(self, function, *args):
        if False:
            while True:
                i = 10
        return super().execute(function, *self.formatargs(args))

    def insertdocument(self, uid, data, tags, entry):
        if False:
            i = 10
            return i + 15
        self.cursor.execute(DuckDB.DELETE_DOCUMENT, [uid])
        super().insertdocument(uid, data, tags, entry)

    def insertobject(self, uid, data, tags, entry):
        if False:
            for i in range(10):
                print('nop')
        self.cursor.execute(DuckDB.DELETE_OBJECT, [uid])
        super().insertobject(uid, data, tags, entry)

    def connect(self, path=':memory:'):
        if False:
            while True:
                i = 10
        connection = duckdb.connect(path)
        connection.begin()
        return connection

    def getcursor(self):
        if False:
            return 10
        return self.connection

    def jsonprefix(self):
        if False:
            i = 10
            return i + 15
        return 'json_extract_string(data'

    def jsoncolumn(self, name):
        if False:
            while True:
                i = 10
        return f"json_extract_string(data, '$.{name}')"

    def rows(self):
        if False:
            return 10
        batch = 256
        rows = self.cursor.fetchmany(batch)
        while rows:
            yield from rows
            rows = self.cursor.fetchmany(batch)

    def addfunctions(self):
        if False:
            for i in range(10):
                print('nop')
        return

    def copy(self, path):
        if False:
            return 10
        if os.path.exists(path):
            os.remove(path)
        connection = duckdb.connect(path)
        tables = ['documents', 'objects', 'sections']
        with TemporaryDirectory() as directory:
            for table in tables:
                self.connection.execute(f"COPY {table} TO '{directory}/{table}.parquet' (FORMAT parquet)")
            for schema in [Statement.CREATE_DOCUMENTS, Statement.CREATE_OBJECTS, Statement.CREATE_SECTIONS % 'sections']:
                connection.execute(schema)
            for table in tables:
                connection.execute(f"COPY {table} FROM '{directory}/{table}.parquet' (FORMAT parquet)")
            connection.execute(Statement.CREATE_SECTIONS_INDEX)
            connection.execute('CHECKPOINT')
        connection.begin()
        return connection

    def formatargs(self, args):
        if False:
            return 10
        "\n        DuckDB doesn't support named parameters. This method replaces named parameters with question marks\n        and makes parameters a list.\n\n        Args:\n            args: input arguments\n\n        Returns:\n            DuckDB compatible args\n        "
        if args and len(args) > 1:
            (query, parameters) = args
            params = []
            for (key, value) in parameters.items():
                pattern = f'\\:{key}(?=\\s|$)'
                match = re.search(pattern, query)
                if match:
                    query = re.sub(pattern, '?', query, count=1)
                    params.append((match.start(), value))
            args = (query, [value for (_, value) in sorted(params, key=lambda x: x[0])])
        return args