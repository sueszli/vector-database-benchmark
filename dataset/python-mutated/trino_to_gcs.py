from __future__ import annotations
from typing import TYPE_CHECKING, Any
from airflow.providers.google.cloud.transfers.sql_to_gcs import BaseSQLToGCSOperator
from airflow.providers.trino.hooks.trino import TrinoHook
if TYPE_CHECKING:
    from trino.client import TrinoResult
    from trino.dbapi import Cursor as TrinoCursor

class _TrinoToGCSTrinoCursorAdapter:
    """
    An adapter that adds additional feature to the Trino cursor.

    The implementation of cursor in the trino library is not sufficient.
    The following changes have been made:

    * The poke mechanism for row. You can look at the next row without consuming it.
    * The description attribute is available before reading the first row. Thanks to the poke mechanism.
    * the iterator interface has been implemented.

    A detailed description of the class methods is available in
    `PEP-249 <https://www.python.org/dev/peps/pep-0249/>`__.
    """

    def __init__(self, cursor: TrinoCursor):
        if False:
            i = 10
            return i + 15
        self.cursor: TrinoCursor = cursor
        self.rows: list[Any] = []
        self.initialized: bool = False

    @property
    def description(self) -> list[tuple]:
        if False:
            while True:
                i = 10
        '\n        This read-only attribute is a sequence of 7-item sequences.\n\n        Each of these sequences contains information describing one result column:\n\n        * ``name``\n        * ``type_code``\n        * ``display_size``\n        * ``internal_size``\n        * ``precision``\n        * ``scale``\n        * ``null_ok``\n\n        The first two items (``name`` and ``type_code``) are mandatory, the other\n        five are optional and are set to None if no meaningful values can be provided.\n        '
        if not self.initialized:
            self.peekone()
        return self.cursor.description

    @property
    def rowcount(self) -> int:
        if False:
            print('Hello World!')
        'The read-only attribute specifies the number of rows.'
        return self.cursor.rowcount

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        'Close the cursor now.'
        self.cursor.close()

    def execute(self, *args, **kwargs) -> TrinoResult:
        if False:
            while True:
                i = 10
        'Prepare and execute a database operation (query or command).'
        self.initialized = False
        self.rows = []
        return self.cursor.execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Prepare and execute a database query.\n\n        Prepare a database operation (query or command) and then execute it against\n        all parameter sequences or mappings found in the sequence seq_of_parameters.\n        '
        self.initialized = False
        self.rows = []
        return self.cursor.executemany(*args, **kwargs)

    def peekone(self) -> Any:
        if False:
            print('Hello World!')
        'Return the next row without consuming it.'
        self.initialized = True
        element = self.cursor.fetchone()
        self.rows.insert(0, element)
        return element

    def fetchone(self) -> Any:
        if False:
            while True:
                i = 10
        'Fetch the next row of a query result set, returning a single sequence, or ``None``.'
        if self.rows:
            return self.rows.pop(0)
        return self.cursor.fetchone()

    def fetchmany(self, size=None) -> list:
        if False:
            while True:
                i = 10
        '\n        Fetch the next set of rows of a query result, returning a sequence of sequences.\n\n        An empty sequence is returned when no more rows are available.\n        '
        if size is None:
            size = self.cursor.arraysize
        result = []
        for _ in range(size):
            row = self.fetchone()
            if row is None:
                break
            result.append(row)
        return result

    def __next__(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the next row from the current SQL statement using the same semantics as ``.fetchone()``.\n\n        A ``StopIteration`` exception is raised when the result set is exhausted.\n        '
        result = self.fetchone()
        if result is None:
            raise StopIteration()
        return result

    def __iter__(self) -> _TrinoToGCSTrinoCursorAdapter:
        if False:
            for i in range(10):
                print('nop')
        'Return self to make cursors compatible to the iteration protocol.'
        return self

class TrinoToGCSOperator(BaseSQLToGCSOperator):
    """Copy data from TrinoDB to Google Cloud Storage in JSON, CSV or Parquet format.

    :param trino_conn_id: Reference to a specific Trino hook.
    """
    ui_color = '#a0e08c'
    type_map = {'BOOLEAN': 'BOOL', 'TINYINT': 'INT64', 'SMALLINT': 'INT64', 'INTEGER': 'INT64', 'BIGINT': 'INT64', 'REAL': 'FLOAT64', 'DOUBLE': 'FLOAT64', 'DECIMAL': 'NUMERIC', 'VARCHAR': 'STRING', 'CHAR': 'STRING', 'VARBINARY': 'BYTES', 'JSON': 'STRING', 'DATE': 'DATE', 'TIME': 'TIME', 'TIME WITH TIME ZONE': 'STRING', 'TIMESTAMP': 'TIMESTAMP', 'TIMESTAMP WITH TIME ZONE': 'STRING', 'IPADDRESS': 'STRING', 'UUID': 'STRING'}

    def __init__(self, *, trino_conn_id: str='trino_default', **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.trino_conn_id = trino_conn_id

    def query(self):
        if False:
            i = 10
            return i + 15
        'Queries trino and returns a cursor to the results.'
        trino = TrinoHook(trino_conn_id=self.trino_conn_id)
        conn = trino.get_conn()
        cursor = conn.cursor()
        self.log.info('Executing: %s', self.sql)
        cursor.execute(self.sql)
        return _TrinoToGCSTrinoCursorAdapter(cursor)

    def field_to_bigquery(self, field) -> dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        'Convert trino field type to BigQuery field type.'
        clear_field_type = field[1].upper()
        (clear_field_type, _, _) = clear_field_type.partition('(')
        new_field_type = self.type_map.get(clear_field_type, 'STRING')
        return {'name': field[0], 'type': new_field_type}

    def convert_type(self, value, schema_type, **kwargs):
        if False:
            print('Hello World!')
        '\n        Do nothing. Trino uses JSON on the transport layer, so types are simple.\n\n        :param value: Trino column value\n        :param schema_type: BigQuery data type\n        '
        return value