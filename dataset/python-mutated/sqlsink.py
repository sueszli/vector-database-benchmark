"""Sink classes load data to SQL targets."""
from __future__ import annotations
import re
import typing as t
from collections import defaultdict
from copy import copy
from textwrap import dedent
import sqlalchemy
from pendulum import now
from singer_sdk.connectors import SQLConnector
from singer_sdk.exceptions import ConformedNameClashException
from singer_sdk.helpers._conformers import replace_leading_digit
from sqlalchemy.sql.expression import bindparam
from mage_integrations.destinations.sink import BatchSink
if t.TYPE_CHECKING:
    from singer_sdk.plugin_base import PluginBase
    from sqlalchemy.sql import Executable

class SQLSink(BatchSink):
    """SQL-type sink type."""
    connector_class: type[SQLConnector]
    soft_delete_column_name = '_sdc_deleted_at'
    version_column_name = '_sdc_table_version'

    def __init__(self, target: PluginBase, stream_name: str, schema: dict, key_properties: list[str] | None, connector: SQLConnector | None=None) -> None:
        if False:
            print('Hello World!')
        "Initialize SQL Sink.\n\n        Args:\n            target: The target object.\n            stream_name: The source tap's stream name.\n            schema: The JSON Schema definition.\n            key_properties: The primary key columns.\n            connector: Optional connector to reuse.\n        "
        self._connector: SQLConnector
        self._connector = connector or self.connector_class(dict(target.config))
        super().__init__(target, stream_name, schema, key_properties)

    @property
    def connector(self) -> SQLConnector:
        if False:
            i = 10
            return i + 15
        'The connector object.\n\n        Returns:\n            The connector object.\n        '
        return self._connector

    @property
    def connection(self) -> sqlalchemy.engine.Connection:
        if False:
            for i in range(10):
                print('nop')
        'Get or set the SQLAlchemy connection for this sink.\n\n        Returns:\n            A connection object.\n        '
        return self.connector.connection

    @property
    def table_name(self) -> str:
        if False:
            while True:
                i = 10
        'Return the table name, with no schema or database part.\n\n        Returns:\n            The target table name.\n        '
        parts = self.stream_name.split('-')
        table = self.stream_name if len(parts) == 1 else parts[-1]
        return self.conform_name(table, 'table')

    @property
    def schema_name(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        'Return the schema name or `None` if using names with no schema part.\n\n        Returns:\n            The target schema name.\n        '
        default_target_schema: str = self.config.get('default_target_schema', None)
        parts = self.stream_name.split('-')
        if default_target_schema:
            return default_target_schema
        if len(parts) in {2, 3}:
            return self.conform_name(parts[-2], 'schema')
        return None

    @property
    def database_name(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        'Return the DB name or `None` if using names with no database part.'

    @property
    def full_table_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return the fully qualified table name.\n\n        Returns:\n            The fully qualified table name.\n        '
        return self.connector.get_fully_qualified_name(table_name=self.table_name, schema_name=self.schema_name, db_name=self.database_name)

    @property
    def full_schema_name(self) -> str:
        if False:
            print('Hello World!')
        'Return the fully qualified schema name.\n\n        Returns:\n            The fully qualified schema name.\n        '
        return self.connector.get_fully_qualified_name(schema_name=self.schema_name, db_name=self.database_name)

    def conform_name(self, name: str, object_type: str | None=None) -> str:
        if False:
            while True:
                i = 10
        "Conform a stream property name to one suitable for the target system.\n\n        Transforms names to snake case by default, applicable to most common DBMSs'.\n        Developers may override this method to apply custom transformations\n        to database/schema/table/column names.\n\n        Args:\n            name: Property name.\n            object_type: One of ``database``, ``schema``, ``table`` or ``column``.\n\n\n        Returns:\n            The name transformed to snake case.\n        "
        name = re.sub('[^a-zA-Z0-9_\\-\\.\\s]', '', name)
        name = name.lower().lstrip().rstrip().replace('.', '_').replace('-', '_').replace(' ', '_')
        return replace_leading_digit(name)

    @staticmethod
    def _check_conformed_names_not_duplicated(conformed_property_names: dict[str, str]) -> None:
        if False:
            print('Hello World!')
        'Check if conformed names produce duplicate keys.\n\n        Args:\n            conformed_property_names: A name:conformed_name dict map.\n\n        Raises:\n            ConformedNameClashException: if duplicates found.\n        '
        grouped = defaultdict(list)
        for (k, v) in conformed_property_names.items():
            grouped[v].append(k)
        duplicates = list(filter(lambda p: len(p[1]) > 1, grouped.items()))
        if duplicates:
            msg = f'Duplicate stream properties produced when conforming property names: {duplicates}'
            raise ConformedNameClashException(msg)

    def conform_schema(self, schema: dict) -> dict:
        if False:
            print('Hello World!')
        'Return schema dictionary with property names conformed.\n\n        Args:\n            schema: JSON schema dictionary.\n\n        Returns:\n            A schema dictionary with the property names conformed.\n        '
        conformed_schema = copy(schema)
        conformed_property_names = {key: self.conform_name(key) for key in conformed_schema['properties']}
        self._check_conformed_names_not_duplicated(conformed_property_names)
        conformed_schema['properties'] = {conformed_property_names[key]: value for (key, value) in conformed_schema['properties'].items()}
        return conformed_schema

    def conform_record(self, record: dict) -> dict:
        if False:
            i = 10
            return i + 15
        'Return record dictionary with property names conformed.\n\n        Args:\n            record: Dictionary representing a single record.\n\n        Returns:\n            New record dictionary with conformed column names.\n        '
        conformed_property_names = {key: self.conform_name(key) for key in record}
        self._check_conformed_names_not_duplicated(conformed_property_names)
        return {conformed_property_names[key]: value for (key, value) in record.items()}

    def setup(self) -> None:
        if False:
            return 10
        'Set up Sink.\n\n        This method is called on Sink creation, and creates the required Schema and\n        Table entities in the target database.\n        '
        if self.schema_name:
            self.connector.prepare_schema(self.schema_name)
        self.connector.prepare_table(full_table_name=self.full_table_name, schema=self.conform_schema(self.schema), primary_keys=self.key_properties, as_temp_table=False)

    @property
    def key_properties(self) -> list[str]:
        if False:
            return 10
        'Return key properties, conformed to target system naming requirements.\n\n        Returns:\n            A list of key properties, conformed with `self.conform_name()`\n        '
        return [self.conform_name(key, 'column') for key in super().key_properties]

    def process_batch(self, context: dict) -> None:
        if False:
            while True:
                i = 10
        'Process a batch with the given batch context.\n\n        Writes a batch to the SQL target. Developers may override this method\n        in order to provide a more efficient upload/upsert process.\n\n        Args:\n            context: Stream partition or context dictionary.\n        '
        self.bulk_insert_records(full_table_name=self.full_table_name, schema=self.schema, records=context['records'])

    def generate_insert_statement(self, full_table_name: str, schema: dict) -> str | Executable:
        if False:
            while True:
                i = 10
        'Generate an insert statement for the given records.\n\n        Args:\n            full_table_name: the target table name.\n            schema: the JSON schema for the new table.\n\n        Returns:\n            An insert statement.\n        '
        property_names = list(self.conform_schema(schema)['properties'].keys())
        statement = dedent(f"            INSERT INTO {full_table_name}\n            ({', '.join(property_names)})\n            VALUES ({', '.join([f':{name}' for name in property_names])})\n            ")
        return statement.rstrip()

    def bulk_insert_records(self, full_table_name: str, schema: dict, records: t.Iterable[dict[str, t.Any]]) -> int | None:
        if False:
            print('Hello World!')
        'Bulk insert records to an existing destination table.\n\n        The default implementation uses a generic SQLAlchemy bulk insert operation.\n        This method may optionally be overridden by developers in order to provide\n        faster, native bulk uploads.\n\n        Args:\n            full_table_name: the target table name.\n            schema: the JSON schema for the new table, to be used when inferring column\n                names.\n            records: the input records.\n\n        Returns:\n            True if table exists, False if not, None if unsure or undetectable.\n        '
        insert_sql = self.generate_insert_statement(full_table_name, schema)
        if isinstance(insert_sql, str):
            insert_sql = sqlalchemy.text(insert_sql)
        conformed_records = [self.conform_record(record) for record in records] if isinstance(records, list) else (self.conform_record(record) for record in records)
        self.logger.info(f'Inserting with SQL: {insert_sql}')
        with self.connector._connect() as conn, conn.begin():
            conn.execute(insert_sql, conformed_records)
        return len(conformed_records) if isinstance(conformed_records, list) else None

    def merge_upsert_from_table(self, target_table_name: str, from_table_name: str, join_keys: list[str]) -> int | None:
        if False:
            return 10
        'Merge upsert data from one table to another.\n\n        Args:\n            target_table_name: The destination table name.\n            from_table_name: The source table name.\n            join_keys: The merge upsert keys, or `None` to append.\n\n        Return:\n            The number of records copied, if detectable, or `None` if the API does not\n            report number of records affected/inserted.\n\n        Raises:\n            NotImplementedError: if the merge upsert capability does not exist or is\n                undefined.\n        '
        raise NotImplementedError

    def activate_version(self, new_version: int) -> None:
        if False:
            return 10
        'Bump the active version of the target table.\n\n        Args:\n            new_version: The version number to activate.\n        '
        if not self.connector.table_exists(self.full_table_name):
            return
        deleted_at = now()
        if not self.connector.column_exists(full_table_name=self.full_table_name, column_name=self.version_column_name):
            self.connector.prepare_column(self.full_table_name, self.version_column_name, sql_type=sqlalchemy.types.Integer())
        if self.config.get('hard_delete', True):
            with self.connector._connect() as conn, conn.begin():
                conn.execute(sqlalchemy.text(f'DELETE FROM {self.full_table_name} WHERE {self.version_column_name} <= {new_version}'))
            return
        if not self.connector.column_exists(full_table_name=self.full_table_name, column_name=self.soft_delete_column_name):
            self.connector.prepare_column(self.full_table_name, self.soft_delete_column_name, sql_type=sqlalchemy.types.DateTime())
        query = sqlalchemy.text(f'UPDATE {self.full_table_name}\nSET {self.soft_delete_column_name} = :deletedate \nWHERE {self.version_column_name} < :version \n  AND {self.soft_delete_column_name} IS NULL\n')
        query = query.bindparams(bindparam('deletedate', value=deleted_at, type_=sqlalchemy.types.DateTime), bindparam('version', value=new_version, type_=sqlalchemy.types.Integer))
        with self.connector._connect() as conn, conn.begin():
            conn.execute(query)
__all__ = ['SQLSink', 'SQLConnector']