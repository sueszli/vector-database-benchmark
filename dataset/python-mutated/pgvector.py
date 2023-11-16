from __future__ import annotations
from airflow.providers.postgres.hooks.postgres import PostgresHook

class PgVectorHook(PostgresHook):
    """Extend PostgresHook for working with PostgreSQL and pgvector extension for vector data types."""

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize a PgVectorHook.'
        super().__init__(*args, **kwargs)

    def create_table(self, table_name: str, columns: list[str], if_not_exists: bool=True) -> None:
        if False:
            while True:
                i = 10
        '\n        Create a table in the Postgres database.\n\n        :param table_name: The name of the table to create.\n        :param columns: A list of column definitions for the table.\n        :param if_not_exists: If True, only create the table if it does not already exist.\n        '
        create_table_sql = 'CREATE TABLE'
        if if_not_exists:
            create_table_sql = f'{create_table_sql} IF NOT EXISTS'
        create_table_sql = f"{create_table_sql} {table_name} ({', '.join(columns)})"
        self.run(create_table_sql)

    def create_extension(self, extension_name: str, if_not_exists: bool=True) -> None:
        if False:
            return 10
        '\n        Create a PostgreSQL extension.\n\n        :param extension_name: The name of the extension to create.\n        :param if_not_exists: If True, only create the extension if it does not already exist.\n        '
        create_extension_sql = 'CREATE EXTENSION'
        if if_not_exists:
            create_extension_sql = f'{create_extension_sql} IF NOT EXISTS'
        create_extension_sql = f'{create_extension_sql} {extension_name}'
        self.run(create_extension_sql)

    def drop_table(self, table_name: str, if_exists: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Drop a table from the Postgres database.\n\n        :param table_name: The name of the table to drop.\n        :param if_exists: If True, only drop the table if it exists.\n        '
        drop_table_sql = 'DROP TABLE'
        if if_exists:
            drop_table_sql = f'{drop_table_sql} IF EXISTS'
        drop_table_sql = f'{drop_table_sql} {table_name}'
        self.run(drop_table_sql)

    def truncate_table(self, table_name: str, restart_identity: bool=True) -> None:
        if False:
            return 10
        '\n        Truncate a table, removing all rows.\n\n        :param table_name: The name of the table to truncate.\n        :param restart_identity: If True, restart the serial sequence if the table has one.\n        '
        truncate_sql = f'TRUNCATE TABLE {table_name}'
        if restart_identity:
            truncate_sql = f'{truncate_sql} RESTART IDENTITY'
        self.run(truncate_sql)