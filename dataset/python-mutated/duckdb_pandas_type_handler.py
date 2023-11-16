from typing import Optional, Sequence, Type
import pandas as pd
from dagster import InputContext, MetadataValue, OutputContext, TableColumn, TableSchema
from dagster._core.storage.db_io_manager import DbTypeHandler, TableSlice
from dagster_duckdb.io_manager import DuckDbClient, DuckDBIOManager, build_duckdb_io_manager

class DuckDBPandasTypeHandler(DbTypeHandler[pd.DataFrame]):
    """Stores and loads Pandas DataFrames in DuckDB.

    To use this type handler, return it from the ``type_handlers` method of an I/O manager that inherits from ``DuckDBIOManager``.

    Example:
        .. code-block:: python

            from dagster_duckdb import DuckDBIOManager
            from dagster_duckdb_pandas import DuckDBPandasTypeHandler

            class MyDuckDBIOManager(DuckDBIOManager):
                @staticmethod
                def type_handlers() -> Sequence[DbTypeHandler]:
                    return [DuckDBPandasTypeHandler()]

            @asset(
                key_prefix=["my_schema"]  # will be used as the schema in duckdb
            )
            def my_table() -> pd.DataFrame:  # the name of the asset will be the table name
                ...

            defs = Definitions(
                assets=[my_table],
                resources={"io_manager": MyDuckDBIOManager(database="my_db.duckdb")}
            )

    """

    def handle_output(self, context: OutputContext, table_slice: TableSlice, obj: pd.DataFrame, connection):
        if False:
            return 10
        'Stores the pandas DataFrame in duckdb.'
        connection.execute(f'create table if not exists {table_slice.schema}.{table_slice.table} as select * from obj;')
        if not connection.fetchall():
            connection.execute(f'insert into {table_slice.schema}.{table_slice.table} select * from obj')
        context.add_output_metadata({'row_count': obj.shape[0], 'dataframe_columns': MetadataValue.table_schema(TableSchema(columns=[TableColumn(name=name, type=str(dtype)) for (name, dtype) in obj.dtypes.items()]))})

    def load_input(self, context: InputContext, table_slice: TableSlice, connection) -> pd.DataFrame:
        if False:
            while True:
                i = 10
        'Loads the input as a Pandas DataFrame.'
        if table_slice.partition_dimensions and len(context.asset_partition_keys) == 0:
            return pd.DataFrame()
        return connection.execute(DuckDbClient.get_select_statement(table_slice)).fetchdf()

    @property
    def supported_types(self):
        if False:
            while True:
                i = 10
        return [pd.DataFrame]
duckdb_pandas_io_manager = build_duckdb_io_manager([DuckDBPandasTypeHandler()], default_load_type=pd.DataFrame)
duckdb_pandas_io_manager.__doc__ = '\nAn I/O manager definition that reads inputs from and writes Pandas DataFrames to DuckDB. When\nusing the duckdb_pandas_io_manager, any inputs and outputs without type annotations will be loaded\nas Pandas DataFrames.\n\nReturns:\n    IOManagerDefinition\n\nExamples:\n\n    .. code-block:: python\n\n        from dagster_duckdb_pandas import duckdb_pandas_io_manager\n\n        @asset(\n            key_prefix=["my_schema"]  # will be used as the schema in DuckDB\n        )\n        def my_table() -> pd.DataFrame:  # the name of the asset will be the table name\n            ...\n\n        @repository\n        def my_repo():\n            return with_resources(\n                [my_table],\n                {"io_manager": duckdb_pandas_io_manager.configured({"database": "my_db.duckdb"})}\n            )\n\n    If you do not provide a schema, Dagster will determine a schema based on the assets and ops using\n    the I/O Manager. For assets, the schema will be determined from the asset key.\n    For ops, the schema can be specified by including a "schema" entry in output metadata. If "schema" is not provided\n    via config or on the asset/op, "public" will be used for the schema.\n\n    .. code-block:: python\n\n        @op(\n            out={"my_table": Out(metadata={"schema": "my_schema"})}\n        )\n        def make_my_table() -> pd.DataFrame:\n            # the returned value will be stored at my_schema.my_table\n            ...\n\n    To only use specific columns of a table as input to a downstream op or asset, add the metadata "columns" to the\n    In or AssetIn.\n\n    .. code-block:: python\n\n        @asset(\n            ins={"my_table": AssetIn("my_table", metadata={"columns": ["a"]})}\n        )\n        def my_table_a(my_table: pd.DataFrame) -> pd.DataFrame:\n            # my_table will just contain the data from column "a"\n            ...\n\n'

class DuckDBPandasIOManager(DuckDBIOManager):
    """An I/O manager definition that reads inputs from and writes Pandas DataFrames to DuckDB. When
    using the DuckDBPandasIOManager, any inputs and outputs without type annotations will be loaded
    as Pandas DataFrames.

    Returns:
        IOManagerDefinition

    Examples:
        .. code-block:: python

            from dagster_duckdb_pandas import DuckDBPandasIOManager

            @asset(
                key_prefix=["my_schema"]  # will be used as the schema in DuckDB
            )
            def my_table() -> pd.DataFrame:  # the name of the asset will be the table name
                ...

            defs = Definitions(
                assets=[my_table],
                resources={"io_manager": DuckDBPandasIOManager(database="my_db.duckdb")}
            )

        If you do not provide a schema, Dagster will determine a schema based on the assets and ops using
        the I/O Manager. For assets, the schema will be determined from the asset key, as in the above example.
        For ops, the schema can be specified by including a "schema" entry in output metadata. If "schema" is not provided
        via config or on the asset/op, "public" will be used for the schema.

        .. code-block:: python

            @op(
                out={"my_table": Out(metadata={"schema": "my_schema"})}
            )
            def make_my_table() -> pd.DataFrame:
                # the returned value will be stored at my_schema.my_table
                ...

        To only use specific columns of a table as input to a downstream op or asset, add the metadata "columns" to the
        In or AssetIn.

        .. code-block:: python

            @asset(
                ins={"my_table": AssetIn("my_table", metadata={"columns": ["a"]})}
            )
            def my_table_a(my_table: pd.DataFrame) -> pd.DataFrame:
                # my_table will just contain the data from column "a"
                ...

    """

    @classmethod
    def _is_dagster_maintained(cls) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def type_handlers() -> Sequence[DbTypeHandler]:
        if False:
            print('Hello World!')
        return [DuckDBPandasTypeHandler()]

    @staticmethod
    def default_load_type() -> Optional[Type]:
        if False:
            return 10
        return pd.DataFrame