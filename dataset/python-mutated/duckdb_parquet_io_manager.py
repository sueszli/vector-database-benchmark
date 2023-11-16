import os
import duckdb
import pandas as pd
from dagster import PartitionKeyRange, _check as check
from dagster._seven.temp_dir import get_system_temp_directory
from .parquet_io_manager import PartitionedParquetIOManager

class DuckDBPartitionedParquetIOManager(PartitionedParquetIOManager):
    """Stores data in parquet files and creates duckdb views over those files."""
    duckdb_path: str
    base_path: str = get_system_temp_directory()

    @property
    def _base_path(self):
        if False:
            i = 10
            return i + 15
        return self.base_path

    def handle_output(self, context, obj):
        if False:
            return 10
        if obj is not None:
            super().handle_output(context, obj)
            con = self._connect_duckdb()
            path = self._get_path(context)
            if context.has_asset_partitions:
                to_scan = os.path.join(os.path.dirname(path), '*.pq', '*.parquet')
            else:
                to_scan = path
            con.execute(f'create schema if not exists {self._schema(context)};')
            con.execute(f"create or replace view {self._table_path(context)} as select * from parquet_scan('{to_scan}');")

    def load_input(self, context):
        if False:
            print('Hello World!')
        check.invariant(not context.has_asset_partitions or context.asset_partition_key_range == PartitionKeyRange(context.asset_partitions_def.get_first_partition_key(), context.asset_partitions_def.get_last_partition_key()), 'Loading a subselection of partitions is not yet supported')
        if context.dagster_type.typing_type == pd.DataFrame:
            con = self._connect_duckdb()
            return con.execute(f'SELECT * FROM {self._table_path(context)}').fetchdf()
        check.failed(f'Inputs of type {context.dagster_type} not supported. Please specify a valid type for this input either on the argument of the @asset-decorated function.')

    def _table_path(self, context) -> str:
        if False:
            print('Hello World!')
        return f'{self._schema(context)}.{context.asset_key.path[-1]}'

    def _schema(self, context) -> str:
        if False:
            while True:
                i = 10
        return f'{context.asset_key.path[-2]}'

    def _connect_duckdb(self):
        if False:
            while True:
                i = 10
        return duckdb.connect(database=self.duckdb_path, read_only=False)