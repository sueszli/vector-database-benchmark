from dagster import InMemoryIOManager, TableSchema, load_assets_from_modules, materialize
from pandas import DataFrame, Series
from assets_smoke_test import pure_python_assets

def empty_dataframe_from_column_schema(column_schema: TableSchema) -> DataFrame:
    if False:
        for i in range(10):
            print('nop')
    return DataFrame({column.name: Series(dtype=column.type) for column in column_schema.columns})

class SmokeIOManager(InMemoryIOManager):

    def load_input(self, context):
        if False:
            while True:
                i = 10
        if context.asset_key not in context.step_context.job_def.asset_layer.asset_keys:
            column_schema = context.upstream_output.metadata['column_schema']
            return empty_dataframe_from_column_schema(column_schema)
        else:
            return super().load_input(context)

def test_smoke_all():
    if False:
        i = 10
        return i + 15
    assets = load_assets_from_modules([pure_python_assets])
    materialize(assets, resources={'io_manager': SmokeIOManager()})