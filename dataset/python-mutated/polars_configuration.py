from dagster import asset

@asset
def iris_dataset():
    if False:
        return 10
    return None
from dagster_duckdb_polars import DuckDBPolarsIOManager
from dagster import Definitions
defs = Definitions(assets=[iris_dataset], resources={'io_manager': DuckDBPolarsIOManager(database='path/to/my_duckdb_database.duckdb', schema='IRIS')})