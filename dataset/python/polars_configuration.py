from dagster import asset


@asset
def iris_dataset():
    return None


# start_configuration

from dagster_duckdb_polars import DuckDBPolarsIOManager

from dagster import Definitions

defs = Definitions(
    assets=[iris_dataset],
    resources={
        "io_manager": DuckDBPolarsIOManager(
            database="path/to/my_duckdb_database.duckdb",  # required
            schema="IRIS",  # optional, defaults to PUBLIC
        )
    },
)

# end_configuration
