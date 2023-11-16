def get_iris_data_for_date(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    pass
import pandas as pd
from dagster import AssetExecutionContext, DailyPartitionsDefinition, asset

@asset(partitions_def=DailyPartitionsDefinition(start_date='2023-01-01'), metadata={'partition_expr': 'TIMESTAMP_SECONDS(TIME)'})
def iris_data_per_day(context: AssetExecutionContext) -> pd.DataFrame:
    if False:
        print('Hello World!')
    partition = context.asset_partition_key_for_output()
    return get_iris_data_for_date(partition)

@asset
def iris_cleaned(iris_data_per_day: pd.DataFrame):
    if False:
        i = 10
        return i + 15
    return iris_data_per_day.dropna().drop_duplicates()