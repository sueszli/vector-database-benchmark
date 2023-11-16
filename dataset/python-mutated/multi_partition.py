def get_iris_data_for_date(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    pass
import pandas as pd
from dagster import AssetExecutionContext, DailyPartitionsDefinition, MultiPartitionsDefinition, StaticPartitionsDefinition, asset

@asset(partitions_def=MultiPartitionsDefinition({'date': DailyPartitionsDefinition(start_date='2023-01-01'), 'species': StaticPartitionsDefinition(['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'])}), metadata={'partition_expr': {'date': 'TIMESTAMP_SECONDS(TIME)', 'species': 'SPECIES'}})
def iris_data_partitioned(context: AssetExecutionContext) -> pd.DataFrame:
    if False:
        print('Hello World!')
    partition = context.partition_key.keys_by_dimension
    species = partition['species']
    date = partition['date']
    full_df = get_iris_data_for_date(date)
    return full_df[full_df['species'] == species]

@asset
def iris_cleaned(iris_data_partitioned: pd.DataFrame):
    if False:
        while True:
            i = 10
    return iris_data_partitioned.dropna().drop_duplicates()