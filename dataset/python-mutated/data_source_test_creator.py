import contextlib
import random
import tempfile
import time
from typing import Iterator
from google.cloud import bigquery
from feast import BigQuerySource, FileSource
from feast.data_format import ParquetFormat

@contextlib.contextmanager
def prep_file_source(df, timestamp_field=None) -> Iterator[FileSource]:
    if False:
        print('Hello World!')
    with tempfile.NamedTemporaryFile(suffix='.parquet') as f:
        f.close()
        df.to_parquet(f.name)
        file_source = FileSource(file_format=ParquetFormat(), path=f.name, timestamp_field=timestamp_field)
        yield file_source

def simple_bq_source_using_table_arg(df, timestamp_field=None) -> BigQuerySource:
    if False:
        while True:
            i = 10
    client = bigquery.Client()
    gcp_project = client.project
    bigquery_dataset = f'ds_{time.time_ns()}'
    dataset = bigquery.Dataset(f'{gcp_project}.{bigquery_dataset}')
    client.create_dataset(dataset, exists_ok=True)
    dataset.default_table_expiration_ms = 1000 * 60 * 60
    client.update_dataset(dataset, ['default_table_expiration_ms'])
    table = f'{gcp_project}.{bigquery_dataset}.table_{random.randrange(100, 999)}'
    job = client.load_table_from_dataframe(df, table)
    job.result()
    return BigQuerySource(table=table, timestamp_field=timestamp_field)

def simple_bq_source_using_query_arg(df, timestamp_field=None) -> BigQuerySource:
    if False:
        for i in range(10):
            print('nop')
    bq_source_using_table = simple_bq_source_using_table_arg(df, timestamp_field)
    return BigQuerySource(name=bq_source_using_table.table, query=f'SELECT * FROM {bq_source_using_table.table}', timestamp_field=timestamp_field)