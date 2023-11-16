from __future__ import annotations
import itertools
from typing import TYPE_CHECKING
from google.cloud.bigquery.table import Row, RowIterator
if TYPE_CHECKING:
    from collections.abc import Iterator
    from logging import Logger
    from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook

def bigquery_get_data(logger: Logger, dataset_id: str, table_id: str, big_query_hook: BigQueryHook, batch_size: int, selected_fields: list[str] | str | None) -> Iterator:
    if False:
        i = 10
        return i + 15
    logger.info('Fetching Data from:')
    logger.info('Dataset: %s ; Table: %s', dataset_id, table_id)
    for start_index in itertools.count(step=batch_size):
        rows: list[Row] | RowIterator = big_query_hook.list_rows(dataset_id=dataset_id, table_id=table_id, max_results=batch_size, selected_fields=selected_fields, start_index=start_index)
        if isinstance(rows, RowIterator):
            raise TypeError('BigQueryHook.list_rows() returns iterator when return_iterator=False (default)')
        if len(rows) == 0:
            logger.info('Job Finished')
            return
        logger.info('Total Extracted rows: %s', len(rows) + start_index)
        yield [row.values() for row in rows]