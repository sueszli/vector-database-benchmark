from typing import Tuple
from pyspark.sql import DataFrame
from ydata_profiling.config import Settings
from ydata_profiling.model.summary_algorithms import describe_counts

@describe_counts.register
def describe_counts_spark(config: Settings, series: DataFrame, summary: dict) -> Tuple[Settings, DataFrame, dict]:
    if False:
        for i in range(10):
            print('nop')
    'Counts the values in a series (with and without NaN, distinct).\n\n    Args:\n        series: Series for which we want to calculate the values.\n\n    Returns:\n        A dictionary with the count values (with and without NaN, distinct).\n    '
    value_counts = series.groupBy(series.columns).count()
    value_counts = value_counts.sort('count', ascending=False).persist()
    value_counts_index_sorted = value_counts.sort(series.columns[0], ascending=True)
    n_missing = value_counts.where(value_counts[series.columns[0]].isNull()).first()
    if n_missing is None:
        n_missing = 0
    else:
        n_missing = n_missing['count']
    value_counts_index_sorted = value_counts_index_sorted.limit(200).toPandas().set_index(series.columns[0], drop=True).squeeze(axis='columns')
    summary['n_missing'] = n_missing
    summary['value_counts'] = value_counts.persist()
    summary['value_counts_index_sorted'] = value_counts_index_sorted
    summary['value_counts_without_nan'] = value_counts.dropna().limit(200).toPandas().set_index(series.columns[0], drop=True).squeeze(axis='columns')
    return (config, series, summary)