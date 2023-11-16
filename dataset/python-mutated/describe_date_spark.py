from typing import Tuple
import pyspark.sql.functions as F
from numpy import array
from pyspark.sql import DataFrame
from ydata_profiling.config import Settings
from ydata_profiling.model.summary_algorithms import describe_date_1d

def date_stats_spark(df: DataFrame, summary: dict) -> dict:
    if False:
        for i in range(10):
            print('nop')
    column = df.columns[0]
    expr = [F.min(F.col(column)).alias('min'), F.max(F.col(column)).alias('max')]
    return df.agg(*expr).first().asDict()

@describe_date_1d.register
def describe_date_1d_spark(config: Settings, df: DataFrame, summary: dict) -> Tuple[Settings, DataFrame, dict]:
    if False:
        for i in range(10):
            print('nop')
    'Describe a date series.\n\n    Args:\n        series: The Series to describe.\n        summary: The dict containing the series description so far.\n\n    Returns:\n        A dict containing calculated series description values.\n    '
    col_name = df.columns[0]
    stats = date_stats_spark(df, summary)
    summary.update({'min': stats['min'], 'max': stats['max']})
    summary['range'] = summary['max'] - summary['min']
    df = df.withColumn(col_name, F.unix_timestamp(df[col_name]))
    bins = config.plot.histogram.bins
    bins_arg = 'auto' if bins == 0 else min(bins, summary['n_distinct'])
    (bin_edges, hist) = df.select(col_name).rdd.flatMap(lambda x: x).histogram(bins_arg)
    summary.update({'histogram': (array(hist), array(bin_edges))})
    return (config, df, summary)