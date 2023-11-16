from typing import Tuple
from pyspark.sql import DataFrame
from ydata_profiling.config import Settings
from ydata_profiling.model.summary_algorithms import describe_supported

@describe_supported.register
def describe_supported_spark(config: Settings, series: DataFrame, summary: dict) -> Tuple[Settings, DataFrame, dict]:
    if False:
        while True:
            i = 10
    'Describe a supported series.\n    Args:\n        series: The Series to describe.\n        series_description: The dict containing the series description so far.\n    Returns:\n        A dict containing calculated series description values.\n    '
    count = summary['count']
    n_distinct = summary['value_counts'].count()
    summary['n_distinct'] = n_distinct
    summary['p_distinct'] = n_distinct / count if count > 0 else 0
    n_unique = summary['value_counts'].where('count == 1').count()
    summary['is_unique'] = n_unique == count
    summary['n_unique'] = n_unique
    summary['p_unique'] = n_unique / count
    return (config, series, summary)