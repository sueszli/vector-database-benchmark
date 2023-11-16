from typing import Tuple
from pyspark.sql import DataFrame
from ydata_profiling.config import Settings
from ydata_profiling.model.summary_algorithms import describe_boolean_1d

@describe_boolean_1d.register
def describe_boolean_1d_spark(config: Settings, df: DataFrame, summary: dict) -> Tuple[Settings, DataFrame, dict]:
    if False:
        print('Hello World!')
    'Describe a boolean series.\n\n    Args:\n        series: The Series to describe.\n        summary: The dict containing the series description so far.\n\n    Returns:\n        A dict containing calculated series description values.\n    '
    value_counts = summary['value_counts']
    top = value_counts.first()
    summary.update({'top': top[0], 'freq': top[1]})
    return (config, df, summary)