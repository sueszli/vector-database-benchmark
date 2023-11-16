from typing import Tuple
from pyspark.sql import DataFrame
from ydata_profiling.config import Settings
from ydata_profiling.model.summary_algorithms import describe_generic

@describe_generic.register
def describe_generic_spark(config: Settings, df: DataFrame, summary: dict) -> Tuple[Settings, DataFrame, dict]:
    if False:
        for i in range(10):
            print('nop')
    'Describe generic series.\n    Args:\n        series: The Series to describe.\n        summary: The dict containing the series description so far.\n    Returns:\n        A dict containing calculated series description values.\n    '
    length = df.count()
    summary['n'] = length
    summary['p_missing'] = summary['n_missing'] / length
    summary['count'] = length - summary['n_missing']
    summary['memory_size'] = 0
    return (config, df, summary)