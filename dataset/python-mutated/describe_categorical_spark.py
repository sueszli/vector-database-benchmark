from typing import Tuple
from pyspark.sql import DataFrame
from ydata_profiling.config import Settings
from ydata_profiling.model.summary_algorithms import describe_categorical_1d

@describe_categorical_1d.register
def describe_categorical_1d_spark(config: Settings, df: DataFrame, summary: dict) -> Tuple[Settings, DataFrame, dict]:
    if False:
        for i in range(10):
            print('nop')
    'Describe a categorical series.\n\n    Args:\n        series: The Series to describe.\n        summary: The dict containing the series description so far.\n\n    Returns:\n        A dict containing calculated series description values.\n    '
    redact = config.vars.cat.redact
    if not redact:
        summary['first_rows'] = df.limit(5).toPandas().squeeze('columns')
    return (config, df, summary)