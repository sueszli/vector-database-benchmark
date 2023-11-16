from typing import Tuple
import pandas as pd
from ydata_profiling.config import Settings
from ydata_profiling.model.summary_algorithms import describe_supported, series_hashable

@describe_supported.register
@series_hashable
def pandas_describe_supported(config: Settings, series: pd.Series, series_description: dict) -> Tuple[Settings, pd.Series, dict]:
    if False:
        for i in range(10):
            print('nop')
    'Describe a supported series.\n\n    Args:\n        config: report Settings object\n        series: The Series to describe.\n        series_description: The dict containing the series description so far.\n\n    Returns:\n        A dict containing calculated series description values.\n    '
    count = series_description['count']
    value_counts = series_description['value_counts_without_nan']
    distinct_count = len(value_counts)
    unique_count = value_counts.where(value_counts == 1).count()
    stats = {'n_distinct': distinct_count, 'p_distinct': distinct_count / count if count > 0 else 0, 'is_unique': unique_count == count and count > 0, 'n_unique': unique_count, 'p_unique': unique_count / count if count > 0 else 0}
    stats.update(series_description)
    return (config, series, stats)