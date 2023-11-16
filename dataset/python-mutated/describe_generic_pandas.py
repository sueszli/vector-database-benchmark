from typing import Tuple
import pandas as pd
from ydata_profiling.config import Settings
from ydata_profiling.model.summary_algorithms import describe_generic

@describe_generic.register
def pandas_describe_generic(config: Settings, series: pd.Series, summary: dict) -> Tuple[Settings, pd.Series, dict]:
    if False:
        return 10
    'Describe generic series.\n\n    Args:\n        config: report Settings object\n        series: The Series to describe.\n        summary: The dict containing the series description so far.\n\n    Returns:\n        A dict containing calculated series description values.\n    '
    length = len(series)
    summary.update({'n': length, 'p_missing': summary['n_missing'] / length if length > 0 else 0, 'count': length - summary['n_missing'], 'memory_size': series.memory_usage(deep=config.memory_deep)})
    return (config, series, summary)