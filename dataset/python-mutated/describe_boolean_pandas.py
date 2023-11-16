from typing import Tuple
import pandas as pd
from ydata_profiling.config import Settings
from ydata_profiling.model.pandas.imbalance_pandas import column_imbalance_score
from ydata_profiling.model.summary_algorithms import describe_boolean_1d, series_hashable

@describe_boolean_1d.register
@series_hashable
def pandas_describe_boolean_1d(config: Settings, series: pd.Series, summary: dict) -> Tuple[Settings, pd.Series, dict]:
    if False:
        i = 10
        return i + 15
    'Describe a boolean series.\n\n    Args:\n        config: report Settings object\n        series: The Series to describe.\n        summary: The dict containing the series description so far.\n\n    Returns:\n        A dict containing calculated series description values.\n    '
    value_counts = summary['value_counts_without_nan']
    summary.update({'top': value_counts.index[0], 'freq': value_counts.iloc[0]})
    summary['imbalance'] = column_imbalance_score(value_counts, len(value_counts))
    return (config, series, summary)