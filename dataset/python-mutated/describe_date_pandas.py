from typing import Tuple
import numpy as np
import pandas as pd
from ydata_profiling.config import Settings
from ydata_profiling.model.summary_algorithms import chi_square, describe_date_1d, histogram_compute, series_handle_nulls, series_hashable

@describe_date_1d.register
@series_hashable
@series_handle_nulls
def pandas_describe_date_1d(config: Settings, series: pd.Series, summary: dict) -> Tuple[Settings, pd.Series, dict]:
    if False:
        print('Hello World!')
    'Describe a date series.\n\n    Args:\n        config: report Settings object\n        series: The Series to describe.\n        summary: The dict containing the series description so far.\n\n    Returns:\n        A dict containing calculated series description values.\n    '
    summary.update({'min': pd.Timestamp.to_pydatetime(series.min()), 'max': pd.Timestamp.to_pydatetime(series.max())})
    summary['range'] = summary['max'] - summary['min']
    values = series.values.astype(np.int64) // 10 ** 9
    if config.vars.num.chi_squared_threshold > 0.0:
        summary['chi_squared'] = chi_square(values)
    summary.update(histogram_compute(config, values, summary['n_distinct']))
    return (config, values, summary)