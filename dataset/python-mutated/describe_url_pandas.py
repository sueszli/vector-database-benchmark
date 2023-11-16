from typing import Tuple
from urllib.parse import urlsplit
import pandas as pd
from ydata_profiling.config import Settings
from ydata_profiling.model.summary_algorithms import describe_url_1d

def url_summary(series: pd.Series) -> dict:
    if False:
        i = 10
        return i + 15
    '\n\n    Args:\n        series: series to summarize\n\n    Returns:\n\n    '
    summary = {'scheme_counts': series.map(lambda x: x.scheme).value_counts(), 'netloc_counts': series.map(lambda x: x.netloc).value_counts(), 'path_counts': series.map(lambda x: x.path).value_counts(), 'query_counts': series.map(lambda x: x.query).value_counts(), 'fragment_counts': series.map(lambda x: x.fragment).value_counts()}
    return summary

@describe_url_1d.register
def pandas_describe_url_1d(config: Settings, series: pd.Series, summary: dict) -> Tuple[Settings, pd.Series, dict]:
    if False:
        return 10
    'Describe a url series.\n\n    Args:\n        config: report Settings object\n        series: The Series to describe.\n        summary: The dict containing the series description so far.\n\n    Returns:\n        A dict containing calculated series description values.\n    '
    if series.hasnans:
        raise ValueError('May not contain NaNs')
    if not hasattr(series, 'str'):
        raise ValueError('series should have .str accessor')
    series = series.apply(urlsplit)
    summary.update(url_summary(series))
    return (config, series, summary)