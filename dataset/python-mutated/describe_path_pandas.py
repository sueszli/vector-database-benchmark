import os
from typing import Tuple
import pandas as pd
from ydata_profiling.config import Settings
from ydata_profiling.model.summary_algorithms import describe_path_1d

def path_summary(series: pd.Series) -> dict:
    if False:
        for i in range(10):
            print('nop')
    '\n\n    Args:\n        series: series to summarize\n\n    Returns:\n\n    '
    summary = {'common_prefix': os.path.commonprefix(series.values.tolist()) or 'No common prefix', 'stem_counts': series.map(lambda x: os.path.splitext(x)[0]).value_counts(), 'suffix_counts': series.map(lambda x: os.path.splitext(x)[1]).value_counts(), 'name_counts': series.map(lambda x: os.path.basename(x)).value_counts(), 'parent_counts': series.map(lambda x: os.path.dirname(x)).value_counts(), 'anchor_counts': series.map(lambda x: os.path.splitdrive(x)[0]).value_counts()}
    summary['n_stem_unique'] = len(summary['stem_counts'])
    summary['n_suffix_unique'] = len(summary['suffix_counts'])
    summary['n_name_unique'] = len(summary['name_counts'])
    summary['n_parent_unique'] = len(summary['parent_counts'])
    summary['n_anchor_unique'] = len(summary['anchor_counts'])
    return summary

@describe_path_1d.register
def pandas_describe_path_1d(config: Settings, series: pd.Series, summary: dict) -> Tuple[Settings, pd.Series, dict]:
    if False:
        return 10
    'Describe a path series.\n\n    Args:\n        config: report Settings object\n        series: The Series to describe.\n        summary: The dict containing the series description so far.\n\n    Returns:\n        A dict containing calculated series description values.\n    '
    if series.hasnans:
        raise ValueError('May not contain NaNs')
    if not hasattr(series, 'str'):
        raise ValueError('series should have .str accessor')
    summary.update(path_summary(series))
    return (config, series, summary)