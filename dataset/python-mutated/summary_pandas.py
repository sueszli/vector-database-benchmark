"""Compute statistical description of datasets."""
import multiprocessing
import multiprocessing.pool
from typing import Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from visions import VisionsTypeset
from ydata_profiling.config import Settings
from ydata_profiling.model.summarizer import BaseSummarizer
from ydata_profiling.model.summary import describe_1d, get_series_descriptions
from ydata_profiling.model.typeset import ProfilingTypeSet
from ydata_profiling.utils.dataframe import sort_column_names

@describe_1d.register
def pandas_describe_1d(config: Settings, series: pd.Series, summarizer: BaseSummarizer, typeset: VisionsTypeset) -> dict:
    if False:
        while True:
            i = 10
    'Describe a series (infer the variable type, then calculate type-specific values).\n\n    Args:\n        config: report Settings object\n        series: The Series to describe.\n        summarizer: Summarizer object\n        typeset: Typeset\n\n    Returns:\n        A Series containing calculated series description values.\n    '
    series = series.fillna(np.nan)
    if isinstance(typeset, ProfilingTypeSet) and typeset.type_schema and (series.name in typeset.type_schema):
        vtype = typeset.type_schema[series.name]
    elif config.infer_dtypes:
        vtype = typeset.infer_type(series)
        series = typeset.cast_to_inferred(series)
    else:
        vtype = typeset.detect_type(series)
    typeset.type_schema[series.name] = vtype
    return summarizer.summarize(config, series, dtype=vtype)

@get_series_descriptions.register
def pandas_get_series_descriptions(config: Settings, df: pd.DataFrame, summarizer: BaseSummarizer, typeset: VisionsTypeset, pbar: tqdm) -> dict:
    if False:
        for i in range(10):
            print('nop')

    def multiprocess_1d(args: tuple) -> Tuple[str, dict]:
        if False:
            while True:
                i = 10
        'Wrapper to process series in parallel.\n\n        Args:\n            column: The name of the column.\n            series: The series values.\n\n        Returns:\n            A tuple with column and the series description.\n        '
        (column, series) = args
        return (column, describe_1d(config, series, summarizer, typeset))
    pool_size = config.pool_size
    if pool_size <= 0:
        pool_size = multiprocessing.cpu_count()
    args = [(name, series) for (name, series) in df.items()]
    series_description = {}
    if pool_size == 1:
        for arg in args:
            pbar.set_postfix_str(f'Describe variable:{arg[0]}')
            (column, description) = multiprocess_1d(arg)
            series_description[column] = description
            pbar.update()
    else:
        with multiprocessing.pool.ThreadPool(pool_size) as executor:
            for (i, (column, description)) in enumerate(executor.imap_unordered(multiprocess_1d, args)):
                pbar.set_postfix_str(f'Describe variable:{column}')
                series_description[column] = description
                pbar.update()
        series_description = {k: series_description[k] for k in df.columns}
    series_description = sort_column_names(series_description, config.sort)
    return series_description