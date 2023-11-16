"""Compute statistical description of datasets."""
import multiprocessing
from typing import Tuple
import numpy as np
from pyspark.sql import DataFrame
from tqdm import tqdm
from visions import VisionsTypeset
from ydata_profiling.config import Settings
from ydata_profiling.model.summarizer import BaseSummarizer
from ydata_profiling.model.summary import describe_1d, get_series_descriptions
from ydata_profiling.utils.dataframe import sort_column_names

@describe_1d.register
def spark_describe_1d(config: Settings, series: DataFrame, summarizer: BaseSummarizer, typeset: VisionsTypeset) -> dict:
    if False:
        print('Hello World!')
    'Describe a series (infer the variable type, then calculate type-specific values).\n\n    Args:\n        config: report Settings object\n        series: The Series to describe.\n        summarizer: Summarizer object\n        typeset: Typeset\n\n    Returns:\n        A Series containing calculated series description values.\n    '
    series = series.fillna(np.nan)
    if config.infer_dtypes:
        vtype = typeset.infer_type(series)
        series = typeset.cast_to_inferred(series)
    else:
        if str(series.schema[0].dataType).startswith('ArrayType'):
            dtype = 'ArrayType'
        else:
            dtype = series.schema[0].dataType.simpleString()
        vtype = {'float': 'Numeric', 'int': 'Numeric', 'bigint': 'Numeric', 'double': 'Numeric', 'string': 'Categorical', 'ArrayType': 'Categorical', 'boolean': 'Boolean', 'date': 'DateTime', 'timestamp': 'DateTime'}[dtype]
    return summarizer.summarize(config, series, dtype=vtype)

@get_series_descriptions.register
def spark_get_series_descriptions(config: Settings, df: DataFrame, summarizer: BaseSummarizer, typeset: VisionsTypeset, pbar: tqdm) -> dict:
    if False:
        i = 10
        return i + 15
    series_description = {}

    def multiprocess_1d(args: tuple) -> Tuple[str, dict]:
        if False:
            while True:
                i = 10
        'Wrapper to process series in parallel.\n\n        Args:\n            column: The name of the column.\n            series: The series values.\n\n        Returns:\n            A tuple with column and the series description.\n        '
        (column, df) = args
        return (column, describe_1d(config, df.select(column), summarizer, typeset))
    args = [(name, df) for name in df.columns]
    with multiprocessing.pool.ThreadPool(12) as executor:
        for (i, (column, description)) in enumerate(executor.imap_unordered(multiprocess_1d, args)):
            pbar.set_postfix_str(f'Describe variable:{column}')
            description.pop('value_counts')
            series_description[column] = description
            pbar.update()
        series_description = {k: series_description[k] for k in df.columns}
    series_description = sort_column_names(series_description, config.sort)
    return series_description