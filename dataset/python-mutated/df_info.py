import pandas as pd
from typing import Union
polars_imported = False
try:
    import polars as pl
    polars_imported = True
    DataFrameType = Union[pd.DataFrame, pl.DataFrame, str]
except ImportError:
    DataFrameType = Union[pd.DataFrame, str]

def df_type(df: DataFrameType) -> Union[str, None]:
    if False:
        while True:
            i = 10
    '\n    Returns the type of the dataframe.\n\n    Args:\n        df (DataFrameType): Pandas or Polars dataframe\n\n    Returns:\n        str: Type of the dataframe\n    '
    if polars_imported and isinstance(df, pl.DataFrame):
        return 'polars'
    elif isinstance(df, pd.DataFrame):
        return 'pandas'
    else:
        return None