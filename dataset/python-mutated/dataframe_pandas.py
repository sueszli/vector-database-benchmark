import warnings
import pandas as pd
from ydata_profiling.config import Settings
from ydata_profiling.model.dataframe import check_dataframe, preprocess
from ydata_profiling.utils.dataframe import rename_index

@check_dataframe.register
def pandas_check_dataframe(df: pd.DataFrame) -> None:
    if False:
        while True:
            i = 10
    if not isinstance(df, pd.DataFrame):
        warnings.warn('df is not of type pandas.DataFrame')

@preprocess.register
def pandas_preprocess(config: Settings, df: pd.DataFrame) -> pd.DataFrame:
    if False:
        i = 10
        return i + 15
    'Preprocess the dataframe\n\n    - Appends the index to the dataframe when it contains information\n    - Rename the "index" column to "df_index", if exists\n    - Convert the DataFrame\'s columns to str\n\n    Args:\n        config: report Settings object\n        df: the pandas DataFrame\n\n    Returns:\n        The preprocessed DataFrame\n    '
    df = rename_index(df)
    df.columns = df.columns.astype('str')
    return df