import warnings
from pyspark.sql import DataFrame
from ydata_profiling.config import Settings
from ydata_profiling.model.dataframe import check_dataframe, preprocess

@check_dataframe.register
def spark_check_dataframe(df: DataFrame) -> None:
    if False:
        print('Hello World!')
    if not isinstance(df, DataFrame):
        warnings.warn('df is not of type pyspark.sql.dataframe.DataFrame')

@preprocess.register
def spark_preprocess(config: Settings, df: DataFrame) -> DataFrame:
    if False:
        print('Hello World!')
    'Preprocess the dataframe\n\n    - Appends the index to the dataframe when it contains information\n    - Rename the "index" column to "df_index", if exists\n    - Convert the DataFrame\'s columns to str\n\n    Args:\n        config: report Settings object\n        df: the pandas DataFrame\n\n    Returns:\n        The preprocessed DataFrame\n    '

    def _check_column_map_type(df: DataFrame, column_name: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return str(df.select(column_name).schema[0].dataType).startswith('MapType')
    columns_to_remove = list(filter(lambda x: _check_column_map_type(df, x), df.columns))
    if columns_to_remove:
        warnings.warn(f"spark dataframes profiling does not handle MapTypes. Column(s) {','.join(columns_to_remove)} will be ignored.\n            To fix this, consider converting your MapType into a StructTypes of StructFields i.e.\n            {{'key1':'value1',...}} -> [('key1','value1'), ...], or extracting the key,value pairs out\n            into individual columns using pyspark.sql.functions.explode.\n            ")
        columns_to_keep = list(filter(lambda x: not _check_column_map_type(df, x), df.columns))
        return df.select(*columns_to_keep)
    else:
        return df