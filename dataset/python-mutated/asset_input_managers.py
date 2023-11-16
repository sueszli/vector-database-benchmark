from typing import List
import pandas as pd
from dagster import AssetIn, Definitions, asset

def store_pandas_dataframe(*_args, **_kwargs):
    if False:
        return 10
    pass

def load_pandas_dataframe(*_args, **_kwargs):
    if False:
        i = 10
        return i + 15
    pass

def load_numpy_array(*_args, **_kwargs):
    if False:
        print('Hello World!')
    pass

class PandasSeriesIOManager:
    pass

@asset
def first_asset() -> List[int]:
    if False:
        while True:
            i = 10
    return [1, 2, 3]

@asset
def second_asset() -> List[int]:
    if False:
        return 10
    return [4, 5, 6]

@asset(ins={'first_asset': AssetIn(input_manager_key='pandas_series'), 'second_asset': AssetIn(input_manager_key='pandas_series')})
def third_asset(first_asset: pd.Series, second_asset: pd.Series) -> pd.Series:
    if False:
        i = 10
        return i + 15
    return pd.concat([first_asset, second_asset, pd.Series([7, 8])])
defs = Definitions(assets=[first_asset, second_asset, third_asset], resources={'pandas_series': PandasSeriesIOManager()})