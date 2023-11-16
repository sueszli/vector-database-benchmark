import io
import pandas as pd
import requests
from pandas import DataFrame
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader
def load_data_from_api(**kwargs) -> DataFrame:
    if False:
        for i in range(10):
            print('nop')
    '\n    Template for loading data from API\n    '
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv?raw=True'
    return pd.read_csv(url)

@test
def test_output(df) -> None:
    if False:
        return 10
    '\n    Template code for testing the output of the block.\n    '
    assert df is not None, 'The output is undefined'