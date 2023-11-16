from pandas import DataFrame
import math
if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def select_number_columns(df: DataFrame) -> DataFrame:
    if False:
        return 10
    return df[['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Survived']]

def fill_missing_values_with_median(df: DataFrame) -> DataFrame:
    if False:
        print('Hello World!')
    for col in df.columns:
        values = sorted(df[col].dropna().tolist())
        median_age = values[math.floor(len(values) / 2)]
        df[[col]] = df[[col]].fillna(median_age)
    return df

@transformer
def transform_df(df: DataFrame, *args, **kwargs) -> DataFrame:
    if False:
        return 10
    '\n    Template code for a transformer block.\n\n    Add more parameters to this function if this block has multiple parent blocks.\n    There should be one parameter for each output variable from each parent block.\n\n    Args:\n        df (DataFrame): Data frame from parent block.\n\n    Returns:\n        DataFrame: Transformed data frame\n    '
    return fill_missing_values_with_median(select_number_columns(df))

@test
def test_output(df) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Template code for testing the output of the block.\n    '
    assert df is not None, 'The output is undefined'