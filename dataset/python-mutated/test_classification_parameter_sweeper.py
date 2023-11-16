import pytest
import pandas as pd
from utils_cv.classification.parameter_sweeper import ParameterSweeper, clean_sweeper_df, plot_sweeper_df

def _test_sweeper_run(df: pd.DataFrame, df_length: int):
    if False:
        while True:
            i = 10
    ' Performs basic tests that all df should pass.\n    Args:\n        df (pd.DataFame): the df to check\n        df_length (int): to assert the len(df) == df_length\n    '
    assert len(df) == df_length
    assert isinstance(df.index, pd.MultiIndex)
    df = clean_sweeper_df(df)
    assert isinstance(df.index, pd.MultiIndex)
    plot_sweeper_df(df)

def test_default_sweeper_single_dataset(tiny_ic_data_path):
    if False:
        for i in range(10):
            print('nop')
    ' Test default sweeper on a single dataset. '
    sweeper = ParameterSweeper().update_parameters(epochs=[3], im_size=[50])
    df = sweeper.run([tiny_ic_data_path], reps=1)
    _test_sweeper_run(df, df_length=1)
    assert df.mean(level=1)['accuracy'][0] > 0.0

def test_default_sweeper_benchmark_dataset(tiny_ic_multidata_path):
    if False:
        while True:
            i = 10
    '\n    Test default sweeper on benchmark dataset.\n    WARNING: This test can take a while to execute since we run the sweeper\n    across all benchmark datasets.\n    '
    sweeper = ParameterSweeper().update_parameters(epochs=[1], im_size=[50])
    df = sweeper.run(tiny_ic_multidata_path, reps=1)
    _test_sweeper_run(df, df_length=len(tiny_ic_multidata_path))
    assert df.mean(level=2).loc['fridgeObjectsTiny', 'accuracy'] > 0.0
    assert df.mean(level=2).loc['fridgeObjectsWatermarkTiny', 'accuracy'] > 0.0

def test_update_parameters_01(tiny_ic_data_path):
    if False:
        for i in range(10):
            print('nop')
    ' Tests updating parameters. '
    sweeper = ParameterSweeper()
    assert len(sweeper.permutations) == 1
    sweeper.update_parameters(learning_rate=[0.0001], im_size=[50, 55], epochs=[1])
    assert len(sweeper.permutations) == 2
    df = sweeper.run([tiny_ic_data_path], reps=1)
    _test_sweeper_run(df, df_length=2)

def test_update_parameters_02():
    if False:
        for i in range(10):
            print('nop')
    ' Tests exception when updating parameters. '
    sweeper = ParameterSweeper()
    with pytest.raises(Exception):
        sweeper.update_parameters(bad_key=[0.001, 0.0001, 1e-05])