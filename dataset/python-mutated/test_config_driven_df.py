import os
import pandas as pd
import pytest
from dagster import DagsterInvalidConfigError, In, Out, graph, op
from dagster._utils import file_relative_path
from dagster_pandas import DataFrame

def check_parquet_support():
    if False:
        for i in range(10):
            print('nop')
    try:
        return
    except ImportError:
        pass
    try:
        import fastparquet
        return
    except ImportError:
        pytest.skip('Skipping parquet test as neither pyarrow nor fastparquet is present.')

def test_dataframe_csv_from_inputs():
    if False:
        i = 10
        return i + 15
    called = {}

    @op(ins={'df': In(DataFrame)})
    def df_as_config(_context, df):
        if False:
            print('Hello World!')
        assert df.to_dict('list') == {'num1': [1, 3], 'num2': [2, 4]}
        called['yup'] = True

    @graph
    def test_graph():
        if False:
            while True:
                i = 10
        return df_as_config()
    result = test_graph.execute_in_process(run_config={'ops': {'df_as_config': {'inputs': {'df': {'csv': {'path': file_relative_path(__file__, 'num.csv')}}}}}})
    assert result.success
    assert called['yup']

def test_dataframe_wrong_sep_from_inputs():
    if False:
        print('Hello World!')
    called = {}

    @op(ins={'df': In(DataFrame)})
    def df_as_config(_context, df):
        if False:
            i = 10
            return i + 15
        assert df.to_dict('list') == {'num1,num2': ['1,2', '3,4']}
        called['yup'] = True

    @graph
    def test_graph():
        if False:
            print('Hello World!')
        return df_as_config()
    result = test_graph.execute_in_process(run_config={'ops': {'df_as_config': {'inputs': {'df': {'csv': {'path': file_relative_path(__file__, 'num.csv'), 'sep': '|'}}}}}})
    assert result.success
    assert called['yup']

def test_dataframe_pipe_sep_csv_from_inputs():
    if False:
        print('Hello World!')
    called = {}

    @op(ins={'df': In(DataFrame)})
    def df_as_config(_context, df):
        if False:
            while True:
                i = 10
        assert df.to_dict('list') == {'num1': [1, 3], 'num2': [2, 4]}
        called['yup'] = True

    @graph
    def test_graph():
        if False:
            return 10
        return df_as_config()
    result = test_graph.execute_in_process(run_config={'ops': {'df_as_config': {'inputs': {'df': {'csv': {'path': file_relative_path(__file__, 'num_pipes.csv'), 'sep': '|'}}}}}})
    assert result.success
    assert called['yup']

def test_dataframe_csv_missing_inputs():
    if False:
        while True:
            i = 10
    called = {}

    @op(ins={'df': In(DataFrame)})
    def df_as_input(_context, df):
        if False:
            return 10
        called['yup'] = True

    @graph
    def missing_inputs():
        if False:
            while True:
                i = 10
        return df_as_input()
    with pytest.raises(DagsterInvalidConfigError) as exc_info:
        missing_inputs.execute_in_process()
    assert len(exc_info.value.errors) == 1
    expected_suggested_config = {'df_as_input': {'inputs': {'df': '<selector>'}}}
    assert exc_info.value.errors[0].message.startswith('Missing required config entry "ops" at the root.')
    assert str(expected_suggested_config) in exc_info.value.errors[0].message
    assert 'yup' not in called

def test_dataframe_csv_missing_input_collision():
    if False:
        for i in range(10):
            print('nop')
    called = {}

    @op(out=Out(DataFrame))
    def df_as_output(_context):
        if False:
            i = 10
            return i + 15
        return pd.DataFrame()

    @op(ins={'df': In(DataFrame)})
    def df_as_input(_context, df):
        if False:
            print('Hello World!')
        called['yup'] = True

    @graph
    def overlapping():
        if False:
            return 10
        return df_as_input(df_as_output())
    with pytest.raises(DagsterInvalidConfigError) as exc_info:
        overlapping.execute_in_process(run_config={'ops': {'df_as_input': {'inputs': {'df': {'csv': {'path': file_relative_path(__file__, 'num.csv')}}}}}})
    assert 'Error 1: Received unexpected config entry "inputs" at path root:ops:df_as_input.' in str(exc_info.value)
    assert 'yup' not in called

def test_dataframe_parquet_from_inputs():
    if False:
        return 10
    check_parquet_support()
    called = {}

    @op(ins={'df': In(DataFrame)})
    def df_as_config(_context, df):
        if False:
            for i in range(10):
                print('nop')
        assert df.to_dict('list') == {'num1': [1, 3], 'num2': [2, 4]}
        called['yup'] = True

    @graph
    def test_graph():
        if False:
            return 10
        df_as_config()
    result = test_graph.execute_in_process(run_config={'ops': {'df_as_config': {'inputs': {'df': {'parquet': {'path': file_relative_path(__file__, 'num.parquet')}}}}}})
    assert result.success
    assert called['yup']

def test_dataframe_table_from_inputs():
    if False:
        for i in range(10):
            print('nop')
    called = {}

    @op(ins={'df': In(DataFrame)})
    def df_as_config(_context, df):
        if False:
            for i in range(10):
                print('nop')
        assert df.to_dict('list') == {'num1': [1, 3], 'num2': [2, 4]}
        called['yup'] = True

    @graph
    def test_graph():
        if False:
            return 10
        df_as_config()
    result = test_graph.execute_in_process(run_config={'ops': {'df_as_config': {'inputs': {'df': {'table': {'path': file_relative_path(__file__, 'num_table.txt')}}}}}})
    assert result.success
    assert called['yup']

def test_dataframe_pickle_from_inputs():
    if False:
        while True:
            i = 10
    pickle_path = file_relative_path(__file__, 'num.pickle')
    df = pd.DataFrame({'num1': [1, 3], 'num2': [2, 4]})
    df.to_pickle(pickle_path)
    called = {}

    @op(ins={'df': In(DataFrame)})
    def df_as_config(_context, df):
        if False:
            print('Hello World!')
        assert df.to_dict('list') == {'num1': [1, 3], 'num2': [2, 4]}
        called['yup'] = True

    @graph
    def test_graph():
        if False:
            for i in range(10):
                print('nop')
        df_as_config()
    result = test_graph.execute_in_process(run_config={'ops': {'df_as_config': {'inputs': {'df': {'pickle': {'path': pickle_path}}}}}})
    assert result.success
    assert called['yup']
    os.remove(pickle_path)