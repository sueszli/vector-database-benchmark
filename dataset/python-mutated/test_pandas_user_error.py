import dagster_pandas as dagster_pd
import pandas as pd
import pytest
from dagster import DagsterTypeCheckDidNotPass, In, Out, graph, op

def test_wrong_output_value():
    if False:
        for i in range(10):
            print('nop')

    @op(ins={'num_csv': In(dagster_pd.DataFrame)}, out=Out(dagster_pd.DataFrame))
    def wrong_output(num_csv):
        if False:
            return 10
        return 'not a dataframe'

    @op
    def pass_df():
        if False:
            print('Hello World!')
        return pd.DataFrame()

    @graph
    def output_fails():
        if False:
            while True:
                i = 10
        return wrong_output(pass_df())
    with pytest.raises(DagsterTypeCheckDidNotPass):
        output_fails.execute_in_process()

def test_wrong_input_value():
    if False:
        return 10

    @op(ins={'foo': In(dagster_pd.DataFrame)})
    def wrong_input(foo):
        if False:
            while True:
                i = 10
        return foo

    @op
    def pass_str():
        if False:
            print('Hello World!')
        'Not a dataframe.'

    @graph
    def input_fails():
        if False:
            while True:
                i = 10
        wrong_input(pass_str())
    with pytest.raises(DagsterTypeCheckDidNotPass):
        input_fails.execute_in_process()