import os
import numpy as np
import pandas as pd
from dagster import Field, In, IOManager, Noneable, graph, io_manager, op

class PandasCsvIOManager(IOManager):

    def __init__(self, base_dir=None):
        if False:
            while True:
                i = 10
        self.base_dir = os.getenv('DAGSTER_HOME') if base_dir is None else base_dir

    def _get_path(self, output_context):
        if False:
            i = 10
            return i + 15
        return os.path.join(self.base_dir, 'storage', f'{output_context.step_key}_{output_context.name}.csv')

    def handle_output(self, context, obj: pd.DataFrame):
        if False:
            i = 10
            return i + 15
        file_path = self._get_path(context)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if obj is not None:
            obj.to_csv(file_path, index=False)

    def load_input(self, context) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        return pd.read_csv(self._get_path(context.upstream_output))

@io_manager(config_schema={'base_dir': Field(Noneable(str), default_value=None, is_required=False)})
def pandas_io_manager(init_context):
    if False:
        print('Hello World!')
    return PandasCsvIOManager(base_dir=init_context.resource_config['base_dir'])

class NumpyCsvIOManager(PandasCsvIOManager):

    def load_input(self, context) -> np.ndarray:
        if False:
            return 10
        if context.upstream_output:
            file_path = self._get_path(context.upstream_output)
            df = np.genfromtxt(file_path, delimiter=',', dtype=None)
            return df
        else:
            multiplier = context.config['multiplier']
            df = pd.DataFrame({'ints': [10 * multiplier, 20 * multiplier, 30 * multiplier, 40 * multiplier], 'floats': [10.0 * multiplier, 20.0 * multiplier, 30.0 * multiplier, 40.0 * multiplier], 'strings': ['ten', 'twenty', 'thirty', 'forty']})
            return df.to_numpy()

@io_manager(config_schema={'base_dir': Field(Noneable(str), default_value=None, is_required=False)}, input_config_schema={'multiplier': Field(int, is_required=False, default_value=1)})
def numpy_io_manager(init_context):
    if False:
        print('Hello World!')
    return NumpyCsvIOManager(base_dir=init_context.resource_config['base_dir'])

@op
def make_a_df():
    if False:
        print('Hello World!')
    df = pd.DataFrame({'ints': [1, 2, 3, 4], 'floats': [1.0, 2.0, 3.0, 4.0], 'strings': ['one', 'two', 'three', 'four']})
    return df

@op
def avg_ints(context, df):
    if False:
        return 10
    avg = df['ints'].mean().item()
    context.log.info(f'Dataframe with type {type(df)} has average of the ints is {avg}')

@op(ins={'df': In(input_manager_key='numpy_csv_mgr')})
def median_floats(context, df):
    if False:
        return 10
    med = df['floats'].median().item()
    context.log.info(f'Dataframe with type {type(df)} has median of the floats is {med}')

@op(ins={'df': In(input_manager_key='numpy_csv_mgr')})
def count_rows(context, df: np.ndarray):
    if False:
        for i in range(10):
            print('nop')
    num_rows = df.shape[0]
    context.log.info(f'Dataframe with type {type(df)} has {num_rows} rows')

@graph
def df_stats():
    if False:
        print('Hello World!')
    df = make_a_df()
    avg_ints(df)
    median_floats()
    count_rows(df)
df_stats_job = df_stats.to_job(name='df_stats_job', resource_defs={'io_manager': pandas_io_manager, 'numpy_csv_mgr': numpy_io_manager})