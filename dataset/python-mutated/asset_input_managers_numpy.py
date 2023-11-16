import os
import numpy as np
import pandas as pd
from dagster import AssetIn, ConfigurableIOManager, Definitions, InputContext, IOManager, OutputContext, asset, io_manager
from .asset_input_managers import load_numpy_array, load_pandas_dataframe, store_pandas_dataframe

class PandasAssetIOManager(ConfigurableIOManager):

    def handle_output(self, context: OutputContext, obj):
        if False:
            print('Hello World!')
        file_path = self._get_path(context)
        store_pandas_dataframe(name=file_path, table=obj)

    def _get_path(self, context):
        if False:
            return 10
        return os.path.join('storage', f'{context.asset_key.path[-1]}.csv')

    def load_input(self, context: InputContext) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        file_path = self._get_path(context)
        return load_pandas_dataframe(name=file_path)

class NumpyAssetIOManager(PandasAssetIOManager):

    def load_input(self, context: InputContext) -> np.ndarray:
        if False:
            return 10
        file_path = self._get_path(context)
        return load_numpy_array(name=file_path)

@asset(io_manager_key='pandas_manager')
def upstream_asset() -> pd.DataFrame:
    if False:
        return 10
    return pd.DataFrame([1, 2, 3])

@asset(ins={'upstream': AssetIn(key_prefix='public', input_manager_key='numpy_manager')})
def downstream_asset(upstream: np.ndarray) -> tuple:
    if False:
        i = 10
        return i + 15
    return upstream.shape
defs = Definitions(assets=[upstream_asset, downstream_asset], resources={'pandas_manager': PandasAssetIOManager(), 'numpy_manager': NumpyAssetIOManager()})