import os
import pandas as pd
from dagster import AssetKey, ConfigurableIOManager
from pandas import DataFrame

class LocalFileSystemIOManager(ConfigurableIOManager):
    """Translates between Pandas DataFrames and CSVs on the local filesystem."""

    def _get_fs_path(self, asset_key: AssetKey) -> str:
        if False:
            i = 10
            return i + 15
        rpath = os.path.join(*asset_key.path) + '.csv'
        return os.path.abspath(rpath)

    def handle_output(self, context, obj: DataFrame):
        if False:
            return 10
        'This saves the dataframe as a CSV.'
        fpath = self._get_fs_path(context.asset_key)
        obj.to_csv(fpath)

    def load_input(self, context):
        if False:
            return 10
        'This reads a dataframe from a CSV.'
        fpath = self._get_fs_path(context.asset_key)
        return pd.read_csv(fpath)