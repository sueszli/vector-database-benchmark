import glob
import os
from typing import Union
import pandas as pd
from dagster import AssetKey, ConfigurableIOManager, _check as check
from pandas import DataFrame as PandasDF
from pyspark.sql import DataFrame as SparkDF, SparkSession

class LocalFileSystemIOManager(ConfigurableIOManager):

    def _get_fs_path(self, asset_key: AssetKey) -> str:
        if False:
            print('Hello World!')
        return os.path.abspath(os.path.join(*asset_key.path))

    def handle_output(self, context, obj: Union[PandasDF, SparkDF]):
        if False:
            return 10
        'This saves the DataFrame as a CSV using the layout written and expected by Spark/Hadoop.\n\n        E.g. if the given storage maps the asset\'s path to the filesystem path "/a/b/c", a directory\n        will be created with two files inside it:\n\n            /a/b/c/\n                part-00000.csv\n         2       _SUCCESS\n        '
        if isinstance(obj, PandasDF):
            directory = self._get_fs_path(context.asset_key)
            os.makedirs(directory, exist_ok=True)
            open(os.path.join(directory, '_SUCCESS'), 'wb').close()
            csv_path = os.path.join(directory, 'part-00000.csv')
            obj.to_csv(csv_path)
        elif isinstance(obj, SparkDF):
            obj.write.format('csv').options(header='true').save(self._get_fs_path(context.asset_key), mode='overwrite')
        else:
            raise ValueError('Unexpected input type')

    def load_input(self, context) -> Union[PandasDF, SparkDF]:
        if False:
            print('Hello World!')
        'This reads a DataFrame from a CSV using the layout written and expected by Spark/Hadoop.\n\n        E.g. if the given storage maps the asset\'s path to the filesystem path "/a/b/c", and that\n        directory contains:\n\n            /a/b/c/\n                part-00000.csv\n                part-00001.csv\n                _SUCCESS\n\n        then the produced dataframe will contain the concatenated contents of the two CSV files.\n        '
        if context.dagster_type.typing_type == PandasDF:
            fs_path = os.path.abspath(self._get_fs_path(context.asset_key))
            paths = glob.glob(os.path.join(fs_path, '*.csv'))
            check.invariant(len(paths) > 0, f'No csv files found under {fs_path}')
            return pd.concat(map(pd.read_csv, paths))
        elif context.dagster_type.typing_type == SparkDF:
            return SparkSession.builder.getOrCreate().read.format('csv').options(header='true').load(self._get_fs_path(context.asset_key))
        else:
            raise ValueError('Unexpected input type')