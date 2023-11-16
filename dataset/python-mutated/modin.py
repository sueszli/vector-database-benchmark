import os
import modin.pandas as pd
import numpy as np
from ludwig.data.dataframe.base import DataFrameEngine
from ludwig.globals import PREDICTIONS_SHAPES_FILE_NAME
from ludwig.utils.data_utils import get_pa_schema, load_json, save_json, split_by_slices
from ludwig.utils.dataframe_utils import flatten_df, unflatten_df

class ModinEngine(DataFrameEngine):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def df_like(self, df, proc_cols):
        if False:
            while True:
                i = 10
        return pd.DataFrame(proc_cols)

    def parallelize(self, data):
        if False:
            while True:
                i = 10
        return data

    def persist(self, data):
        if False:
            i = 10
            return i + 15
        return data

    def compute(self, data):
        if False:
            return 10
        return data

    def from_pandas(self, df):
        if False:
            for i in range(10):
                print('nop')
        return pd.DataFrame(df)

    def map_objects(self, series, map_fn, meta=None):
        if False:
            return 10
        return series.map(map_fn)

    def map_batches(self, df, map_fn, enable_tensor_extension_casting=True):
        if False:
            while True:
                i = 10
        return map_fn(df)

    def map_partitions(self, series, map_fn, meta=None):
        if False:
            for i in range(10):
                print('nop')
        return map_fn(series)

    def apply_objects(self, df, apply_fn, meta=None):
        if False:
            print('Hello World!')
        return df.apply(apply_fn, axis=1)

    def reduce_objects(self, series, reduce_fn):
        if False:
            i = 10
            return i + 15
        return reduce_fn(series)

    def split(self, df, probabilities):
        if False:
            i = 10
            return i + 15
        return split_by_slices(df.iloc, len(df), probabilities)

    def remove_empty_partitions(self, df):
        if False:
            i = 10
            return i + 15
        return df

    def to_parquet(self, df, path, index=False):
        if False:
            i = 10
            return i + 15
        schema = get_pa_schema(df)
        df.to_parquet(path, engine='pyarrow', index=index, schema=schema)

    def write_predictions(self, df: pd.DataFrame, path: str):
        if False:
            for i in range(10):
                print('nop')
        (df, column_shapes) = flatten_df(df, self)
        self.to_parquet(df, path)
        save_json(os.path.join(os.path.dirname(path), PREDICTIONS_SHAPES_FILE_NAME), column_shapes)

    def read_predictions(self, path: str) -> pd.DataFrame:
        if False:
            print('Hello World!')
        pred_df = pd.read_parquet(path)
        column_shapes = load_json(os.path.join(os.path.dirname(path), PREDICTIONS_SHAPES_FILE_NAME))
        return unflatten_df(pred_df, column_shapes, self)

    def to_ray_dataset(self, df):
        if False:
            print('Hello World!')
        from ray.data import from_modin
        return from_modin(df)

    def from_ray_dataset(self, dataset) -> pd.DataFrame:
        if False:
            print('Hello World!')
        return dataset.to_modin()

    def reset_index(self, df):
        if False:
            while True:
                i = 10
        return df.reset_index(drop=True)

    @property
    def array_lib(self):
        if False:
            i = 10
            return i + 15
        return np

    @property
    def df_lib(self):
        if False:
            print('Hello World!')
        return pd

    @property
    def partitioned(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def set_parallelism(self, parallelism):
        if False:
            print('Hello World!')
        pass