from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import DASK_MODULE_NAME
from ludwig.data.dataframe.base import DataFrameEngine
from ludwig.utils.types import DataFrame

@DeveloperAPI
def is_dask_lib(df_lib) -> bool:
    if False:
        i = 10
        return i + 15
    'Returns whether the dataframe library is dask.'
    return df_lib.__name__ == DASK_MODULE_NAME

@DeveloperAPI
def is_dask_backend(backend: Optional['Backend']) -> bool:
    if False:
        i = 10
        return i + 15
    "Returns whether the backend's dataframe is dask."
    return backend is not None and is_dask_lib(backend.df_engine.df_lib)

@DeveloperAPI
def is_dask_series_or_df(df: DataFrame, backend: Optional['Backend']) -> bool:
    if False:
        return 10
    if is_dask_backend(backend):
        import dask.dataframe as dd
        return isinstance(df, dd.Series) or isinstance(df, dd.DataFrame)
    return False

@DeveloperAPI
def flatten_df(df: DataFrame, df_engine: DataFrameEngine) -> Tuple[DataFrame, Dict[str, Tuple]]:
    if False:
        print('Hello World!')
    'Returns a flattened dataframe with a dictionary of the original shapes, keyed by dataframe columns.'
    column_shapes = {}
    for c in df.columns:
        df = df_engine.persist(df)
        shape = df_engine.compute(df_engine.map_objects(df[c], lambda x: np.array(x).shape).max())
        if len(shape) > 1:
            column_shapes[c] = shape
            df[c] = df_engine.map_objects(df[c], lambda x: np.array(x).reshape(-1))
    return (df, column_shapes)

@DeveloperAPI
def unflatten_df(df: DataFrame, column_shapes: Dict[str, Tuple], df_engine: DataFrameEngine) -> DataFrame:
    if False:
        for i in range(10):
            print('nop')
    'Returns an unflattened dataframe, the reverse of flatten_df.'
    for c in df.columns:
        shape = column_shapes.get(c)
        if shape:
            df[c] = df_engine.map_objects(df[c], lambda x: np.array(x).reshape(shape))
    return df

@DeveloperAPI
def to_numpy_dataset(df: DataFrame, backend: Optional['Backend']=None) -> Dict[str, np.ndarray]:
    if False:
        i = 10
        return i + 15
    'Returns a dictionary of numpy arrays, keyed by the columns of the given dataframe.'
    dataset = {}
    for col in df.columns:
        res = df[col]
        if backend and is_dask_backend(backend):
            res = res.compute()
        if len(df.index) != 0:
            dataset[col] = np.stack(res.to_numpy())
        else:
            dataset[col] = res.to_list()
    return dataset

@DeveloperAPI
def from_numpy_dataset(dataset) -> pd.DataFrame:
    if False:
        return 10
    'Returns a pandas dataframe from the dataset.'
    col_mapping = {}
    for (k, v) in dataset.items():
        if len(v.shape) > 1:
            (*vals,) = v
        else:
            vals = v
        col_mapping[k] = vals
    return pd.DataFrame.from_dict(col_mapping)

@DeveloperAPI
def set_index_name(pd_df: pd.DataFrame, name: str) -> pd.DataFrame:
    if False:
        print('Hello World!')
    pd_df.index.name = name
    return pd_df

@DeveloperAPI
def to_batches(df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
    if False:
        for i in range(10):
            print('nop')
    return [df[i:i + batch_size].copy() for i in range(0, df.shape[0], batch_size)]

@DeveloperAPI
def from_batches(batches: List[pd.DataFrame]) -> pd.DataFrame:
    if False:
        for i in range(10):
            print('nop')
    return pd.concat(batches)

@DeveloperAPI
def to_scalar_df(df: pd.DataFrame) -> pd.DataFrame:
    if False:
        return 10
    "Converts all columns in a pd.DataFrame to be scalar types.\n\n    For object columns of lists, each element of the list is expanded into its own column named {column}_{index}. We\n    assume all object columns are lists of the same length (i.e., tensor format output from preprocessing). It's also\n    important that the relative order of the columns is preserved, to maintain consistency with other conversions like\n    the one for Hummingbird.\n    "
    scalar_df = df
    column_ordering = []
    for (c, s) in df.items():
        if s.dtype == 'object':
            s_list = s.to_list()
            try:
                ncols = s_list[0].shape[0]
                split_cols = [f'{c}_{k}' for k in range(ncols)]
                sdf = pd.DataFrame(s_list, columns=split_cols)
                scalar_df = pd.concat([scalar_df, sdf], axis=1)
                column_ordering += split_cols
            except AttributeError as e:
                raise ValueError(f'Expected series of lists, but found {s_list[0]}') from e
        else:
            column_ordering.append(c)
    return scalar_df[column_ordering]