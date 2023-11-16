from enum import Enum
from typing import Dict, Union, List, TYPE_CHECKING
import warnings
import numpy as np
from ray.air.data_batch_type import DataBatchType
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.util.annotations import Deprecated, DeveloperAPI
if TYPE_CHECKING:
    import pandas as pd
try:
    import pyarrow
except ImportError:
    pyarrow = None
_pandas = None

def _lazy_import_pandas():
    if False:
        for i in range(10):
            print('nop')
    global _pandas
    if _pandas is None:
        import pandas
        _pandas = pandas
    return _pandas

@DeveloperAPI
class BatchFormat(str, Enum):
    PANDAS = 'pandas'
    ARROW = 'arrow'
    NUMPY = 'numpy'

@DeveloperAPI
class BlockFormat(str, Enum):
    """Internal Dataset block format enum."""
    PANDAS = 'pandas'
    ARROW = 'arrow'
    SIMPLE = 'simple'

def _convert_batch_type_to_pandas(data: DataBatchType, cast_tensor_columns: bool=False) -> 'pd.DataFrame':
    if False:
        while True:
            i = 10
    'Convert the provided data to a Pandas DataFrame.\n\n    Args:\n        data: Data of type DataBatchType\n        cast_tensor_columns: Whether tensor columns should be cast to NumPy ndarrays.\n\n    Returns:\n        A pandas Dataframe representation of the input data.\n\n    '
    pd = _lazy_import_pandas()
    if isinstance(data, np.ndarray):
        data = pd.DataFrame({TENSOR_COLUMN_NAME: _ndarray_to_column(data)})
    elif isinstance(data, dict):
        tensor_dict = {}
        for (col_name, col) in data.items():
            if not isinstance(col, np.ndarray):
                raise ValueError(f'All values in the provided dict must be of type np.ndarray. Found type {type(col)} for key {col_name} instead.')
            tensor_dict[col_name] = _ndarray_to_column(col)
        data = pd.DataFrame(tensor_dict)
    elif pyarrow is not None and isinstance(data, pyarrow.Table):
        data = data.to_pandas()
    elif not isinstance(data, pd.DataFrame):
        raise ValueError(f'Received data of type: {type(data)}, but expected it to be one of {DataBatchType}')
    if cast_tensor_columns:
        data = _cast_tensor_columns_to_ndarrays(data)
    return data

def _convert_pandas_to_batch_type(data: 'pd.DataFrame', type: BatchFormat, cast_tensor_columns: bool=False) -> DataBatchType:
    if False:
        while True:
            i = 10
    'Convert the provided Pandas dataframe to the provided ``type``.\n\n    Args:\n        data: A Pandas DataFrame\n        type: The specific ``BatchFormat`` to convert to.\n        cast_tensor_columns: Whether tensor columns should be cast to our tensor\n            extension type.\n\n    Returns:\n        The input data represented with the provided type.\n    '
    if cast_tensor_columns:
        data = _cast_ndarray_columns_to_tensor_extension(data)
    if type == BatchFormat.PANDAS:
        return data
    elif type == BatchFormat.NUMPY:
        if len(data.columns) == 1:
            return data.iloc[:, 0].to_numpy()
        else:
            output_dict = {}
            for column in data:
                output_dict[column] = data[column].to_numpy()
            return output_dict
    elif type == BatchFormat.ARROW:
        if not pyarrow:
            raise ValueError('Attempted to convert data to Pyarrow Table but Pyarrow is not installed. Please do `pip install pyarrow` to install Pyarrow.')
        return pyarrow.Table.from_pandas(data)
    else:
        raise ValueError(f'Received type {type}, but expected it to be one of {DataBatchType}')

@Deprecated
def convert_batch_type_to_pandas(data: DataBatchType, cast_tensor_columns: bool=False):
    if False:
        return 10
    'Convert the provided data to a Pandas DataFrame.\n\n    This API is deprecated from Ray 2.4.\n\n    Args:\n        data: Data of type DataBatchType\n        cast_tensor_columns: Whether tensor columns should be cast to NumPy ndarrays.\n\n    Returns:\n        A pandas Dataframe representation of the input data.\n\n    '
    warnings.warn('`convert_batch_type_to_pandas` is deprecated as a developer API starting from Ray 2.4. All batch format conversions should be done manually instead of relying on this API.', PendingDeprecationWarning)
    return _convert_batch_type_to_pandas(data=data, cast_tensor_columns=cast_tensor_columns)

@Deprecated
def convert_pandas_to_batch_type(data: 'pd.DataFrame', type: BatchFormat, cast_tensor_columns: bool=False):
    if False:
        for i in range(10):
            print('nop')
    'Convert the provided Pandas dataframe to the provided ``type``.\n\n    Args:\n        data: A Pandas DataFrame\n        type: The specific ``BatchFormat`` to convert to.\n        cast_tensor_columns: Whether tensor columns should be cast to our tensor\n            extension type.\n\n    Returns:\n        The input data represented with the provided type.\n    '
    warnings.warn('`convert_pandas_to_batch_type` is deprecated as a developer API starting from Ray 2.4. All batch format conversions should be done manually instead of relying on this API.', PendingDeprecationWarning)
    return _convert_pandas_to_batch_type(data=data, type=type, cast_tensor_columns=cast_tensor_columns)

def _convert_batch_type_to_numpy(data: DataBatchType) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    if False:
        print('Hello World!')
    'Convert the provided data to a NumPy ndarray or dict of ndarrays.\n\n    Args:\n        data: Data of type DataBatchType\n\n    Returns:\n        A numpy representation of the input data.\n    '
    pd = _lazy_import_pandas()
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, dict):
        for (col_name, col) in data.items():
            if not isinstance(col, np.ndarray):
                raise ValueError(f'All values in the provided dict must be of type np.ndarray. Found type {type(col)} for key {col_name} instead.')
        return data
    elif pyarrow is not None and isinstance(data, pyarrow.Table):
        from ray.air.util.tensor_extensions.arrow import ArrowTensorType
        from ray.air.util.transform_pyarrow import _is_column_extension_type, _concatenate_extension_column
        if data.column_names == [TENSOR_COLUMN_NAME] and isinstance(data.schema.types[0], ArrowTensorType):
            return _concatenate_extension_column(data[TENSOR_COLUMN_NAME]).to_numpy(zero_copy_only=False)
        else:
            output_dict = {}
            for col_name in data.column_names:
                col = data[col_name]
                if col.num_chunks == 0:
                    col = pyarrow.array([], type=col.type)
                elif _is_column_extension_type(col):
                    col = _concatenate_extension_column(col)
                else:
                    col = col.combine_chunks()
                output_dict[col_name] = col.to_numpy(zero_copy_only=False)
            return output_dict
    elif isinstance(data, pd.DataFrame):
        return _convert_pandas_to_batch_type(data, BatchFormat.NUMPY)
    else:
        raise ValueError(f'Received data of type: {type(data)}, but expected it to be one of {DataBatchType}')

def _ndarray_to_column(arr: np.ndarray) -> Union['pd.Series', List[np.ndarray]]:
    if False:
        print('Hello World!')
    'Convert a NumPy ndarray into an appropriate column format for insertion into a\n    pandas DataFrame.\n\n    If conversion to a pandas Series fails (e.g. if the ndarray is multi-dimensional),\n    fall back to a list of NumPy ndarrays.\n    '
    pd = _lazy_import_pandas()
    try:
        return pd.Series(arr)
    except ValueError:
        return list(arr)

def _unwrap_ndarray_object_type_if_needed(arr: np.ndarray) -> np.ndarray:
    if False:
        print('Hello World!')
    'Unwrap an object-dtyped NumPy ndarray containing ndarray pointers into a single\n    contiguous ndarray, if needed/possible.\n    '
    if arr.dtype.type is np.object_:
        try:
            arr = np.array([np.asarray(v) for v in arr])
        except Exception:
            pass
    return arr

def _cast_ndarray_columns_to_tensor_extension(df: 'pd.DataFrame') -> 'pd.DataFrame':
    if False:
        for i in range(10):
            print('nop')
    '\n    Cast all NumPy ndarray columns in df to our tensor extension type, TensorArray.\n    '
    pd = _lazy_import_pandas()
    try:
        SettingWithCopyWarning = pd.core.common.SettingWithCopyWarning
    except AttributeError:
        SettingWithCopyWarning = pd.errors.SettingWithCopyWarning
    from ray.air.util.tensor_extensions.pandas import TensorArray, column_needs_tensor_extension
    for (col_name, col) in df.items():
        if column_needs_tensor_extension(col):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=FutureWarning)
                    warnings.simplefilter('ignore', category=SettingWithCopyWarning)
                    df.loc[:, col_name] = TensorArray(col)
            except Exception as e:
                raise ValueError(f'Tried to cast column {col_name} to the TensorArray tensor extension type but the conversion failed. To disable automatic casting to this tensor extension, set ctx = DataContext.get_current(); ctx.enable_tensor_extension_casting = False.') from e
    return df

def _cast_tensor_columns_to_ndarrays(df: 'pd.DataFrame') -> 'pd.DataFrame':
    if False:
        print('Hello World!')
    'Cast all tensor extension columns in df to NumPy ndarrays.'
    pd = _lazy_import_pandas()
    try:
        SettingWithCopyWarning = pd.core.common.SettingWithCopyWarning
    except AttributeError:
        SettingWithCopyWarning = pd.errors.SettingWithCopyWarning
    from ray.air.util.tensor_extensions.pandas import TensorDtype
    for (col_name, col) in df.items():
        if isinstance(col.dtype, TensorDtype):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=FutureWarning)
                warnings.simplefilter('ignore', category=SettingWithCopyWarning)
                df.loc[:, col_name] = pd.Series(list(col.to_numpy()))
    return df