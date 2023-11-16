"""``MemoryDataset`` is a data set implementation which handles in-memory data.
"""
from __future__ import annotations
import copy
import warnings
from typing import Any
from kedro import KedroDeprecationWarning
from kedro.io.core import AbstractDataset, DatasetError
_EMPTY = object()
MemoryDataSet: type[MemoryDataset]

class MemoryDataset(AbstractDataset):
    """``MemoryDataset`` loads and saves data from/to an in-memory
    Python object.

    Example:
    ::

        >>> from kedro.io import MemoryDataset
        >>> import pandas as pd
        >>>
        >>> data = pd.DataFrame({'col1': [1, 2], 'col2': [4, 5],
        >>>                      'col3': [5, 6]})
        >>> data_set = MemoryDataset(data=data)
        >>>
        >>> loaded_data = data_set.load()
        >>> assert loaded_data.equals(data)
        >>>
        >>> new_data = pd.DataFrame({'col1': [1, 2], 'col2': [4, 5]})
        >>> data_set.save(new_data)
        >>> reloaded_data = data_set.load()
        >>> assert reloaded_data.equals(new_data)

    """

    def __init__(self, data: Any=_EMPTY, copy_mode: str=None, metadata: dict[str, Any]=None):
        if False:
            i = 10
            return i + 15
        'Creates a new instance of ``MemoryDataset`` pointing to the\n        provided Python object.\n\n        Args:\n            data: Python object containing the data.\n            copy_mode: The copy mode used to copy the data. Possible\n                values are: "deepcopy", "copy" and "assign". If not\n                provided, it is inferred based on the data type.\n            metadata: Any arbitrary metadata.\n                This is ignored by Kedro, but may be consumed by users or external plugins.\n        '
        self._data = _EMPTY
        self._copy_mode = copy_mode
        self.metadata = metadata
        if data is not _EMPTY:
            self._save(data)

    def _load(self) -> Any:
        if False:
            while True:
                i = 10
        if self._data is _EMPTY:
            raise DatasetError('Data for MemoryDataset has not been saved yet.')
        copy_mode = self._copy_mode or _infer_copy_mode(self._data)
        data = _copy_with_mode(self._data, copy_mode=copy_mode)
        return data

    def _save(self, data: Any):
        if False:
            print('Hello World!')
        copy_mode = self._copy_mode or _infer_copy_mode(data)
        self._data = _copy_with_mode(data, copy_mode=copy_mode)

    def _exists(self) -> bool:
        if False:
            while True:
                i = 10
        return self._data is not _EMPTY

    def _release(self) -> None:
        if False:
            return 10
        self._data = _EMPTY

    def _describe(self) -> dict[str, Any]:
        if False:
            return 10
        if self._data is not _EMPTY:
            return {'data': f'<{type(self._data).__name__}>'}
        return {'data': None}

def _infer_copy_mode(data: Any) -> str:
    if False:
        return 10
    'Infers the copy mode to use given the data type.\n\n    Args:\n        data: The data whose type will be used to infer the copy mode.\n\n    Returns:\n        One of "copy", "assign" or "deepcopy" as the copy mode to use.\n    '
    try:
        import pandas as pd
    except ImportError:
        pd = None
    try:
        import numpy as np
    except ImportError:
        np = None
    if pd and isinstance(data, pd.DataFrame) or (np and isinstance(data, np.ndarray)):
        copy_mode = 'copy'
    elif type(data).__name__ == 'DataFrame':
        copy_mode = 'assign'
    else:
        copy_mode = 'deepcopy'
    return copy_mode

def _copy_with_mode(data: Any, copy_mode: str) -> Any:
    if False:
        i = 10
        return i + 15
    'Returns the copied data using the copy mode specified.\n    If no copy mode is provided, then it is inferred based on the type of the data.\n\n    Args:\n        data: The data to copy.\n        copy_mode: The copy mode to use, one of "deepcopy", "copy" and "assign".\n\n    Raises:\n        DatasetError: If copy_mode is specified, but isn\'t valid\n            (i.e: not one of deepcopy, copy, assign)\n\n    Returns:\n        The data copied according to the specified copy mode.\n    '
    if copy_mode == 'deepcopy':
        copied_data = copy.deepcopy(data)
    elif copy_mode == 'copy':
        copied_data = data.copy()
    elif copy_mode == 'assign':
        copied_data = data
    else:
        raise DatasetError(f'Invalid copy mode: {copy_mode}. Possible values are: deepcopy, copy, assign.')
    return copied_data

def __getattr__(name):
    if False:
        while True:
            i = 10
    if name == 'MemoryDataSet':
        alias = MemoryDataset
        warnings.warn(f'{repr(name)} has been renamed to {repr(alias.__name__)}, and the alias will be removed in Kedro 0.19.0', KedroDeprecationWarning, stacklevel=2)
        return alias
    raise AttributeError(f'module {repr(__name__)} has no attribute {repr(name)}')