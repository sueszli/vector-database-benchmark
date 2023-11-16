""" feather-format compat """
from __future__ import annotations
from typing import TYPE_CHECKING, Any
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.util._validators import check_dtype_backend
import pandas as pd
from pandas.core.api import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io._util import arrow_string_types_mapper
from pandas.io.common import get_handle
if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from pandas._typing import DtypeBackend, FilePath, ReadBuffer, StorageOptions, WriteBuffer

@doc(storage_options=_shared_docs['storage_options'])
def to_feather(df: DataFrame, path: FilePath | WriteBuffer[bytes], storage_options: StorageOptions | None=None, **kwargs: Any) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Write a DataFrame to the binary Feather format.\n\n    Parameters\n    ----------\n    df : DataFrame\n    path : str, path object, or file-like object\n    {storage_options}\n\n        .. versionadded:: 1.2.0\n\n    **kwargs :\n        Additional keywords passed to `pyarrow.feather.write_feather`.\n\n    '
    import_optional_dependency('pyarrow')
    from pyarrow import feather
    if not isinstance(df, DataFrame):
        raise ValueError('feather only support IO with DataFrames')
    with get_handle(path, 'wb', storage_options=storage_options, is_text=False) as handles:
        feather.write_feather(df, handles.handle, **kwargs)

@doc(storage_options=_shared_docs['storage_options'])
def read_feather(path: FilePath | ReadBuffer[bytes], columns: Sequence[Hashable] | None=None, use_threads: bool=True, storage_options: StorageOptions | None=None, dtype_backend: DtypeBackend | lib.NoDefault=lib.no_default) -> DataFrame:
    if False:
        return 10
    '\n    Load a feather-format object from the file path.\n\n    Parameters\n    ----------\n    path : str, path object, or file-like object\n        String, path object (implementing ``os.PathLike[str]``), or file-like\n        object implementing a binary ``read()`` function. The string could be a URL.\n        Valid URL schemes include http, ftp, s3, and file. For file URLs, a host is\n        expected. A local file could be: ``file://localhost/path/to/table.feather``.\n    columns : sequence, default None\n        If not provided, all columns are read.\n    use_threads : bool, default True\n        Whether to parallelize reading using multiple threads.\n    {storage_options}\n\n        .. versionadded:: 1.2.0\n\n    dtype_backend : {{\'numpy_nullable\', \'pyarrow\'}}, default \'numpy_nullable\'\n        Back-end data type applied to the resultant :class:`DataFrame`\n        (still experimental). Behaviour is as follows:\n\n        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`\n          (default).\n        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`\n          DataFrame.\n\n        .. versionadded:: 2.0\n\n    Returns\n    -------\n    type of object stored in file\n\n    Examples\n    --------\n    >>> df = pd.read_feather("path/to/file.feather")  # doctest: +SKIP\n    '
    import_optional_dependency('pyarrow')
    from pyarrow import feather
    import pandas.core.arrays.arrow.extension_types
    check_dtype_backend(dtype_backend)
    with get_handle(path, 'rb', storage_options=storage_options, is_text=False) as handles:
        if dtype_backend is lib.no_default and (not using_pyarrow_string_dtype()):
            return feather.read_feather(handles.handle, columns=columns, use_threads=bool(use_threads))
        pa_table = feather.read_table(handles.handle, columns=columns, use_threads=bool(use_threads))
        if dtype_backend == 'numpy_nullable':
            from pandas.io._util import _arrow_dtype_mapping
            return pa_table.to_pandas(types_mapper=_arrow_dtype_mapping().get)
        elif dtype_backend == 'pyarrow':
            return pa_table.to_pandas(types_mapper=pd.ArrowDtype)
        elif using_pyarrow_string_dtype():
            return pa_table.to_pandas(types_mapper=arrow_string_types_mapper())
        else:
            raise NotImplementedError