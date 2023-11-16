from __future__ import annotations
import gzip
import io
import pathlib
import tarfile
from typing import TYPE_CHECKING, Any, Callable
import uuid
import zipfile
from pandas.compat import get_bz2_file, get_lzma_file
from pandas.compat._optional import import_optional_dependency
import pandas as pd
from pandas._testing.contexts import ensure_clean
if TYPE_CHECKING:
    from pandas._typing import FilePath, ReadPickleBuffer
    from pandas import DataFrame, Series

def round_trip_pickle(obj: Any, path: FilePath | ReadPickleBuffer | None=None) -> DataFrame | Series:
    if False:
        for i in range(10):
            print('nop')
    '\n    Pickle an object and then read it again.\n\n    Parameters\n    ----------\n    obj : any object\n        The object to pickle and then re-read.\n    path : str, path object or file-like object, default None\n        The path where the pickled object is written and then read.\n\n    Returns\n    -------\n    pandas object\n        The original object that was pickled and then re-read.\n    '
    _path = path
    if _path is None:
        _path = f'__{uuid.uuid4()}__.pickle'
    with ensure_clean(_path) as temp_path:
        pd.to_pickle(obj, temp_path)
        return pd.read_pickle(temp_path)

def round_trip_pathlib(writer, reader, path: str | None=None):
    if False:
        i = 10
        return i + 15
    '\n    Write an object to file specified by a pathlib.Path and read it back\n\n    Parameters\n    ----------\n    writer : callable bound to pandas object\n        IO writing function (e.g. DataFrame.to_csv )\n    reader : callable\n        IO reading function (e.g. pd.read_csv )\n    path : str, default None\n        The path where the object is written and then read.\n\n    Returns\n    -------\n    pandas object\n        The original object that was serialized and then re-read.\n    '
    Path = pathlib.Path
    if path is None:
        path = '___pathlib___'
    with ensure_clean(path) as path:
        writer(Path(path))
        obj = reader(Path(path))
    return obj

def round_trip_localpath(writer, reader, path: str | None=None):
    if False:
        i = 10
        return i + 15
    '\n    Write an object to file specified by a py.path LocalPath and read it back.\n\n    Parameters\n    ----------\n    writer : callable bound to pandas object\n        IO writing function (e.g. DataFrame.to_csv )\n    reader : callable\n        IO reading function (e.g. pd.read_csv )\n    path : str, default None\n        The path where the object is written and then read.\n\n    Returns\n    -------\n    pandas object\n        The original object that was serialized and then re-read.\n    '
    import pytest
    LocalPath = pytest.importorskip('py.path').local
    if path is None:
        path = '___localpath___'
    with ensure_clean(path) as path:
        writer(LocalPath(path))
        obj = reader(LocalPath(path))
    return obj

def write_to_compressed(compression, path, data, dest: str='test') -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Write data to a compressed file.\n\n    Parameters\n    ----------\n    compression : {\'gzip\', \'bz2\', \'zip\', \'xz\', \'zstd\'}\n        The compression type to use.\n    path : str\n        The file path to write the data.\n    data : str\n        The data to write.\n    dest : str, default "test"\n        The destination file (for ZIP only)\n\n    Raises\n    ------\n    ValueError : An invalid compression value was passed in.\n    '
    args: tuple[Any, ...] = (data,)
    mode = 'wb'
    method = 'write'
    compress_method: Callable
    if compression == 'zip':
        compress_method = zipfile.ZipFile
        mode = 'w'
        args = (dest, data)
        method = 'writestr'
    elif compression == 'tar':
        compress_method = tarfile.TarFile
        mode = 'w'
        file = tarfile.TarInfo(name=dest)
        bytes = io.BytesIO(data)
        file.size = len(data)
        args = (file, bytes)
        method = 'addfile'
    elif compression == 'gzip':
        compress_method = gzip.GzipFile
    elif compression == 'bz2':
        compress_method = get_bz2_file()
    elif compression == 'zstd':
        compress_method = import_optional_dependency('zstandard').open
    elif compression == 'xz':
        compress_method = get_lzma_file()
    else:
        raise ValueError(f'Unrecognized compression type: {compression}')
    with compress_method(path, mode=mode) as f:
        getattr(f, method)(*args)