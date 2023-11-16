"""
Module houses `FileDispatcher` class.

`FileDispatcher` can be used as abstract base class for dispatchers of specific file formats or
for direct files processing.
"""
import os
import fsspec
import numpy as np
from pandas.io.common import is_fsspec_url, is_url
from modin.config import AsyncReadMode
from modin.logging import ClassLogger
from modin.utils import ModinAssumptionError
NOT_IMPLEMENTED_MESSAGE = 'Implement in children classes!'

class OpenFile:
    """
    OpenFile is a context manager for an input file.

    OpenFile uses fsspec to open files on __enter__. On __exit__, it closes the
    fsspec file. This class exists to encapsulate the special behavior in
    __enter__ around anon=False and anon=True for s3 buckets.

    Parameters
    ----------
    file_path : str
        String that represents the path to the file (paths to S3 buckets
        are also acceptable).
    mode : str, default: "rb"
        String, which defines which mode file should be open.
    compression : str, default: "infer"
        File compression name.
    **kwargs : dict
        Keywords arguments to be passed into ``fsspec.open`` function.

    Attributes
    ----------
    file_path : str
        String that represents the path to the file
    mode : str
        String that defines which mode the file should be opened in.
    compression : str
        File compression name.
    file : fsspec.core.OpenFile
        The opened file.
    kwargs : dict
        Keywords arguments to be passed into ``fsspec.open`` function.
    """

    def __init__(self, file_path, mode='rb', compression='infer', **kwargs):
        if False:
            print('Hello World!')
        self.file_path = file_path
        self.mode = mode
        self.compression = compression
        self.kwargs = kwargs

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Open the file with fsspec and return the opened file.\n\n        Returns\n        -------\n        fsspec.core.OpenFile\n            The opened file.\n        '
        try:
            from botocore.exceptions import NoCredentialsError
            credential_error_type = (NoCredentialsError, PermissionError)
        except ModuleNotFoundError:
            credential_error_type = (PermissionError,)
        args = (self.file_path, self.mode, self.compression)
        self.file = fsspec.open(*args, **self.kwargs)
        try:
            return self.file.open()
        except credential_error_type:
            self.kwargs['anon'] = True
            self.file = fsspec.open(*args, **self.kwargs)
        return self.file.open()

    def __exit__(self, *args):
        if False:
            return 10
        '\n        Close the file.\n\n        Parameters\n        ----------\n        *args : any type\n            Variable positional arguments, all unused.\n        '
        self.file.close()

class FileDispatcher(ClassLogger):
    """
    Class handles util functions for reading data from different kinds of files.

    Notes
    -----
    `_read`, `deploy`, `parse` and `materialize` are abstract methods and should be
    implemented in the child classes (functions signatures can differ between child
    classes).
    """
    BUFFER_UNSUPPORTED_MSG = 'Reading from buffers or other non-path-like objects is not supported'
    frame_cls = None
    frame_partition_cls = None
    query_compiler_cls = None

    @classmethod
    def read(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Read data according passed `args` and `kwargs`.\n\n        Parameters\n        ----------\n        *args : iterable\n            Positional arguments to be passed into `_read` function.\n        **kwargs : dict\n            Keywords arguments to be passed into `_read` function.\n\n        Returns\n        -------\n        query_compiler : BaseQueryCompiler\n            Query compiler with imported data for further processing.\n\n        Notes\n        -----\n        `read` is high-level function that calls specific for defined storage format, engine and\n        dispatcher class `_read` function with passed parameters and performs some\n        postprocessing work on the resulting query_compiler object.\n        '
        try:
            query_compiler = cls._read(*args, **kwargs)
        except ModinAssumptionError as err:
            param_name = 'path_or_buf' if 'path_or_buf' in kwargs else 'fname'
            fname = kwargs.pop(param_name)
            return cls.single_worker_read(fname, *args, reason=str(err), **kwargs)
        if not AsyncReadMode.get() and hasattr(query_compiler, 'dtypes'):
            _ = query_compiler.dtypes
        return query_compiler

    @classmethod
    def _read(cls, *args, **kwargs):
        if False:
            return 10
        '\n        Perform reading of the data from file.\n\n        Should be implemented in the child class.\n\n        Parameters\n        ----------\n        *args : iterable\n            Positional arguments of the function.\n        **kwargs : dict\n            Keywords arguments of the function.\n        '
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    def get_path(cls, file_path):
        if False:
            i = 10
            return i + 15
        "\n        Process `file_path` in accordance to it's type.\n\n        Parameters\n        ----------\n        file_path : str, os.PathLike[str] object or file-like object\n            The file, or a path to the file. Paths to S3 buckets are also\n            acceptable.\n\n        Returns\n        -------\n        str\n            Updated or verified `file_path` parameter.\n\n        Notes\n        -----\n        if `file_path` is a URL, parameter will be returned as is, otherwise\n        absolute path will be returned.\n        "
        if is_fsspec_url(file_path) or is_url(file_path):
            return file_path
        else:
            return os.path.abspath(file_path)

    @classmethod
    def file_size(cls, f):
        if False:
            while True:
                i = 10
        '\n        Get the size of file associated with file handle `f`.\n\n        Parameters\n        ----------\n        f : file-like object\n            File-like object, that should be used to get file size.\n\n        Returns\n        -------\n        int\n            File size in bytes.\n        '
        cur_pos = f.tell()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(cur_pos, os.SEEK_SET)
        return size

    @classmethod
    def file_exists(cls, file_path, storage_options=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if `file_path` exists.\n\n        Parameters\n        ----------\n        file_path : str\n            String that represents the path to the file (paths to S3 buckets\n            are also acceptable).\n        storage_options : dict, optional\n            Keyword from `read_*` functions.\n\n        Returns\n        -------\n        bool\n            Whether file exists or not.\n        '
        if not is_fsspec_url(file_path) and (not is_url(file_path)):
            return os.path.exists(file_path)
        try:
            from botocore.exceptions import ConnectTimeoutError, EndpointConnectionError, NoCredentialsError
            credential_error_type = (NoCredentialsError, PermissionError, EndpointConnectionError, ConnectTimeoutError)
        except ModuleNotFoundError:
            credential_error_type = (PermissionError,)
        if storage_options is not None:
            new_storage_options = dict(storage_options)
            new_storage_options.pop('anon', None)
        else:
            new_storage_options = {}
        (fs, _) = fsspec.core.url_to_fs(file_path, **new_storage_options)
        exists = False
        try:
            exists = fs.exists(file_path)
        except credential_error_type:
            (fs, _) = fsspec.core.url_to_fs(file_path, anon=True, **new_storage_options)
            exists = fs.exists(file_path)
        return exists

    @classmethod
    def deploy(cls, func, *args, num_returns=1, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Deploy remote task.\n\n        Should be implemented in the task class (for example in the `RayWrapper`).\n        '
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def parse(self, func, args, num_returns):
        if False:
            i = 10
            return i + 15
        "\n        Parse file's data in the worker process.\n\n        Should be implemented in the parser class (for example in the `PandasCSVParser`).\n        "
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    def materialize(cls, obj_id):
        if False:
            print('Hello World!')
        '\n        Get results from worker.\n\n        Should be implemented in the task class (for example in the `RayWrapper`).\n        '
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    def build_partition(cls, partition_ids, row_lengths, column_widths):
        if False:
            i = 10
            return i + 15
        '\n        Build array with partitions of `cls.frame_partition_cls` class.\n\n        Parameters\n        ----------\n        partition_ids : list\n            Array with references to the partitions data.\n        row_lengths : list\n            Partitions rows lengths.\n        column_widths : list\n            Number of columns in each partition.\n\n        Returns\n        -------\n        np.ndarray\n            array with shape equals to the shape of `partition_ids` and\n            filed with partition objects.\n        '
        return np.array([[cls.frame_partition_cls(partition_ids[i][j], length=row_lengths[i], width=column_widths[j]) for j in range(len(partition_ids[i]))] for i in range(len(partition_ids))])

    @classmethod
    def _file_not_found_msg(cls, filename: str):
        if False:
            i = 10
            return i + 15
        return f"No such file: '{filename}'"