"""Module houses `ExperimentalCSVGlobDispatcher` class, that is used for reading multiple `.csv` files simultaneously."""
import csv
import glob
import os
import warnings
from contextlib import ExitStack
from typing import List, Tuple
import fsspec
import pandas
import pandas._libs.lib as lib
from pandas.io.common import is_fsspec_url, is_url, stringify_path
from modin.config import NPartitions
from modin.core.io.file_dispatcher import OpenFile
from modin.core.io.text.csv_dispatcher import CSVDispatcher

class ExperimentalCSVGlobDispatcher(CSVDispatcher):
    """Class contains utils for reading multiple `.csv` files simultaneously."""

    @classmethod
    def _read(cls, filepath_or_buffer, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read data from multiple `.csv` files passed with `filepath_or_buffer` simultaneously.\n\n        Parameters\n        ----------\n        filepath_or_buffer : str, path object or file-like object\n            `filepath_or_buffer` parameter of ``read_csv`` function.\n        **kwargs : dict\n            Parameters of ``read_csv`` function.\n\n        Returns\n        -------\n        new_query_compiler : BaseQueryCompiler\n            Query compiler with imported data for further processing.\n        '
        filepath_or_buffer = cls.get_path_or_buffer(stringify_path(filepath_or_buffer))
        if isinstance(filepath_or_buffer, str):
            is_folder = any((filepath_or_buffer.endswith(sep) for sep in (os.sep, os.altsep) if sep))
            if '*' not in filepath_or_buffer and (not is_folder):
                warnings.warn("Shell-style wildcard '*' must be in the filename pattern in order to read multiple " + f"files at once. Did you forget it? Passed filename: '{filepath_or_buffer}'")
            if not cls.file_exists(filepath_or_buffer, kwargs.get('storage_options')):
                return cls.single_worker_read(filepath_or_buffer, reason=cls._file_not_found_msg(filepath_or_buffer), **kwargs)
            filepath_or_buffer = cls.get_path(filepath_or_buffer)
        elif not cls.pathlib_or_pypath(filepath_or_buffer):
            return cls.single_worker_read(filepath_or_buffer, reason=cls.BUFFER_UNSUPPORTED_MSG, **kwargs)
        glob_filepaths = filepath_or_buffer
        filepath_or_buffer = filepath_or_buffer[0]
        compression_type = cls.infer_compression(filepath_or_buffer, kwargs.get('compression'))
        chunksize = kwargs.get('chunksize')
        if chunksize is not None:
            return cls.single_worker_read(filepath_or_buffer, reason='`chunksize` parameter is not supported', **kwargs)
        skiprows = kwargs.get('skiprows')
        if skiprows is not None and (not isinstance(skiprows, int)):
            return cls.single_worker_read(filepath_or_buffer, reason='Non-integer `skiprows` value not supported', **kwargs)
        nrows = kwargs.pop('nrows', None)
        names = kwargs.get('names', lib.no_default)
        index_col = kwargs.get('index_col', None)
        usecols = kwargs.get('usecols', None)
        encoding = kwargs.get('encoding', None)
        if names in [lib.no_default, None]:
            names = pandas.read_csv(filepath_or_buffer, **dict(kwargs, usecols=None, nrows=0, skipfooter=0, index_col=None)).columns
        elif index_col is None and (not usecols):
            empty_pd_df = pandas.read_csv(filepath_or_buffer, nrows=0, encoding=encoding)
            num_cols = len(empty_pd_df.columns)
            if num_cols > len(names):
                index_col = list(range(num_cols - len(names)))
                if len(index_col) == 1:
                    index_col = index_col[0]
                kwargs['index_col'] = index_col
        pd_df_metadata = pandas.read_csv(filepath_or_buffer, **dict(kwargs, nrows=1, skipfooter=0))
        column_names = pd_df_metadata.columns
        skipfooter = kwargs.get('skipfooter', None)
        skiprows = kwargs.pop('skiprows', None)
        usecols_md = cls._validate_usecols_arg(usecols)
        if usecols is not None and usecols_md[1] != 'integer':
            del kwargs['usecols']
            all_cols = pandas.read_csv(filepath_or_buffer, **dict(kwargs, nrows=0, skipfooter=0)).columns
            usecols = all_cols.get_indexer_for(list(usecols_md[0]))
        parse_dates = kwargs.pop('parse_dates', False)
        partition_kwargs = dict(kwargs, header=None, names=names, skipfooter=0, skiprows=None, parse_dates=parse_dates, usecols=usecols)
        encoding = kwargs.get('encoding', None)
        quotechar = kwargs.get('quotechar', '"').encode(encoding if encoding is not None else 'UTF-8')
        is_quoting = kwargs.get('quoting', '') != csv.QUOTE_NONE
        with ExitStack() as stack:
            files = [stack.enter_context(OpenFile(fname, 'rb', compression_type, **kwargs.get('storage_options', None) or {})) for fname in glob_filepaths]
            if isinstance(skiprows, int) or skiprows is None:
                if skiprows is None:
                    skiprows = 0
                header = kwargs.get('header', 'infer')
                if header == 'infer' and kwargs.get('names', lib.no_default) in [lib.no_default, None]:
                    skip_header = 1
                elif isinstance(header, int):
                    skip_header = header + 1
                elif hasattr(header, '__iter__') and (not isinstance(header, str)):
                    skip_header = max(header) + 1
                else:
                    skip_header = 0
            if kwargs.get('encoding', None) is not None:
                partition_kwargs['skiprows'] = 1
            (column_widths, num_splits) = cls._define_metadata(pd_df_metadata, column_names)
            args = {'num_splits': num_splits, **partition_kwargs}
            splits = cls.partitioned_file(files, glob_filepaths, num_partitions=NPartitions.get(), nrows=nrows, skiprows=skiprows, skip_header=skip_header, quotechar=quotechar, is_quoting=is_quoting)
            partition_ids = [None] * len(splits)
            index_ids = [None] * len(splits)
            dtypes_ids = [None] * len(splits)
            for (idx, chunks) in enumerate(splits):
                args.update({'chunks': chunks})
                (*partition_ids[idx], index_ids[idx], dtypes_ids[idx]) = cls.deploy(func=cls.parse, f_kwargs=args, num_returns=num_splits + 2)
        if index_col is None:
            row_lengths = cls.materialize(index_ids)
            new_index = pandas.RangeIndex(sum(row_lengths))
        else:
            index_objs = cls.materialize(index_ids)
            row_lengths = [len(o) for o in index_objs]
            new_index = index_objs[0].append(index_objs[1:])
            new_index.name = pd_df_metadata.index.name
        partition_ids = cls.build_partition(partition_ids, row_lengths, column_widths)
        dtypes = cls.get_dtypes(dtypes_ids, column_names)
        new_frame = cls.frame_cls(partition_ids, new_index, column_names, row_lengths, column_widths, dtypes=dtypes)
        new_query_compiler = cls.query_compiler_cls(new_frame)
        if skipfooter:
            new_query_compiler = new_query_compiler.drop(new_query_compiler.index[-skipfooter:])
        if kwargs.get('squeeze', False) and len(new_query_compiler.columns) == 1:
            return new_query_compiler[new_query_compiler.columns[0]]
        if index_col is None:
            new_query_compiler._modin_frame.synchronize_labels(axis=0)
        return new_query_compiler

    @classmethod
    def file_exists(cls, file_path: str, storage_options=None) -> bool:
        if False:
            while True:
                i = 10
        '\n        Check if the `file_path` is valid.\n\n        Parameters\n        ----------\n        file_path : str\n            String representing a path.\n        storage_options : dict, optional\n            Keyword from `read_*` functions.\n\n        Returns\n        -------\n        bool\n            True if the path is valid.\n        '
        if is_url(file_path):
            raise NotImplementedError('`read_csv_glob` does not support urllib paths.')
        if not is_fsspec_url(file_path):
            return len(glob.glob(file_path)) > 0
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
        return exists or len(fs.glob(file_path)) > 0

    @classmethod
    def get_path(cls, file_path: str) -> list:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the path of the file(s).\n\n        Parameters\n        ----------\n        file_path : str\n            String representing a path.\n\n        Returns\n        -------\n        list\n            List of strings of absolute file paths.\n        '
        if not is_fsspec_url(file_path) and (not is_url(file_path)):
            relative_paths = glob.glob(file_path)
            abs_paths = [os.path.abspath(path) for path in relative_paths]
            return abs_paths
        try:
            from botocore.exceptions import ConnectTimeoutError, EndpointConnectionError, NoCredentialsError
            credential_error_type = (NoCredentialsError, PermissionError, EndpointConnectionError, ConnectTimeoutError)
        except ModuleNotFoundError:
            credential_error_type = (PermissionError,)

        def get_file_path(fs_handle) -> List[str]:
            if False:
                return 10
            if '*' in file_path:
                file_paths = fs_handle.glob(file_path)
            else:
                file_paths = [f for f in fs_handle.find(file_path) if not f.endswith('/')]
            if len(file_paths) == 0 and (not fs_handle.exists(file_path)):
                raise FileNotFoundError(f"Path <{file_path}> isn't available.")
            fs_addresses = [fs_handle.unstrip_protocol(path) for path in file_paths]
            return fs_addresses
        (fs, _) = fsspec.core.url_to_fs(file_path)
        try:
            return get_file_path(fs)
        except credential_error_type:
            (fs, _) = fsspec.core.url_to_fs(file_path, anon=True)
        return get_file_path(fs)

    @classmethod
    def partitioned_file(cls, files, fnames: List[str], num_partitions: int=None, nrows: int=None, skiprows: int=None, skip_header: int=None, quotechar: bytes=b'"', is_quoting: bool=True) -> List[List[Tuple[str, int, int]]]:
        if False:
            return 10
        '\n        Compute chunk sizes in bytes for every partition.\n\n        Parameters\n        ----------\n        files : file or list of files\n            File(s) to be partitioned.\n        fnames : str or list of str\n            File name(s) to be partitioned.\n        num_partitions : int, optional\n            For what number of partitions split a file.\n            If not specified grabs the value from `modin.config.NPartitions.get()`.\n        nrows : int, optional\n            Number of rows of file to read.\n        skiprows : int, optional\n            Specifies rows to skip.\n        skip_header : int, optional\n            Specifies header rows to skip.\n        quotechar : bytes, default: b\'"\'\n            Indicate quote in a file.\n        is_quoting : bool, default: True\n            Whether or not to consider quotes.\n\n        Returns\n        -------\n        list\n            List, where each element of the list is a list of tuples. The inner lists\n            of tuples contains the data file name of the chunk, chunk start offset, and\n            chunk end offsets for its corresponding file.\n\n        Notes\n        -----\n        The logic gets really complicated if we try to use the `TextFileDispatcher.partitioned_file`.\n        '
        if type(files) is not list:
            files = [files]
        if num_partitions is None:
            num_partitions = NPartitions.get()
        file_sizes = [cls.file_size(f) for f in files]
        partition_size = max(1, num_partitions, (nrows if nrows else sum(file_sizes)) // num_partitions)
        result = []
        split_result = []
        split_size = 0
        read_rows_counter = 0
        for (f, fname, f_size) in zip(files, fnames, file_sizes):
            if skiprows or skip_header:
                skip_amount = (skiprows if skiprows else 0) + (skip_header if skip_header else 0)
                (outside_quotes, read_rows) = cls._read_rows(f, nrows=skip_amount, quotechar=quotechar, is_quoting=is_quoting)
                if skiprows:
                    skiprows -= read_rows
                    if skiprows > 0:
                        continue
            start = f.tell()
            while f.tell() < f_size:
                if split_size >= partition_size:
                    result.append(split_result)
                    split_result = []
                    split_size = 0
                read_size = partition_size - split_size
                if nrows:
                    if read_rows_counter >= nrows:
                        if len(split_result) > 0:
                            result.append(split_result)
                        return result
                    elif read_rows_counter + read_size > nrows:
                        read_size = nrows - read_rows_counter
                    (outside_quotes, read_rows) = cls._read_rows(f, nrows=read_size, quotechar=quotechar, is_quoting=is_quoting)
                    split_size += read_rows
                    read_rows_counter += read_rows
                else:
                    outside_quotes = cls.offset(f, offset_size=read_size, quotechar=quotechar, is_quoting=is_quoting)
                split_result.append((fname, start, f.tell()))
                split_size += f.tell() - start
                start = f.tell()
                if is_quoting and (not outside_quotes):
                    warnings.warn('File has mismatched quotes')
        if len(split_result) > 0:
            result.append(split_result)
        return result