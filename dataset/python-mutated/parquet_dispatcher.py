"""Module houses `ParquetDispatcher` class, that is used for reading `.parquet` files."""
import json
import os
import re
from typing import TYPE_CHECKING
import fsspec
import numpy as np
import pandas
import pandas._libs.lib as lib
from fsspec.core import url_to_fs
from fsspec.spec import AbstractBufferedFile
from packaging import version
from pandas.io.common import stringify_path
from modin.config import NPartitions
from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
if TYPE_CHECKING:
    from modin.core.storage_formats.pandas.parsers import ParquetFileToRead

class ColumnStoreDataset:
    """
    Base class that encapsulates Parquet engine-specific details.

    This class exposes a set of functions that are commonly used in the
    `read_parquet` implementation.

    Attributes
    ----------
    path : str, path object or file-like object
        The filepath of the parquet file in local filesystem or hdfs.
    storage_options : dict
        Parameters for specific storage engine.
    _fs_path : str, path object or file-like object
        The filepath or handle of the parquet dataset specific to the
        filesystem implementation. E.g. for `s3://test/example`, _fs
        would be set to S3FileSystem and _fs_path would be `test/example`.
    _fs : Filesystem
        Filesystem object specific to the given parquet file/dataset.
    dataset : ParquetDataset or ParquetFile
        Underlying dataset implementation for PyArrow and fastparquet
        respectively.
    _row_groups_per_file : list
        List that contains the number of row groups for each file in the
        given parquet dataset.
    _files : list
        List that contains the full paths of the parquet files in the dataset.
    """

    def __init__(self, path, storage_options):
        if False:
            print('Hello World!')
        self.path = path.__fspath__() if isinstance(path, os.PathLike) else path
        self.storage_options = storage_options
        self._fs_path = None
        self._fs = None
        self.dataset = self._init_dataset()
        self._row_groups_per_file = None
        self._files = None

    @property
    def pandas_metadata(self):
        if False:
            return 10
        'Return the pandas metadata of the dataset.'
        raise NotImplementedError

    @property
    def columns(self):
        if False:
            return 10
        'Return the list of columns in the dataset.'
        raise NotImplementedError

    @property
    def engine(self):
        if False:
            for i in range(10):
                print('nop')
        'Return string representing what engine is being used.'
        raise NotImplementedError

    @property
    def files(self):
        if False:
            while True:
                i = 10
        'Return the list of formatted file paths of the dataset.'
        raise NotImplementedError

    @property
    def row_groups_per_file(self):
        if False:
            while True:
                i = 10
        'Return a list with the number of row groups per file.'
        raise NotImplementedError

    @property
    def fs(self):
        if False:
            return 10
        '\n        Return the filesystem object associated with the dataset path.\n\n        Returns\n        -------\n        filesystem\n            Filesystem object.\n        '
        if self._fs is None:
            if isinstance(self.path, AbstractBufferedFile):
                self._fs = self.path.fs
            else:
                (self._fs, self._fs_path) = url_to_fs(self.path, **self.storage_options)
        return self._fs

    @property
    def fs_path(self):
        if False:
            while True:
                i = 10
        '\n        Return the filesystem-specific path or file handle.\n\n        Returns\n        -------\n        fs_path : str, path object or file-like object\n            String path specific to filesystem or a file handle.\n        '
        if self._fs_path is None:
            if isinstance(self.path, AbstractBufferedFile):
                self._fs_path = self.path
            else:
                (self._fs, self._fs_path) = url_to_fs(self.path, **self.storage_options)
        return self._fs_path

    def to_pandas_dataframe(self, columns):
        if False:
            while True:
                i = 10
        '\n        Read the given columns as a pandas dataframe.\n\n        Parameters\n        ----------\n        columns : list\n            List of columns that should be read from file.\n        '
        raise NotImplementedError

    def _get_files(self, files):
        if False:
            while True:
                i = 10
        '\n        Retrieve list of formatted file names in dataset path.\n\n        Parameters\n        ----------\n        files : list\n            List of files from path.\n\n        Returns\n        -------\n        fs_files : list\n            List of files from path with fs-protocol prepended.\n        '

        def _unstrip_protocol(protocol, path):
            if False:
                while True:
                    i = 10
            protos = (protocol,) if isinstance(protocol, str) else protocol
            for protocol in protos:
                if path.startswith(f'{protocol}://'):
                    return path
            return f'{protos[0]}://{path}'
        if isinstance(self.path, AbstractBufferedFile):
            return [self.path]
        if version.parse(fsspec.__version__) < version.parse('2022.5.0'):
            fs_files = [_unstrip_protocol(self.fs.protocol, fpath) for fpath in files]
        else:
            fs_files = [self.fs.unstrip_protocol(fpath) for fpath in files]
        return fs_files

@_inherit_docstrings(ColumnStoreDataset)
class PyArrowDataset(ColumnStoreDataset):

    def _init_dataset(self):
        if False:
            i = 10
            return i + 15
        from pyarrow.parquet import ParquetDataset
        return ParquetDataset(self.fs_path, filesystem=self.fs, use_legacy_dataset=False)

    @property
    def pandas_metadata(self):
        if False:
            while True:
                i = 10
        return self.dataset.schema.pandas_metadata

    @property
    def columns(self):
        if False:
            return 10
        return self.dataset.schema.names

    @property
    def engine(self):
        if False:
            for i in range(10):
                print('nop')
        return 'pyarrow'

    @property
    def row_groups_per_file(self):
        if False:
            while True:
                i = 10
        from pyarrow.parquet import ParquetFile
        if self._row_groups_per_file is None:
            row_groups_per_file = []
            for file in self.files:
                with self.fs.open(file) as f:
                    row_groups = ParquetFile(f).num_row_groups
                    row_groups_per_file.append(row_groups)
            self._row_groups_per_file = row_groups_per_file
        return self._row_groups_per_file

    @property
    def files(self):
        if False:
            return 10
        if self._files is None:
            try:
                files = self.dataset.files
            except AttributeError:
                files = self.dataset._dataset.files
            self._files = self._get_files(files)
        return self._files

    def to_pandas_dataframe(self, columns):
        if False:
            print('Hello World!')
        from pyarrow.parquet import read_table
        return read_table(self._fs_path, columns=columns, filesystem=self.fs).to_pandas()

@_inherit_docstrings(ColumnStoreDataset)
class FastParquetDataset(ColumnStoreDataset):

    def _init_dataset(self):
        if False:
            return 10
        from fastparquet import ParquetFile
        return ParquetFile(self.fs_path, fs=self.fs)

    @property
    def pandas_metadata(self):
        if False:
            print('Hello World!')
        if 'pandas' not in self.dataset.key_value_metadata:
            return {}
        return json.loads(self.dataset.key_value_metadata['pandas'])

    @property
    def columns(self):
        if False:
            return 10
        return self.dataset.columns

    @property
    def engine(self):
        if False:
            print('Hello World!')
        return 'fastparquet'

    @property
    def row_groups_per_file(self):
        if False:
            for i in range(10):
                print('nop')
        from fastparquet import ParquetFile
        if self._row_groups_per_file is None:
            row_groups_per_file = []
            for file in self.files:
                with self.fs.open(file) as f:
                    row_groups = ParquetFile(f).info['row_groups']
                    row_groups_per_file.append(row_groups)
            self._row_groups_per_file = row_groups_per_file
        return self._row_groups_per_file

    @property
    def files(self):
        if False:
            return 10
        if self._files is None:
            self._files = self._get_files(self._get_fastparquet_files())
        return self._files

    def to_pandas_dataframe(self, columns):
        if False:
            for i in range(10):
                print('nop')
        return self.dataset.to_pandas(columns=columns)

    def _get_fastparquet_files(self):
        if False:
            return 10
        if '*' in self.path:
            files = self.fs.glob(self.path)
        else:
            files = [f for f in self.fs.find(self.path) if f.endswith('.parquet') or f.endswith('.parq')]
        return files

class ParquetDispatcher(ColumnStoreDispatcher):
    """Class handles utils for reading `.parquet` files."""
    index_regex = re.compile('__index_level_\\d+__')

    @classmethod
    def get_dataset(cls, path, engine, storage_options):
        if False:
            while True:
                i = 10
        "\n        Retrieve Parquet engine specific Dataset implementation.\n\n        Parameters\n        ----------\n        path : str, path object or file-like object\n            The filepath of the parquet file in local filesystem or hdfs.\n        engine : str\n            Parquet library to use (only 'PyArrow' is supported for now).\n        storage_options : dict\n            Parameters for specific storage engine.\n\n        Returns\n        -------\n        Dataset\n            Either a PyArrowDataset or FastParquetDataset object.\n        "
        if engine == 'auto':
            engine_classes = [PyArrowDataset, FastParquetDataset]
            error_msgs = ''
            for engine_class in engine_classes:
                try:
                    return engine_class(path, storage_options)
                except ImportError as err:
                    error_msgs += '\n - ' + str(err)
            raise ImportError('Unable to find a usable engine; ' + "tried using: 'pyarrow', 'fastparquet'.\n" + 'A suitable version of ' + 'pyarrow or fastparquet is required for parquet ' + 'support.\n' + 'Trying to import the above resulted in these errors:' + f'{error_msgs}')
        elif engine == 'pyarrow':
            return PyArrowDataset(path, storage_options)
        elif engine == 'fastparquet':
            return FastParquetDataset(path, storage_options)
        else:
            raise ValueError("engine must be one of 'pyarrow', 'fastparquet'")

    @classmethod
    def _determine_partitioning(cls, dataset: ColumnStoreDataset) -> 'list[list[ParquetFileToRead]]':
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine which partition will read certain files/row groups of the dataset.\n\n        Parameters\n        ----------\n        dataset : ColumnStoreDataset\n\n        Returns\n        -------\n        list[list[ParquetFileToRead]]\n            Each element in the returned list describes a list of files that a partition has to read.\n        '
        from modin.core.storage_formats.pandas.parsers import ParquetFileToRead
        parquet_files = dataset.files
        row_groups_per_file = dataset.row_groups_per_file
        num_row_groups = sum(row_groups_per_file)
        if num_row_groups == 0:
            return []
        num_splits = min(NPartitions.get(), num_row_groups)
        part_size = num_row_groups // num_splits
        reminder = num_row_groups % num_splits
        part_sizes = [part_size] * (num_splits - reminder) + [part_size + 1] * reminder
        partition_files = []
        file_idx = 0
        row_group_idx = 0
        row_groups_left_in_current_file = row_groups_per_file[file_idx]
        total_row_groups_added = 0
        for size in part_sizes:
            row_groups_taken = 0
            part_files = []
            while row_groups_taken != size:
                if row_groups_left_in_current_file < 1:
                    file_idx += 1
                    row_group_idx = 0
                    row_groups_left_in_current_file = row_groups_per_file[file_idx]
                to_take = min(size - row_groups_taken, row_groups_left_in_current_file)
                part_files.append(ParquetFileToRead(parquet_files[file_idx], row_group_start=row_group_idx, row_group_end=row_group_idx + to_take))
                row_groups_left_in_current_file -= to_take
                row_groups_taken += to_take
                row_group_idx += to_take
            total_row_groups_added += row_groups_taken
            partition_files.append(part_files)
        sanity_check = len(partition_files) == num_splits and total_row_groups_added == num_row_groups
        ErrorMessage.catch_bugs_and_request_email(failure_condition=not sanity_check, extra_log='row groups added does not match total num of row groups across parquet files')
        return partition_files

    @classmethod
    def call_deploy(cls, partition_files: 'list[list[ParquetFileToRead]]', col_partitions: 'list[list[str]]', storage_options: dict, engine: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deploy remote tasks to the workers with passed parameters.\n\n        Parameters\n        ----------\n        partition_files : list[list[ParquetFileToRead]]\n            List of arrays with files that should be read by each partition.\n        col_partitions : list[list[str]]\n            List of arrays with columns names that should be read\n            by each partition.\n        storage_options : dict\n            Parameters for specific storage engine.\n        engine : {"auto", "pyarrow", "fastparquet"}\n            Parquet library to use for reading.\n        **kwargs : dict\n            Parameters of deploying read_* function.\n\n        Returns\n        -------\n        List\n            Array with references to the task deploy result for each partition.\n        '
        if len(col_partitions) == 0:
            return []
        all_partitions = []
        for files_to_read in partition_files:
            all_partitions.append([cls.deploy(func=cls.parse, f_kwargs={'files_for_parser': files_to_read, 'columns': cols, 'engine': engine, 'storage_options': storage_options, **kwargs}, num_returns=3) for cols in col_partitions])
        return all_partitions

    @classmethod
    def build_partition(cls, partition_ids, column_widths):
        if False:
            return 10
        '\n        Build array with partitions of `cls.frame_partition_cls` class.\n\n        Parameters\n        ----------\n        partition_ids : list\n            Array with references to the partitions data.\n        column_widths : list\n            Number of columns in each partition.\n\n        Returns\n        -------\n        np.ndarray\n            array with shape equals to the shape of `partition_ids` and\n            filed with partition objects.\n\n        Notes\n        -----\n        The second level of partitions_ids contains a list of object references\n        for each read call:\n        partition_ids[i][j] -> [ObjectRef(df), ObjectRef(df.index), ObjectRef(len(df))].\n        '
        return np.array([[cls.frame_partition_cls(part_id[0], length=part_id[2], width=col_width) for (part_id, col_width) in zip(part_ids, column_widths)] for part_ids in partition_ids])

    @classmethod
    def build_index(cls, dataset, partition_ids, index_columns, filters):
        if False:
            print('Hello World!')
        "\n        Compute index and its split sizes of resulting Modin DataFrame.\n\n        Parameters\n        ----------\n        dataset : Dataset\n            Dataset object of Parquet file/files.\n        partition_ids : list\n            Array with references to the partitions data.\n        index_columns : list\n            List of index columns specified by pandas metadata.\n        filters : list\n            List of filters to be used in reading the Parquet file/files.\n\n        Returns\n        -------\n        index : pandas.Index\n            Index of resulting Modin DataFrame.\n        needs_index_sync : bool\n            Whether the partition indices need to be synced with frame\n            index because there's no index column, or at least one\n            index column is a RangeIndex.\n\n        Notes\n        -----\n        See `build_partition` for more detail on the contents of partitions_ids.\n        "
        range_index = True
        range_index_metadata = None
        column_names_to_read = []
        for column in index_columns:
            if isinstance(column, str):
                column_names_to_read.append(column)
                range_index = False
            elif column['kind'] == 'range':
                range_index_metadata = column
        if range_index and filters is None or (len(partition_ids) == 0 and len(column_names_to_read) != 0):
            complete_index = dataset.to_pandas_dataframe(columns=column_names_to_read).index
        elif len(partition_ids) == 0:
            return ([], False)
        else:
            index_ids = [part_id[0][1] for part_id in partition_ids if len(part_id) > 0]
            index_objs = cls.materialize(index_ids)
            if range_index:
                total_filtered_length = sum((len(index_part) for index_part in index_objs))
                metadata_length_mismatch = False
                if range_index_metadata is not None:
                    metadata_implied_length = (range_index_metadata['stop'] - range_index_metadata['start']) / range_index_metadata['step']
                    metadata_length_mismatch = total_filtered_length != metadata_implied_length
                if range_index_metadata is None or (isinstance(dataset, PyArrowDataset) and metadata_length_mismatch):
                    complete_index = pandas.RangeIndex(total_filtered_length)
                else:
                    complete_index = pandas.RangeIndex(start=range_index_metadata['start'], step=range_index_metadata['step'], stop=range_index_metadata['start'] + total_filtered_length * range_index_metadata['step'], name=range_index_metadata['name'])
            else:
                complete_index = index_objs[0].append(index_objs[1:])
        return (complete_index, range_index or len(index_columns) == 0)

    @classmethod
    def build_query_compiler(cls, dataset, columns, index_columns, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Build query compiler from deployed tasks outputs.\n\n        Parameters\n        ----------\n        dataset : Dataset\n            Dataset object of Parquet file/files.\n        columns : list\n            List of columns that should be read from file.\n        index_columns : list\n            List of index columns specified by pandas metadata.\n        **kwargs : dict\n            Parameters of deploying read_* function.\n\n        Returns\n        -------\n        new_query_compiler : BaseQueryCompiler\n            Query compiler with imported data for further processing.\n        '
        storage_options = kwargs.pop('storage_options', {}) or {}
        filters = kwargs.get('filters', None)
        partition_files = cls._determine_partitioning(dataset)
        (col_partitions, column_widths) = cls.build_columns(columns, num_row_parts=len(partition_files))
        partition_ids = cls.call_deploy(partition_files, col_partitions, storage_options, dataset.engine, **kwargs)
        (index, sync_index) = cls.build_index(dataset, partition_ids, index_columns, filters)
        remote_parts = cls.build_partition(partition_ids, column_widths)
        if len(partition_ids) > 0:
            row_lengths = [part.length() for part in remote_parts.T[0]]
        else:
            row_lengths = None
        frame = cls.frame_cls(remote_parts, index, columns, row_lengths=row_lengths, column_widths=column_widths, dtypes=None)
        if sync_index:
            frame.synchronize_labels(axis=0)
        return cls.query_compiler_cls(frame)

    @classmethod
    def _read(cls, path, engine, columns, use_nullable_dtypes, dtype_backend, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Load a parquet object from the file path, returning a query compiler.\n\n        Parameters\n        ----------\n        path : str, path object or file-like object\n            The filepath of the parquet file in local filesystem or hdfs.\n        engine : {"auto", "pyarrow", "fastparquet"}\n            Parquet library to use.\n        columns : list\n            If not None, only these columns will be read from the file.\n        use_nullable_dtypes : Union[bool, lib.NoDefault]\n        dtype_backend : {"numpy_nullable", "pyarrow", lib.no_default}\n        **kwargs : dict\n            Keyword arguments.\n\n        Returns\n        -------\n        BaseQueryCompiler\n            A new Query Compiler.\n\n        Notes\n        -----\n        ParquetFile API is used. Please refer to the documentation here\n        https://arrow.apache.org/docs/python/parquet.html\n        '
        if set(kwargs) - {'storage_options', 'filters', 'filesystem'} or use_nullable_dtypes != lib.no_default or kwargs.get('filesystem') is not None:
            return cls.single_worker_read(path, engine=engine, columns=columns, use_nullable_dtypes=use_nullable_dtypes, dtype_backend=dtype_backend, reason='Parquet options that are not currently supported', **kwargs)
        path = stringify_path(path)
        if isinstance(path, list):
            compilers: list[cls.query_compiler_cls] = [cls._read(p, engine, columns, use_nullable_dtypes, dtype_backend, **kwargs) for p in path]
            return compilers[0].concat(axis=0, other=compilers[1:], ignore_index=True)
        if isinstance(path, str):
            if os.path.isdir(path):
                path_generator = os.walk(path)
            else:
                storage_options = kwargs.get('storage_options')
                if storage_options is not None:
                    (fs, fs_path) = url_to_fs(path, **storage_options)
                else:
                    (fs, fs_path) = url_to_fs(path)
                path_generator = fs.walk(fs_path)
            partitioned_columns = set()
            for (_, dir_names, files) in path_generator:
                if dir_names:
                    partitioned_columns.add(dir_names[0].split('=')[0])
                if files:
                    if len(files[0]) > 0 and files[0][0] == '.':
                        continue
                    break
            partitioned_columns = list(partitioned_columns)
            if len(partitioned_columns):
                return cls.single_worker_read(path, engine=engine, columns=columns, use_nullable_dtypes=use_nullable_dtypes, dtype_backend=dtype_backend, reason='Mixed partitioning columns in Parquet', **kwargs)
        dataset = cls.get_dataset(path, engine, kwargs.get('storage_options') or {})
        index_columns = dataset.pandas_metadata.get('index_columns', []) if dataset.pandas_metadata else []
        column_names = columns if columns else dataset.columns
        columns = [c for c in column_names if c not in index_columns and (not cls.index_regex.match(c))]
        return cls.build_query_compiler(dataset, columns, index_columns, dtype_backend=dtype_backend, **kwargs)

    @classmethod
    def write(cls, qc, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Write a ``DataFrame`` to the binary parquet format.\n\n        Parameters\n        ----------\n        qc : BaseQueryCompiler\n            The query compiler of the Modin dataframe that we want to run `to_parquet` on.\n        **kwargs : dict\n            Parameters for `pandas.to_parquet(**kwargs)`.\n        '
        output_path = kwargs['path']
        if not isinstance(output_path, str):
            return cls.base_io.to_parquet(qc, **kwargs)
        client_kwargs = (kwargs.get('storage_options') or {}).get('client_kwargs', {})
        (fs, url) = fsspec.core.url_to_fs(output_path, client_kwargs=client_kwargs)
        fs.mkdirs(url, exist_ok=True)

        def func(df, **kw):
            if False:
                i = 10
                return i + 15
            '\n            Dump a chunk of rows as parquet, then save them to target maintaining order.\n\n            Parameters\n            ----------\n            df : pandas.DataFrame\n                A chunk of rows to write to a parquet file.\n            **kw : dict\n                Arguments to pass to ``pandas.to_parquet(**kwargs)`` plus an extra argument\n                `partition_idx` serving as chunk index to maintain rows order.\n            '
            compression = kwargs['compression']
            partition_idx = kw['partition_idx']
            kwargs['path'] = f'{output_path}/part-{partition_idx:04d}.{compression}.parquet'
            df.to_parquet(**kwargs)
            return pandas.DataFrame()
        qc._modin_frame._propagate_index_objs(axis=None)
        result = qc._modin_frame._partition_mgr_cls.map_axis_partitions(axis=1, partitions=qc._modin_frame._partitions, map_func=func, keep_partitioning=True, lengths=None, enumerate_partitions=True)
        cls.materialize([part.list_of_blocks[0] for row in result for part in row])