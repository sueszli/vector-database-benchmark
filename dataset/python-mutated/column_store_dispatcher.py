"""
Module houses `ColumnStoreDispatcher` class.

`ColumnStoreDispatcher` contains utils for handling columnar store format files,
inherits util functions for handling files from `FileDispatcher` class and can be
used as base class for dipatchers of specific columnar store formats.
"""
import numpy as np
import pandas
from modin.config import MinPartitionSize, NPartitions
from modin.core.io.file_dispatcher import FileDispatcher
from modin.core.storage_formats.pandas.utils import compute_chunksize

class ColumnStoreDispatcher(FileDispatcher):
    """
    Class handles utils for reading columnar store format files.

    Inherits some util functions for processing files from `FileDispatcher` class.
    """

    @classmethod
    def call_deploy(cls, fname, col_partitions, **kwargs):
        if False:
            return 10
        '\n        Deploy remote tasks to the workers with passed parameters.\n\n        Parameters\n        ----------\n        fname : str, path object or file-like object\n            Name of the file to read.\n        col_partitions : list\n            List of arrays with columns names that should be read\n            by each partition.\n        **kwargs : dict\n            Parameters of deploying read_* function.\n\n        Returns\n        -------\n        np.ndarray\n            Array with references to the task deploy result for each partition.\n        '
        return np.array([cls.deploy(func=cls.parse, f_kwargs={'fname': fname, 'columns': cols, 'num_splits': NPartitions.get(), **kwargs}, num_returns=NPartitions.get() + 2) for cols in col_partitions]).T

    @classmethod
    def build_partition(cls, partition_ids, row_lengths, column_widths):
        if False:
            while True:
                i = 10
        '\n        Build array with partitions of `cls.frame_partition_cls` class.\n\n        Parameters\n        ----------\n        partition_ids : list\n            Array with references to the partitions data.\n        row_lengths : list\n            Partitions rows lengths.\n        column_widths : list\n            Number of columns in each partition.\n\n        Returns\n        -------\n        np.ndarray\n            array with shape equals to the shape of `partition_ids` and\n            filed with partition objects.\n        '
        return np.array([[cls.frame_partition_cls(partition_ids[i][j], length=row_lengths[i], width=column_widths[j]) for j in range(len(partition_ids[i]))] for i in range(len(partition_ids))])

    @classmethod
    def build_index(cls, partition_ids):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute index and its split sizes of resulting Modin DataFrame.\n\n        Parameters\n        ----------\n        partition_ids : list\n            Array with references to the partitions data.\n\n        Returns\n        -------\n        index : pandas.Index\n            Index of resulting Modin DataFrame.\n        row_lengths : list\n            List with lengths of index chunks.\n        '
        num_partitions = NPartitions.get()
        index_len = 0 if len(partition_ids) == 0 else cls.materialize(partition_ids[-2][0])
        if isinstance(index_len, int):
            index = pandas.RangeIndex(index_len)
        else:
            index = index_len
            index_len = len(index)
        index_chunksize = compute_chunksize(index_len, num_partitions)
        if index_chunksize > index_len:
            row_lengths = [index_len] + [0 for _ in range(num_partitions - 1)]
        else:
            row_lengths = [index_chunksize if (i + 1) * index_chunksize < index_len else max(0, index_len - index_chunksize * i) for i in range(num_partitions)]
        return (index, row_lengths)

    @classmethod
    def build_columns(cls, columns, num_row_parts=None):
        if False:
            while True:
                i = 10
        "\n        Split columns into chunks that should be read by workers.\n\n        Parameters\n        ----------\n        columns : list\n            List of columns that should be read from file.\n        num_row_parts : int, optional\n            Number of parts the dataset is split into. This parameter is used\n            to align the column partitioning with it so we won't end up with an\n            over partitioned frame.\n\n        Returns\n        -------\n        col_partitions : list\n            List of lists with columns for reading by workers.\n        column_widths : list\n            List with lengths of `col_partitions` subarrays\n            (number of columns that should be read by workers).\n        "
        columns_length = len(columns)
        if columns_length == 0:
            return ([], [])
        if num_row_parts is None:
            min_block_size = 1
        else:
            num_remaining_parts = round(NPartitions.get() / num_row_parts)
            min_block_size = min(columns_length // num_remaining_parts, MinPartitionSize.get())
        column_splits = compute_chunksize(columns_length, NPartitions.get(), max(1, min_block_size))
        col_partitions = [columns[i:i + column_splits] for i in range(0, columns_length, column_splits)]
        column_widths = [len(c) for c in col_partitions]
        return (col_partitions, column_widths)

    @classmethod
    def build_dtypes(cls, partition_ids, columns):
        if False:
            print('Hello World!')
        '\n        Compute common for all partitions `dtypes` for each of the DataFrame column.\n\n        Parameters\n        ----------\n        partition_ids : list\n            Array with references to the partitions data.\n        columns : list\n            List of columns that should be read from file.\n\n        Returns\n        -------\n        dtypes : pandas.Series\n            Series with dtypes for columns.\n        '
        dtypes = pandas.concat(cls.materialize(list(partition_ids)), axis=0)
        dtypes.index = columns
        return dtypes

    @classmethod
    def build_query_compiler(cls, path, columns, **kwargs):
        if False:
            return 10
        '\n        Build query compiler from deployed tasks outputs.\n\n        Parameters\n        ----------\n        path : str, path object or file-like object\n            Path to the file to read.\n        columns : list\n            List of columns that should be read from file.\n        **kwargs : dict\n            Parameters of deploying read_* function.\n\n        Returns\n        -------\n        new_query_compiler : BaseQueryCompiler\n            Query compiler with imported data for further processing.\n        '
        (col_partitions, column_widths) = cls.build_columns(columns)
        partition_ids = cls.call_deploy(path, col_partitions, **kwargs)
        (index, row_lens) = cls.build_index(partition_ids)
        remote_parts = cls.build_partition(partition_ids[:-2], row_lens, column_widths)
        dtypes = cls.build_dtypes(partition_ids[-1], columns) if len(partition_ids) > 0 else None
        new_query_compiler = cls.query_compiler_cls(cls.frame_cls(remote_parts, index, columns, row_lens, column_widths, dtypes=dtypes))
        return new_query_compiler