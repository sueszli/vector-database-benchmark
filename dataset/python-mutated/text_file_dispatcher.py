"""
Module houses `TextFileDispatcher` class.

`TextFileDispatcher` contains utils for text formats files, inherits util functions for
files from `FileDispatcher` class and can be used as base class for dipatchers of SQL queries.
"""
import codecs
import io
import os
import warnings
from csv import QUOTE_NONE
from typing import Callable, Optional, Sequence, Tuple, Union
import numpy as np
import pandas
import pandas._libs.lib as lib
from pandas.core.dtypes.common import is_list_like
from modin.config import NPartitions
from modin.core.io.file_dispatcher import FileDispatcher, OpenFile
from modin.core.io.text.utils import CustomNewlineIterator
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.utils import _inherit_docstrings
ColumnNamesTypes = Tuple[Union[pandas.Index, pandas.MultiIndex]]
IndexColType = Union[int, str, bool, Sequence[int], Sequence[str], None]

class TextFileDispatcher(FileDispatcher):
    """Class handles utils for reading text formats files."""

    @classmethod
    def get_path_or_buffer(cls, filepath_or_buffer):
        if False:
            print('Hello World!')
        '\n        Extract path from `filepath_or_buffer`.\n\n        Parameters\n        ----------\n        filepath_or_buffer : str, path object or file-like object\n            `filepath_or_buffer` parameter of `read_csv` function.\n\n        Returns\n        -------\n        str or path object\n            verified `filepath_or_buffer` parameter.\n\n        Notes\n        -----\n        Given a buffer, try and extract the filepath from it so that we can\n        use it without having to fall back to pandas and share file objects between\n        workers. Given a filepath, return it immediately.\n        '
        if hasattr(filepath_or_buffer, 'name') and hasattr(filepath_or_buffer, 'seekable') and filepath_or_buffer.seekable() and (filepath_or_buffer.tell() == 0):
            buffer_filepath = filepath_or_buffer.name
            if cls.file_exists(buffer_filepath):
                warnings.warn('For performance reasons, the filepath will be ' + 'used in place of the file handle passed in ' + 'to load the data')
                return cls.get_path(buffer_filepath)
        return filepath_or_buffer

    @classmethod
    def build_partition(cls, partition_ids, row_lengths, column_widths):
        if False:
            i = 10
            return i + 15
        '\n        Build array with partitions of `cls.frame_partition_cls` class.\n\n        Parameters\n        ----------\n        partition_ids : list\n                Array with references to the partitions data.\n        row_lengths : list\n                Partitions rows lengths.\n        column_widths : list\n                Number of columns in each partition.\n\n        Returns\n        -------\n        np.ndarray\n            array with shape equals to the shape of `partition_ids` and\n            filed with partitions objects.\n        '
        return np.array([[cls.frame_partition_cls(partition_ids[i][j], length=row_lengths[i], width=column_widths[j]) for j in range(len(partition_ids[i]))] for i in range(len(partition_ids))])

    @classmethod
    def pathlib_or_pypath(cls, filepath_or_buffer):
        if False:
            return 10
        '\n        Check if `filepath_or_buffer` is instance of `py.path.local` or `pathlib.Path`.\n\n        Parameters\n        ----------\n        filepath_or_buffer : str, path object or file-like object\n            `filepath_or_buffer` parameter of `read_csv` function.\n\n        Returns\n        -------\n        bool\n            Whether or not `filepath_or_buffer` is instance of `py.path.local`\n            or `pathlib.Path`.\n        '
        try:
            import py
            if isinstance(filepath_or_buffer, py.path.local):
                return True
        except ImportError:
            pass
        try:
            import pathlib
            if isinstance(filepath_or_buffer, pathlib.Path):
                return True
        except ImportError:
            pass
        return False

    @classmethod
    def offset(cls, f, offset_size: int, quotechar: bytes=b'"', is_quoting: bool=True, encoding: str=None, newline: bytes=None):
        if False:
            print('Hello World!')
        '\n        Move the file offset at the specified amount of bytes.\n\n        Parameters\n        ----------\n        f : file-like object\n            File handle that should be used for offset movement.\n        offset_size : int\n            Number of bytes to read and ignore.\n        quotechar : bytes, default: b\'"\'\n            Indicate quote in a file.\n        is_quoting : bool, default: True\n            Whether or not to consider quotes.\n        encoding : str, optional\n            Encoding of `f`.\n        newline : bytes, optional\n            Byte or sequence of bytes indicating line endings.\n\n        Returns\n        -------\n        bool\n            If file pointer reached the end of the file, but did not find\n            closing quote returns `False`. `True` in any other case.\n        '
        if is_quoting:
            chunk = f.read(offset_size)
            outside_quotes = not chunk.count(quotechar) % 2
        else:
            f.seek(offset_size, os.SEEK_CUR)
            outside_quotes = True
        (outside_quotes, _) = cls._read_rows(f, nrows=1, quotechar=quotechar, is_quoting=is_quoting, outside_quotes=outside_quotes, encoding=encoding, newline=newline)
        return outside_quotes

    @classmethod
    def partitioned_file(cls, f, num_partitions: int=None, nrows: int=None, skiprows: int=None, quotechar: bytes=b'"', is_quoting: bool=True, encoding: str=None, newline: bytes=None, header_size: int=0, pre_reading: int=0, read_callback_kw: dict=None):
        if False:
            while True:
                i = 10
        '\n        Compute chunk sizes in bytes for every partition.\n\n        Parameters\n        ----------\n        f : file-like object\n            File handle of file to be partitioned.\n        num_partitions : int, optional\n            For what number of partitions split a file.\n            If not specified grabs the value from `modin.config.NPartitions.get()`.\n        nrows : int, optional\n            Number of rows of file to read.\n        skiprows : int, optional\n            Specifies rows to skip.\n        quotechar : bytes, default: b\'"\'\n            Indicate quote in a file.\n        is_quoting : bool, default: True\n            Whether or not to consider quotes.\n        encoding : str, optional\n            Encoding of `f`.\n        newline : bytes, optional\n            Byte or sequence of bytes indicating line endings.\n        header_size : int, default: 0\n            Number of rows, that occupied by header.\n        pre_reading : int, default: 0\n            Number of rows between header and skipped rows, that should be read.\n        read_callback_kw : dict, optional\n            Keyword arguments for `cls.read_callback` to compute metadata if needed.\n            This option is not compatible with `pre_reading!=0`.\n\n        Returns\n        -------\n        list\n            List with the next elements:\n                int : partition start read byte\n                int : partition end read byte\n        pandas.DataFrame or None\n            Dataframe from which metadata can be retrieved. Can be None if `read_callback_kw=None`.\n        '
        if read_callback_kw is not None and pre_reading != 0:
            raise ValueError(f'Incompatible combination of parameters: read_callback_kw={read_callback_kw!r}, pre_reading={pre_reading!r}')
        read_rows_counter = 0
        outside_quotes = True
        if num_partitions is None:
            num_partitions = NPartitions.get() - 1 if pre_reading else NPartitions.get()
        rows_skipper = cls.rows_skipper_builder(f, quotechar, is_quoting=is_quoting, encoding=encoding, newline=newline)
        result = []
        file_size = cls.file_size(f)
        pd_df_metadata = None
        if pre_reading:
            rows_skipper(header_size)
            pre_reading_start = f.tell()
            (outside_quotes, read_rows) = cls._read_rows(f, nrows=pre_reading, quotechar=quotechar, is_quoting=is_quoting, outside_quotes=outside_quotes, encoding=encoding, newline=newline)
            read_rows_counter += read_rows
            result.append((pre_reading_start, f.tell()))
            if is_quoting and (not outside_quotes):
                warnings.warn('File has mismatched quotes')
            rows_skipper(skiprows)
        else:
            rows_skipper(skiprows)
            if read_callback_kw:
                start = f.tell()
                pd_df_metadata = cls.read_callback(f, **read_callback_kw)
                f.seek(start)
            rows_skipper(header_size)
        start = f.tell()
        if nrows:
            partition_size = max(1, num_partitions, nrows // num_partitions)
            while f.tell() < file_size and read_rows_counter < nrows:
                if read_rows_counter + partition_size > nrows:
                    partition_size = nrows - read_rows_counter
                (outside_quotes, read_rows) = cls._read_rows(f, nrows=partition_size, quotechar=quotechar, is_quoting=is_quoting, encoding=encoding, newline=newline)
                result.append((start, f.tell()))
                start = f.tell()
                read_rows_counter += read_rows
                if is_quoting and (not outside_quotes):
                    warnings.warn('File has mismatched quotes')
        else:
            partition_size = max(1, num_partitions, file_size // num_partitions)
            while f.tell() < file_size:
                outside_quotes = cls.offset(f, offset_size=partition_size, quotechar=quotechar, is_quoting=is_quoting, encoding=encoding, newline=newline)
                result.append((start, f.tell()))
                start = f.tell()
                if is_quoting and (not outside_quotes):
                    warnings.warn('File has mismatched quotes')
        return (result, pd_df_metadata)

    @classmethod
    def _read_rows(cls, f, nrows: int, quotechar: bytes=b'"', is_quoting: bool=True, outside_quotes: bool=True, encoding: str=None, newline: bytes=None):
        if False:
            return 10
        '\n        Move the file offset at the specified amount of rows.\n\n        Parameters\n        ----------\n        f : file-like object\n            File handle that should be used for offset movement.\n        nrows : int\n            Number of rows to read.\n        quotechar : bytes, default: b\'"\'\n            Indicate quote in a file.\n        is_quoting : bool, default: True\n            Whether or not to consider quotes.\n        outside_quotes : bool, default: True\n            Whether the file pointer is within quotes or not at the time this function is called.\n        encoding : str, optional\n            Encoding of `f`.\n        newline : bytes, optional\n            Byte or sequence of bytes indicating line endings.\n\n        Returns\n        -------\n        bool\n            If file pointer reached the end of the file, but did not find closing quote\n            returns `False`. `True` in any other case.\n        int\n            Number of rows that were read.\n        '
        if nrows is not None and nrows <= 0:
            return (True, 0)
        rows_read = 0
        if encoding and ('utf' in encoding and '8' not in encoding or encoding == 'unicode_escape' or encoding.replace('-', '_') == 'utf_8_sig'):
            iterator = CustomNewlineIterator(f, newline)
        else:
            iterator = f
        for line in iterator:
            if is_quoting and line.count(quotechar) % 2:
                outside_quotes = not outside_quotes
            if outside_quotes:
                rows_read += 1
                if rows_read >= nrows:
                    break
        if isinstance(iterator, CustomNewlineIterator):
            iterator.seek()
        if not outside_quotes:
            rows_read += 1
        return (outside_quotes, rows_read)

    @classmethod
    def compute_newline(cls, file_like, encoding, quotechar):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute byte or sequence of bytes indicating line endings.\n\n        Parameters\n        ----------\n        file_like : file-like object\n            File handle that should be used for line endings computing.\n        encoding : str\n            Encoding of `file_like`.\n        quotechar : str\n            Quotechar used for parsing `file-like`.\n\n        Returns\n        -------\n        bytes\n            line endings\n        '
        newline = None
        if encoding is None:
            return (newline, quotechar.encode('UTF-8'))
        quotechar = quotechar.encode(encoding)
        encoding = encoding.replace('-', '_')
        if 'utf' in encoding and '8' not in encoding or encoding == 'unicode_escape' or encoding == 'utf_8_sig':
            file_like.readline()
            newline = file_like.newlines.encode(encoding)
            boms = ()
            if encoding == 'utf_8_sig':
                boms = (codecs.BOM_UTF8,)
            elif '16' in encoding:
                boms = (codecs.BOM_UTF16_BE, codecs.BOM_UTF16_LE)
            elif '32' in encoding:
                boms = (codecs.BOM_UTF32_BE, codecs.BOM_UTF32_LE)
            for bom in boms:
                if newline.startswith(bom):
                    bom_len = len(bom)
                    newline = newline[bom_len:]
                    quotechar = quotechar[bom_len:]
                    break
        return (newline, quotechar)

    @classmethod
    def rows_skipper_builder(cls, f, quotechar, is_quoting, encoding=None, newline=None):
        if False:
            i = 10
            return i + 15
        '\n        Build object for skipping passed number of lines.\n\n        Parameters\n        ----------\n        f : file-like object\n            File handle that should be used for offset movement.\n        quotechar : bytes\n            Indicate quote in a file.\n        is_quoting : bool\n            Whether or not to consider quotes.\n        encoding : str, optional\n            Encoding of `f`.\n        newline : bytes, optional\n            Byte or sequence of bytes indicating line endings.\n\n        Returns\n        -------\n        object\n            skipper object.\n        '

        def skipper(n):
            if False:
                print('Hello World!')
            if n == 0 or n is None:
                return 0
            else:
                return cls._read_rows(f, quotechar=quotechar, is_quoting=is_quoting, nrows=n, encoding=encoding, newline=newline)[1]
        return skipper

    @classmethod
    def _define_header_size(cls, header: Union[int, Sequence[int], str, None]='infer', names: Optional[Sequence]=lib.no_default) -> int:
        if False:
            return 10
        '\n        Define the number of rows that are used by header.\n\n        Parameters\n        ----------\n        header : int, list of int or str, default: "infer"\n            Original `header` parameter of `read_csv` function.\n        names :  array-like, optional\n            Original names parameter of `read_csv` function.\n\n        Returns\n        -------\n        header_size : int\n            The number of rows that are used by header.\n        '
        header_size = 0
        if header == 'infer' and names in [lib.no_default, None]:
            header_size += 1
        elif isinstance(header, int):
            header_size += header + 1
        elif hasattr(header, '__iter__') and (not isinstance(header, str)):
            header_size += max(header) + 1
        return header_size

    @classmethod
    def _define_metadata(cls, df: pandas.DataFrame, column_names: ColumnNamesTypes) -> Tuple[list, int]:
        if False:
            i = 10
            return i + 15
        '\n        Define partitioning metadata.\n\n        Parameters\n        ----------\n        df : pandas.DataFrame\n            The DataFrame to split.\n        column_names : ColumnNamesTypes\n            Column names of df.\n\n        Returns\n        -------\n        column_widths : list\n            Column width to use during new frame creation (number of\n            columns for each partition).\n        num_splits : int\n            The maximum number of splits to separate the DataFrame into.\n        '
        num_splits = min(len(column_names) or 1, NPartitions.get())
        column_chunksize = compute_chunksize(df.shape[1], num_splits)
        if column_chunksize > len(column_names):
            column_widths = [len(column_names)]
            num_splits = 1
        else:
            column_widths = [column_chunksize if len(column_names) > column_chunksize * (i + 1) else 0 if len(column_names) < column_chunksize * i else len(column_names) - column_chunksize * i for i in range(num_splits)]
        return (column_widths, num_splits)
    _parse_func = None

    @classmethod
    def preprocess_func(cls):
        if False:
            while True:
                i = 10
        'Prepare a function for transmission to remote workers.'
        if cls._parse_func is None:
            cls._parse_func = cls.put(cls.parse)
        return cls._parse_func

    @classmethod
    def _launch_tasks(cls, splits: list, *partition_args, **partition_kwargs) -> Tuple[list, list, list]:
        if False:
            while True:
                i = 10
        '\n        Launch tasks to read partitions.\n\n        Parameters\n        ----------\n        splits : list\n            List of tuples with partitions data, which defines\n            parser task (start/end read bytes and etc.).\n        *partition_args : tuple\n            Positional arguments to be passed to the parser function.\n        **partition_kwargs : dict\n            `kwargs` that should be passed to the parser function.\n\n        Returns\n        -------\n        partition_ids : list\n            array with references to the partitions data.\n        index_ids : list\n            array with references to the partitions index objects.\n        dtypes_ids : list\n            array with references to the partitions dtypes objects.\n        '
        partition_ids = [None] * len(splits)
        index_ids = [None] * len(splits)
        dtypes_ids = [None] * len(splits)
        func = cls.preprocess_func()
        for (idx, (start, end)) in enumerate(splits):
            partition_kwargs.update({'start': start, 'end': end})
            (*partition_ids[idx], index_ids[idx], dtypes_ids[idx]) = cls.deploy(func=func, f_args=partition_args, f_kwargs=partition_kwargs, num_returns=partition_kwargs.get('num_splits') + 2)
        return (partition_ids, index_ids, dtypes_ids)

    @classmethod
    def check_parameters_support(cls, filepath_or_buffer, read_kwargs: dict, skiprows_md: Union[Sequence, callable, int], header_size: int) -> Tuple[bool, Optional[str]]:
        if False:
            while True:
                i = 10
        '\n        Check support of only general parameters of `read_*` function.\n\n        Parameters\n        ----------\n        filepath_or_buffer : str, path object or file-like object\n            `filepath_or_buffer` parameter of `read_*` function.\n        read_kwargs : dict\n            Parameters of `read_*` function.\n        skiprows_md : int, array or callable\n            `skiprows` parameter modified for easier handling by Modin.\n        header_size : int\n            Number of rows that are used by header.\n\n        Returns\n        -------\n        bool\n            Whether passed parameters are supported or not.\n        Optional[str]\n            `None` if parameters are supported, otherwise an error\n            message describing why parameters are not supported.\n        '
        skiprows = read_kwargs.get('skiprows')
        if isinstance(filepath_or_buffer, str):
            if not cls.file_exists(filepath_or_buffer, read_kwargs.get('storage_options')):
                return (False, cls._file_not_found_msg(filepath_or_buffer))
        elif not cls.pathlib_or_pypath(filepath_or_buffer):
            return (False, cls.BUFFER_UNSUPPORTED_MSG)
        if read_kwargs['chunksize'] is not None:
            return (False, '`chunksize` parameter is not supported')
        if read_kwargs.get('iterator'):
            return (False, '`iterator==True` parameter is not supported')
        if read_kwargs.get('dialect') is not None:
            return (False, '`dialect` parameter is not supported')
        if read_kwargs['lineterminator'] is not None:
            return (False, '`lineterminator` parameter is not supported')
        if read_kwargs['escapechar'] is not None:
            return (False, '`escapechar` parameter is not supported')
        if read_kwargs.get('skipfooter'):
            if read_kwargs.get('nrows') or read_kwargs.get('engine') == 'c':
                return (False, 'Exception is raised by pandas itself')
        skiprows_supported = True
        if is_list_like(skiprows_md) and skiprows_md[0] < header_size:
            skiprows_supported = False
        elif callable(skiprows):
            is_intersection = any(cls._get_skip_mask(pandas.RangeIndex(header_size), skiprows))
            if is_intersection:
                skiprows_supported = False
        if not skiprows_supported:
            return (False, 'Values of `header` and `skiprows` parameters have intersections; ' + 'this case is unsupported by Modin')
        return (True, None)

    @classmethod
    @_inherit_docstrings(pandas.io.parsers.base_parser.ParserBase._validate_usecols_arg)
    def _validate_usecols_arg(cls, usecols):
        if False:
            return 10
        msg = "'usecols' must either be list-like of all strings, all unicode, " + 'all integers or a callable.'
        if usecols is not None:
            if callable(usecols):
                return (usecols, None)
            if not is_list_like(usecols):
                raise ValueError(msg)
            usecols_dtype = lib.infer_dtype(usecols, skipna=False)
            if usecols_dtype not in ('empty', 'integer', 'string'):
                raise ValueError(msg)
            usecols = set(usecols)
            return (usecols, usecols_dtype)
        return (usecols, None)

    @classmethod
    def _manage_skiprows_parameter(cls, skiprows: Union[int, Sequence[int], Callable, None]=None, header_size: int=0) -> Tuple[Union[int, Sequence, Callable], bool, int]:
        if False:
            print('Hello World!')
        '\n        Manage `skiprows` parameter of read_csv and read_fwf functions.\n\n        Change `skiprows` parameter in the way Modin could more optimally\n        process it. `csv_dispatcher` and `fwf_dispatcher` have two mechanisms of rows skipping:\n\n        1) During file partitioning (setting of file limits that should be read\n        by each partition) exact rows can be excluded from partitioning scope,\n        thus they won\'t be read at all and can be considered as skipped. This is\n        the most effective way of rows skipping (since it doesn\'t require any\n        actual data reading and postprocessing), but in this case `skiprows`\n        parameter can be an integer only. When it possible Modin always uses\n        this approach by setting of `skiprows_partitioning` return value.\n\n        2) Rows for skipping can be dropped after full dataset import. This is\n        more expensive way since it requires extra IO work and postprocessing\n        afterwards, but `skiprows` parameter can be of any non-integer type\n        supported by any pandas read function. These rows is\n        specified by setting of `skiprows_md` return value.\n\n        In some cases, if `skiprows` is uniformly distributed array (e.g. [1,2,3]),\n        `skiprows` can be "squashed" and represented as integer to make a fastpath.\n        If there is a gap between the first row for skipping and the last line of\n        the header (that will be skipped too), then assign to read this gap first\n        (assign the first partition to read these rows be setting of `pre_reading`\n        return value). See `Examples` section for details.\n\n        Parameters\n        ----------\n        skiprows : int, array or callable, optional\n            Original `skiprows` parameter of any pandas read function.\n        header_size : int, default: 0\n            Number of rows that are used by header.\n\n        Returns\n        -------\n        skiprows_md : int, array or callable\n            Updated skiprows parameter. If `skiprows` is an array, this\n            array will be sorted. Also parameter will be aligned to\n            actual data in the `query_compiler` (which, for example,\n            doesn\'t contain header rows)\n        pre_reading : int\n            The number of rows that should be read before data file\n            splitting for further reading (the number of rows for\n            the first partition).\n        skiprows_partitioning : int\n            The number of rows that should be skipped virtually (skipped during\n            data file partitioning).\n\n        Examples\n        --------\n        Let\'s consider case when `header`="infer" and `skiprows`=[3,4,5]. In\n        this specific case fastpath can be done since `skiprows` is uniformly\n        distributed array, so we can "squash" it to integer and set\n        `skiprows_partitioning`=3. But if no additional action will be done,\n        these three rows will be skipped right after header line, that corresponds\n        to `skiprows`=[1,2,3]. Now, to avoid this discrepancy, we need to assign\n        the first partition to read data between header line and the first\n        row for skipping by setting of `pre_reading` parameter, so setting\n        `pre_reading`=2. During data file partitiong, these lines will be assigned\n        for reading for the first partition, and then file position will be set at\n        the beginning of rows that should be skipped by `skiprows_partitioning`.\n        After skipping of these rows, the rest data will be divided between the\n        rest of partitions, see rows assignement below:\n\n        0 - header line (skip during partitioning)\n        1 - pre_reading (assign to read by the first partition)\n        2 - pre_reading (assign to read by the first partition)\n        3 - skiprows_partitioning (skip during partitioning)\n        4 - skiprows_partitioning (skip during partitioning)\n        5 - skiprows_partitioning (skip during partitioning)\n        6 - data to partition (divide between the rest of partitions)\n        7 - data to partition (divide between the rest of partitions)\n        '
        pre_reading = skiprows_partitioning = skiprows_md = 0
        if isinstance(skiprows, int):
            skiprows_partitioning = skiprows
        elif is_list_like(skiprows) and len(skiprows) > 0:
            skiprows_md = np.sort(skiprows)
            if np.all(np.diff(skiprows_md) == 1):
                pre_reading = skiprows_md[0] - header_size if skiprows_md[0] > header_size else 0
                skiprows_partitioning = len(skiprows_md)
                skiprows_md = 0
            elif skiprows_md[0] > header_size:
                skiprows_md = skiprows_md - header_size
        elif callable(skiprows):

            def skiprows_func(x):
                if False:
                    return 10
                return skiprows(x + header_size)
            skiprows_md = skiprows_func
        return (skiprows_md, pre_reading, skiprows_partitioning)

    @classmethod
    def _define_index(cls, index_ids: list, index_name: str) -> Tuple[IndexColType, list]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the resulting DataFrame index and index lengths for each of partitions.\n\n        Parameters\n        ----------\n        index_ids : list\n            Array with references to the partitions index objects.\n        index_name : str\n            Name that should be assigned to the index if `index_col`\n            is not provided.\n\n        Returns\n        -------\n        new_index : IndexColType\n            Index that should be passed to the new_frame constructor.\n        row_lengths : list\n            Partitions rows lengths.\n        '
        index_objs = cls.materialize(index_ids)
        if len(index_objs) == 0 or isinstance(index_objs[0], int):
            row_lengths = index_objs
            new_index = pandas.RangeIndex(sum(index_objs))
        else:
            row_lengths = [len(o) for o in index_objs]
            new_index = index_objs[0].append(index_objs[1:])
            new_index.name = index_name
        return (new_index, row_lengths)

    @classmethod
    def _get_new_qc(cls, partition_ids: list, index_ids: list, dtypes_ids: list, index_col: IndexColType, index_name: str, column_widths: list, column_names: ColumnNamesTypes, skiprows_md: Union[Sequence, callable, None]=None, header_size: int=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get new query compiler from data received from workers.\n\n        Parameters\n        ----------\n        partition_ids : list\n            Array with references to the partitions data.\n        index_ids : list\n            Array with references to the partitions index objects.\n        dtypes_ids : list\n            Array with references to the partitions dtypes objects.\n        index_col : IndexColType\n            `index_col` parameter of `read_csv` function.\n        index_name : str\n            Name that should be assigned to the index if `index_col`\n            is not provided.\n        column_widths : list\n            Number of columns in each partition.\n        column_names : ColumnNamesTypes\n            Array with columns names.\n        skiprows_md : array-like or callable, optional\n            Specifies rows to skip.\n        header_size : int, default: 0\n            Number of rows, that occupied by header.\n        **kwargs : dict\n            Parameters of `read_csv` function needed for postprocessing.\n\n        Returns\n        -------\n        new_query_compiler : BaseQueryCompiler\n            New query compiler, created from `new_frame`.\n        '
        partition_ids = cls.build_partition(partition_ids, [None] * len(index_ids), column_widths)
        new_frame = cls.frame_cls(partition_ids, lambda : cls._define_index(index_ids, index_name), column_names, None, column_widths, dtypes=lambda : cls.get_dtypes(dtypes_ids, column_names))
        new_query_compiler = cls.query_compiler_cls(new_frame)
        skipfooter = kwargs.get('skipfooter', None)
        if skipfooter:
            new_query_compiler = new_query_compiler.drop(new_query_compiler.index[-skipfooter:])
        if skiprows_md is not None:
            nrows = kwargs.get('nrows', None)
            index_range = pandas.RangeIndex(len(new_query_compiler.index))
            if is_list_like(skiprows_md):
                new_query_compiler = new_query_compiler.take_2d_positional(index=index_range.delete(skiprows_md))
            elif callable(skiprows_md):
                skip_mask = cls._get_skip_mask(index_range, skiprows_md)
                if not isinstance(skip_mask, np.ndarray):
                    skip_mask = skip_mask.to_numpy('bool')
                view_idx = index_range[~skip_mask]
                new_query_compiler = new_query_compiler.take_2d_positional(index=view_idx)
            else:
                raise TypeError(f'Not acceptable type of `skiprows` parameter: {type(skiprows_md)}')
            if not isinstance(new_query_compiler.index, pandas.MultiIndex):
                new_query_compiler = new_query_compiler.reset_index(drop=True)
            if nrows:
                new_query_compiler = new_query_compiler.take_2d_positional(pandas.RangeIndex(len(new_query_compiler.index))[:nrows])
        if index_col is None or index_col is False:
            new_query_compiler._modin_frame.synchronize_labels(axis=0)
        return new_query_compiler

    @classmethod
    def _read(cls, filepath_or_buffer, **kwargs):
        if False:
            return 10
        '\n        Read data from `filepath_or_buffer` according to `kwargs` parameters.\n\n        Used in `read_csv` and `read_fwf` Modin implementations.\n\n        Parameters\n        ----------\n        filepath_or_buffer : str, path object or file-like object\n            `filepath_or_buffer` parameter of read functions.\n        **kwargs : dict\n            Parameters of read functions.\n\n        Returns\n        -------\n        new_query_compiler : BaseQueryCompiler\n            Query compiler with imported data for further processing.\n        '
        filepath_or_buffer_md = cls.get_path(filepath_or_buffer) if isinstance(filepath_or_buffer, str) else cls.get_path_or_buffer(filepath_or_buffer)
        compression_infered = cls.infer_compression(filepath_or_buffer, kwargs['compression'])
        names = kwargs['names']
        index_col = kwargs['index_col']
        encoding = kwargs['encoding']
        skiprows = kwargs['skiprows']
        header = kwargs['header']
        header_size = cls._define_header_size(header, names)
        (skiprows_md, pre_reading, skiprows_partitioning) = cls._manage_skiprows_parameter(skiprows, header_size)
        should_handle_skiprows = skiprows_md is not None and (not isinstance(skiprows_md, int))
        (use_modin_impl, fallback_reason) = cls.check_parameters_support(filepath_or_buffer_md, kwargs, skiprows_md, header_size)
        if not use_modin_impl:
            return cls.single_worker_read(filepath_or_buffer, kwargs, reason=fallback_reason)
        is_quoting = kwargs['quoting'] != QUOTE_NONE
        usecols = kwargs['usecols']
        use_inferred_column_names = cls._uses_inferred_column_names(names, skiprows, kwargs['skipfooter'], usecols)
        can_compute_metadata_while_skipping_rows = isinstance(skiprows, int) and (usecols is None or skiprows is None) and (pre_reading == 0)
        read_callback_kw = dict(kwargs, nrows=1, skipfooter=0, index_col=index_col)
        if not can_compute_metadata_while_skipping_rows:
            pd_df_metadata = cls.read_callback(filepath_or_buffer_md, **read_callback_kw)
            column_names = pd_df_metadata.columns
            (column_widths, num_splits) = cls._define_metadata(pd_df_metadata, column_names)
            read_callback_kw = None
        else:
            read_callback_kw = dict(read_callback_kw, skiprows=None)
            read_callback_kw.pop('memory_map', None)
            read_callback_kw.pop('storage_options', None)
            read_callback_kw.pop('compression', None)
        with OpenFile(filepath_or_buffer_md, 'rb', compression_infered, **kwargs.get('storage_options', None) or {}) as f:
            old_pos = f.tell()
            fio = io.TextIOWrapper(f, encoding=encoding, newline='')
            (newline, quotechar) = cls.compute_newline(fio, encoding, kwargs.get('quotechar', '"'))
            f.seek(old_pos)
            (splits, pd_df_metadata_temp) = cls.partitioned_file(f, num_partitions=NPartitions.get(), nrows=kwargs['nrows'] if not should_handle_skiprows else None, skiprows=skiprows_partitioning, quotechar=quotechar, is_quoting=is_quoting, encoding=encoding, newline=newline, header_size=header_size, pre_reading=pre_reading, read_callback_kw=read_callback_kw)
            if can_compute_metadata_while_skipping_rows:
                pd_df_metadata = pd_df_metadata_temp
        common_dtypes = None
        if kwargs['dtype'] is None:
            most_common_dtype = (object,)
            common_dtypes = {}
            for (col, dtype) in pd_df_metadata.dtypes.to_dict().items():
                if dtype in most_common_dtype:
                    common_dtypes[col] = dtype
        column_names = pd_df_metadata.columns
        (column_widths, num_splits) = cls._define_metadata(pd_df_metadata, column_names)
        partition_kwargs = dict(kwargs, header_size=0 if use_inferred_column_names else header_size, names=column_names if use_inferred_column_names else names, header='infer' if use_inferred_column_names else header, skipfooter=0, skiprows=None, nrows=None, compression=compression_infered, common_dtypes=common_dtypes)
        filepath_or_buffer_md_ref = cls.put(filepath_or_buffer_md)
        kwargs_ref = cls.put(partition_kwargs)
        (partition_ids, index_ids, dtypes_ids) = cls._launch_tasks(splits, filepath_or_buffer_md_ref, kwargs_ref, num_splits=num_splits)
        new_query_compiler = cls._get_new_qc(partition_ids=partition_ids, index_ids=index_ids, dtypes_ids=dtypes_ids, index_col=index_col, index_name=pd_df_metadata.index.name, column_widths=column_widths, column_names=column_names, skiprows_md=skiprows_md if should_handle_skiprows else None, header_size=header_size, skipfooter=kwargs['skipfooter'], parse_dates=kwargs['parse_dates'], nrows=kwargs['nrows'] if should_handle_skiprows else None)
        return new_query_compiler

    @classmethod
    def _get_skip_mask(cls, rows_index: pandas.Index, skiprows: Callable):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get mask of skipped by callable `skiprows` rows.\n\n        Parameters\n        ----------\n        rows_index : pandas.Index\n            Rows index to get mask for.\n        skiprows : Callable\n            Callable to check whether row index should be skipped.\n\n        Returns\n        -------\n        pandas.Index\n        '
        try:
            mask = skiprows(rows_index)
            assert is_list_like(mask)
        except (ValueError, TypeError, AssertionError):
            mask = rows_index.map(skiprows)
        return mask

    @staticmethod
    def _uses_inferred_column_names(names, skiprows, skipfooter, usecols):
        if False:
            print('Hello World!')
        "\n        Tell whether need to use inferred column names in workers or not.\n\n        1) ``False`` is returned in 2 cases and means next:\n            1.a) `names` parameter was provided from the API layer. In this case parameter\n            `names` must be provided as `names` parameter for ``read_csv`` in the workers.\n            1.b) `names` parameter wasn't provided from the API layer. In this case column names\n            inference must happen in each partition.\n        2) ``True`` is returned in case when inferred column names from pre-reading stage must be\n            provided as `names` parameter for ``read_csv`` in the workers.\n\n        In case `names` was provided, the other parameters aren't checked. Otherwise, inferred column\n        names should be used in a case of not full data reading which is defined by `skipfooter` parameter,\n        when need to skip lines at the bottom of file or by `skiprows` parameter, when need to skip lines at\n        the top of file (but if `usecols` was provided, column names inference must happen in the workers).\n\n        Parameters\n        ----------\n        names : array-like\n            List of column names to use.\n        skiprows : list-like, int or callable\n            Line numbers to skip (0-indexed) or number of lines to skip (int) at\n            the start of the file. If callable, the callable function will be\n            evaluated against the row indices, returning ``True`` if the row should\n            be skipped and ``False`` otherwise.\n        skipfooter : int\n            Number of lines at bottom of the file to skip.\n        usecols : list-like or callable\n            Subset of the columns.\n\n        Returns\n        -------\n        bool\n            Whether to use inferred column names in ``read_csv`` of the workers or not.\n        "
        if names not in [None, lib.no_default]:
            return False
        if skipfooter != 0:
            return True
        if isinstance(skiprows, int) and skiprows == 0:
            return False
        if is_list_like(skiprows):
            return usecols is None
        return skiprows is not None