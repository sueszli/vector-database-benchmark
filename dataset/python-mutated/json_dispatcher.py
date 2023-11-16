"""Module houses `JSONDispatcher` class, that is used for reading `.json` files."""
from io import BytesIO
import numpy as np
import pandas
from modin.config import NPartitions
from modin.core.io.file_dispatcher import OpenFile
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher

class JSONDispatcher(TextFileDispatcher):
    """Class handles utils for reading `.json` files."""

    @classmethod
    def _read(cls, path_or_buf, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read data from `path_or_buf` according to the passed `read_json` `kwargs` parameters.\n\n        Parameters\n        ----------\n        path_or_buf : str, path object or file-like object\n            `path_or_buf` parameter of `read_json` function.\n        **kwargs : dict\n            Parameters of `read_json` function.\n\n        Returns\n        -------\n        BaseQueryCompiler\n            Query compiler with imported data for further processing.\n        '
        path_or_buf = cls.get_path_or_buffer(path_or_buf)
        if isinstance(path_or_buf, str):
            if not cls.file_exists(path_or_buf):
                return cls.single_worker_read(path_or_buf, reason=cls._file_not_found_msg(path_or_buf), **kwargs)
            path_or_buf = cls.get_path(path_or_buf)
        elif not cls.pathlib_or_pypath(path_or_buf):
            return cls.single_worker_read(path_or_buf, reason=cls.BUFFER_UNSUPPORTED_MSG, **kwargs)
        if not kwargs.get('lines', False):
            return cls.single_worker_read(path_or_buf, reason='`lines` argument not supported', **kwargs)
        with OpenFile(path_or_buf, 'rb') as f:
            columns = pandas.read_json(BytesIO(b'' + f.readline()), lines=True).columns
        kwargs['columns'] = columns
        empty_pd_df = pandas.DataFrame(columns=columns)
        with OpenFile(path_or_buf, 'rb', kwargs.get('compression', 'infer')) as f:
            (column_widths, num_splits) = cls._define_metadata(empty_pd_df, columns)
            args = {'fname': path_or_buf, 'num_splits': num_splits, **kwargs}
            (splits, _) = cls.partitioned_file(f, num_partitions=NPartitions.get())
            partition_ids = [None] * len(splits)
            index_ids = [None] * len(splits)
            dtypes_ids = [None] * len(splits)
            for (idx, (start, end)) in enumerate(splits):
                args.update({'start': start, 'end': end})
                (*partition_ids[idx], index_ids[idx], dtypes_ids[idx], _) = cls.deploy(func=cls.parse, f_kwargs=args, num_returns=num_splits + 3)
        row_lengths = cls.materialize(index_ids)
        new_index = pandas.RangeIndex(sum(row_lengths))
        partition_ids = cls.build_partition(partition_ids, row_lengths, column_widths)
        dtypes = cls.get_dtypes(dtypes_ids, columns)
        new_frame = cls.frame_cls(np.array(partition_ids), new_index, columns, row_lengths, column_widths, dtypes=dtypes)
        new_frame.synchronize_labels(axis=0)
        return cls.query_compiler_cls(new_frame)