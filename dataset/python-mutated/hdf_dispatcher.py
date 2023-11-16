"""Module houses `HDFDispatcher` class, that is used for reading hdf data."""
import pandas
from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher

class HDFDispatcher(ColumnStoreDispatcher):
    """
    Class handles utils for reading hdf data.

    Inherits some common for columnar store files util functions from
    `ColumnStoreDispatcher` class.
    """

    @classmethod
    def _validate_hdf_format(cls, path_or_buf):
        if False:
            for i in range(10):
                print('nop')
        '\n        Validate `path_or_buf` and then return `table_type` parameter of store group attribute.\n\n        Parameters\n        ----------\n        path_or_buf : str, buffer or path object\n            Path to the file to open, or an open :class:`pandas.HDFStore` object.\n\n        Returns\n        -------\n        str\n            `table_type` parameter of store group attribute.\n        '
        s = pandas.HDFStore(path_or_buf)
        groups = s.groups()
        if len(groups) == 0:
            raise ValueError('No dataset in HDF5 file.')
        candidate_only_group = groups[0]
        format = getattr(candidate_only_group._v_attrs, 'table_type', None)
        s.close()
        return format

    @classmethod
    def _read(cls, path_or_buf, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Load an h5 file from the file path or buffer, returning a query compiler.\n\n        Parameters\n        ----------\n        path_or_buf : str, buffer or path object\n            Path to the file to open, or an open :class:`pandas.HDFStore` object.\n        **kwargs : dict\n            Pass into pandas.read_hdf function.\n\n        Returns\n        -------\n        BaseQueryCompiler\n            Query compiler with imported data for further processing.\n        '
        if cls._validate_hdf_format(path_or_buf=path_or_buf) is None:
            return cls.single_worker_read(path_or_buf, reason='File format seems to be `fixed`. For better distribution consider ' + 'saving the file in `table` format. df.to_hdf(format=`table`).', **kwargs)
        columns = kwargs.pop('columns', None)
        kwargs['_key'] = kwargs.pop('key', None)
        if not columns:
            start = kwargs.pop('start', None)
            stop = kwargs.pop('stop', None)
            empty_pd_df = pandas.read_hdf(path_or_buf, start=0, stop=0, **kwargs)
            if start is not None:
                kwargs['start'] = start
            if stop is not None:
                kwargs['stop'] = stop
            columns = empty_pd_df.columns
        return cls.build_query_compiler(path_or_buf, columns, **kwargs)