"""Module houses `FeatherDispatcher` class, that is used for reading `.feather` files."""
from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher
from modin.core.io.file_dispatcher import OpenFile
from modin.utils import import_optional_dependency

class FeatherDispatcher(ColumnStoreDispatcher):
    """Class handles utils for reading `.feather` files."""

    @classmethod
    def _read(cls, path, columns=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read data from the file path, returning a query compiler.\n\n        Parameters\n        ----------\n        path : str or file-like object\n            The filepath of the feather file.\n        columns : array-like, optional\n            Columns to read from file. If not provided, all columns are read.\n        **kwargs : dict\n            `read_feather` function kwargs.\n\n        Returns\n        -------\n        BaseQueryCompiler\n            Query compiler with imported data for further processing.\n\n        Notes\n        -----\n        `PyArrow` engine and local files only are supported for now,\n        multi threading is set to False by default.\n        PyArrow feather is used. Please refer to the documentation here\n        https://arrow.apache.org/docs/python/api.html#feather-format\n        '
        path = cls.get_path(path)
        if columns is None:
            import_optional_dependency('pyarrow', 'pyarrow is required to read feather files.')
            from pyarrow import ipc
            with OpenFile(path, **kwargs.get('storage_options', None) or {}) as file:
                reader = ipc.open_file(file)
            index_cols = frozenset((col for col in reader.schema.pandas_metadata['index_columns'] if isinstance(col, str)))
            columns = [col for col in reader.schema.names if col not in index_cols]
        return cls.build_query_compiler(path, columns, use_threads=False, dtype_backend=kwargs['dtype_backend'])