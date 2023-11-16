import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
if TYPE_CHECKING:
    import pyarrow
logger = logging.getLogger(__name__)

@PublicAPI
class ParquetBaseDatasource(FileBasedDatasource):
    """Minimal Parquet datasource, for reading and writing Parquet files."""
    _FILE_EXTENSIONS = ['parquet']

    def __init__(self, paths: Union[str, List[str]], read_table_args: Optional[Dict[str, Any]]=None, **file_based_datasource_kwargs):
        if False:
            return 10
        super().__init__(paths, **file_based_datasource_kwargs)
        if read_table_args is None:
            read_table_args = {}
        self.read_table_args = read_table_args

    def get_name(self):
        if False:
            print('Hello World!')
        'Return a human-readable name for this datasource.\n        This will be used as the names of the read tasks.\n        Note: overrides the base `FileBasedDatasource` method.\n        '
        return 'ParquetBulk'

    def _read_file(self, f: 'pyarrow.NativeFile', path: str):
        if False:
            print('Hello World!')
        import pyarrow.parquet as pq
        use_threads = self.read_table_args.pop('use_threads', False)
        return pq.read_table(f, use_threads=use_threads, **self.read_table_args)

    def _open_input_source(self, filesystem: 'pyarrow.fs.FileSystem', path: str, **open_args) -> 'pyarrow.NativeFile':
        if False:
            return 10
        return filesystem.open_input_file(path, **open_args)