from typing import TYPE_CHECKING, Optional
from ray.data._internal.util import _check_import
from ray.data.block import BlockAccessor
from ray.data.datasource.file_datasink import BlockBasedFileDatasink
from ray.data.datasource.tfrecords_datasource import _convert_arrow_table_to_examples, _write_record
if TYPE_CHECKING:
    import pyarrow
    from tensorflow_metadata.proto.v0 import schema_pb2

class _TFRecordDatasink(BlockBasedFileDatasink):

    def __init__(self, path: str, *, tf_schema: Optional['schema_pb2.Schema']=None, file_format: str='tar', **file_datasink_kwargs):
        if False:
            while True:
                i = 10
        super().__init__(path, file_format=file_format, **file_datasink_kwargs)
        _check_import(self, module='crc32c', package='crc32c')
        self.tf_schema = tf_schema

    def write_block_to_file(self, block: BlockAccessor, file: 'pyarrow.NativeFile'):
        if False:
            while True:
                i = 10
        arrow_table = block.to_arrow()
        examples = _convert_arrow_table_to_examples(arrow_table, self.tf_schema)
        for example in examples:
            _write_record(file, example)