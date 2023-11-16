"""Generates ManyField test data for Merger.

Constructs chunked proto test data containing various field types for
Merger::Read and Merger::Merge.

Usage: bazel run tensorflow/tools/proto_splitter/testdata:many_field_gen -- \\
    --path=/tmp/many_field
"""
from collections.abc import Sequence
import os
from absl import app
from absl import flags
from tensorflow.python.lib.io import file_io
from tensorflow.tools.proto_splitter import split
from tensorflow.tools.proto_splitter.testdata import test_message_pb2
SPLITTER_TESTDATA_PATH = flags.DEFINE_string('path', None, help='Path to testdata directory.')

class ManyFieldSplitter(split.ComposableSplitter):
    """Splitter for ManyField proto."""

    def build_chunks(self):
        if False:
            return 10
        self.add_chunk(self._proto.field_one, [test_message_pb2.ManyFields.DESCRIPTOR.fields_by_name['field_one'].number])
        self._proto.ClearField('field_one')
        for (map_key, map_value) in self._proto.nested_map_bool.items():
            self.add_chunk(map_value, [test_message_pb2.ManyFields.DESCRIPTOR.fields_by_name['nested_map_bool'].number, map_key])
        self._proto.ClearField('nested_map_bool')

def main(argv: Sequence[str]) -> None:
    if False:
        print('Hello World!')
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    proto = test_message_pb2.ManyFields(field_one=test_message_pb2.ManyFields(repeated_field=[test_message_pb2.ManyFields(), test_message_pb2.ManyFields(string_field='inner_inner_string', map_field_uint32={324: 'map_value_324', 543: 'map_value_543'})]), map_field_int64={-1345: 'map_value_-1345'}, nested_map_bool={True: test_message_pb2.ManyFields(string_field='string_true'), False: test_message_pb2.ManyFields(string_field='string_false')})
    file_io.write_string_to_file(os.path.join(SPLITTER_TESTDATA_PATH.value, 'many-field.pbtxt'), str(proto))
    ManyFieldSplitter(proto).write(os.path.join(SPLITTER_TESTDATA_PATH.value, 'many-field'))
if __name__ == '__main__':
    app.run(main)