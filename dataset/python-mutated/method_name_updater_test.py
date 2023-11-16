"""Tests for method name utils."""
import os
import tempfile
from google.protobuf import text_format
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import test
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader_impl as loader
from tensorflow.python.saved_model import method_name_updater
from tensorflow.python.util import compat
_SAVED_MODEL_PROTO = text_format.Parse('\nsaved_model_schema_version: 1\nmeta_graphs {\n  meta_info_def {\n    tags: "serve"\n  }\n  signature_def: {\n    key: "serving_default"\n    value: {\n      inputs: {\n        key: "inputs"\n        value { name: "input_node:0" }\n      }\n      method_name: "predict"\n      outputs: {\n        key: "outputs"\n        value {\n          dtype: DT_FLOAT\n          tensor_shape {\n            dim { size: -1 }\n            dim { size: 100 }\n          }\n        }\n      }\n    }\n  }\n  signature_def: {\n    key: "foo"\n    value: {\n      inputs: {\n        key: "inputs"\n        value { name: "input_node:0" }\n      }\n      method_name: "predict"\n      outputs: {\n        key: "outputs"\n        value {\n          dtype: DT_FLOAT\n          tensor_shape { dim { size: 1 } }\n        }\n      }\n    }\n  }\n}\nmeta_graphs {\n  meta_info_def {\n    tags: "serve"\n    tags: "gpu"\n  }\n  signature_def: {\n    key: "serving_default"\n    value: {\n      inputs: {\n        key: "inputs"\n        value { name: "input_node:0" }\n      }\n      method_name: "predict"\n      outputs: {\n        key: "outputs"\n        value {\n          dtype: DT_FLOAT\n          tensor_shape {\n            dim { size: -1 }\n          }\n        }\n      }\n    }\n  }\n  signature_def: {\n    key: "bar"\n    value: {\n      inputs: {\n        key: "inputs"\n        value { name: "input_node:0" }\n      }\n      method_name: "predict"\n      outputs: {\n        key: "outputs"\n        value {\n          dtype: DT_FLOAT\n          tensor_shape { dim { size: 1 } }\n        }\n      }\n    }\n  }\n}\n', saved_model_pb2.SavedModel())

class MethodNameUpdaterTest(test.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(MethodNameUpdaterTest, self).setUp()
        self._saved_model_path = tempfile.mkdtemp(prefix=test.get_temp_dir())

    def testBasic(self):
        if False:
            print('Hello World!')
        path = os.path.join(compat.as_bytes(self._saved_model_path), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
        file_io.write_string_to_file(path, _SAVED_MODEL_PROTO.SerializeToString(deterministic=True))
        updater = method_name_updater.MethodNameUpdater(self._saved_model_path)
        updater.replace_method_name(signature_key='serving_default', method_name='classify')
        updater.save()
        actual = loader.parse_saved_model(self._saved_model_path)
        self.assertProtoEquals(actual, text_format.Parse('\n        saved_model_schema_version: 1\n        meta_graphs {\n          meta_info_def {\n            tags: "serve"\n          }\n          signature_def: {\n            key: "serving_default"\n            value: {\n              inputs: {\n                key: "inputs"\n                value { name: "input_node:0" }\n              }\n              method_name: "classify"\n              outputs: {\n                key: "outputs"\n                value {\n                  dtype: DT_FLOAT\n                  tensor_shape {\n                    dim { size: -1 }\n                    dim { size: 100 }\n                  }\n                }\n              }\n            }\n          }\n          signature_def: {\n            key: "foo"\n            value: {\n              inputs: {\n                key: "inputs"\n                value { name: "input_node:0" }\n              }\n              method_name: "predict"\n              outputs: {\n                key: "outputs"\n                value {\n                  dtype: DT_FLOAT\n                  tensor_shape { dim { size: 1 } }\n                }\n              }\n            }\n          }\n        }\n        meta_graphs {\n          meta_info_def {\n            tags: "serve"\n            tags: "gpu"\n          }\n          signature_def: {\n            key: "serving_default"\n            value: {\n              inputs: {\n                key: "inputs"\n                value { name: "input_node:0" }\n              }\n              method_name: "classify"\n              outputs: {\n                key: "outputs"\n                value {\n                  dtype: DT_FLOAT\n                  tensor_shape {\n                    dim { size: -1 }\n                  }\n                }\n              }\n            }\n          }\n          signature_def: {\n            key: "bar"\n            value: {\n              inputs: {\n                key: "inputs"\n                value { name: "input_node:0" }\n              }\n              method_name: "predict"\n              outputs: {\n                key: "outputs"\n                value {\n                  dtype: DT_FLOAT\n                  tensor_shape { dim { size: 1 } }\n                }\n              }\n            }\n          }\n        }\n    ', saved_model_pb2.SavedModel()))

    def testTextFormatAndNewExportDir(self):
        if False:
            return 10
        path = os.path.join(compat.as_bytes(self._saved_model_path), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
        file_io.write_string_to_file(path, str(_SAVED_MODEL_PROTO))
        updater = method_name_updater.MethodNameUpdater(self._saved_model_path)
        updater.replace_method_name(signature_key='foo', method_name='regress', tags='serve')
        updater.replace_method_name(signature_key='bar', method_name='classify', tags=['gpu', 'serve'])
        new_export_dir = tempfile.mkdtemp(prefix=test.get_temp_dir())
        updater.save(new_export_dir)
        self.assertTrue(file_io.file_exists(os.path.join(compat.as_bytes(new_export_dir), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))))
        actual = loader.parse_saved_model(new_export_dir)
        self.assertProtoEquals(actual, text_format.Parse('\n        saved_model_schema_version: 1\n        meta_graphs {\n          meta_info_def {\n            tags: "serve"\n          }\n          signature_def: {\n            key: "serving_default"\n            value: {\n              inputs: {\n                key: "inputs"\n                value { name: "input_node:0" }\n              }\n              method_name: "predict"\n              outputs: {\n                key: "outputs"\n                value {\n                  dtype: DT_FLOAT\n                  tensor_shape {\n                    dim { size: -1 }\n                    dim { size: 100 }\n                  }\n                }\n              }\n            }\n          }\n          signature_def: {\n            key: "foo"\n            value: {\n              inputs: {\n                key: "inputs"\n                value { name: "input_node:0" }\n              }\n              method_name: "regress"\n              outputs: {\n                key: "outputs"\n                value {\n                  dtype: DT_FLOAT\n                  tensor_shape { dim { size: 1 } }\n                }\n              }\n            }\n          }\n        }\n        meta_graphs {\n          meta_info_def {\n            tags: "serve"\n            tags: "gpu"\n          }\n          signature_def: {\n            key: "serving_default"\n            value: {\n              inputs: {\n                key: "inputs"\n                value { name: "input_node:0" }\n              }\n              method_name: "predict"\n              outputs: {\n                key: "outputs"\n                value {\n                  dtype: DT_FLOAT\n                  tensor_shape {\n                    dim { size: -1 }\n                  }\n                }\n              }\n            }\n          }\n          signature_def: {\n            key: "bar"\n            value: {\n              inputs: {\n                key: "inputs"\n                value { name: "input_node:0" }\n              }\n              method_name: "classify"\n              outputs: {\n                key: "outputs"\n                value {\n                  dtype: DT_FLOAT\n                  tensor_shape { dim { size: 1 } }\n                }\n              }\n            }\n          }\n        }\n    ', saved_model_pb2.SavedModel()))

    def testExceptions(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(IOError):
            updater = method_name_updater.MethodNameUpdater(tempfile.mkdtemp(prefix=test.get_temp_dir()))
        path = os.path.join(compat.as_bytes(self._saved_model_path), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
        file_io.write_string_to_file(path, _SAVED_MODEL_PROTO.SerializeToString(deterministic=True))
        updater = method_name_updater.MethodNameUpdater(self._saved_model_path)
        with self.assertRaisesRegex(ValueError, '`signature_key` must be defined'):
            updater.replace_method_name(signature_key=None, method_name='classify')
        with self.assertRaisesRegex(ValueError, '`method_name` must be defined'):
            updater.replace_method_name(signature_key='foobar', method_name='')
        with self.assertRaisesRegex(ValueError, "MetaGraphDef associated with tags \\['gpu'\\] could not be found"):
            updater.replace_method_name(signature_key='bar', method_name='classify', tags=['gpu'])
        with self.assertRaisesRegex(ValueError, "MetaGraphDef associated with tags \\['serve'\\] does not have a signature_def with key: 'baz'"):
            updater.replace_method_name(signature_key='baz', method_name='classify', tags=['serve'])
if __name__ == '__main__':
    test.main()