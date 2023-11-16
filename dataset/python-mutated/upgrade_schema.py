"""Upgrade script to move from pre-release schema to new schema.

Usage examples:

bazel run tensorflow/lite/schema/upgrade_schema -- in.json out.json
bazel run tensorflow/lite/schema/upgrade_schema -- in.bin out.bin
bazel run tensorflow/lite/schema/upgrade_schema -- in.bin out.json
bazel run tensorflow/lite/schema/upgrade_schema -- in.json out.bin
bazel run tensorflow/lite/schema/upgrade_schema -- in.tflite out.tflite
"""
import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tensorflow as tf
from tensorflow.python.platform import resource_loader
parser = argparse.ArgumentParser(description='Script to move TFLite models from pre-release schema to new schema.')
parser.add_argument('input', type=str, help='Input TensorFlow lite file in `.json`, `.bin` or `.tflite` format.')
parser.add_argument('output', type=str, help='Output json or bin TensorFlow lite model compliant with the new schema. Extension must be `.json`, `.bin` or `.tflite`.')

@contextlib.contextmanager
def TemporaryDirectoryResource():
    if False:
        for i in range(10):
            print('nop')
    temporary = tempfile.mkdtemp()
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary)

class Converter:
    """Converts TensorFlow flatbuffer models from old to new version of schema.

  This can convert between any version to the latest version. It uses
  an incremental upgrade strategy to go from version to version.

  Usage:
    converter = Converter()
    converter.Convert("a.tflite", "a.json")
    converter.Convert("b.json", "b.tflite")
  """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        paths_to_try = ['../../../../flatbuffers/flatc', '../../../../external/flatbuffers/flatc']
        for p in paths_to_try:
            self._flatc_path = resource_loader.get_path_to_datafile(p)
            if os.path.exists(self._flatc_path):
                break

        def FindSchema(base_name):
            if False:
                for i in range(10):
                    print('nop')
            return resource_loader.get_path_to_datafile('%s' % base_name)
        self._schemas = [(0, FindSchema('schema_v0.fbs'), True, self._Upgrade0To1), (1, FindSchema('schema_v1.fbs'), True, self._Upgrade1To2), (2, FindSchema('schema_v2.fbs'), True, self._Upgrade2To3), (3, FindSchema('schema_v3.fbs'), False, None)]
        self._schemas.sort()
        (self._new_version, self._new_schema) = self._schemas[-1][:2]
        self._upgrade_dispatch = {version: dispatch for (version, unused1, unused2, dispatch) in self._schemas}

    def _Read(self, input_file, schema, raw_binary=False):
        if False:
            print('Hello World!')
        'Read a tflite model assuming the given flatbuffer schema.\n\n    If `input_file` is in bin, then we must use flatc to convert the schema\n    from binary to json.\n\n    Args:\n      input_file: a binary (flatbuffer) or json file to read from. Extension\n        must  be `.tflite`, `.bin`, or `.json` for FlatBuffer Binary or\n        FlatBuffer JSON.\n      schema: which schema to use for reading\n      raw_binary: whether to assume raw_binary (versions previous to v3)\n        that lacked file_identifier require this.\n\n    Raises:\n      RuntimeError: 1. When flatc cannot be invoked.\n                    2. When json file does not exists.\n      ValueError: When the extension is not json or bin.\n\n    Returns:\n      A dictionary representing the read tflite model.\n    '
        raw_binary = ['--raw-binary'] if raw_binary else []
        with TemporaryDirectoryResource() as tempdir:
            basename = os.path.basename(input_file)
            (basename_no_extension, extension) = os.path.splitext(basename)
            if extension in ['.bin', '.tflite']:
                returncode = subprocess.call([self._flatc_path, '-t', '--strict-json', '--defaults-json'] + raw_binary + ['-o', tempdir, schema, '--', input_file])
                if returncode != 0:
                    raise RuntimeError('flatc failed to convert from binary to json.')
                json_file = os.path.join(tempdir, basename_no_extension + '.json')
                if not os.path.exists(json_file):
                    raise RuntimeError('Could not find %r' % json_file)
            elif extension == '.json':
                json_file = input_file
            else:
                raise ValueError('Invalid extension on input file %r' % input_file)
            return json.load(open(json_file))

    def _Write(self, data, output_file):
        if False:
            print('Hello World!')
        'Output a json or bin version of the flatbuffer model.\n\n    Args:\n      data: Dict representing the TensorFlow Lite model to write.\n      output_file: filename to write the converted flatbuffer to. (json,\n        tflite, or bin extension is required).\n    Raises:\n      ValueError: When the extension is not json or bin\n      RuntimeError: When flatc fails to convert json data to binary.\n    '
        (_, extension) = os.path.splitext(output_file)
        with TemporaryDirectoryResource() as tempdir:
            if extension == '.json':
                json.dump(data, open(output_file, 'w'), sort_keys=True, indent=2)
            elif extension in ['.tflite', '.bin']:
                input_json = os.path.join(tempdir, 'temp.json')
                with open(input_json, 'w') as fp:
                    json.dump(data, fp, sort_keys=True, indent=2)
                returncode = subprocess.call([self._flatc_path, '-b', '--defaults-json', '--strict-json', '-o', tempdir, self._new_schema, input_json])
                if returncode != 0:
                    raise RuntimeError('flatc failed to convert upgraded json to binary.')
                shutil.copy(os.path.join(tempdir, 'temp.tflite'), output_file)
            else:
                raise ValueError('Invalid extension on output file %r' % output_file)

    def _Upgrade0To1(self, data):
        if False:
            print('Hello World!')
        'Upgrade data from Version 0 to Version 1.\n\n    Changes: Added subgraphs (which contains a subset of formally global\n    entries).\n\n    Args:\n      data: Dictionary representing the TensorFlow lite data to be upgraded.\n        This will be modified in-place to be an upgraded version.\n    '
        subgraph = {}
        for key_to_promote in ['tensors', 'operators', 'inputs', 'outputs']:
            subgraph[key_to_promote] = data[key_to_promote]
            del data[key_to_promote]
        data['subgraphs'] = [subgraph]

    def _Upgrade1To2(self, data):
        if False:
            while True:
                i = 10
        'Upgrade data from Version 1 to Version 2.\n\n    Changes: Rename operators to Conform to NN API.\n\n    Args:\n      data: Dictionary representing the TensorFlow lite data to be upgraded.\n        This will be modified in-place to be an upgraded version.\n    Raises:\n      ValueError: Throws when model builtins are numeric rather than symbols.\n    '

        def RemapOperator(opcode_name):
            if False:
                print('Hello World!')
            'Go from old schema op name to new schema op name.\n\n      Args:\n        opcode_name: String representing the ops (see :schema.fbs).\n      Returns:\n        Converted opcode_name from V1 to V2.\n      '
            old_name_to_new_name = {'CONVOLUTION': 'CONV_2D', 'DEPTHWISE_CONVOLUTION': 'DEPTHWISE_CONV_2D', 'AVERAGE_POOL': 'AVERAGE_POOL_2D', 'MAX_POOL': 'MAX_POOL_2D', 'L2_POOL': 'L2_POOL_2D', 'SIGMOID': 'LOGISTIC', 'L2NORM': 'L2_NORMALIZATION', 'LOCAL_RESPONSE_NORM': 'LOCAL_RESPONSE_NORMALIZATION', 'Basic_RNN': 'RNN'}
            return old_name_to_new_name[opcode_name] if opcode_name in old_name_to_new_name else opcode_name

        def RemapOperatorType(operator_type):
            if False:
                i = 10
                return i + 15
            'Remap operator structs from old names to new names.\n\n      Args:\n        operator_type: String representing the builtin operator data type\n          string. (see :schema.fbs).\n      Raises:\n        ValueError: When the model has consistency problems.\n      Returns:\n        Upgraded builtin operator data type as a string.\n      '
            old_to_new = {'PoolOptions': 'Pool2DOptions', 'DepthwiseConvolutionOptions': 'DepthwiseConv2DOptions', 'ConvolutionOptions': 'Conv2DOptions', 'LocalResponseNormOptions': 'LocalResponseNormalizationOptions', 'BasicRNNOptions': 'RNNOptions'}
            return old_to_new[operator_type] if operator_type in old_to_new else operator_type
        for subgraph in data['subgraphs']:
            for ops in subgraph['operators']:
                ops['builtin_options_type'] = RemapOperatorType(ops['builtin_options_type'])
        for operator_code in data['operator_codes']:
            if not isinstance(operator_code['builtin_code'], type(u'')):
                raise ValueError('builtin_code %r is non-string. this usually means your model has consistency problems.' % operator_code['builtin_code'])
            operator_code['builtin_code'] = RemapOperator(operator_code['builtin_code'])

    def _Upgrade2To3(self, data):
        if False:
            print('Hello World!')
        'Upgrade data from Version 2 to Version 3.\n\n    Changed actual read-only tensor data to be in a buffers table instead\n    of inline with the tensor.\n\n    Args:\n      data: Dictionary representing the TensorFlow lite data to be upgraded.\n        This will be modified in-place to be an upgraded version.\n    '
        buffers = [{'data': []}]
        for subgraph in data['subgraphs']:
            if 'tensors' not in subgraph:
                continue
            for tensor in subgraph['tensors']:
                if 'data_buffer' not in tensor:
                    tensor['buffer'] = 0
                else:
                    if tensor['data_buffer']:
                        tensor[u'buffer'] = len(buffers)
                        buffers.append({'data': tensor['data_buffer']})
                    else:
                        tensor['buffer'] = 0
                    del tensor['data_buffer']
        data['buffers'] = buffers

    def _PerformUpgrade(self, data):
        if False:
            print('Hello World!')
        'Manipulate the `data` (parsed JSON) based on changes in format.\n\n    This incrementally will upgrade from version to version within data.\n\n    Args:\n      data: Dictionary representing the TensorFlow data. This will be upgraded\n        in place.\n    '
        while data['version'] < self._new_version:
            self._upgrade_dispatch[data['version']](data)
            data['version'] += 1

    def Convert(self, input_file, output_file):
        if False:
            return 10
        'Perform schema conversion from input_file to output_file.\n\n    Args:\n      input_file: Filename of TensorFlow Lite data to convert from. Must\n        be `.json` or `.bin` extension files for JSON or Binary forms of\n        the TensorFlow FlatBuffer schema.\n      output_file: Filename to write to. Extension also must be `.json`\n        or `.bin`.\n\n    Raises:\n      RuntimeError: Generated when none of the upgrader supported schemas\n        matche the `input_file` data.\n    '
        for (version, schema, raw_binary, _) in self._schemas:
            try:
                data_candidate = self._Read(input_file, schema, raw_binary)
            except RuntimeError:
                continue
            if 'version' not in data_candidate:
                data_candidate['version'] = 1
            elif data_candidate['version'] == 0:
                data_candidate['version'] = 1
            if data_candidate['version'] == version:
                self._PerformUpgrade(data_candidate)
                self._Write(data_candidate, output_file)
                return
        raise RuntimeError('No schema that the converter understands worked with the data file you provided.')

def main(argv):
    if False:
        while True:
            i = 10
    del argv
    Converter().Convert(FLAGS.input, FLAGS.output)
if __name__ == '__main__':
    (FLAGS, unparsed) = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)