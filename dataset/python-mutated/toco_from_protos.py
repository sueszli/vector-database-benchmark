"""Python console command to invoke TOCO from serialized protos."""
import argparse
import sys
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import _pywrap_toco_api
from absl import app
FLAGS = None

def execute(unused_args):
    if False:
        while True:
            i = 10
    'Runs the converter.'
    with open(FLAGS.model_proto_file, 'rb') as model_file:
        model_str = model_file.read()
    with open(FLAGS.toco_proto_file, 'rb') as toco_file:
        toco_str = toco_file.read()
    with open(FLAGS.model_input_file, 'rb') as input_file:
        input_str = input_file.read()
    debug_info_str = None
    if FLAGS.debug_proto_file:
        with open(FLAGS.debug_proto_file, 'rb') as debug_info_file:
            debug_info_str = debug_info_file.read()
    enable_mlir_converter = FLAGS.enable_mlir_converter
    output_str = _pywrap_toco_api.TocoConvert(model_str, toco_str, input_str, False, debug_info_str, enable_mlir_converter)
    open(FLAGS.model_output_file, 'wb').write(output_str)
    sys.exit(0)

def main():
    if False:
        while True:
            i = 10
    global FLAGS
    parser = argparse.ArgumentParser(description='Invoke toco using protos as input.')
    parser.add_argument('model_proto_file', type=str, help='File containing serialized proto that describes the model.')
    parser.add_argument('toco_proto_file', type=str, help='File containing serialized proto describing how TOCO should run.')
    parser.add_argument('model_input_file', type=str, help='Input model is read from this file.')
    parser.add_argument('model_output_file', type=str, help='Result of applying TOCO conversion is written here.')
    parser.add_argument('--debug_proto_file', type=str, default='', help='File containing serialized `GraphDebugInfo` proto that describes logging information.')
    parser.add_argument('--enable_mlir_converter', action='store_true', help='Boolean indicating whether to enable MLIR-based conversion instead of TOCO conversion. (default False)')
    (FLAGS, unparsed) = parser.parse_known_args()
    app.run(main=execute, argv=[sys.argv[0]] + unparsed)
if __name__ == '__main__':
    main()