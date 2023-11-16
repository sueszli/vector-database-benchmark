"""Reverses xxd dump, i.e, converts a C++ source file back to a TFLite file.

This script is used to convert a model from a C++ source file (dumped with xxd)
back to it's original TFLite file format in order to analyze it with either a
model visualizer like Netron (https://github.com/lutzroeder/netron) or to
evaluate the model using the Python TensorFlow Lite Interpreter API.

The xxd command to dump the TFLite file to a C++ source file looks like:
xxd -i model_data.tflite > model_data.cc

"""
from absl import app
from absl import flags
from tensorflow.lite.tools import flatbuffer_utils
FLAGS = flags.FLAGS
flags.DEFINE_string('input_cc_file', None, 'Full path name to the input C++ source file.')
flags.DEFINE_string('output_tflite_file', None, 'Full path name to the output TFLite file.')
flags.mark_flag_as_required('input_cc_file')
flags.mark_flag_as_required('output_tflite_file')

def main(_):
    if False:
        i = 10
        return i + 15
    model = flatbuffer_utils.xxd_output_to_object(FLAGS.input_cc_file)
    flatbuffer_utils.write_model(model, FLAGS.output_tflite_file)
if __name__ == '__main__':
    app.run(main)