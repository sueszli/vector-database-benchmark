"""Strips all nonessential strings from a TFLite file."""
from absl import app
from absl import flags
from tensorflow.lite.tools import flatbuffer_utils
FLAGS = flags.FLAGS
flags.DEFINE_string('input_tflite_file', None, 'Full path name to the input TFLite file.')
flags.DEFINE_string('output_tflite_file', None, 'Full path name to the output stripped TFLite file.')
flags.mark_flag_as_required('input_tflite_file')
flags.mark_flag_as_required('output_tflite_file')

def main(_):
    if False:
        for i in range(10):
            print('nop')
    model = flatbuffer_utils.read_model(FLAGS.input_tflite_file)
    flatbuffer_utils.strip_strings(model)
    flatbuffer_utils.write_model(model, FLAGS.output_tflite_file)
if __name__ == '__main__':
    app.run(main)