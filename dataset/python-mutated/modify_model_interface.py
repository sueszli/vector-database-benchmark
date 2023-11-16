"""Modify a quantized model's interface from float to integer."""
from absl import app
from absl import flags
from tensorflow.lite.tools.optimize.python import modify_model_interface_constants as mmi_constants
from tensorflow.lite.tools.optimize.python import modify_model_interface_lib as mmi_lib
FLAGS = flags.FLAGS
flags.DEFINE_string('input_tflite_file', None, 'Full path name to the input TFLite file.')
flags.DEFINE_string('output_tflite_file', None, 'Full path name to the output TFLite file.')
flags.DEFINE_enum('input_type', mmi_constants.DEFAULT_STR_TYPE, mmi_constants.STR_TYPES, 'Modified input integer interface type.')
flags.DEFINE_enum('output_type', mmi_constants.DEFAULT_STR_TYPE, mmi_constants.STR_TYPES, 'Modified output integer interface type.')
flags.mark_flag_as_required('input_tflite_file')
flags.mark_flag_as_required('output_tflite_file')

def main(_):
    if False:
        for i in range(10):
            print('nop')
    input_type = mmi_constants.STR_TO_TFLITE_TYPES[FLAGS.input_type]
    output_type = mmi_constants.STR_TO_TFLITE_TYPES[FLAGS.output_type]
    mmi_lib.modify_model_interface(FLAGS.input_tflite_file, FLAGS.output_tflite_file, input_type, output_type)
    print('Successfully modified the model input type from FLOAT to {input_type} and output type from FLOAT to {output_type}.'.format(input_type=FLAGS.input_type, output_type=FLAGS.output_type))
if __name__ == '__main__':
    app.run(main)