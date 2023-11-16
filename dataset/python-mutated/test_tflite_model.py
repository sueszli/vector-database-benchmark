"""Test a tflite model using random input data."""
from __future__ import print_function
from absl import flags
import numpy as np
import tensorflow as tf
flags.DEFINE_string('model_path', None, 'Path to model.')
FLAGS = flags.FLAGS

def main(_):
    if False:
        print('Hello World!')
    flags.mark_flag_as_required('model_path')
    interpreter = tf.lite.Interpreter(model_path=FLAGS.model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    print('input_details:', input_details)
    output_details = interpreter.get_output_details()
    print('output_details:', output_details)
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
if __name__ == '__main__':
    tf.app.run()