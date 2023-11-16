"""Tests for modify_model_interface_lib.py."""
import os
import numpy as np
import tensorflow as tf
from tensorflow.lite.tools.optimize.python import modify_model_interface_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

def build_tflite_model_with_full_integer_quantization(supported_ops=tf.lite.OpsSet.TFLITE_BUILTINS_INT8):
    if False:
        for i in range(10):
            print('nop')
    input_size = 3
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(input_size,), dtype=tf.float32), tf.keras.layers.Dense(units=5, activation=tf.nn.relu), tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        if False:
            return 10
        for i in range(10):
            yield [np.array([i] * input_size, dtype=np.float32)]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [supported_ops]
    tflite_model = converter.convert()
    return tflite_model

class ModifyModelInterfaceTest(test_util.TensorFlowTestCase):

    def testInt8Interface(self):
        if False:
            print('Hello World!')
        temp_dir = self.get_temp_dir()
        initial_file = os.path.join(temp_dir, 'initial_model.tflite')
        final_file = os.path.join(temp_dir, 'final_model.tflite')
        initial_model = build_tflite_model_with_full_integer_quantization()
        with open(initial_file, 'wb') as model_file:
            model_file.write(initial_model)
        modify_model_interface_lib.modify_model_interface(initial_file, final_file, tf.int8, tf.int8)
        initial_interpreter = tf.lite.Interpreter(model_path=initial_file)
        initial_interpreter.allocate_tensors()
        final_interpreter = tf.lite.Interpreter(model_path=final_file)
        final_interpreter.allocate_tensors()
        initial_input_dtype = initial_interpreter.get_input_details()[0]['dtype']
        initial_output_dtype = initial_interpreter.get_output_details()[0]['dtype']
        final_input_dtype = final_interpreter.get_input_details()[0]['dtype']
        final_output_dtype = final_interpreter.get_output_details()[0]['dtype']
        self.assertEqual(initial_input_dtype, np.float32)
        self.assertEqual(initial_output_dtype, np.float32)
        self.assertEqual(final_input_dtype, np.int8)
        self.assertEqual(final_output_dtype, np.int8)

    def testInt16Interface(self):
        if False:
            while True:
                i = 10
        temp_dir = self.get_temp_dir()
        initial_file = os.path.join(temp_dir, 'initial_model.tflite')
        final_file = os.path.join(temp_dir, 'final_model.tflite')
        initial_model = build_tflite_model_with_full_integer_quantization(supported_ops=tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8)
        with open(initial_file, 'wb') as model_file:
            model_file.write(initial_model)
        modify_model_interface_lib.modify_model_interface(initial_file, final_file, tf.int16, tf.int16)
        initial_interpreter = tf.lite.Interpreter(model_path=initial_file)
        initial_interpreter.allocate_tensors()
        final_interpreter = tf.lite.Interpreter(model_path=final_file)
        final_interpreter.allocate_tensors()
        initial_input_dtype = initial_interpreter.get_input_details()[0]['dtype']
        initial_output_dtype = initial_interpreter.get_output_details()[0]['dtype']
        final_input_dtype = final_interpreter.get_input_details()[0]['dtype']
        final_output_dtype = final_interpreter.get_output_details()[0]['dtype']
        self.assertEqual(initial_input_dtype, np.float32)
        self.assertEqual(initial_output_dtype, np.float32)
        self.assertEqual(final_input_dtype, np.int16)
        self.assertEqual(final_output_dtype, np.int16)

    def testUInt8Interface(self):
        if False:
            for i in range(10):
                print('nop')
        temp_dir = self.get_temp_dir()
        initial_file = os.path.join(temp_dir, 'initial_model.tflite')
        final_file = os.path.join(temp_dir, 'final_model.tflite')
        initial_model = build_tflite_model_with_full_integer_quantization()
        with open(initial_file, 'wb') as model_file:
            model_file.write(initial_model)
        modify_model_interface_lib.modify_model_interface(initial_file, final_file, tf.uint8, tf.uint8)
        initial_interpreter = tf.lite.Interpreter(model_path=initial_file)
        initial_interpreter.allocate_tensors()
        final_interpreter = tf.lite.Interpreter(model_path=final_file)
        final_interpreter.allocate_tensors()
        initial_input_dtype = initial_interpreter.get_input_details()[0]['dtype']
        initial_output_dtype = initial_interpreter.get_output_details()[0]['dtype']
        final_input_dtype = final_interpreter.get_input_details()[0]['dtype']
        final_output_dtype = final_interpreter.get_output_details()[0]['dtype']
        self.assertEqual(initial_input_dtype, np.float32)
        self.assertEqual(initial_output_dtype, np.float32)
        self.assertEqual(final_input_dtype, np.uint8)
        self.assertEqual(final_output_dtype, np.uint8)
if __name__ == '__main__':
    test.main()