"""Tests for lite.py."""
import io
import logging
import os
import tempfile
from absl.testing import parameterized
import numpy as np
from tensorflow import keras
from tensorflow.lite.python import conversion_metadata_schema_py_generated as metadata_fb
from tensorflow.lite.python import lite
from tensorflow.lite.python import lite_constants
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import util
from tensorflow.lite.python.convert import ConverterError
from tensorflow.lite.python.convert import mlir_quantize
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.util import get_conversion_metadata
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.variables import global_variables_initializer as _global_variables_initializer
from tensorflow.python.platform import gfile
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.training.training_util import write_graph

class LiteTest(test_util.TensorFlowTestCase):
    """Base class of all the tests in this module."""

class TestModels(LiteTest):

    def assertValidDebugInfo(self, debug_info):
        if False:
            i = 10
            return i + 15
        'Verify the DebugInfo is valid.'
        file_names = set()
        for file_path in debug_info.files:
            file_names.add(os.path.basename(file_path))
        self.assertIn('lite_test.py', file_names)
        self.assertNotIn('lite_v2_test.py', file_names)

class FromConstructor(TestModels):

    def testInvalidConstructor(self):
        if False:
            for i in range(10):
                print('nop')
        message = 'If input_tensors and output_tensors are None, both input_arrays_with_shape and output_arrays|control_output_arrays must be defined.'
        with self.assertRaises(ValueError) as error:
            lite.TFLiteConverter(None, None, [], input_arrays_with_shape=[('input', [3, 9])]).convert()
        self.assertEqual(message, str(error.exception))
        with self.assertRaises(ValueError) as error:
            lite.TFLiteConverter(None, [], None, output_arrays=['output']).convert()
        self.assertEqual(message, str(error.exception))

    def testValidConstructor(self):
        if False:
            return 10
        converter = lite.TFLiteConverter(None, None, None, input_arrays_with_shape=[('input', [3, 9])], output_arrays=['output'])
        self.assertFalse(converter._has_valid_tensors())
        self.assertEqual(converter.get_input_arrays(), ['input'])
        with self.assertRaises(ValueError) as error:
            converter._set_batch_size(1)
        self.assertEqual('The batch size cannot be set for this model. Please use input_shapes parameter.', str(error.exception))
        converter = lite.TFLiteConverter(None, ['input_tensor'], ['output_tensor'])
        self.assertTrue(converter._has_valid_tensors())

    def testRedundantArgumentsWarning(self):
        if False:
            print('Hello World!')
        'Test if the warning message when there are redundant arguments.'
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor')
            out_tensor = math_ops.add(in_tensor, in_tensor, name='add')
            sess = session.Session()
        frozen_graph_def = convert_to_constants.convert_variables_to_constants_from_session_graph(sess, sess.graph_def, ['add'])
        log = io.StringIO()
        handler = logging.StreamHandler(log)
        logging.root.addHandler(handler)
        converter = lite.TFLiteConverter(frozen_graph_def, [in_tensor], [out_tensor], [('in_tensor', [2, 16, 16, 3])], ['add'])
        input_warning_message = 'input_arrays_with_shape will be ignored'
        output_warning_message = 'output_arrays will be ignored'
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        self.assertIn(input_warning_message, log.getvalue())
        self.assertIn(output_warning_message, log.getvalue())
        logging.root.removeHandler(handler)

    def testShapeOverriding(self):
        if False:
            i = 10
            return i + 15
        'Test a shape overriding case via the constructor.'
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor')
            math_ops.add(in_tensor, in_tensor, name='add')
            sess = session.Session()
        frozen_graph_def = convert_to_constants.convert_variables_to_constants_from_session_graph(sess, sess.graph_def, ['add'])
        converter = lite.TFLiteConverter(frozen_graph_def, None, None, [('in_tensor', [2, 16, 16, 3])], ['add'])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('in_tensor', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([2, 16, 16, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual('add', output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([2, 16, 16, 3], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

    def testPartialShapeOverriding(self):
        if False:
            while True:
                i = 10
        'Test a partial shape overriding case via the constructor.'
        with ops.Graph().as_default():
            in_tensor_a = array_ops.placeholder(shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor_a')
            in_tensor_b = array_ops.placeholder(shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor_b')
            math_ops.add(in_tensor_a, in_tensor_b, name='add')
            sess = session.Session()
        frozen_graph_def = convert_to_constants.convert_variables_to_constants_from_session_graph(sess, sess.graph_def, ['add'])
        converter = lite.TFLiteConverter(frozen_graph_def, None, None, [('in_tensor_a', [2, 16, 16, 3])], ['add'])
        with self.assertRaises(ConverterError):
            converter.convert()

    def testInvalidShapeOverriding(self):
        if False:
            return 10
        'Test an invalid shape overriding case via the constructor.'
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[None, 16, 16, 3], dtype=dtypes.float32, name='in_tensor')
            math_ops.add(in_tensor, in_tensor, name='add')
            sess = session.Session()
        frozen_graph_def = convert_to_constants.convert_variables_to_constants_from_session_graph(sess, sess.graph_def, ['add'])
        converter = lite.TFLiteConverter(frozen_graph_def, None, None, [('wrong_tensor', [2, 16, 16, 3])], ['add'])
        with self.assertRaises(ConverterError):
            converter.convert()

class FromSessionTest(TestModels, parameterized.TestCase):

    def testFloatModel(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('Placeholder', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual('add', output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

    def testFloatModelQuantizedInput(self):
        if False:
            return 10
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        converter.inference_input_type = dtypes.uint8
        converter.inference_type = dtypes.float32
        converter.quantized_input_stats = {'Placeholder': (0.0, 1.0)}
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('Placeholder', input_details[0]['name'])
        self.assertEqual(np.uint8, input_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
        self.assertEqual((1.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual('add', output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

    def testForgottenCallToAllocateTensors(self):
        if False:
            return 10
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        input_index = interpreter.get_input_details()[0]['index']
        dummy_tensor = np.ones(shape=[1, 16, 16, 3], dtype=np.float32)
        with self.assertRaises(ValueError):
            interpreter.set_tensor(input_index, dummy_tensor)

    @parameterized.named_parameters(('_INT8InputOutput', False, False, dtypes.int8), ('_UINT8InputOutput', False, False, dtypes.uint8), ('_INT16Quantize_INT16InputOutput', False, True, dtypes.int16), ('_IntOnly_INT8InputOutput', True, False, dtypes.int8), ('_IntOnly_UINT8InputOutput', True, False, dtypes.uint8), ('_IntOnly_INT16Quantize_INT16InputOutput', True, True, dtypes.int16), ('_IntOnly_INT8InputOutputMlirQuant', True, False, dtypes.int8, True), ('_IntOnly_UINT8InputOutputMlirQuant', True, False, dtypes.uint8, True))
    def testIntegerQuantizationWithUnsupportedOps(self, is_int_only, is_int16_quantize, inference_input_output_type, enable_mlir_quantizer=False):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            in_tensor_a = array_ops.placeholder(shape=[3], dtype=dtypes.float32)
            in_tensor_b = array_ops.placeholder(shape=[3], dtype=dtypes.float32)
            left = math_ops.ceil(in_tensor_a)
            out_tensor_b = math_ops.tanh(in_tensor_b)
            add = math_ops.add(left, out_tensor_b)
            out_tensor_a = math_ops.ceil(add)
            sess = session.Session()

        def calibration_gen():
            if False:
                while True:
                    i = 10
            for _ in range(5):
                yield [np.random.uniform(-1, 1, size=3).astype(np.float32), np.random.uniform(-1, 1, size=3).astype(np.float32)]
        quantized_converter = lite.TFLiteConverter.from_session(sess, [in_tensor_a, in_tensor_b], [out_tensor_a, out_tensor_b])
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.representative_dataset = calibration_gen
        if is_int_only:
            if is_int16_quantize:
                quantized_converter.target_spec.supported_ops = [lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8, lite.OpsSet.TFLITE_BUILTINS]
            else:
                quantized_converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8, lite.OpsSet.TFLITE_BUILTINS]
        elif is_int16_quantize:
            quantized_converter.target_spec.supported_ops = [lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8, lite.OpsSet.TFLITE_BUILTINS]
        else:
            quantized_converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS]
        quantized_converter.inference_input_type = inference_input_output_type
        quantized_converter.inference_output_type = inference_input_output_type
        quantized_converter.experimental_new_quantizer = enable_mlir_quantizer
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        expected_dtype = inference_input_output_type.as_numpy_dtype
        expected_ceil_dtype = expected_dtype if enable_mlir_quantizer else dtypes.float32
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 2)
        self.assertEqual(input_details[0]['dtype'], expected_ceil_dtype)
        self.assertEqual(input_details[1]['dtype'], expected_dtype)
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 2)
        self.assertEqual(output_details[0]['dtype'], expected_ceil_dtype)
        self.assertEqual(output_details[1]['dtype'], expected_dtype)

    @parameterized.named_parameters(('_PerChannelQuant', False, False), ('_PerChannelMlirQuant', False, True), ('_PerTensorQuant', True, False), ('_PerTensorMlirQuant', True, True), ('_PerChannelMlirDynamicRangeQuant', False, False, False), ('_PerTensorMlirDynamicRangeQuant', True, False, False))
    def testDisablePerChannelQuantization(self, disable_per_channel=False, enable_mlir_quantizer=False, representative_dataset=True):
        if False:
            print('Hello World!')
        k_conv_name = 'Conv2D1'
        k_num_filters = 38
        with ops.Graph().as_default():
            (inp, output, calibration_gen) = self._getIntegerQuantizeModel(k_num_filters)
            sess = session.Session()
        quantized_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        if representative_dataset:
            quantized_converter.representative_dataset = calibration_gen
        quantized_converter.experimental_new_quantizer = enable_mlir_quantizer
        if disable_per_channel:
            quantized_converter._experimental_disable_per_channel = disable_per_channel
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        detail = next((d for d in interpreter.get_tensor_details() if d['name'] == k_conv_name))
        quant_params = detail['quantization_parameters']
        expected_num_params = 1 if disable_per_channel else k_num_filters
        self.assertLen(quant_params['scales'], expected_num_params)
        self.assertLen(quant_params['zero_points'], expected_num_params)

    def testString(self):
        if False:
            return 10
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[4], dtype=dtypes.string)
            out_tensor = array_ops.reshape(in_tensor, shape=[2, 2])
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('Placeholder', input_details[0]['name'])
        self.assertEqual(np.string_, input_details[0]['dtype'])
        self.assertAllEqual([4], input_details[0]['shape'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual('Reshape', output_details[0]['name'])
        self.assertEqual(np.string_, output_details[0]['dtype'])
        self.assertAllEqual([2, 2], output_details[0]['shape'])

    def testIntermediateInputArray(self):
        if False:
            for i in range(10):
                print('nop')
        'Convert a model from an intermediate input array.'
        with ops.Graph().as_default():
            in_tensor_init = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            in_tensor_final = in_tensor_init + in_tensor_init
            out_tensor = in_tensor_final + in_tensor_final
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor_final], [out_tensor])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('add', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual('add_1', output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

    def testSizeNoneInvalid(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        converter.experimental_new_converter = False
        with self.assertRaises(ValueError) as error:
            converter.convert()
        self.assertEqual("Provide an input shape for input array 'Placeholder'.", str(error.exception))

    def testScalarValid(self):
        if False:
            return 10
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(dtype=dtypes.float32, shape=[])
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('Placeholder', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertEmpty(input_details[0]['shape'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual('add', output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertEmpty(input_details[0]['shape'])
        test_input = np.array(4.0, dtype=np.float32)
        expected_output = np.array(8.0, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(expected_output, output_data)

    def testSizeInvalid(self):
        if False:
            return 10
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, None, 16, 3], dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        converter.experimental_new_converter = False
        with self.assertRaises(ValueError) as error:
            converter.convert()
        self.assertEqual("None is only supported in the 1st dimension. Tensor 'Placeholder' has invalid shape '[1, None, 16, 3]'.", str(error.exception))

    def testSizeNone(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, None, 16, 3], dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('Placeholder', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 1, 16, 3], input_details[0]['shape'])
        self.assertAllEqual([1, -1, 16, 3], input_details[0]['shape_signature'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        with self.assertRaises(RuntimeError) as error:
            interpreter.resize_tensor_input(0, [3, 16, 16, 3], strict=True)
        self.assertIn('ResizeInputTensorStrict only allows mutating unknown dimensions identified by -1.', str(error.exception))
        interpreter.resize_tensor_input(0, [1, 16, 16, 3], strict=True)
        interpreter.allocate_tensors()
        test_input = np.full([1, 16, 16, 3], 1.0, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
        self.assertAllEqual([1, -1, 16, 3], input_details[0]['shape_signature'])
        output_details = interpreter.get_output_details()
        self.assertAllEqual([1, -1, 16, 3], output_details[0]['shape_signature'])

    def testResizeTensorInputStrict(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        with self.assertRaises(RuntimeError) as error:
            interpreter.resize_tensor_input(0, [3, 16, 16, 3], strict=True)
        self.assertIn('ResizeInputTensorStrict only allows mutating unknown dimensions identified by -1.', str(error.exception))
        interpreter.resize_tensor_input(0, [1, 16, 16, 3], strict=True)
        interpreter.allocate_tensors()

    def testBatchSizeValid(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[None, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('Placeholder', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual('add', output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

    def testBatchSizeNonZero(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            in_tensor_1 = array_ops.placeholder(shape=[None, 4], dtype=dtypes.float32, name='input1')
            in_tensor_2 = array_ops.placeholder(shape=[4, 10], dtype=dtypes.float32, name='input2')
            out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2)
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor_1, in_tensor_2], [out_tensor])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 2)
        self.assertEqual('input1', input_details[0]['name'])
        self.assertAllEqual([1, 4], input_details[0]['shape'])
        self.assertEqual('input2', input_details[1]['name'])
        self.assertAllEqual([4, 10], input_details[1]['shape'])

    def testFreezeGraph(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            var = variable_scope.get_variable('weights', shape=[1, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = nn_ops.top_k(in_tensor + var, name='top_k')[1]
            sess = session.Session()
            sess.run(_global_variables_initializer())
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('Placeholder', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual('top_k:1', output_details[0]['name'])
        self.assertEqual(np.int32, output_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 1], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

    def testGraphviz(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        converter.output_format = lite_constants.GRAPHVIZ_DOT
        graphviz_output = converter.convert()
        self.assertIsNotNone(graphviz_output)

    def testDumpGraphviz(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        graphviz_dir = self.get_temp_dir()
        converter.dump_graphviz_dir = graphviz_dir
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        num_items_graphviz = len(os.listdir(graphviz_dir))
        self.assertIsNotNone(num_items_graphviz)
        self.assertIsNotNone(os.path.exists(os.path.join(graphviz_dir, 'toco_AT_IMPORT.dot')))
        self.assertIsNotNone(os.path.exists(os.path.join(graphviz_dir, 'toco_AFTER_TRANSFORMATIONS.dot')))

    def testDumpConversionSummary(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        log_dir = self.get_temp_dir()
        converter.conversion_summary_dir = log_dir
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        self.assertNotEmpty(os.listdir(log_dir))

    def testDumpConversionSummaryWithOldConverter(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        converter.experimental_new_converter = False
        log_dir = self.get_temp_dir()
        converter.conversion_summary_dir = log_dir
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        num_items_conversion_summary = len(os.listdir(log_dir))
        self.assertEqual(num_items_conversion_summary, 0)

    def testQuantizeDynamicRange(self):
        if False:
            return 10
        np.random.seed(0)
        with ops.Graph().as_default():
            in_tensor_1 = array_ops.placeholder(shape=[33, 33], dtype=dtypes.float32, name='inputA')
            in_tensor_2 = constant_op.constant(np.random.uniform(low=-10.0, high=10.0, size=(33, 33)), shape=[33, 33], dtype=dtypes.float32, name='inputB')
            out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2, name='output')
            sess = session.Session()
        float_converter = lite.TFLiteConverter.from_session(sess, [in_tensor_1], [out_tensor])
        float_tflite_model = float_converter.convert()
        self.assertIsNotNone(float_tflite_model)
        quantized_converter = lite.TFLiteConverter.from_session(sess, [in_tensor_1], [out_tensor])
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

    def testQuantizeDynamicRangeDeprecatedPostTrainingQuantizeAttribute(self):
        if False:
            return 10
        with ops.Graph().as_default():
            in_tensor_1 = array_ops.placeholder(shape=[33, 33], dtype=dtypes.float32, name='inputA')
            in_tensor_2 = constant_op.constant(np.random.uniform(low=-10.0, high=10.0, size=(33, 33)), shape=[33, 33], dtype=dtypes.float32, name='inputB')
            out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2, name='output')
            sess = session.Session()
        quantized_converter = lite.TFLiteConverter.from_session(sess, [in_tensor_1], [out_tensor])
        self.assertFalse(quantized_converter.post_training_quantize)
        quantized_converter.post_training_quantize = True
        self.assertTrue(quantized_converter.post_training_quantize)
        self.assertEqual(quantized_converter.optimizations, [lite.Optimize.DEFAULT])
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)

    def _getIntegerQuantizeModel(self, num_filters=16):
        if False:
            i = 10
            return i + 15
        np.random.seed(0)
        inp = array_ops.placeholder(dtype=dtypes.float32, shape=(1, 5, 5, 3), name='input')
        conv = nn_ops.conv2d(inp, filter=array_ops.ones([3, 3, 3, num_filters]), strides=[1, 1, 1, 1], padding='SAME')
        output = nn_ops.relu(conv, name='output')

        def calibration_gen():
            if False:
                while True:
                    i = 10
            for _ in range(5):
                yield [np.random.uniform(-1, 1, size=(1, 5, 5, 3)).astype(np.float32)]
        return (inp, output, calibration_gen)

    def testQuantizeInt8AllowFloat(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            (inp, output, calibration_gen) = self._getIntegerQuantizeModel()
            sess = session.Session()
        float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
        float_tflite_model = float_converter.convert()
        self.assertIsNotNone(float_tflite_model)
        metadata = get_conversion_metadata(float_tflite_model)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.environment.tensorflowVersion.decode('utf-8'), versions.__version__)
        self.assertEqual(metadata.environment.apiVersion, 1)
        self.assertEqual(metadata.environment.modelType, metadata_fb.ModelType.TF_SESSION)
        self.assertEqual(metadata.options.allowCustomOps, False)
        self.assertEqual(metadata.options.enableSelectTfOps, False)
        self.assertEqual(metadata.options.forceSelectTfOps, False)
        self.assertAllEqual([], metadata.options.modelOptimizationModes)
        quantized_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.representative_dataset = calibration_gen
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        metadata = get_conversion_metadata(quantized_tflite_model)
        self.assertIsNotNone(metadata)
        self.assertAllEqual([metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER], metadata.options.modelOptimizationModes)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual(np.float32, input_details[0]['dtype'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

    @parameterized.named_parameters(('UseTfliteBuiltinsInt', [lite.OpsSet.TFLITE_BUILTINS_INT8], [metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER]), ('UseTfliteBuiltinsInt16', [lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8], [metadata_fb.ModelOptimizationMode.PTQ_INT16]))
    def testQuantizeInt8And16x8(self, supported_ops, expected_opt_modes):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            (inp, output, calibration_gen) = self._getIntegerQuantizeModel()
            sess = session.Session()
        float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
        float_tflite_model = float_converter.convert()
        self.assertIsNotNone(float_tflite_model)
        quantized_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.target_spec.supported_ops = supported_ops
        quantized_converter.representative_dataset = calibration_gen
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        metadata = get_conversion_metadata(quantized_tflite_model)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.environment.tensorflowVersion.decode('utf-8'), versions.__version__)
        self.assertEqual(metadata.environment.apiVersion, 1)
        self.assertEqual(metadata.environment.modelType, metadata_fb.ModelType.TF_SESSION)
        self.assertEqual(metadata.options.allowCustomOps, False)
        self.assertEqual(metadata.options.enableSelectTfOps, False)
        self.assertEqual(metadata.options.forceSelectTfOps, False)
        self.assertAllEqual(expected_opt_modes, metadata.options.modelOptimizationModes)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual(np.float32, input_details[0]['dtype'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

    def testQuantizeInt8InputOutput(self):
        if False:
            return 10
        with ops.Graph().as_default():
            (inp, output, calibration_gen) = self._getIntegerQuantizeModel()
            sess = session.Session()
        float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
        float_tflite_model = float_converter.convert()
        self.assertIsNotNone(float_tflite_model)
        quantized_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
        quantized_converter.inference_input_type = dtypes.int8
        quantized_converter.inference_output_type = dtypes.int8
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.representative_dataset = calibration_gen
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual(np.int8, input_details[0]['dtype'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual(np.int8, output_details[0]['dtype'])
        self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

    def testInvalidQuantizeInt8(self):
        if False:
            i = 10
            return i + 15
        np.random.seed(0)
        with ops.Graph().as_default():
            in_tensor_1 = array_ops.placeholder(shape=[33, 33], dtype=dtypes.float32, name='inputA')
            in_tensor_2 = constant_op.constant(np.random.uniform(low=-10.0, high=10.0, size=(33, 33)), shape=[33, 33], dtype=dtypes.float32, name='inputB')
            out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2, name='output')
            sess = session.Session()
        quantized_converter = lite.TFLiteConverter.from_session(sess, [in_tensor_1], [out_tensor])
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.target_spec.supported_types = [dtypes.int8]
        with self.assertRaises(ValueError) as error:
            quantized_converter.convert()
        self.assertEqual('For full integer quantization, a `representative_dataset` must be specified.', str(error.exception))

    def testQuantizeUInt8(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            in_tensor_1 = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputA')
            in_tensor_2 = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputB')
            out_tensor = array_ops.fake_quant_with_min_max_args(in_tensor_1 + in_tensor_2, min=0.0, max=1.0, name='output')
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor_1, in_tensor_2], [out_tensor])
        converter.inference_type = dtypes.uint8
        converter.quantized_input_stats = {'inputA': (0.0, 1.0), 'inputB': (0.0, 1.0)}
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 2)
        self.assertEqual('inputA', input_details[0]['name'])
        self.assertEqual(np.uint8, input_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
        self.assertEqual((1.0, 0.0), input_details[0]['quantization'])
        self.assertEqual('inputB', input_details[1]['name'])
        self.assertEqual(np.uint8, input_details[1]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[1]['shape'])
        self.assertEqual((1.0, 0.0), input_details[1]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual(np.uint8, output_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
        self.assertGreater(output_details[0]['quantization'][0], 0)

    def testQuantizeUInt8UsingDefaultRangeStats(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        converter.inference_type = dtypes.uint8
        converter.quantized_input_stats = {'Placeholder': (0.0, 1.0)}
        converter.default_ranges_stats = (0, 6)
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('Placeholder', input_details[0]['name'])
        self.assertEqual(np.uint8, input_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
        self.assertEqual((1.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual('add', output_details[0]['name'])
        self.assertEqual(np.uint8, output_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
        self.assertGreater(output_details[0]['quantization'][0], 0)

    @parameterized.named_parameters(('UseRepresentativeData', True, False, True, False, False, False, [metadata_fb.ModelOptimizationMode.PTQ_FLOAT16]), ('NoRepresentativeData', False, False, True, False, False, False, [metadata_fb.ModelOptimizationMode.PTQ_FLOAT16]), ('SampleDataIncludeInt8', True, True, False, False, True, False, [metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER]), ('SampleDataIncludeInt8Quant', True, True, False, False, True, True, [metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER]))
    def testQuantizeFloat16(self, use_rep_data, include_int8, is_float16_quantized, is_float16_accumulation, is_post_training_quantized, enable_mlir_quantizer, expected_opt_modes):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            (inp, output, calibration_gen) = self._getIntegerQuantizeModel()
            sess = session.Session()
        bias_idx = 1
        bias_name = 'Conv2D'
        float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
        float_tflite_model = float_converter.convert()
        self.assertIsNotNone(float_tflite_model)
        interpreter = Interpreter(model_content=float_tflite_model)
        interpreter.allocate_tensors()
        self.assertEqual(interpreter.get_tensor_details()[bias_idx]['name'], bias_name)
        self.assertEqual(interpreter.get_tensor_details()[bias_idx]['dtype'], dtypes.float32)
        quantized_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
        quantized_converter.experimental_new_quantizer = enable_mlir_quantizer
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.target_spec.supported_types = [dtypes.float16]
        if include_int8:
            quantized_converter.target_spec.supported_types.append(dtypes.int8)
        if use_rep_data:
            quantized_converter.representative_dataset = calibration_gen
        if is_float16_accumulation:
            quantized_converter.target_spec.experimental_supported_accumulation_type = dtypes.float16
        else:
            quantized_tflite_model = quantized_converter.convert()
            self.assertIsNotNone(quantized_tflite_model)
            metadata = get_conversion_metadata(quantized_tflite_model)
            self.assertIsNotNone(metadata)
            self.assertAllEqual(expected_opt_modes, metadata.options.modelOptimizationModes)
            interpreter = Interpreter(model_content=quantized_tflite_model)
            interpreter.allocate_tensors()
            bias_tensor = [tensor for tensor in interpreter.get_tensor_details() if tensor['name'] == bias_name]
            self.assertLen(bias_tensor, 1)
            if is_float16_quantized:
                self.assertEqual(bias_tensor[0]['dtype'], dtypes.float16)
            elif is_post_training_quantized:
                self.assertEqual(bias_tensor[0]['dtype'], dtypes.int32)
            else:
                raise ValueError('Invalid test options.')

    def testInvalidQuantizeFloat16(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            (inp, output, _) = self._getIntegerQuantizeModel()
            sess = session.Session()
        quantized_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.target_spec.supported_types = [dtypes.float16]
        quantized_converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
        with self.assertRaises(ValueError) as error:
            quantized_converter.convert()
        self.assertEqual('As full integer quantization has been enabled by setting `target_spec.supported_ops`={tf.lite.OpsSet.TFLITE_BUILTINS_INT8}, thus `target_spec.supported_types` should be left uninitizalized or set to {tf.int8}.', str(error.exception))

    @parameterized.named_parameters(('InferenceType_INT8', dtypes.int8), ('InferenceType_UINT8', dtypes.uint8))
    def testInvalidQuantizeQATModelRequiresInputStats(self, quantized_type):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = array_ops.fake_quant_with_min_max_args(in_tensor + in_tensor, min=0.0, max=1.0)
            sess = session.Session()
        quantized_converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        with self.assertRaises(ValueError) as error:
            quantized_converter.inference_type = quantized_type
            quantized_converter.convert()
        self.assertEqual('The `quantized_input_stats` flag must be defined when either `inference_type` flag or `inference_input_type` flag is set to tf.int8 or tf.uint8. Currently, `inference_type=tf.{}` and `inference_input_type=None`.'.format(quantized_type.name), str(error.exception))
        with self.assertRaises(ValueError) as error:
            quantized_converter.inference_type = dtypes.float32
            quantized_converter.inference_input_type = quantized_type
            quantized_converter.convert()
        self.assertEqual('The `quantized_input_stats` flag must be defined when either `inference_type` flag or `inference_input_type` flag is set to tf.int8 or tf.uint8. Currently, `inference_type=tf.float32` and `inference_input_type=tf.{}`.'.format(quantized_type.name), str(error.exception))
        quantized_converter.inference_type = quantized_type
        quantized_converter.inference_input_type = quantized_type
        input_arrays = quantized_converter.get_input_arrays()
        quantized_converter.quantized_input_stats = {input_arrays[0]: (0.0, 1.0)}
        quantized_converter.convert()

    def testInvalidQuantizeQATModelMissingInputStats(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            in_tensor_1 = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputA')
            in_tensor_2 = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputB')
            out_tensor = array_ops.fake_quant_with_min_max_args(in_tensor_1 + in_tensor_2, min=0.0, max=1.0, name='output')
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor_1, in_tensor_2], [out_tensor])
        converter.inference_type = dtypes.uint8
        converter.quantized_input_stats = {'inputA': (0.0, 1.0)}
        with self.assertRaises(ValueError) as error:
            converter.convert()
        self.assertEqual("Quantization input stats are not available for input tensors 'inputB'.", str(error.exception))

    def testTrainingTimeAndPostTrainingCalibrateAndQuantize(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            (inp, output, calibration_gen) = self._getIntegerQuantizeModel()
            sess = session.Session()
        float_converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
        float_tflite_model = float_converter.convert()
        self.assertIsNotNone(float_tflite_model)
        converter = lite.TFLiteConverter.from_session(sess, [inp], [output])
        converter.inference_type = dtypes.int8
        converter.inference_input_type = dtypes.float32
        converter.inference_output_type = dtypes.float32
        input_arrays = converter.get_input_arrays()
        converter.quantized_input_stats = {input_arrays[0]: (0.0, 1.0)}
        converter.optimizations = [lite.Optimize.DEFAULT]
        converter.representative_dataset = calibration_gen
        converter.experimental_new_quantizer = True
        quantized_tflite_model = converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        self.assertLess(len(quantized_tflite_model), len(float_tflite_model))
        converter._experimental_calibrate_only = True
        calibrated_tflite = converter.convert()
        quantized_tflite_model = mlir_quantize(calibrated_tflite, fully_quantize=True)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertEqual(np.int8, input_details[0]['dtype'])
        self.assertEqual((1.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertEqual(np.int8, output_details[0]['dtype'])

    def testFloatTocoConverter(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests deprecated test TocoConverter.'
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TocoConverter.from_session(sess, [in_tensor], [out_tensor])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

    def testMultipleOutputNodeNames(self):
        if False:
            while True:
                i = 10
        'Tests converting a graph with an op that have multiple outputs.'
        with ops.Graph().as_default():
            input_tensor = array_ops.placeholder(shape=[4], dtype=dtypes.float32)
            (out0, out1, out2, out3) = array_ops.split(input_tensor, [1, 1, 1, 1], axis=0)
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [input_tensor], [out0, out1, out2, out3])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        interpreter.set_tensor(input_details[0]['index'], np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        interpreter.invoke()
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 4)
        self.assertEqual(1.0, interpreter.get_tensor(output_details[0]['index']))
        self.assertEqual(2.0, interpreter.get_tensor(output_details[1]['index']))
        self.assertEqual(3.0, interpreter.get_tensor(output_details[2]['index']))
        self.assertEqual(4.0, interpreter.get_tensor(output_details[3]['index']))

    @test_util.run_in_graph_and_eager_modes
    def testFunctions(self):
        if False:
            return 10
        'Tests tf.function in 1.X.'

        @def_function.function
        def plus_placeholder(x, placeholder):
            if False:
                print('Hello World!')
            return x + placeholder
        with ops.Graph().as_default():
            placeholder = array_ops.placeholder(dtype=dtypes.float32, shape=[1], name='input')
            variable_node = variables.Variable(1.0, name='variable_node')
            defun_node = plus_placeholder(variable_node, placeholder)
            output_node = math_ops.multiply(defun_node, 2.0, name='output_node')
            sess = session.Session()
            sess.run(variables.variables_initializer([variable_node]))
        converter = lite.TFLiteConverter.from_session(sess, [placeholder], [output_node])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('input', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual('output_node', output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

    def testInferenceInputOutputTypeFloatDefault(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('Placeholder', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual('add', output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])

    def testInferenceInputOutputTypeQuantizedUint8Default(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = array_ops.fake_quant_with_min_max_args(in_tensor + in_tensor, min=0.0, max=1.0, name='output')
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        converter.inference_type = dtypes.uint8
        converter.quantized_input_stats = {'Placeholder': (0.0, 1.0)}
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('Placeholder', input_details[0]['name'])
        self.assertEqual(np.uint8, input_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual('output', output_details[0]['name'])
        self.assertEqual(np.uint8, output_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])

    def testReusingConverterWithDifferentPostTrainingQuantization(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            out_tensor = array_ops.fake_quant_with_min_max_args(in_tensor + in_tensor, min=0.0, max=1.0, name='output')
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        converter.post_training_quantize = True
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        converter.post_training_quantize = False
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)

    def testResizeWithShape(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[None, None], dtype=dtypes.float32)
            in_tensor2 = [[1, 2], [3, 4]]
            out_tensor = array_ops.reshape(in_tensor2, array_ops.shape(in_tensor))
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertAllEqual([1, 1], input_details[0]['shape'])
        self.assertAllEqual([-1, -1], input_details[0]['shape_signature'])
        interpreter.resize_tensor_input(0, [4])
        interpreter.allocate_tensors()
        interpreter.invoke()
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual(np.int32, output_details[0]['dtype'])
        self.assertAllEqual([4], output_details[0]['shape'])
        output_data = interpreter.get_tensor(output_details[0]['index'])
        self.assertAllEqual([1, 2, 3, 4], output_data)

    def testResizingIntermediateDynamicTensor(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            input_tensor = array_ops.placeholder(shape=[1, 1], dtype=dtypes.float32)
            input2_tensor = array_ops.placeholder(shape=[1], dtype=dtypes.float32)
            neg = math_ops.negative(input2_tensor)
            padding = array_ops.placeholder(shape=[2, 2], dtype=dtypes.int32)
            output_tensor = array_ops.pad(input_tensor, padding) + neg
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [input_tensor, padding, input2_tensor], [output_tensor])
        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[1]['index'], np.array([[1, 1], [1, 1]], dtype=np.int32))
        interpreter.invoke()
        interpreter.set_tensor(input_details[1]['index'], np.array([[2, 2], [2, 2]], dtype=np.int32))
        interpreter.invoke()

    def testGraphDebugInfo(self):
        if False:
            print('Hello World!')
        'Test a session has debug info captured.'

        @def_function.function
        def plus_placeholder(x, placeholder):
            if False:
                for i in range(10):
                    print('nop')
            return x + placeholder
        with ops.Graph().as_default():
            placeholder = array_ops.placeholder(dtype=dtypes.float32, shape=[1], name='input')
            variable_node = variables.Variable(1.0, name='variable_node')
            defun_node = plus_placeholder(variable_node, placeholder)
            output_node = math_ops.multiply(defun_node, 2.0, name='output_node')
            sess = session.Session()
            sess.run(variables.variables_initializer([variable_node]))
        converter = lite.TFLiteConverter.from_session(sess, [placeholder], [output_node])
        converter.convert()
        self.assertValidDebugInfo(converter._debug_info)
        func = sess.graph.as_graph_def().library.function[0].signature.name
        self.assertIn('add@' + func, repr(converter._debug_info))

    def testOutputOnlyModel(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            out_tensor = random_ops.random_normal(shape=[3])
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [], [out_tensor])
        converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS, lite.OpsSet.SELECT_TF_OPS]
        self.assertTrue(converter._has_valid_tensors())
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)

class FromFrozenGraphFile(LiteTest):

    def testFloat(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            _ = in_tensor + in_tensor
            sess = session.Session()
        graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
        write_graph(sess.graph_def, '', graph_def_file, False)
        sess.close()
        converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file, ['Placeholder'], ['add'])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('Placeholder', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual('add', output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

    def testFloatWithShapesArray(self):
        if False:
            print('Hello World!')
        'Test a shape overriding case.'
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[None, 16, 16, 3], dtype=dtypes.float32)
            _ = in_tensor + in_tensor
            sess = session.Session()
        graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
        write_graph(sess.graph_def, '', graph_def_file, False)
        sess.close()
        converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file, ['Placeholder'], ['add'], input_shapes={'Placeholder': [2, 16, 16, 3]})
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertAllEqual([2, 16, 16, 3], input_details[0]['shape'])

    def testInvalidShapesArray(self):
        if False:
            return 10
        'Test an invalid shape overriding case, which has a wrong input name.'
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[None, 16, 16, 3], dtype=dtypes.float32)
            _ = in_tensor + in_tensor
            sess = session.Session()
        graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
        write_graph(sess.graph_def, '', graph_def_file, False)
        sess.close()
        with self.assertRaises(ValueError):
            lite.TFLiteConverter.from_frozen_graph(graph_def_file, ['Placeholder'], ['add'], input_shapes={'wrong_input': [2, 16, 16, 3]})

    def testPartialShapesArray(self):
        if False:
            return 10
        'Test a shape overriding case, with the only one input among two.'
        with ops.Graph().as_default():
            a = array_ops.placeholder(shape=[None, 16, 16, 3], dtype=dtypes.float32, name='a')
            b = array_ops.placeholder(shape=[None, 16, 16, 3], dtype=dtypes.float32, name='b')
            _ = math_ops.add(a, b, name='add')
            sess = session.Session()
        graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
        write_graph(sess.graph_def, '', graph_def_file, False)
        sess.close()
        converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file, ['a', 'b'], ['add'], input_shapes={'a': [2, 16, 16, 3]})
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 2)
        self.assertAllEqual([2, 16, 16, 3], input_details[0]['shape'])
        self.assertAllEqual([1, 16, 16, 3], input_details[1]['shape'])

    def testFreezeGraph(self):
        if False:
            return 10
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            var = variable_scope.get_variable('weights', shape=[1, 16, 16, 3], dtype=dtypes.float32)
            _ = in_tensor + var
            sess = session.Session()
        graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
        write_graph(sess.graph_def, '', graph_def_file, False)
        sess.close()
        with self.assertRaises(ValueError) as error:
            lite.TFLiteConverter.from_frozen_graph(graph_def_file, ['Placeholder'], ['add'])
        self.assertEqual('Please freeze the graph using freeze_graph.py.', str(error.exception))

    def testPbtxt(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            _ = in_tensor + in_tensor
            sess = session.Session()
        graph_def_file = os.path.join(self.get_temp_dir(), 'model.pbtxt')
        write_graph(sess.graph_def, '', graph_def_file, True)
        sess.close()
        converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file, ['Placeholder'], ['add'])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('Placeholder', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual('add', output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

    def testInvalidFileNotFound(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(IOError) as error:
            lite.TFLiteConverter.from_frozen_graph('invalid_file', ['Placeholder'], ['add'])
        self.assertEqual("File 'invalid_file' does not exist.", str(error.exception))

    def testInvalidFileBadData(self):
        if False:
            print('Hello World!')
        graph_def_file = os.path.join(self.get_temp_dir(), 'invalid_file')
        with gfile.Open(graph_def_file, 'wb') as temp_file:
            temp_file.write('bad data')
            temp_file.flush()
        with self.assertRaises(IOError) as error:
            lite.TFLiteConverter.from_frozen_graph(graph_def_file, ['Placeholder'], ['add'])
        self.assertEqual("Unable to parse input file '{}'.".format(graph_def_file), str(error.exception))

    def testFloatTocoConverter(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            _ = in_tensor + in_tensor
            sess = session.Session()
        graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
        write_graph(sess.graph_def, '', graph_def_file, False)
        sess.close()
        converter = lite.TocoConverter.from_frozen_graph(graph_def_file, ['Placeholder'], ['add'])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

    def testGraphDebugInfo(self):
        if False:
            print('Hello World!')
        "Test a frozen graph doesn't have debug info captured."
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            _ = in_tensor + in_tensor
            sess = session.Session()
        graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
        write_graph(sess.graph_def, '', graph_def_file, False)
        sess.close()
        converter = lite.TocoConverter.from_frozen_graph(graph_def_file, ['Placeholder'], ['add'])
        converter.convert()
        self.assertFalse(converter._debug_info)

    def testExcludeConversionMetadata(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3], dtype=dtypes.float32)
            _ = in_tensor + in_tensor
            sess = session.Session()
        graph_def_file = os.path.join(self.get_temp_dir(), 'model.pb')
        write_graph(sess.graph_def, '', graph_def_file, False)
        sess.close()
        converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file, ['Placeholder'], ['add'])
        converter.exclude_conversion_metadata = True
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        metadata = get_conversion_metadata(tflite_model)
        self.assertIsNone(metadata)

class FromFrozenGraphObjectDetection(LiteTest):

    def _initObjectDetectionArgs(self):
        if False:
            for i in range(10):
                print('nop')
        filename = resource_loader.get_path_to_datafile('testdata/tflite_graph.pb')
        if not os.path.exists(filename):
            filename = os.path.join(resource_loader.get_root_dir_with_all_resources(), '../tflite_mobilenet_ssd_quant_protobuf/tflite_graph.pb')
            if not os.path.exists(filename):
                raise IOError("File '{0}' does not exist.".format(filename))
        self._graph_def_file = filename
        self._input_arrays = ['normalized_input_image_tensor']
        self._output_arrays = ['TFLite_Detection_PostProcess', 'TFLite_Detection_PostProcess:1', 'TFLite_Detection_PostProcess:2', 'TFLite_Detection_PostProcess:3']
        self._input_shapes = {'normalized_input_image_tensor': [1, 300, 300, 3]}

    def testTFLiteGraphDef(self):
        if False:
            print('Hello World!')
        self._initObjectDetectionArgs()
        converter = lite.TFLiteConverter.from_frozen_graph(self._graph_def_file, self._input_arrays, self._output_arrays, self._input_shapes)
        converter.allow_custom_ops = True
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('normalized_input_image_tensor', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 300, 300, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 4)
        self.assertEqual('TFLite_Detection_PostProcess', output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 10, 4], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])
        self.assertEqual('TFLite_Detection_PostProcess:1', output_details[1]['name'])
        self.assertAllEqual([1, 10], output_details[1]['shape'])
        self.assertEqual('TFLite_Detection_PostProcess:2', output_details[2]['name'])
        self.assertAllEqual([1, 10], output_details[2]['shape'])
        self.assertEqual('TFLite_Detection_PostProcess:3', output_details[3]['name'])
        self.assertAllEqual([1], output_details[3]['shape'])

    def testTFLiteGraphDefWithControlOutput(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[5, 5], dtype=dtypes.float32, name='input')
            out_tensor = in_tensor + in_tensor
            logging_ops.print_v2(out_tensor)
            sess = session.Session()
        converter = lite.TFLiteConverter(sess.graph_def, input_tensors=None, output_tensors=None, input_arrays_with_shape=[('input', [5, 5])], output_arrays=None, experimental_debug_info_func=None)
        converter._control_output_arrays = ['PrintV2']
        converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS, lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        model = util._convert_model_from_bytearray_to_object(tflite_model)
        self.assertEqual(model.operatorCodes[0].builtinCode, schema_fb.BuiltinOperator.ADD)
        self.assertEqual(model.operatorCodes[1].builtinCode, schema_fb.BuiltinOperator.CUSTOM)
        self.assertEqual(model.operatorCodes[1].customCode, b'FlexStringFormat')
        self.assertEqual(model.operatorCodes[2].builtinCode, schema_fb.BuiltinOperator.CUSTOM)
        self.assertEqual(model.operatorCodes[2].customCode, b'FlexPrintV2')
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('input', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([5, 5], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 0)

    def testModifyIOToUint8(self):
        if False:
            while True:
                i = 10
        self._initObjectDetectionArgs()

        def representative_dataset_gen():
            if False:
                for i in range(10):
                    print('nop')
            for _ in range(2):
                yield [np.random.uniform(low=0, high=1, size=(1, 300, 300, 3)).astype(np.float32)]
        converter = lite.TFLiteConverter.from_frozen_graph(self._graph_def_file, self._input_arrays, self._output_arrays, self._input_shapes)
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = {lite.OpsSet.TFLITE_BUILTINS_INT8}
        converter.inference_type = dtypes.int8
        converter.inference_input_type = dtypes.uint8
        converter.inference_output_type = dtypes.uint8
        converter.experimental_new_quantizer = True
        converter.quantized_input_stats = {'normalized_input_image_tensor': (0.0, 1.0)}
        converter.allow_custom_ops = True
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        model = util._convert_model_from_bytearray_to_object(tflite_model)
        quant_opcode_idxs = util.get_quantize_opcode_idx(model)
        subgraph = model.subgraphs[0]
        tensors = subgraph.tensors
        operators = subgraph.operators
        for op in operators:
            if op.opcodeIndex in quant_opcode_idxs:
                input_type = util._convert_tflite_enum_type_to_tf_type(tensors[op.inputs[0]].type)
                if op.outputs[0] in subgraph.outputs:
                    self.assertEqual(input_type, dtypes.float32)

class FromSavedModelTest(TestModels):

    def _createSavedModel(self, shape):
        if False:
            i = 10
            return i + 15
        'Create a simple SavedModel.'
        saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_savedmodel')
        with ops.Graph().as_default():
            with session.Session() as sess:
                in_tensor_1 = array_ops.placeholder(shape=shape, dtype=dtypes.float32, name='inputB')
                in_tensor_2 = array_ops.placeholder(shape=shape, dtype=dtypes.float32, name='inputA')
                out_tensor = in_tensor_1 + in_tensor_2
                inputs = {'x': in_tensor_1, 'y': in_tensor_2}
                outputs = {'z': out_tensor}
                saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
        return saved_model_dir

    def testSimpleModel(self):
        if False:
            return 10
        'Test a SavedModel.'
        saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])
        converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 2)
        self.assertStartsWith(input_details[0]['name'], 'inputA')
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        self.assertStartsWith(input_details[1]['name'], 'inputB')
        self.assertEqual(np.float32, input_details[1]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[1]['shape'])
        self.assertEqual((0.0, 0.0), input_details[1]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertStartsWith(output_details[0]['name'], 'add')
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

    def testNoneBatchSize(self):
        if False:
            i = 10
            return i + 15
        "Test a SavedModel, with None in input tensor's shape."
        saved_model_dir = self._createSavedModel(shape=[None, 16, 16, 3])
        converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 2)
        self.assertStartsWith(input_details[0]['name'], 'inputA')
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        self.assertStartsWith(input_details[1]['name'], 'inputB')
        self.assertEqual(np.float32, input_details[1]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[1]['shape'])
        self.assertEqual((0.0, 0.0), input_details[1]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertStartsWith(output_details[0]['name'], 'add')
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

    def testOrderInputArrays(self):
        if False:
            for i in range(10):
                print('nop')
        'Test a SavedModel ordering of input arrays.'
        saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])
        converter = lite.TFLiteConverter.from_saved_model(saved_model_dir, input_arrays=['inputB', 'inputA'])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 2)
        self.assertStartsWith(input_details[0]['name'], 'inputA')
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        self.assertStartsWith(input_details[1]['name'], 'inputB')
        self.assertEqual(np.float32, input_details[1]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], input_details[1]['shape'])
        self.assertEqual((0.0, 0.0), input_details[1]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertStartsWith(output_details[0]['name'], 'add')
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 16, 16, 3], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

    def testShapeOverriding(self):
        if False:
            for i in range(10):
                print('nop')
        'Test a SavedModel with the input_shapes arugment.'
        saved_model_dir = self._createSavedModel(shape=[None, 16, 16, 3])
        converter = lite.TFLiteConverter.from_saved_model(saved_model_dir, input_shapes={'inputA': [2, 16, 16, 3], 'inputB': [2, 16, 16, 3]})
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 2)
        self.assertStartsWith(input_details[0]['name'], 'inputA')
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([2, 16, 16, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        self.assertStartsWith(input_details[1]['name'], 'inputB')
        self.assertEqual(np.float32, input_details[1]['dtype'])
        self.assertAllEqual([2, 16, 16, 3], input_details[1]['shape'])
        self.assertEqual((0.0, 0.0), input_details[1]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertStartsWith(output_details[0]['name'], 'add')
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([2, 16, 16, 3], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

    def testWrongInputShapes(self):
        if False:
            while True:
                i = 10
        'Test a SavedModel with a wrong name in the input_shapes argument.'
        saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])
        with self.assertRaises(ValueError):
            lite.TFLiteConverter.from_saved_model(saved_model_dir, input_arrays=['inputA'], input_shapes={'wrong_input': [1, 16, 16, 3]})

    def testSubsetInputShaapes(self):
        if False:
            i = 10
            return i + 15
        'Test a SavedModel with a subset of the input array names of the model.'
        saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])
        converter = lite.TFLiteConverter.from_saved_model(saved_model_dir, input_arrays=['inputA'], input_shapes={'inputA': [1, 16, 16, 3]})
        with self.assertRaises(ConverterError):
            _ = converter.convert()
        converter = lite.TFLiteConverter.from_saved_model(saved_model_dir, input_arrays=['inputA'], input_shapes={'inputA': None})
        with self.assertRaises(ConverterError):
            _ = converter.convert()

    def testSimpleModelTocoConverter(self):
        if False:
            print('Hello World!')
        'Test a SavedModel with deprecated TocoConverter.'
        saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])
        converter = lite.TocoConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

    def testGraphDebugInfo(self):
        if False:
            i = 10
            return i + 15
        'Test a SavedModel has debug info captured.'
        self.skipTest('b/221093690: The debug info is not from self._createSavedModel(), but from saved_model.loader_impl().')
        saved_model_dir = self._createSavedModel(shape=[1, 16, 16, 3])
        converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.convert()
        self.assertValidDebugInfo(converter._debug_info)

class MyAddLayer(keras.layers.Layer):

    def __init__(self, increment, **kwargs):
        if False:
            return 10
        super(MyAddLayer, self).__init__(**kwargs)
        self._increment = increment

    def call(self, inputs):
        if False:
            while True:
                i = 10
        return inputs + self._increment

    def get_config(self):
        if False:
            i = 10
            return i + 15
        config = super(MyAddLayer, self).get_config()
        config['increment'] = self._increment
        return config

class FromKerasFile(TestModels, parameterized.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(FromKerasFile, self).setUp()
        self._keras_file = None
        self._custom_objects = None
        if not context.executing_eagerly():
            keras.backend.clear_session()

    def tearDown(self):
        if False:
            while True:
                i = 10
        if self._keras_file:
            os.remove(self._keras_file)
        super(FromKerasFile, self).tearDown()

    def _getSequentialModel(self, include_custom_layer=False):
        if False:
            return 10
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(2, input_shape=(3,)))
        if include_custom_layer:
            model.add(MyAddLayer(1.0))
        model.add(keras.layers.RepeatVector(3))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
        model.compile(loss=keras.losses.MSE, optimizer='sgd', metrics=[keras.metrics.categorical_accuracy], sample_weight_mode='temporal')
        x = np.random.random((1, 3))
        y = np.random.random((1, 3, 3))
        model.train_on_batch(x, y)
        model.predict(x)
        try:
            (fd, self._keras_file) = tempfile.mkstemp('.h5')
            keras.models.save_model(model, self._keras_file)
        finally:
            os.close(fd)
        if include_custom_layer:
            self._custom_objects = {'MyAddLayer': MyAddLayer}

    @parameterized.named_parameters(('_graph', context.graph_mode), ('_eager', context.eager_mode))
    def testSequentialModel(self, test_context):
        if False:
            for i in range(10):
                print('nop')
        'Test a Sequential tf.keras model with default inputs.'
        with test_context():
            self._getSequentialModel()
            converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
            tflite_model = converter.convert()
            self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEndsWith(input_details[0]['name'], 'dense_input')
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 3, 3], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])
        input_data = np.array([[1, 2, 3]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        tflite_result = interpreter.get_tensor(output_details[0]['index'])
        keras_model = keras.models.load_model(self._keras_file)
        keras_result = keras_model.predict(input_data)
        np.testing.assert_almost_equal(tflite_result, keras_result, 5)

    @parameterized.named_parameters(('_graph', context.graph_mode), ('_eager', context.eager_mode))
    def testCustomLayer(self, test_context):
        if False:
            return 10
        'Test a Sequential tf.keras model with default inputs.'
        with test_context():
            self._getSequentialModel(include_custom_layer=True)
            converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file, custom_objects=self._custom_objects)
            tflite_model = converter.convert()
            self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_data = np.array([[1, 2, 3]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        tflite_result = interpreter.get_tensor(output_details[0]['index'])
        keras_model = keras.models.load_model(self._keras_file, custom_objects=self._custom_objects)
        keras_result = keras_model.predict(input_data)
        np.testing.assert_almost_equal(tflite_result, keras_result, 5)

    def testSequentialModelInputArray(self):
        if False:
            print('Hello World!')
        'Test a Sequential tf.keras model testing input arrays argument.'
        ops.disable_eager_execution()
        self._getSequentialModel()
        with self.assertRaises(ValueError) as error:
            lite.TFLiteConverter.from_keras_model_file(self._keras_file, input_arrays=['invalid-input'])
        self.assertEqual("Invalid tensors 'invalid-input' were found.", str(error.exception))
        converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file, input_arrays=['dense_input'])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)

    def testSequentialModelInputShape(self):
        if False:
            i = 10
            return i + 15
        'Test a Sequential tf.keras model testing input shapes argument.'
        self._getSequentialModel()
        with self.assertRaises(ValueError) as error:
            converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file, input_shapes={'invalid-input': [2, 3]})
        self.assertEqual("Invalid tensor 'invalid-input' found in tensor shapes map.", str(error.exception))
        converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file, input_shapes={'dense_input': [2, 3]})
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEndsWith(input_details[0]['name'], 'dense_input')
        self.assertAllEqual([2, 3], input_details[0]['shape'])

    def testSequentialModelOutputArray(self):
        if False:
            return 10
        'Test a Sequential tf.keras model testing output arrays argument.'
        ops.disable_eager_execution()
        self._getSequentialModel()
        with self.assertRaises(ValueError) as error:
            lite.TFLiteConverter.from_keras_model_file(self._keras_file, output_arrays=['invalid-output'])
        self.assertEqual("Invalid tensors 'invalid-output' were found.", str(error.exception))
        converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file, output_arrays=['time_distributed/Reshape_1'])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)

    @parameterized.named_parameters(('_graph', context.graph_mode), ('_eager', context.eager_mode))
    def testFunctionalModel(self, test_context):
        if False:
            while True:
                i = 10
        'Test a Functional tf.keras model with default inputs.'
        with test_context():
            inputs = keras.layers.Input(shape=(3,), name='input')
            x = keras.layers.Dense(2)(inputs)
            output = keras.layers.Dense(3)(x)
            model = keras.models.Model(inputs, output)
            model.compile(loss=keras.losses.MSE, optimizer='sgd', metrics=[keras.metrics.categorical_accuracy])
            x = np.random.random((1, 3))
            y = np.random.random((1, 3))
            model.train_on_batch(x, y)
            model.predict(x)
            (fd, self._keras_file) = tempfile.mkstemp('.h5')
            try:
                keras.models.save_model(model, self._keras_file)
            finally:
                os.close(fd)
            converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
            tflite_model = converter.convert()
            self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('input', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 3], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])
        input_data = np.array([[1, 2, 3]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        tflite_result = interpreter.get_tensor(output_details[0]['index'])
        keras_model = keras.models.load_model(self._keras_file)
        keras_result = keras_model.predict(input_data)
        np.testing.assert_almost_equal(tflite_result, keras_result, 5)

    def _getFunctionalModelMultipleInputs(self):
        if False:
            for i in range(10):
                print('nop')
        a = keras.layers.Input(shape=(3,), name='input_a')
        b = keras.layers.Input(shape=(3,), name='input_b')
        dense = keras.layers.Dense(4, name='dense')
        c = dense(a)
        d = dense(b)
        e = keras.layers.Dropout(0.5, name='dropout')(c)
        model = keras.models.Model([a, b], [d, e])
        model.compile(loss=keras.losses.MSE, optimizer='sgd', metrics=[keras.metrics.mae], loss_weights=[1.0, 0.5])
        input_a_np = np.random.random((10, 3))
        input_b_np = np.random.random((10, 3))
        output_d_np = np.random.random((10, 4))
        output_e_np = np.random.random((10, 4))
        model.train_on_batch([input_a_np, input_b_np], [output_d_np, output_e_np])
        model.predict([input_a_np, input_b_np], batch_size=5)
        (fd, self._keras_file) = tempfile.mkstemp('.h5')
        try:
            keras.models.save_model(model, self._keras_file)
        finally:
            os.close(fd)

    def testFunctionalModelMultipleInputs(self):
        if False:
            for i in range(10):
                print('nop')
        'Test a Functional tf.keras model with multiple inputs and outputs.'
        self._getFunctionalModelMultipleInputs()
        converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 2)
        self.assertEndsWith(input_details[0]['name'], 'input_a')
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        self.assertEndsWith(input_details[1]['name'], 'input_b')
        self.assertEqual(np.float32, input_details[1]['dtype'])
        self.assertAllEqual([1, 3], input_details[1]['shape'])
        self.assertEqual((0.0, 0.0), input_details[1]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 2)
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 4], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])
        self.assertEqual(np.float32, output_details[1]['dtype'])
        self.assertAllEqual([1, 4], output_details[1]['shape'])
        self.assertEqual((0.0, 0.0), output_details[1]['quantization'])

    def testShapeOverriding(self):
        if False:
            i = 10
            return i + 15
        'Test a Functional tf.keras model with input shape overriding.'
        self._getFunctionalModelMultipleInputs()
        converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file, input_shapes={'input_a': {2, 3}, 'input_b': {2, 3}})
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 2)
        self.assertEndsWith(input_details[0]['name'], 'input_a')
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([2, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        self.assertEndsWith(input_details[1]['name'], 'input_b')
        self.assertEqual(np.float32, input_details[1]['dtype'])
        self.assertAllEqual([2, 3], input_details[1]['shape'])
        self.assertEqual((0.0, 0.0), input_details[1]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 2)
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([2, 4], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])
        self.assertEqual(np.float32, output_details[1]['dtype'])
        self.assertAllEqual([2, 4], output_details[1]['shape'])
        self.assertEqual((0.0, 0.0), output_details[1]['quantization'])

    def testPartialShapeOverriding(self):
        if False:
            while True:
                i = 10
        'Test a Functional tf.keras model with partial input shape overriding.'
        self._getFunctionalModelMultipleInputs()
        converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file, input_shapes={'input_a': {2, 3}})
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 2)
        self.assertEndsWith(input_details[0]['name'], 'input_a')
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([2, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        self.assertEndsWith(input_details[1]['name'], 'input_b')
        self.assertEqual(np.float32, input_details[1]['dtype'])
        self.assertAllEqual([1, 3], input_details[1]['shape'])
        self.assertEqual((0.0, 0.0), input_details[1]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 2)
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 4], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])
        self.assertEqual(np.float32, output_details[1]['dtype'])
        self.assertAllEqual([2, 4], output_details[1]['shape'])
        self.assertEqual((0.0, 0.0), output_details[1]['quantization'])

    def testWrongShapeOverriding(self):
        if False:
            while True:
                i = 10
        'Test a Functional tf.keras model with wrong input shape overriding.'
        self._getFunctionalModelMultipleInputs()
        with self.assertRaises(ValueError):
            lite.TFLiteConverter.from_keras_model_file(self._keras_file, input_shapes={'wrong_input': {2, 3}})

    def testFunctionalSequentialModel(self):
        if False:
            while True:
                i = 10
        'Test a Functional tf.keras model containing a Sequential model.'
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(2, input_shape=(3,)))
        model.add(keras.layers.RepeatVector(3))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
        model = keras.models.Model(model.input, model.output)
        model.compile(loss=keras.losses.MSE, optimizer='sgd', metrics=[keras.metrics.categorical_accuracy], sample_weight_mode='temporal')
        x = np.random.random((1, 3))
        y = np.random.random((1, 3, 3))
        model.train_on_batch(x, y)
        model.predict(x)
        model.predict(x)
        (fd, self._keras_file) = tempfile.mkstemp('.h5')
        try:
            keras.models.save_model(model, self._keras_file)
        finally:
            os.close(fd)
        converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEndsWith(input_details[0]['name'], 'dense_input')
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1, 3], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1, 3, 3], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])
        input_data = np.array([[1, 2, 3]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        tflite_result = interpreter.get_tensor(output_details[0]['index'])
        keras_model = keras.models.load_model(self._keras_file)
        keras_result = keras_model.predict(input_data)
        np.testing.assert_almost_equal(tflite_result, keras_result, 5)

    def testSequentialModelTocoConverter(self):
        if False:
            return 10
        'Test a Sequential tf.keras model with deprecated TocoConverter.'
        self._getSequentialModel()
        converter = lite.TocoConverter.from_keras_model_file(self._keras_file)
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

    @parameterized.named_parameters(('_graph', context.graph_mode), ('_eager', context.eager_mode))
    def testGraphDebugInfo(self, test_context):
        if False:
            print('Hello World!')
        'Test a Sequential tf.keras model has debug info captured.'
        self.skipTest('TODO(b/291005679): will not be able to fix on OSS')
        with test_context():
            self._getSequentialModel()
            converter = lite.TFLiteConverter.from_keras_model_file(self._keras_file)
            converter.convert()
            self.assertValidDebugInfo(converter._debug_info)

class SparsityTest(TestModels):

    def _getSparsificableModel(self, matrix_b_values):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            in_tensor_1 = array_ops.placeholder(shape=[16, 4], dtype=dtypes.float32, name='input1')
            in_tensor_2 = constant_op.constant(matrix_b_values, shape=[4, 8], dtype=dtypes.float32)
            out_tensor = math_ops.matmul(in_tensor_1, in_tensor_2)
            sess = session.Session()
        return (sess, [in_tensor_1], [out_tensor])

    def testRandomSparsity(self):
        if False:
            while True:
                i = 10
        matrix_b_values = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        (sess, inputs, outputs) = self._getSparsificableModel(matrix_b_values)
        float_converter = lite.TFLiteConverter.from_session(sess, inputs, outputs)
        float_converter.optimizations = [lite.Optimize.EXPERIMENTAL_SPARSITY]
        float_tflite_model = float_converter.convert()
        self.assertIsNotNone(float_tflite_model)
        metadata = get_conversion_metadata(float_tflite_model)
        self.assertIsNotNone(metadata)
        self.assertAllEqual([metadata_fb.ModelOptimizationMode.RANDOM_SPARSITY], metadata.options.modelOptimizationModes)

    def testSparsifyModel(self):
        if False:
            print('Hello World!')
        matrix_b_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        (sess, inputs, outputs) = self._getSparsificableModel(matrix_b_values)
        converter = lite.TFLiteConverter.from_session(sess, inputs, outputs)
        converter.optimizations = {lite.Optimize.EXPERIMENTAL_SPARSITY}
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)
        metadata = get_conversion_metadata(tflite_model)
        self.assertIsNotNone(metadata)
        self.assertAllEqual([metadata_fb.ModelOptimizationMode.BLOCK_SPARSITY], metadata.options.modelOptimizationModes)

    def testSparsifyQuantizedModel(self):
        if False:
            print('Hello World!')
        matrix_b_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        (sess, inputs, outputs) = self._getSparsificableModel(matrix_b_values)
        converter = lite.TFLiteConverter.from_session(sess, inputs, outputs)
        converter.optimizations = {lite.Optimize.DEFAULT, lite.Optimize.EXPERIMENTAL_SPARSITY}
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        metadata = get_conversion_metadata(tflite_model)
        self.assertIsNotNone(metadata)
        self.assertAllEqual([metadata_fb.ModelOptimizationMode.PTQ_DYNAMIC_RANGE, metadata_fb.ModelOptimizationMode.BLOCK_SPARSITY], metadata.options.modelOptimizationModes)

class GrapplerTest(TestModels, parameterized.TestCase):

    def testConstantFolding(self):
        if False:
            i = 10
            return i + 15
        ops.disable_eager_execution()
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[3, 3], dtype=dtypes.float32)
            y_const = constant_op.constant([1.0, 2.0, 3.0])
            y_broadcast = array_ops.broadcast_to(y_const, [3, 3])
            out_tensor = math_ops.matmul(in_tensor, y_broadcast, name='output')
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('Placeholder', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([3, 3], input_details[0]['shape'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual('output', output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([3, 3], output_details[0]['shape'])

    def testInputNodeIsNotFolded(self):
        if False:
            return 10
        ops.disable_eager_execution()
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[3], dtype=dtypes.float32)
            y_const = constant_op.constant([1.0, 2.0, 3.0])
            y_add = y_const + y_const
            out_tensor = in_tensor * y_add
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor, y_const], [out_tensor])
        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 2)
        self.assertEqual('Placeholder', input_details[0]['name'])
        self.assertEqual('Const', input_details[1]['name'])

    def testGrapplerConstFolding(self):
        if False:
            return 10

        @def_function.function
        def plus_placeholder(x, placeholder):
            if False:
                print('Hello World!')
            return x + placeholder
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[2, 2], dtype=dtypes.float32)
            out_tensor = plus_placeholder(array_ops.zeros([2, 2, 2]), array_ops.reshape(in_tensor, shape=[2, 2]))
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('Placeholder', input_details[0]['name'])

class DefaultConverterAttrsTest(LiteTest):

    def testAttrs(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            in_tensor = array_ops.placeholder(shape=[2, 2], dtype=dtypes.float32)
            out_tensor = in_tensor + in_tensor
            sess = session.Session()
        converter = lite.TFLiteConverter.from_session(sess, [in_tensor], [out_tensor])
        self.assertEqual(converter.output_format, lite_constants.TFLITE)
        self.assertEqual(converter.inference_type, dtypes.float32)
        self.assertIsNone(converter.inference_input_type)
        self.assertIsNone(converter.inference_output_type)
        self.assertEqual(converter.quantized_input_stats, {})
        self.assertIsNone(converter.default_ranges_stats)
        self.assertFalse(converter.reorder_across_fake_quant)
        self.assertFalse(converter.change_concat_input_ranges)
        self.assertIsNotNone(converter.drop_control_dependency)
        self.assertIsNone(converter.dump_graphviz_dir)
        self.assertFalse(converter.dump_graphviz_video)
        self.assertIsNone(converter.conversion_summary_dir)

class ControlFlowV1OpsTest(LiteTest):

    def testConverterErrorOnControlFlowV1Ops(self):
        if False:
            for i in range(10):
                print('nop')
        graph_def_file = resource_loader.get_path_to_datafile('testdata/control_flow_v1.pbtxt')
        input_arrays = ['a', 'b', 'c', 'd']
        output_arrays = ['Merge']
        converter = lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
        with self.assertRaises(ConverterError) as error:
            converter.convert()
        self.assertIn('Failed to functionalize Control Flow V1 ops. Consider using Control Flow V2 ops instead. See https://www.tensorflow.org/api_docs/python/tf/compat/v1/enable_control_flow_v2.', str(error.exception))

class QuantizationModeTest(LiteTest, parameterized.TestCase):

    @parameterized.named_parameters(('size', lite.Optimize.OPTIMIZE_FOR_SIZE), ('latency', lite.Optimize.OPTIMIZE_FOR_LATENCY))
    def testDeprecatedOptionWarning(self, optimization):
        if False:
            while True:
                i = 10
        'Test if the warning message when using TOCO is logged.'
        log = io.StringIO()
        handler = logging.StreamHandler(log)
        logging.root.addHandler(handler)
        warning_message = 'please use optimizations=[Optimize.DEFAULT] instead.'
        lite.QuantizationMode([optimization], lite.TargetSpec(), None, None)
        self.assertIn(warning_message, log.getvalue())
        logging.root.removeHandler(handler)
if __name__ == '__main__':
    test.main()