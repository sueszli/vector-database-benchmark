"""Tests for lite.py functionality related to TensorFlow 2.0."""
import ctypes
import functools
import itertools
import os
import sys
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
if hasattr(sys, 'setdlopenflags') and hasattr(sys, 'getdlopenflags'):
    sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
from tensorflow.compiler.mlir.quantization.stablehlo import quantization_options_pb2 as quant_opts_pb2
from tensorflow.lite.python import conversion_metadata_schema_py_generated as metadata_fb
from tensorflow.lite.python import convert
from tensorflow.lite.python import lite
from tensorflow.lite.python import lite_v2_test_util
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import test_util as tflite_test_util
from tensorflow.lite.python import util
from tensorflow.lite.python.convert import mlir_quantize
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import InterpreterWithCustomOps
from tensorflow.lite.python.interpreter import OpResolverType
from tensorflow.lite.python.testdata import _pywrap_test_registerer as test_registerer
from tensorflow.lite.python.testdata import double_op
from tensorflow.lite.python.util import get_conversion_metadata
from tensorflow.lite.tools.flatbuffer_utils import convert_bytearray_to_object as _convert_bytearray_to_object
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import map_ops
from tensorflow.python.ops import rnn
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import saved_model
from tensorflow.python.saved_model.loader_impl import parse_saved_model
from tensorflow.python.saved_model.save import save
from tensorflow.python.trackable import autotrackable
_PresetQuantizationMethod = quant_opts_pb2.PresetQuantizationMethod.PresetMethod
DISABLE_JAX_TEST = False
try:
    import jax
    from jax import numpy as jnp
except ImportError:
    DISABLE_JAX_TEST = True

class FromConcreteFunctionTest(lite_v2_test_util.ModelTest):

    @test_util.run_v2_only
    def testTypeInvalid(self):
        if False:
            while True:
                i = 10
        root = self._getSimpleVariableModel()
        with self.assertRaises(ValueError) as error:
            _ = lite.TFLiteConverterV2.from_concrete_functions([root.f], root)
        self.assertIn('call get_concrete_function', str(error.exception))

    @test_util.run_v2_only
    def testFloat(self):
        if False:
            print('Hello World!')
        root = self._getSimpleVariableModel()
        input_data = tf.constant(1.0, shape=[1])
        concrete_func = root.f.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        tflite_model = converter.convert()
        expected_value = root.f(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
        self.assertEqual(expected_value.numpy(), actual_value)

    @parameterized.named_parameters(('_INT8InputOutput', dtypes.int8), ('_UINT8InputOutput', dtypes.uint8), ('_INT16InputOutput', dtypes.int16))
    @test_util.run_v2_only
    def testInvalidFloat(self, inference_input_output_type):
        if False:
            for i in range(10):
                print('nop')
        root = self._getSimpleVariableModel()
        input_data = tf.constant(1.0, shape=[1])
        concrete_func = root.f.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        with self.assertRaises(ValueError) as error:
            converter.inference_input_type = inference_input_output_type
            converter.inference_output_type = inference_input_output_type
            converter.convert()
        self.assertEqual('The inference_input_type and inference_output_type must be tf.float32.', str(error.exception))

    @test_util.run_v2_only
    def testScalarInput(self):
        if False:
            for i in range(10):
                print('nop')
        root = self._getSimpleVariableModel()
        input_data = tf.constant(1.0, shape=[])
        concrete_func = root.f.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        tflite_model = converter.convert()
        expected_value = root.f(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
        self.assertEqual(expected_value.numpy(), actual_value)

    @test_util.run_v2_only
    def testStringInput(self):
        if False:
            return 10

        class Model(tf.Module):

            @tf.function
            def __call__(self, x):
                if False:
                    while True:
                        i = 10
                return x
        root = Model()
        concrete_func = root.__call__.get_concrete_function(tf.constant([str(x) for x in range(11)]))
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        tflite_model = converter.convert()
        input_data = tf.constant([str(x) for x in range(11)], shape=(11,), dtype=tf.dtypes.string)
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        my_signature = interpreter.get_signature_runner()
        with self.assertRaises(ValueError) as error:
            _ = my_signature(x=input_data)
        self.assertIn('Passed in value type is not a numpy array, got type ', str(error.exception))

    @test_util.run_v2_only
    def testModelWithoutInputs(self):
        if False:
            for i in range(10):
                print('nop')

        def _get_random_number_gen():
            if False:
                for i in range(10):
                    print('nop')
            root = autotrackable.AutoTrackable()

            @tf.function(input_signature=[])
            def func():
                if False:
                    i = 10
                    return i + 15
                return tf.random.uniform(shape=[1], dtype=tf.float32)
            root.f = func
            to_save = root.f.get_concrete_function()
            return (root, to_save)
        (root, concrete_func) = _get_random_number_gen()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)

    @test_util.run_v2_only
    def testMultiFunctionModel(self):
        if False:
            print('Hello World!')
        'Convert a single model in a multi-functional model.'
        root = self._getMultiFunctionModel()
        input_data = tf.constant(1.0, shape=[1])
        concrete_func = root.add.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        tflite_model = converter.convert()
        expected_value = root.add(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
        self.assertEqual(expected_value.numpy(), actual_value)

    @test_util.run_v2_only
    def testConvertMultipleFunctions(self):
        if False:
            for i in range(10):
                print('nop')
        'Convert multiple functions in a multi-functional model.'
        root = self._getMultiFunctionModel()
        input_data = tf.constant(1.0, shape=[1])
        add_func = root.add.get_concrete_function(input_data)
        sub_func = root.sub.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2.from_concrete_functions([add_func, sub_func], root)
        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
        signature_defs = interpreter.get_signature_list()
        self.assertEqual(len(signature_defs), 2)
        self.assertEqual(list(signature_defs.keys()), ['add', 'sub'])
        self.assertEqual(len(signature_defs.values()), 2)
        self.assertEqual(list(signature_defs['add'].keys()), ['inputs', 'outputs'])
        self.assertCountEqual(signature_defs['add']['inputs'], ['x'])
        self.assertEqual(list(signature_defs['add']['outputs']), ['output_0'])
        self.assertEqual(list(signature_defs['sub'].keys()), ['inputs', 'outputs'])
        self.assertCountEqual(signature_defs['sub']['inputs'], ['x'])
        self.assertEqual(list(signature_defs['sub']['outputs']), ['output_0'])
        add_signature_runner = interpreter.get_signature_runner('add')
        add_output = add_signature_runner(x=input_data)
        self.assertEqual(add_output['output_0'], 3)
        input_details = add_signature_runner.get_input_details()
        self.assertEqual(1, len(input_details))
        self.assertEqual('add_x:0', input_details['x']['name'])
        self.assertEqual(np.float32, input_details['x']['dtype'])
        self.assertTrue(([1] == input_details['x']['shape']).all())
        self.assertEqual((0.0, 0), input_details['x']['quantization'])
        sub_signature_runner = interpreter.get_signature_runner('sub')
        sub_output = sub_signature_runner(x=input_data)
        self.assertEqual(sub_output['output_0'], -2)
        output_details = sub_signature_runner.get_output_details()
        self.assertEqual(1, len(output_details))
        self.assertEqual('StatefulPartitionedCall_1:0', output_details['output_0']['name'])
        self.assertEqual(np.float32, output_details['output_0']['dtype'])
        self.assertTrue(([1] == output_details['output_0']['shape']).all())
        self.assertEqual((0.0, 0), output_details['output_0']['quantization'])
        metadata = get_conversion_metadata(tflite_model)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.environment.apiVersion, 2)
        self.assertEqual(metadata.environment.modelType, metadata_fb.ModelType.TF_CONCRETE_FUNCTIONS)
        self.assertAllEqual([], metadata.options.modelOptimizationModes)

    def _getIntegerQuantizeModel(self, num_filters=16):
        if False:
            while True:
                i = 10
        np.random.seed(0)
        root = autotrackable.AutoTrackable()

        @tf.function(input_signature=[tf.TensorSpec(shape=[1, 5, 5, 3], dtype=tf.float32)])
        def func(inp):
            if False:
                while True:
                    i = 10
            conv = tf.nn.conv2d(inp, tf.ones([3, 3, 3, num_filters]), strides=[1, 1, 1, 1], padding='SAME')
            output = tf.nn.relu(conv, name='output')
            return output

        def calibration_gen():
            if False:
                for i in range(10):
                    print('nop')
            for _ in range(5):
                yield [np.random.uniform(-1, 1, size=(1, 5, 5, 3)).astype(np.float32)]
        root.f = func
        to_save = root.f.get_concrete_function()
        return (root, to_save, calibration_gen)

    @parameterized.named_parameters(('EnableMlirQuantizer', True), ('DisableMlirQuantizer', False))
    def testPostTrainingCalibrateAndQuantize(self, mlir_quantizer):
        if False:
            while True:
                i = 10
        (root, func, calibration_gen) = self._getIntegerQuantizeModel()
        float_converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
        float_tflite_model = float_converter.convert()
        self.assertIsNotNone(float_tflite_model)
        quantized_converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.representative_dataset = calibration_gen
        quantized_converter.experimental_new_quantizer = mlir_quantizer
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        metadata = get_conversion_metadata(quantized_tflite_model)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.environment.tensorflowVersion.decode('utf-8'), versions.__version__)
        self.assertEqual(metadata.environment.apiVersion, 2)
        self.assertEqual(metadata.environment.modelType, metadata_fb.ModelType.TF_CONCRETE_FUNCTIONS)
        self.assertEqual(metadata.options.allowCustomOps, False)
        self.assertEqual(metadata.options.enableSelectTfOps, False)
        self.assertEqual(metadata.options.forceSelectTfOps, False)
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

    @parameterized.named_parameters(('_INT8InputOutput', dtypes.int8), ('_UINT8InputOutput', dtypes.uint8), ('_INT16InputOutput', dtypes.int16))
    @test_util.run_v2_only
    def testInvalidPostTrainingDynamicRangeQuantization(self, inference_input_output_type):
        if False:
            i = 10
            return i + 15
        (root, func, _) = self._getIntegerQuantizeModel()
        converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)
        quantized_converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        with self.assertRaises(ValueError) as error:
            quantized_converter.inference_input_type = inference_input_output_type
            quantized_converter.inference_output_type = inference_input_output_type
            quantized_converter.convert()
        self.assertEqual('The inference_input_type and inference_output_type must be tf.float32.', str(error.exception))

    def _createV2QATSavedModelWithFloatOpsAtEnd(self):
        if False:
            return 10
        'Create a simple QAT SavedModel that includes float ops at the end.'
        saved_model_dir = os.path.join(self.get_temp_dir(), 'qat_float_ops_at_end')
        input_tensor = tf.keras.layers.Input((32, 32, 128))
        x = tf.quantization.fake_quant_with_min_max_args(input_tensor, -3.0, 3.0)
        x = tf.keras.layers.Conv2D(1, (3, 3))(x)
        x = tf.quantization.fake_quant_with_min_max_args(x, -3.0, 3.0)
        output_tensor = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(input_tensor, output_tensor)
        model.save(saved_model_dir)
        return saved_model_dir

    def testQuantizationRemovesQDQsForFloatIOInQAT(self):
        if False:
            for i in range(10):
                print('nop')
        saved_model_dir = self._createV2QATSavedModelWithFloatOpsAtEnd()
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_model = converter.convert()
        interpreter = Interpreter(model_content=quantized_model, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        interpreter.allocate_tensors()
        op_details = interpreter._get_ops_details()
        self.assertEqual(op_details[len(op_details) - 1]['op_name'], 'LOGISTIC')

    @parameterized.named_parameters(('EnableMlirQuantizer', True), ('DisableMlirQuantizer', False))
    def testQuantizationRemovesQDQsForFloatIO(self, mlir_quantizer):
        if False:
            while True:
                i = 10
        (func, calibration_gen) = self._getSqrtModel()
        converter = lite.TFLiteConverterV2.from_concrete_functions([func.get_concrete_function()])
        converter.representative_dataset = calibration_gen
        converter.optimizations = [lite.Optimize.DEFAULT]
        converter.experimental_new_quantizer = mlir_quantizer
        quantized_model = converter.convert()
        interpreter = Interpreter(model_content=quantized_model, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        interpreter.allocate_tensors()
        op_details = interpreter._get_ops_details()
        self.assertLen(op_details, 1)
        self.assertEqual(op_details[0]['op_name'], 'SQRT')

    @parameterized.named_parameters(('_Default', False, False, dtypes.float32), ('_INT8InputOutput', False, False, dtypes.int8), ('_UINT8InputOutput', False, False, dtypes.uint8), ('_INT16Quantize', False, True, dtypes.float32), ('_INT16Quantize_INT16InputOutput', False, True, dtypes.int16), ('_IntOnly', True, False, dtypes.float32), ('_IntOnly_INT8InputOutput', True, False, dtypes.int8), ('_IntOnly_UINT8InputOutput', True, False, dtypes.uint8), ('_IntOnly_INT16Quantize', True, True, dtypes.float32), ('_IntOnly_INT16Quantize_INT16InputOutput', True, True, dtypes.int16))
    def testIntegerQuantization(self, is_int_only, is_int16_quantize, inference_input_output_type):
        if False:
            print('Hello World!')
        (root, func, calibration_gen) = self._getIntegerQuantizeModel()
        converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)
        quantized_converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.representative_dataset = calibration_gen
        if is_int_only:
            if is_int16_quantize:
                quantized_converter.target_spec.supported_ops = [lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            else:
                quantized_converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
        elif is_int16_quantize:
            quantized_converter.target_spec.supported_ops = [lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8, lite.OpsSet.TFLITE_BUILTINS]
        quantized_converter.inference_input_type = inference_input_output_type
        quantized_converter.inference_output_type = inference_input_output_type
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        metadata = get_conversion_metadata(quantized_tflite_model)
        self.assertIsNotNone(metadata)
        expected_opt_options = [metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER]
        if is_int16_quantize:
            expected_opt_options = [metadata_fb.ModelOptimizationMode.PTQ_INT16]
        self.assertAllEqual(expected_opt_options, metadata.options.modelOptimizationModes)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual(inference_input_output_type.as_numpy_dtype, input_details[0]['dtype'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual(inference_input_output_type.as_numpy_dtype, output_details[0]['dtype'])
        self.assertLess(len(quantized_tflite_model), len(tflite_model))

    @parameterized.named_parameters(('_INT16Quantize_INT8InputOutput', True, dtypes.int8))
    def testInvalidIntegerQuantization(self, is_int16_quantize, inference_input_output_type):
        if False:
            for i in range(10):
                print('nop')
        (root, func, calibration_gen) = self._getIntegerQuantizeModel()
        quantized_converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.representative_dataset = calibration_gen
        if is_int16_quantize:
            quantized_converter.target_spec.supported_ops = [lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8, lite.OpsSet.TFLITE_BUILTINS]
        with self.assertRaises(ValueError) as error:
            quantized_converter.inference_input_type = dtypes.int8
            quantized_converter.inference_output_type = dtypes.int8
            quantized_converter.convert()
        self.assertEqual("The inference_input_type and inference_output_type must be in ['tf.float32', 'tf.int16'].", str(error.exception))

    def testCalibrateAndQuantizeBuiltinInt16(self):
        if False:
            print('Hello World!')
        (root, func, calibration_gen) = self._getIntegerQuantizeModel()
        float_converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
        float_tflite_model = float_converter.convert()
        self.assertIsNotNone(float_tflite_model)
        converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
        converter.optimizations = [lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        converter.representative_dataset = calibration_gen
        quantized_tflite_model = converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual(np.float32, input_details[0]['dtype'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual(np.float32, output_details[0]['dtype'])
        tensor_details = interpreter.get_tensor_details()
        self.assertEqual(np.int8, tensor_details[2]['dtype'])
        self.assertEqual(np.int64, tensor_details[1]['dtype'])
        self.assertEqual(np.int16, tensor_details[0]['dtype'])
        self.assertEqual(np.int16, tensor_details[3]['dtype'])
        self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

    @test_util.run_v2_only
    def testSignatureDefs(self):
        if False:
            i = 10
            return i + 15
        'Test converting SignatureDef is correct and uses SignatureDef API.'
        root = self._getMultiFunctionModel()
        input_data = tf.constant(1.0, shape=[1])
        add_func = root.add.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2([add_func], trackable_obj=root)
        tflite_model = converter.convert()
        expected_value = add_func(input_data)
        interpreter = Interpreter(model_content=tflite_model)
        signature_defs = interpreter.get_signature_list()
        results = self._evaluateTFLiteModelUsingSignatureDef(tflite_model, 'serving_default', {'x': input_data})
        self.assertLen(list(results.keys()), 1)
        self.assertStartsWith(list(results.keys())[0], 'output')
        self.assertAllClose(expected_value.numpy(), results[signature_defs['serving_default']['outputs'][0]])
        self.assertEqual(len(signature_defs), 1)
        self.assertEqual(list(signature_defs.keys()), ['serving_default'])
        self.assertEqual(len(signature_defs.values()), 1)
        self.assertEqual(list(signature_defs['serving_default'].keys()), ['inputs', 'outputs'])
        self.assertCountEqual(signature_defs['serving_default']['inputs'], ['x'])
        self.assertLen(list(signature_defs['serving_default']['outputs']), 1)
        self.assertStartsWith(list(signature_defs['serving_default']['outputs'])[0], 'output')

    @test_util.run_v2_only
    def testNoSignatureDefsWhenTrackingObjIsNone(self):
        if False:
            while True:
                i = 10
        'Test converting SignatureDef is correct and uses SignatureDef API.'
        root = self._getSimpleVariableModel()
        input_data = tf.constant(1.0, shape=[1])
        concrete_func = root.f.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], None)
        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
        signature_defs = interpreter.get_signature_list()
        self.assertEqual(len(signature_defs), 0)

    @test_util.run_v2_only
    def testNoSignatureDefsWhenInvalidTrackingObjIsGiven(self):
        if False:
            i = 10
            return i + 15
        'Test converting SignatureDef is correct and uses SignatureDef API.'
        root = self._getSimpleVariableModel()
        input_data = tf.constant(1.0, shape=[1])
        concrete_func = root.f.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], trackable_obj=autotrackable.AutoTrackable())
        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
        signature_defs = interpreter.get_signature_list()
        self.assertEqual(len(signature_defs), 0)

    @test_util.run_v2_only
    def testTrackbleObject(self):
        if False:
            for i in range(10):
                print('nop')
        'Test converting with trackable objects.'
        root = self._getMultiFunctionModel()
        input_data = tf.constant(1.0, shape=[1])
        add_func = root.add.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2.from_concrete_functions([add_func], trackable_obj=root)
        tflite_model = converter.convert()
        expected_value = add_func(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
        self.assertEqual(expected_value.numpy(), actual_value)

    def _getTrainingTimeQuantizedModel(self):
        if False:
            i = 10
            return i + 15

        class QLinear(tf.keras.layers.Layer):

            def __init__(self, units=3, **kwargs):
                if False:
                    while True:
                        i = 10
                super(QLinear, self).__init__(**kwargs)
                self.units = units

            def build(self, input_shape):
                if False:
                    for i in range(10):
                        print('nop')
                self.w = self.add_weight('weight', shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
                self.min_var = self.add_weight('min', initializer=tf.keras.initializers.Constant(-6.0), trainable=False)
                self.max_var = self.add_weight('max', initializer=tf.keras.initializers.Constant(6.0), trainable=False)

            def call(self, inputs):
                if False:
                    for i in range(10):
                        print('nop')
                x = tf.quantization.fake_quant_with_min_max_vars(inputs, self.min_var, self.max_var)
                w_fq = tf.quantization.fake_quant_with_min_max_vars(self.w, self.min_var, self.max_var)
                x = tf.matmul(x, w_fq)
                x = tf.quantization.fake_quant_with_min_max_vars(x, self.min_var, self.max_var)
                return x
        return tf.keras.Sequential(QLinear(3, input_shape=(2,)))

    @parameterized.named_parameters(('_DefaultFLOAT32InputOutput', dtypes.float32), ('_INT8InputOutput', dtypes.int8), ('_UINT8InputOutput', dtypes.uint8))
    @test_util.run_v2_only
    def testTrainingTimeQuantization(self, inference_input_output_type):
        if False:
            i = 10
            return i + 15
        model = self._getTrainingTimeQuantizedModel()
        float_converter = lite.TFLiteConverterV2.from_keras_model(model)
        float_tflite_model = float_converter.convert()
        self.assertIsNotNone(float_tflite_model)
        quantized_converter = lite.TFLiteConverterV2.from_keras_model(model)
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.inference_input_type = inference_input_output_type
        quantized_converter.inference_output_type = inference_input_output_type
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        metadata = get_conversion_metadata(quantized_tflite_model)
        self.assertIsNotNone(metadata)
        self.assertAllEqual([metadata_fb.ModelOptimizationMode.QUANTIZATION_AWARE_TRAINING], metadata.options.modelOptimizationModes)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual(inference_input_output_type.as_numpy_dtype, input_details[0]['dtype'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual(inference_input_output_type.as_numpy_dtype, output_details[0]['dtype'])
        self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

    @test_util.run_v2_only
    def testNewQuantizer(self):
        if False:
            i = 10
            return i + 15
        'Test the model quantized by the new converter.'
        (root, func, calibration_gen) = self._getIntegerQuantizeModel()
        quantized_converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
        quantized_converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
        quantized_converter.representative_dataset = calibration_gen
        quantized_converter.experimental_new_quantizer = False
        old_tflite = quantized_converter.convert()
        quantized_converter.experimental_new_quantizer = True
        new_tflite = quantized_converter.convert()
        for _ in range(5):
            input_data = tf.constant(np.random.uniform(-1, 1, size=(1, 5, 5, 3)).astype(np.float32))
            old_value = self._evaluateTFLiteModel(old_tflite, [input_data])
            new_value = self._evaluateTFLiteModel(new_tflite, [input_data])
            self.assertAllClose(old_value, new_value, atol=0.1)

    @test_util.run_v2_only
    def testEmbeddings(self):
        if False:
            for i in range(10):
                print('nop')
        'Test model with embeddings.'
        input_data = tf.constant(np.array(np.random.random_sample(20), dtype=np.int32))

        class EmbeddingModel(tf.keras.Model):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super(EmbeddingModel, self).__init__()
                self.shared_weights = self.add_weight('weights', shape=(2000, 300), dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=300 ** (-0.5)))

            @tf.function(input_signature=[tf.TensorSpec(shape=20, dtype=tf.int32)])
            def func(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return tf.gather(self.shared_weights, x)
        root = EmbeddingModel()
        concrete_func = root.func.get_concrete_function()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        tflite_model = converter.convert()
        expected_value = root.func(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
        self.assertAllClose(expected_value.numpy(), actual_value[0], atol=1e-05)

    @test_util.run_v2_only
    def testGraphDebugInfo(self):
        if False:
            print('Hello World!')
        'Test a concrete function has debug info captured.'
        root = autotrackable.AutoTrackable()
        root.v1 = tf.Variable(3.0)
        root.f = tf.function(lambda x: root.v1 * x)
        input_data = tf.constant(1.0, shape=[1])
        concrete_func = root.f.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        converter.convert()
        self._assertValidDebugInfo(converter._debug_info)

    def _getIntegerQuantizationModelWithFlexOp(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(0)
        root = autotrackable.AutoTrackable()

        @tf.function(input_signature=[tf.TensorSpec(shape=[3, 3, 3, 3, 3], dtype=tf.float32)])
        def func(inp):
            if False:
                while True:
                    i = 10
            tanh = tf.math.tanh(inp)
            conv3d = tf.nn.conv3d(tanh, tf.ones([3, 3, 3, 3, 3]), strides=[1, 1, 1, 1, 1], padding='SAME')
            erf = tf.math.erf(conv3d)
            output = tf.math.tanh(erf)
            return output

        def calibration_gen():
            if False:
                return 10
            for _ in range(5):
                yield [np.random.uniform(-1, 1, size=(3, 3, 3, 3, 3)).astype(np.float32)]
        root.f = func
        return (root, root.f.get_concrete_function(), calibration_gen)

    @parameterized.named_parameters(('_Default', False, False, dtypes.float32), ('_INT8InputOutput', False, False, dtypes.int8), ('_UINT8InputOutput', False, False, dtypes.uint8), ('_INT16Quantize', False, True, dtypes.float32), ('_INT16Quantize_INT16InputOutput', False, True, dtypes.int16), ('_IntOnly', True, False, dtypes.float32), ('_IntOnly_INT8InputOutput', True, False, dtypes.int8), ('_IntOnly_UINT8InputOutput', True, False, dtypes.uint8), ('_IntOnly_INT16Quantize', True, True, dtypes.float32), ('_IntOnly_INT16Quantize_INT16InputOutput', True, True, dtypes.int16))
    @test_util.run_v2_only
    def testIntegerQuantizationWithFlexOp(self, is_int_only, is_int16_quantize, inference_input_output_type):
        if False:
            while True:
                i = 10
        (root, func, calibration_gen) = self._getIntegerQuantizationModelWithFlexOp()
        quantized_converter = tf.lite.TFLiteConverter.from_concrete_functions([func], root)
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.representative_dataset = calibration_gen
        if is_int_only:
            if is_int16_quantize:
                quantized_converter.target_spec.supported_ops = [lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8, lite.OpsSet.SELECT_TF_OPS]
            else:
                quantized_converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8, lite.OpsSet.SELECT_TF_OPS]
        elif is_int16_quantize:
            quantized_converter.target_spec.supported_ops = [lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8, lite.OpsSet.TFLITE_BUILTINS, lite.OpsSet.SELECT_TF_OPS]
        else:
            quantized_converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS, lite.OpsSet.SELECT_TF_OPS]
        quantized_converter.inference_input_type = inference_input_output_type
        quantized_converter.inference_output_type = inference_input_output_type
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        metadata = get_conversion_metadata(quantized_tflite_model)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.options.enableSelectTfOps, True)
        expected_opt_options = [metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER]
        if is_int16_quantize:
            expected_opt_options = [metadata_fb.ModelOptimizationMode.PTQ_INT16]
        self.assertAllEqual(expected_opt_options, metadata.options.modelOptimizationModes)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual(inference_input_output_type.as_numpy_dtype, input_details[0]['dtype'])
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual(inference_input_output_type.as_numpy_dtype, output_details[0]['dtype'])

    def _getIntegerQuantizationModelWithUnsupportedOps(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(0)
        root = autotrackable.AutoTrackable()

        @tf.function(input_signature=[tf.TensorSpec(shape=[3], dtype=tf.float32), tf.TensorSpec(shape=[3], dtype=tf.float32)])
        def func(a, b):
            if False:
                while True:
                    i = 10
            left = tf.math.ceil(a)
            right = tf.nn.tanh(b)
            add = tf.math.add(left, right)
            output = tf.math.ceil(add)
            return (output, right)

        def calibration_gen():
            if False:
                i = 10
                return i + 15
            for _ in range(5):
                yield [np.random.uniform(-1, 1, size=3).astype(np.float32), np.random.uniform(-1, 1, size=3).astype(np.float32)]
        root.f = func
        return (root, root.f.get_concrete_function(), calibration_gen)

    @parameterized.named_parameters(('_INT8InputOutput', False, False, dtypes.int8), ('_UINT8InputOutput', False, False, dtypes.uint8), ('_INT16Quantize_INT16InputOutput', False, True, dtypes.int16), ('_IntOnly_INT8InputOutput', True, False, dtypes.int8), ('_IntOnly_UINT8InputOutput', True, False, dtypes.uint8), ('_IntOnly_INT16Quantize_INT16InputOutput', True, True, dtypes.int16), ('_IntOnly_INT8InputOutputMlirQuant', True, False, dtypes.int8, True), ('_IntOnly_UINT8InputOutputMlirQuant', True, False, dtypes.uint8, True))
    @test_util.run_v2_only
    def testIntegerQuantizationWithUnsupportedOps(self, is_int_only, is_int16_quantize, inference_input_output_type, enable_mlir_quantizer=False):
        if False:
            for i in range(10):
                print('nop')
        (root, func, calib_gen) = self._getIntegerQuantizationModelWithUnsupportedOps()
        quantized_converter = tf.lite.TFLiteConverter.from_concrete_functions([func], root)
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.representative_dataset = calib_gen
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
        self.assertEqual(input_details[0]['dtype'], expected_dtype)
        self.assertEqual(input_details[1]['dtype'], expected_ceil_dtype)
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 2)
        self.assertEqual(output_details[0]['dtype'], expected_dtype)
        self.assertEqual(output_details[1]['dtype'], expected_ceil_dtype)

    def _getIntegerQuantizationModelWithControlFlow(self):
        if False:
            print('Hello World!')

        def true_fn(x):
            if False:
                return 10
            return x

        def false_fn(x):
            if False:
                i = 10
                return i + 15
            return x

        @tf.function(input_signature=[tf.TensorSpec(shape=[1, 2], dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.bool)])
        def model(x, b):
            if False:
                return 10
            x = x + x
            x = tf.cond(b, true_fn=lambda : true_fn(x), false_fn=lambda : false_fn(x))
            return x + x

        def calibration_gen():
            if False:
                print('Hello World!')
            for _ in range(5):
                yield [np.random.uniform(-1, 1, size=(1, 2)).astype(np.float32), tf.constant(True)]
            for _ in range(5):
                yield [np.random.uniform(-1, 1, size=(1, 2)).astype(np.float32), tf.constant(False)]
        return (model, model.get_concrete_function(), calibration_gen)

    @parameterized.named_parameters(('_INT8InputOutput', False, False, dtypes.int8), ('_UINT8InputOutput', False, False, dtypes.uint8), ('_INT16Quantize_INT16InputOutput', False, True, dtypes.int16), ('_IntOnly_INT8InputOutput', True, False, dtypes.int8), ('_IntOnly_UINT8InputOutput', True, False, dtypes.uint8), ('_IntOnly_INT16Quantize_INT16InputOutput', True, True, dtypes.int16))
    @test_util.run_v2_only
    def testIntegerQuantizationWithControlFlow(self, is_int_only, is_int16_quantize, inference_input_output_type, enable_mlir_quantizer=False):
        if False:
            i = 10
            return i + 15
        (root, func, calib_gen) = self._getIntegerQuantizationModelWithControlFlow()
        quantized_converter = tf.lite.TFLiteConverter.from_concrete_functions([func], root)
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.representative_dataset = calib_gen
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
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 2)
        self.assertEqual(input_details[0]['dtype'], expected_dtype)
        self.assertEqual(input_details[1]['dtype'], dtypes.bool)
        output_details = interpreter.get_output_details()
        self.assertLen(output_details, 1)
        self.assertEqual(output_details[0]['dtype'], expected_dtype)

    @parameterized.named_parameters(('_BlocklistedNoneWithLowering', None, None, True), ('_BlocklistedNoneWithoutLowering', None, None, False), ('_BlocklistedOpsWithLowering', {'CONV_2D'}, None, True), ('_BlocklistedOpsWithoutLowering', {'CONV_2D'}, None, False), ('_BlocklistedNodesWithLowering', None, {'PartitionedCall:0'}, True), ('_BlocklistedNodesWithoutLowering', None, {'Identity'}, False))
    @test_util.run_v2_only
    def testNewQuantizerBlocklistingArgs(self, denylisted_ops, denylisted_nodes, lower_to_saved_model):
        if False:
            for i in range(10):
                print('nop')
        'Test the model quantized by the new converter and denylisted options.'
        (root, func, calibration_gen) = self._getIntegerQuantizeModel()
        quantized_converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
        quantized_converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
        quantized_converter.representative_dataset = calibration_gen
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.experimental_new_quantizer = True
        quantized_converter._experimental_calibrate_only = True
        quantized_converter.experimental_lower_to_saved_model = lower_to_saved_model
        calibrated = quantized_converter.convert()
        quantized_tflite_model = mlir_quantize(calibrated, denylisted_ops=denylisted_ops, denylisted_nodes=denylisted_nodes)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        details = interpreter.get_tensor_details()
        num_quantized_tensors = sum([1 for detail in details if len(detail['quantization_parameters']['scales'])])
        if denylisted_nodes or denylisted_ops:
            self.assertEqual(num_quantized_tensors, 0)
            return
        self.assertEqual(num_quantized_tensors, 4)

    @parameterized.named_parameters(('_SingleLayer', False), ('_WholeModel', True))
    @test_util.run_v2_only
    def testNewQuantizerNumericVerificationDebugMode(self, whole_model_verify):
        if False:
            while True:
                i = 10
        'Test the model quantized by the new converter with numeric verify ops.'
        (root, func, calibration_gen) = self._getIntegerQuantizeModel()
        quantized_converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
        quantized_converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
        quantized_converter.representative_dataset = calibration_gen
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.experimental_new_quantizer = True
        production_tflite = quantized_converter.convert()
        quantized_converter._experimental_calibrate_only = True
        calibrated = quantized_converter.convert()
        debug_mode_tflite = mlir_quantize(calibrated, enable_numeric_verify=True, enable_whole_model_verify=whole_model_verify)
        self.assertNotEqual(production_tflite, debug_mode_tflite)
        input_data = tf.constant(np.random.uniform(-1, 1, size=(1, 5, 5, 3)).astype(np.float32))

        def examine_tflite_model(tflite_content, input_data):
            if False:
                return 10
            interpreter = Interpreter(model_content=tflite_content, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            interpreter.set_tensor(input_details[0]['index'], input_data.numpy())
            interpreter.invoke()
            tensor_details = interpreter.get_tensor_details()
            return ({details['name']: interpreter.get_tensor(details['index']) for details in interpreter.get_tensor_details()}, tensor_details)
        (tflite_result, _) = examine_tflite_model(production_tflite, input_data)
        (debug_mode_tflite_result, debug_tensor_details) = examine_tflite_model(debug_mode_tflite, input_data)
        num_production_quantize_ops = len([None for output_tensor_name in tflite_result if 'tfl.quantize' in output_tensor_name])
        self.assertEqual(num_production_quantize_ops, 1)
        num_debug_quantize_ops = len([None for output_tensor_name in debug_mode_tflite_result if 'tfl.quantize' in output_tensor_name])
        self.assertEqual(num_production_quantize_ops, num_debug_quantize_ops)
        num_debug_ops = 0
        for output_tensor_name in debug_mode_tflite_result:
            if 'NumericVerify' in output_tensor_name:
                pos_end_prefix = len('NumericVerify/')
                pos_colon = output_tensor_name.rfind(':')
                self.assertEqual('NumericVerify/', output_tensor_name[:pos_end_prefix])
                tensor_id = int(output_tensor_name[pos_colon + 1:])
                original_tensor_name = output_tensor_name[pos_end_prefix:pos_colon]
                self.assertEqual(original_tensor_name, debug_tensor_details[tensor_id]['name'])
                num_debug_ops += 1
        self.assertEqual(num_debug_ops, 1)
        self.assertEqual(num_debug_ops, num_debug_quantize_ops)

    @parameterized.named_parameters(('_PerChannelQuant', False, False), ('_PerChannelMlirQuant', False, True), ('_PerTensorQuant', True, False), ('_PerTensorMlirQuant', True, True), ('_PerChannelDynamicRange', False, False, False), ('_PerTensorDynamicRange', True, False, False))
    @test_util.run_v2_only
    def testDisablePerChannelQuantization(self, disable_per_channel=False, enable_mlir_quantizer=False, representative_dataset=True):
        if False:
            return 10
        k_conv_name = 'Conv2D'
        k_num_filters = 38
        (root, func, calib_gen) = self._getIntegerQuantizeModel(k_num_filters)
        quantized_converter = tf.lite.TFLiteConverter.from_concrete_functions([func], root)
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.representative_dataset = calib_gen
        quantized_converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS]
        quantized_converter.experimental_new_quantizer = enable_mlir_quantizer
        if disable_per_channel:
            quantized_converter._experimental_disable_per_channel = disable_per_channel
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        detail = next((d for d in interpreter.get_tensor_details() if d['name'].startswith(k_conv_name)))
        quant_params = detail['quantization_parameters']
        expected_num_params = 1 if disable_per_channel else k_num_filters
        self.assertLen(quant_params['scales'], expected_num_params)
        self.assertLen(quant_params['zero_points'], expected_num_params)

    @parameterized.named_parameters(('MlirQuantize', True), ('TocoQuantize', False))
    @test_util.run_v2_only
    def testQuantizeBiasOverflow(self, enable_mlir_quantizer):
        if False:
            for i in range(10):
                print('nop')
        'Tests if the quantizer handles bias overflow by adjusting scales.'
        input_data = np.array([[-0.001, 0.001]], dtype=np.float32)

        def calibration_gen():
            if False:
                for i in range(10):
                    print('nop')
            yield {'x': input_data}
        root = self._getMatMulModelWithSmallWeights()
        input_data = tf.constant([-0.001, 0.001], shape=(1, 2))
        concrete_func = root.matmul.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        converter.optimizations = [lite.Optimize.DEFAULT]
        converter.representative_dataset = calibration_gen
        converter.experimental_new_quantizer = enable_mlir_quantizer
        quantized_model = converter.convert()
        interpreter = Interpreter(model_content=quantized_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_details = interpreter.get_output_details()
        output = interpreter.get_tensor(output_details[0]['index'])
        self.assertAllClose(root.bias, output.flatten())

    @test_util.run_v2_only
    def testOpVersion(self):
        if False:
            for i in range(10):
                print('nop')

        @tf.function(input_signature=[tf.TensorSpec(shape=[5, 5], dtype=tf.float32)])
        def custom_resize(image):
            if False:
                while True:
                    i = 10
            image = image[tf.newaxis, ..., tf.newaxis]
            resize1 = tf.compat.v1.image.resize_bilinear(image, [2, 2], half_pixel_centers=True)
            resize2 = tf.compat.v1.image.resize_bilinear(image, [2, 2])
            return resize1 + resize2
        concrete_func = custom_resize.get_concrete_function()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], custom_resize)
        tflite_model = converter.convert()
        model_object = schema_fb.Model.GetRootAsModel(tflite_model, 0)
        model = schema_fb.ModelT.InitFromObj(model_object)
        for operator in model.operatorCodes:
            if operator.builtinCode == schema_fb.BuiltinOperator.RESIZE_BILINEAR:
                self.assertEqual(operator.version, 3)
                break

    @test_util.run_v2_only
    def testForceSelectTFOps(self):
        if False:
            return 10
        root = self._getSimpleVariableModel()
        input_data = tf.constant(1.0, shape=[1])
        concrete_func = root.f.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        metadata = get_conversion_metadata(tflite_model)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.options.forceSelectTfOps, True)
        expected_value = root.f(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
        self.assertEqual(expected_value.numpy(), actual_value)

    def testExcludeConversionMetadata(self):
        if False:
            return 10
        root = self._getSimpleVariableModel()
        input_data = tf.constant(1.0, shape=[1])
        concrete_func = root.f.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        converter.exclude_conversion_metadata = True
        tflite_model = converter.convert()
        metadata = get_conversion_metadata(tflite_model)
        self.assertIsNone(metadata)

    def testConversionMetadataForDynamicRange(self):
        if False:
            print('Hello World!')
        (func, _) = self._getSqrtModel()
        converter = lite.TFLiteConverterV2.from_concrete_functions([func.get_concrete_function()])
        converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_model = converter.convert()
        metadata = get_conversion_metadata(quantized_model)
        self.assertIsNotNone(metadata)
        self.assertAllEqual([metadata_fb.ModelOptimizationMode.PTQ_DYNAMIC_RANGE], metadata.options.modelOptimizationModes)

    def testConversionMetadataForFloat16(self):
        if False:
            while True:
                i = 10
        (root, func, calibration_gen) = self._getIntegerQuantizeModel()
        converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
        converter.optimizations = [lite.Optimize.DEFAULT]
        converter.representative_dataset = calibration_gen
        converter.target_spec.supported_types = [dtypes.float16]
        quantized_model = converter.convert()
        metadata = get_conversion_metadata(quantized_model)
        self.assertIsNotNone(metadata)
        self.assertAllEqual([metadata_fb.ModelOptimizationMode.PTQ_FLOAT16], metadata.options.modelOptimizationModes)

class FromSavedModelTest(lite_v2_test_util.ModelTest):

    def _createV1SavedModel(self, shape):
        if False:
            print('Hello World!')
        'Create a simple SavedModel.'
        saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_savedmodel')
        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                in_tensor_1 = tf.compat.v1.placeholder(shape=shape, dtype=tf.float32, name='inputB')
                in_tensor_2 = tf.compat.v1.placeholder(shape=shape, dtype=tf.float32, name='inputA')
                variable_node = tf.Variable(1.0, name='variable_node')
                out_tensor = in_tensor_1 + in_tensor_2 * variable_node
                inputs = {'x': in_tensor_1, 'y': in_tensor_2}
                outputs = {'z': out_tensor}
                sess.run(tf.compat.v1.variables_initializer([variable_node]))
                saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
        return saved_model_dir

    def _createV2QATSavedModel(self, shape):
        if False:
            i = 10
            return i + 15
        'Create a simple QAT SavedModel in TF 2.'
        saved_model_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        input_name = 'input'
        output_name = 'scores'
        input_tensor = tf.keras.layers.Input((32, 32, 128), name=input_name)
        x = tf.quantization.fake_quant_with_min_max_args(input_tensor, -3.0, 3.0)
        x = tf.keras.layers.Conv2D(1, (3, 3))(x)
        x = tf.quantization.fake_quant_with_min_max_args(x, -3.0, 3.0)
        scores = tf.keras.layers.Reshape((-1,), name=output_name)(x)
        model = tf.keras.Model(input_tensor, scores)
        model.save(saved_model_dir)
        return (saved_model_dir, input_name, output_name)

    @test_util.run_v2_only
    def testV1SimpleModel(self):
        if False:
            i = 10
            return i + 15
        'Test a SavedModel.'
        with tf.Graph().as_default():
            saved_model_dir = self._createV1SavedModel(shape=[1, 16, 16, 3])
            converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
            tflite_model = converter.convert()
            self.assertTrue(tflite_model)
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
            self.assertTrue([1, 16, 16, 3], input_details[1]['shape'])
            self.assertEqual((0.0, 0.0), input_details[1]['quantization'])
            output_details = interpreter.get_output_details()
            self.assertLen(output_details, 1)
            self.assertStartsWith(output_details[0]['name'], 'add')
            self.assertEqual(np.float32, output_details[0]['dtype'])
            self.assertTrue([1, 16, 16, 3], output_details[0]['shape'])
            self.assertEqual((0.0, 0.0), output_details[0]['quantization'])

    @parameterized.named_parameters(('Default', False), ('UnfoldLargeConstant', True))
    @test_util.run_v2_only
    def testUnfoldLargeConstant(self, unfold_large_constant):
        if False:
            for i in range(10):
                print('nop')
        'Test unfolding large splat constant in a TF Lite model.'
        saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_savedmodel')
        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                in_tensor = tf.compat.v1.placeholder(shape=[1000, 1000], dtype=tf.float32, name='input')
                constant = tf.constant(value=1, dtype=tf.float32, shape=[1000, 1000])
                out_tensor = in_tensor + constant
                inputs = {'x': in_tensor}
                outputs = {'y': out_tensor}
                saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        converter._experimental_unfold_large_splat_constant = unfold_large_constant
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)
        model = util._convert_model_from_bytearray_to_object(tflite_model)
        if unfold_large_constant:
            self.assertEqual(model.operatorCodes[0].builtinCode, schema_fb.BuiltinOperator.FILL)
            self.assertEqual(model.operatorCodes[1].builtinCode, schema_fb.BuiltinOperator.ADD)
        else:
            self.assertEqual(model.operatorCodes[0].builtinCode, schema_fb.BuiltinOperator.ADD)
        interpreter = Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual('input:0', input_details[0]['name'])
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([1000, 1000], input_details[0]['shape'])
        self.assertEqual((0.0, 0.0), input_details[0]['quantization'])
        output_details = interpreter.get_output_details()
        self.assertEqual('add:0', output_details[0]['name'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        self.assertAllEqual([1000, 1000], output_details[0]['shape'])
        self.assertEqual((0.0, 0.0), output_details[0]['quantization'])
        interpreter.set_tensor(input_details[0]['index'], np.ones(shape=[1000, 1000], dtype=np.float32))
        interpreter.invoke()
        self.assertAllEqual(np.full(shape=[1000, 1000], fill_value=2.0, dtype=np.float32), interpreter.get_tensor(output_details[0]['index']))

    @test_util.run_v2_only
    def testPreserveAssert(self):
        if False:
            while True:
                i = 10
        'Test preserving AssertOp in a TF Lite model.'
        saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_savedmodel')
        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                in_tensor = tf.compat.v1.placeholder(shape=[10, 10], dtype=tf.float32, name='input')
                constant = tf.constant(value=1, dtype=tf.float32, shape=[10, 10])
                assert_op = tf.Assert(tf.less_equal(in_tensor, constant), [in_tensor])
                with tf.control_dependencies([assert_op]):
                    out_tensor = in_tensor + constant
                inputs = {'x': in_tensor}
                outputs = {'y': out_tensor}
                saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_preserve_assert_op = True
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)
        model = util._convert_model_from_bytearray_to_object(tflite_model)
        has_assert = False
        for op_code in model.operatorCodes:
            if op_code.customCode == b'FlexAssert':
                has_assert = True
                break
        self.assertTrue(has_assert)

    @test_util.run_v2_only
    def testTF1HubFormattedModel(self):
        if False:
            while True:
                i = 10
        'Test a TF1 hub formatted model.'
        saved_model_dir = self._createV1SavedModel(shape=[1, 16, 16, 3])
        saved_model_proto = parse_saved_model(saved_model_dir)
        saved_model_proto.saved_model_schema_version = 0
        saved_model_pb_file_path = os.path.join(saved_model_dir, 'saved_model.pb')
        with file_io.FileIO(saved_model_pb_file_path, 'wb') as writer:
            writer.write(saved_model_proto.SerializeToString())
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)

    def _createV1ModelWithHashTableInitializer(self):
        if False:
            print('Hello World!')
        tf.compat.v1.disable_eager_execution()
        saved_model_dir = os.path.join(self.get_temp_dir(), 'savedmodel_with_hashtable')
        table_initializer = tf.lookup.KeyValueTensorInitializer(keys=['a', 'b', 'c', 'd'], values=[1, 2, 3, 4], key_dtype=tf.string, value_dtype=tf.int64)
        table = tf.lookup.StaticHashTable(table_initializer, default_value=tf.constant(-1, dtype=tf.int64))
        x = tf.compat.v1.placeholder(tf.string, shape=(), name='input')
        y = table.lookup(x)
        tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
        tensor_info_y = tf.compat.v1.saved_model.utils.build_tensor_info(y)
        (signature_def_map, init_op, assets_collection) = ({'serving_default': tf.compat.v1.saved_model.signature_def_utils.build_signature_def(inputs={'x': tensor_info_x}, outputs={'y': tensor_info_y}, method_name='some_function')}, tf.compat.v1.tables_initializer(), None)
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.initializers.global_variables())
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(saved_model_dir)
        builder.add_meta_graph_and_variables(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], signature_def_map, main_op=init_op, assets_collection=assets_collection, strip_default_attrs=True)
        builder.save()
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.enable_eager_execution()
        return saved_model_dir

    @test_util.run_v2_only
    def testModelWithHashTableInitializer(self):
        if False:
            i = 10
            return i + 15
        "Test a model with saved_model's session initializer for hash tables."
        saved_model_dir = self._createV1ModelWithHashTableInitializer()
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_data = np.array(['a', 'b', 'c', 'z'], dtype=np.string_)
        interpreter.resize_tensor_input(input_details[0]['index'], [4], strict=False)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual([1, 2, 3, -1], list(actual_value))
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual([1, 2, 3, -1], list(actual_value))
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual([1, 2, 3, -1], list(actual_value))

    def _createV1ModelWithMutableHashTable(self):
        if False:
            i = 10
            return i + 15
        tf.compat.v1.disable_eager_execution()
        saved_model_dir = os.path.join(self.get_temp_dir(), 'savedmodel_with_mutable_hashtable')
        table = tf.raw_ops.MutableHashTableV2(key_dtype=tf.string, value_dtype=tf.int64)
        x = tf.compat.v1.placeholder(tf.string, shape=(), name='input')
        keys = tf.constant(['a', 'b'], tf.string)
        values = tf.constant([1, 5], tf.int64)
        default_value = tf.constant(-1, tf.int64)
        insert_call = tf.raw_ops.LookupTableInsertV2(table_handle=table, keys=keys, values=values)
        with tf.control_dependencies([insert_call]):
            y = tf.raw_ops.LookupTableFindV2(table_handle=table, keys=x, default_value=default_value)
        tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
        tensor_info_y = tf.compat.v1.saved_model.utils.build_tensor_info(y)
        (signature_def_map, init_op, assets_collection) = ({'serving_default': tf.compat.v1.saved_model.signature_def_utils.build_signature_def(inputs={'x': tensor_info_x}, outputs={'y': tensor_info_y}, method_name='some_function')}, tf.compat.v1.tables_initializer(), None)
        sess = tf.compat.v1.Session()
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(saved_model_dir)
        builder.add_meta_graph_and_variables(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], signature_def_map, main_op=init_op, assets_collection=assets_collection, strip_default_attrs=True)
        builder.save()
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.enable_eager_execution()
        return saved_model_dir

    @test_util.run_v2_only
    def testModelWithMutableHashTable(self):
        if False:
            return 10
        "Test a model with saved_model's session initializer for hash tables."
        saved_model_dir = self._createV1ModelWithMutableHashTable()
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_data = np.array(['a', 'b', 'c'], dtype=np.string_)
        interpreter.resize_tensor_input(input_details[0]['index'], [3], strict=False)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual([1, 5, -1], list(actual_value))

    @test_util.run_v2_only
    def testReduceSumWithInt16Quant(self):
        if False:
            return 10
        'Test a model with quantized int16 reduce sum op.'
        inp = tf.keras.Input([3, 3], 3, name='x')
        m = tf.keras.Model(inp, tf.reduce_sum(inp, axis=-1))
        converter = tf.lite.TFLiteConverter.from_keras_model(m)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        converter.inference_input_type = tf.int16
        converter.inference_output_type = tf.int16
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        inputs = {i.name: np.random.normal(size=i.shape).astype(np.float32) for i in m.inputs}
        converter.representative_dataset = lambda : [inputs]
        content = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=content)
        runner = interpreter.get_signature_runner('serving_default')
        y = runner(x=np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]).astype(np.int16))
        self.assertEqual([3, 6, 9], list(list(y.values())[0]))

    @test_util.run_v2_only
    def testConstModel(self):
        if False:
            i = 10
            return i + 15
        'Test a basic model with functions to make sure functions are inlined.'
        input_data = tf.constant(1.0, shape=[1])
        root = autotrackable.AutoTrackable()
        root.f = tf.function(lambda x: 2.0 * x)
        to_save = root.f.get_concrete_function(input_data)
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, to_save)
        converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
        tflite_model = converter.convert()
        expected_value = root.f(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
        self.assertEqual(expected_value.numpy(), actual_value)

    @test_util.run_v2_only
    def testVariableModel(self):
        if False:
            i = 10
            return i + 15
        'Test a basic model with Variables with saving/loading the SavedModel.'
        root = self._getSimpleVariableModel()
        input_data = tf.constant(1.0, shape=[1])
        to_save = root.f.get_concrete_function(input_data)
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, to_save)
        converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
        tflite_model = converter.convert()
        metadata = get_conversion_metadata(tflite_model)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.environment.modelType, metadata_fb.ModelType.TF_SAVED_MODEL)
        expected_value = root.f(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
        self.assertEqual(expected_value.numpy(), actual_value)

    @parameterized.named_parameters(('EnableResourceVariables', True), ('DisableResourceVariables', False))
    @test_util.run_v2_only
    def testNativeVariablesModel(self, enable_resource_variables):
        if False:
            i = 10
            return i + 15
        'Test a basic model with Variables with saving/loading the SavedModel.'
        root = self._getSimpleModelWithVariables()
        input_data = tf.constant(1.0, shape=[1, 10])
        to_save = root.assign_add.get_concrete_function(input_data)
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, to_save)
        converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
        converter.experimental_enable_resource_variables = enable_resource_variables
        if not enable_resource_variables:
            with self.assertRaises(convert.ConverterError) as error:
                tflite_model = converter.convert()
            self.assertIn('Variable constant folding is failed. Please consider using enabling `experimental_enable_resource_variables` flag in the TFLite converter object.', str(error.exception))
            return
        tflite_model = converter.convert()
        expected_value = root.assign_add(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
        for (tf_result, tflite_result) in zip(expected_value, actual_value[0]):
            self.assertAllClose(tf_result, tflite_result, atol=1e-05)

    @test_util.run_v2_only
    def testSignatures(self):
        if False:
            print('Hello World!')
        'Test values for `signature_keys` argument.'
        root = self._getSimpleVariableModel()
        input_data = tf.constant(1.0, shape=[1])
        to_save = root.f.get_concrete_function(input_data)
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, to_save)
        with self.assertRaises(ValueError) as error:
            _ = lite.TFLiteConverterV2.from_saved_model(save_dir, signature_keys=['INVALID'])
        self.assertIn("Invalid signature key 'INVALID'", str(error.exception))
        converter = lite.TFLiteConverterV2.from_saved_model(save_dir, signature_keys=[])
        tflite_model = converter.convert()
        expected_value = root.f(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
        self.assertEqual(expected_value.numpy(), actual_value)

    @test_util.run_v2_only
    def testSignatureDefsWithFullIntegerQuantization(self):
        if False:
            while True:
                i = 10
        tf_input_shape = (32, 32, 128)
        tflite_input_shape = (1,) + tf_input_shape
        (tf_saved_model_dir, input_name, output_name) = self._createV2QATSavedModel(tf_input_shape)
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        interpreter.resize_tensor_input(input_details['index'], tflite_input_shape)
        interpreter.allocate_tensors()
        signature_list = interpreter._get_full_signature_list()['serving_default']
        input_data = np.random.random(tflite_input_shape).astype(np.float32)
        result = self._evaluateTFLiteModelUsingSignatureDef(tflite_model, 'serving_default', {input_name: input_data})[output_name]
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model_quant = converter.convert()
        interpreter = Interpreter(model_content=tflite_model_quant)
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        interpreter.resize_tensor_input(input_details['index'], tflite_input_shape)
        interpreter.allocate_tensors()
        all_indices = {item['index'] for item in interpreter.get_tensor_details()}
        signature_list = interpreter._get_full_signature_list()['serving_default']
        input_tensor_indices = set(signature_list['inputs'].values())
        assert input_tensor_indices.issubset(all_indices)
        output_tensor_indices = set(signature_list['outputs'].values())
        assert output_tensor_indices.issubset(all_indices)
        input_data = np.random.random(tflite_input_shape)
        (input_scale, input_zero_point) = input_details['quantization']
        if (input_scale, input_zero_point) != (0.0, 0):
            input_data = input_data / input_scale + input_zero_point
            input_data = input_data.astype(input_details['dtype'])
        result_quant = self._evaluateTFLiteModelUsingSignatureDef(tflite_model_quant, 'serving_default', {input_name: input_data})[output_name]
        (output_scale, output_zero_point) = output_details['quantization']
        if (output_scale, output_zero_point) != (0.0, 0):
            result_quant = result_quant.astype(np.float32)
            result_quant = (result_quant - output_zero_point) * output_scale
        root_mean_squared = np.sqrt(np.mean((result - result_quant) ** 2))
        assert root_mean_squared < 1.0

    @test_util.run_v2_only
    def testSignatureDefs(self):
        if False:
            return 10
        'Test converting SignatureDef is correct and uses SignatureDef API.'
        root = self._getMultiFunctionModel()
        input_data_0 = tf.constant(1.0, shape=[1])
        input_data_1 = tf.constant(3.0, shape=[1])
        mul_add_func = root.mul_add.get_concrete_function(input_data_1, input_data_0)
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, {'mul_add': mul_add_func})
        converter = lite.TFLiteConverterV2.from_saved_model(save_dir, signature_keys=['mul_add'])
        tflite_model = converter.convert()
        expected_value = root.mul_add(input_data_1, input_data_0)
        interpreter = Interpreter(model_content=tflite_model)
        signature_defs = interpreter.get_signature_list()
        results = self._evaluateTFLiteModelUsingSignatureDef(tflite_model, 'mul_add', {'y': input_data_0, 'x': input_data_1})
        self.assertEqual(list(results.keys()), ['output_0'])
        self.assertEqual(expected_value.numpy(), results['output_0'])
        self.assertEqual(len(signature_defs), 1)
        self.assertEqual(list(signature_defs.keys()), ['mul_add'])
        self.assertEqual(len(signature_defs.values()), 1)
        self.assertEqual(list(signature_defs['mul_add'].keys()), ['inputs', 'outputs'])
        self.assertCountEqual(signature_defs['mul_add']['inputs'], ['x', 'y'])
        self.assertEqual(list(signature_defs['mul_add']['outputs']), ['output_0'])

    @test_util.run_v2_only
    def testSignatureDefsWithDefaultValue(self):
        if False:
            print('Hello World!')
        'Test converting SignatureDef is correct and uses SignatureDef API.\n\n    This test uses None as signature_key to test default behavior.\n    '
        root = self._getMultiFunctionModel()
        input_data_0 = tf.constant(1.0, shape=[1])
        input_data_1 = tf.constant(3.0, shape=[1])
        mul_add_func = root.mul_add.get_concrete_function(input_data_1, input_data_0)
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, {'mul_add': mul_add_func})
        converter = lite.TFLiteConverterV2.from_saved_model(save_dir, signature_keys=['mul_add'])
        tflite_model = converter.convert()
        expected_value = root.mul_add(input_data_1, input_data_0)
        interpreter = Interpreter(model_content=tflite_model)
        signature_defs = interpreter.get_signature_list()
        results = self._evaluateTFLiteModelUsingSignatureDef(tflite_model, None, {'y': input_data_0, 'x': input_data_1})
        self.assertEqual(list(results.keys()), ['output_0'])
        self.assertEqual(expected_value.numpy(), results['output_0'])
        self.assertEqual(len(signature_defs), 1)
        self.assertEqual(list(signature_defs.keys()), ['mul_add'])
        self.assertEqual(len(signature_defs.values()), 1)
        self.assertEqual(list(signature_defs['mul_add'].keys()), ['inputs', 'outputs'])
        self.assertCountEqual(signature_defs['mul_add']['inputs'], ['x', 'y'])
        self.assertEqual(list(signature_defs['mul_add']['outputs']), ['output_0'])

    @test_util.run_v2_only
    def testSignatureDefsQuantizedModel(self):
        if False:
            i = 10
            return i + 15
        'Test converting SignatureDef on quantized model.'
        root = self._getMultiFunctionModel()
        input_data_0 = tf.constant(1.0, shape=[1])
        input_data_1 = tf.constant(3.0, shape=[1])
        mul_add_func = root.mul_add.get_concrete_function(input_data_1, input_data_0)
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, {'mul_add': mul_add_func})
        converter = lite.TFLiteConverterV2.from_saved_model(save_dir, signature_keys=['mul_add'])

        def representative_dataset_gen():
            if False:
                print('Hello World!')
            for _ in range(2):
                yield {'x': np.random.uniform(low=0, high=1, size=(1, 1)).astype(np.float32), 'y': np.random.uniform(low=0, high=1, size=(1, 1)).astype(np.float32)}
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
        signature_defs = interpreter.get_signature_list()
        self.assertEqual(len(signature_defs), 1)
        self.assertEqual(list(signature_defs.keys()), ['mul_add'])
        self.assertEqual(len(signature_defs.values()), 1)
        self.assertEqual(list(signature_defs['mul_add'].keys()), ['inputs', 'outputs'])
        self.assertCountEqual(signature_defs['mul_add']['inputs'], ['x', 'y'])
        self.assertEqual(list(signature_defs['mul_add']['outputs']), ['output_0'])

    @test_util.run_v2_only
    def testMultipleFunctionModel(self):
        if False:
            while True:
                i = 10
        'Convert multiple functions in a multi-functional model.'
        root = self._getMultiFunctionModel()
        input_data = tf.constant(1.0, shape=[1])
        add_func = root.add.get_concrete_function(input_data)
        sub_func = root.sub.get_concrete_function(input_data)
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, {'add': add_func, 'sub': sub_func})
        converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        signature_defs = interpreter.get_signature_list()
        self.assertEqual(len(signature_defs), 2)
        self.assertEqual(list(signature_defs.keys()), ['add', 'sub'])
        self.assertEqual(len(signature_defs.values()), 2)
        self.assertEqual(list(signature_defs['add'].keys()), ['inputs', 'outputs'])
        self.assertCountEqual(signature_defs['add']['inputs'], ['x'])
        self.assertEqual(list(signature_defs['add']['outputs']), ['output_0'])
        self.assertEqual(list(signature_defs['sub'].keys()), ['inputs', 'outputs'])
        self.assertCountEqual(signature_defs['sub']['inputs'], ['x'])
        self.assertEqual(list(signature_defs['sub']['outputs']), ['output_0'])
        add_signature_runner = interpreter.get_signature_runner('add')
        add_output = add_signature_runner(x=input_data)
        self.assertEqual(add_output['output_0'], 3)
        sub_signature_runner = interpreter.get_signature_runner('sub')
        sub_output = sub_signature_runner(x=input_data)
        self.assertEqual(sub_output['output_0'], -2)

    @parameterized.named_parameters(('_Default', False, False, dtypes.float32, False), ('_DefaultMlirQuant', False, False, dtypes.float32, True), ('_INT8InputOutput', False, False, dtypes.int8), ('_UINT8InputOutput', False, False, dtypes.uint8), ('_INT16Quantize_INT16InputOutput', False, True, dtypes.int16), ('_IntOnly_INT8InputOutput', True, False, dtypes.int8), ('_IntOnly_UINT8InputOutput', True, False, dtypes.uint8), ('_IntOnly_INT16Quantize_INT16InputOutput', True, True, dtypes.int16), ('_IntOnly_INT8InputOutputMlirQuant', True, False, dtypes.int8, True), ('_IntOnly_UINT8InputOutputMlirQuant', True, False, dtypes.uint8, True))
    @test_util.run_v2_only
    def testMultipleFunctionQuantizedModel(self, is_int_only, is_int16_quantize, inference_input_output_type, enable_mlir_quantizer=False):
        if False:
            i = 10
            return i + 15
        'Convert multiple functions in a multi-functional model.'
        root = self._getMultiFunctionModel()
        input_data = tf.constant(1.0, shape=[1])
        add_func = root.add.get_concrete_function(input_data)
        sub_func = root.sub.get_concrete_function(input_data)
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, {'add': add_func, 'sub': sub_func})
        converter = lite.TFLiteConverterV2.from_saved_model(save_dir)

        def representative_dataset_gen():
            if False:
                print('Hello World!')
            for _ in range(2):
                yield ('add', {'x': np.random.uniform(low=0, high=1, size=(1,)).astype(np.float32)})
            for _ in range(2):
                yield ('sub', {'x': np.random.uniform(low=0, high=1, size=(1,)).astype(np.float32)})
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        if is_int_only:
            if is_int16_quantize:
                converter.target_spec.supported_ops = [lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            else:
                converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
        elif is_int16_quantize:
            converter.target_spec.supported_ops = [lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        else:
            converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = inference_input_output_type
        converter.inference_output_type = inference_input_output_type
        converter.experimental_new_quantizer = enable_mlir_quantizer
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        signature_defs = interpreter.get_signature_list()
        self.assertEqual(len(signature_defs), 2)
        self.assertEqual(list(signature_defs.keys()), ['add', 'sub'])
        self.assertEqual(len(signature_defs.values()), 2)
        self.assertEqual(list(signature_defs['add'].keys()), ['inputs', 'outputs'])
        self.assertCountEqual(signature_defs['add']['inputs'], ['x'])
        self.assertEqual(list(signature_defs['add']['outputs']), ['output_0'])
        self.assertEqual(list(signature_defs['sub'].keys()), ['inputs', 'outputs'])
        self.assertCountEqual(signature_defs['sub']['inputs'], ['x'])
        self.assertEqual(list(signature_defs['sub']['outputs']), ['output_0'])
        input_data = tf.constant(np.random.uniform(-1, 1, size=(1,)).astype(inference_input_output_type.as_numpy_dtype))
        add_signature_runner = interpreter.get_signature_runner('add')
        add_output = add_signature_runner(x=input_data)
        self.assertIsNotNone(add_output['output_0'])
        input_details = add_signature_runner.get_input_details()
        self.assertLen(input_details, 1)
        self.assertStartsWith(input_details['x']['name'], 'add_x:0')
        self.assertEqual(inference_input_output_type.as_numpy_dtype, input_details['x']['dtype'])
        self.assertTrue(([1] == input_details['x']['shape']).all())
        if inference_input_output_type == dtypes.float32:
            self.assertEqual((0.0, 0), input_details['x']['quantization'])
        sub_signature_runner = interpreter.get_signature_runner('sub')
        sub_output = sub_signature_runner(x=input_data)
        self.assertIsNotNone(sub_output['output_0'])
        output_details = sub_signature_runner.get_output_details()
        self.assertLen(output_details, 1)
        self.assertStartsWith(output_details['output_0']['name'], 'StatefulPartitionedCall_1:0')
        self.assertEqual(inference_input_output_type.as_numpy_dtype, output_details['output_0']['dtype'])
        self.assertTrue(([1] == output_details['output_0']['shape']).all())
        if inference_input_output_type == dtypes.float32:
            self.assertEqual((0.0, 0), output_details['output_0']['quantization'])

    @test_util.run_v2_only
    def testMultipleFunctionModelWithSharedWeight(self):
        if False:
            while True:
                i = 10
        'Convert multiple functions with the shared weight.'
        root = self._getMultiFunctionModelWithSharedWeight()
        input_data = tf.constant(1.0, shape=[1])
        add_func = root.add.get_concrete_function(input_data)
        sub_func = root.sub.get_concrete_function(input_data)
        mul_func = root.mul.get_concrete_function(input_data)
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, {'add': add_func, 'sub': sub_func, 'mul': mul_func})
        converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        self.assertLess(len(tflite_model), 1100000)
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        signature_defs = interpreter.get_signature_list()
        self.assertLen(signature_defs, 3)
        add_signature_runner = interpreter.get_signature_runner('add')
        sub_signature_runner = interpreter.get_signature_runner('sub')
        mul_signature_runner = interpreter.get_signature_runner('mul')
        self.assertIsNotNone(add_signature_runner)
        self.assertIsNotNone(sub_signature_runner)
        self.assertIsNotNone(mul_signature_runner)

    @test_util.run_v2_only
    def testNoConcreteFunctionModel(self):
        if False:
            print('Hello World!')
        root = self._getMultiFunctionModel()
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir)
        with self.assertRaises(ValueError) as error:
            _ = lite.TFLiteConverterV2.from_saved_model(save_dir)
        self.assertIn('Only support at least one signature key.', str(error.exception))

    @test_util.run_v2_only
    def testKerasSequentialModel(self):
        if False:
            return 10
        'Test a simple sequential tf.Keras model.'
        input_data = tf.constant(1.0, shape=[1, 1])
        x = np.array([[1.0], [2.0]])
        y = np.array([[2.0], [4.0]])
        model = tf.keras.models.Sequential([tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(1)])
        model.compile(optimizer='sgd', loss='mean_squared_error')
        model.fit(x, y, epochs=1)
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(model, save_dir)
        converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
        tflite_model = converter.convert()
        expected_value = model.predict(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
        self.assertEqual(expected_value, actual_value)

    @test_util.run_v2_only
    def testKerasSequentialModelExport(self):
        if False:
            for i in range(10):
                print('nop')
        'Test a simple sequential tf.Keras model with `model.export` usage.'
        input_data = tf.constant(1.0, shape=[1, 1])
        x = np.array([[1.0], [2.0]])
        y = np.array([[2.0], [4.0]])
        model = tf.keras.models.Sequential([tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(1)])
        model.compile(optimizer='sgd', loss='mean_squared_error')
        model.fit(x, y, epochs=1)
        export_dir = os.path.join(self.get_temp_dir(), 'exported_model')
        model.export(export_dir)
        converter = lite.TFLiteConverterV2.from_saved_model(export_dir)
        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model)
        signature_defs = interpreter.get_signature_list()
        self.assertLen(signature_defs, 1)
        self.assertEqual(next(iter(signature_defs)), 'serving_default')
        expected_value = model.predict(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
        self.assertEqual(expected_value, actual_value)

    @test_util.run_v2_only
    def testGraphDebugInfo(self):
        if False:
            while True:
                i = 10
        'Test a SavedModel has debug info captured.'
        input_data = tf.constant(1.0, shape=[1])
        root = autotrackable.AutoTrackable()
        root.f = tf.function(lambda x: 2.0 * x)
        to_save = root.f.get_concrete_function(input_data)
        options = save_options.SaveOptions(save_debug_info=True)
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, to_save, options)
        converter = lite.TFLiteConverterV2.from_saved_model(save_dir)
        converter.convert()
        self._assertValidDebugInfo(converter._debug_info)

    @test_util.run_v2_only
    def testNonStatefulConvLSTM2D(self):
        if False:
            print('Hello World!')
        'Test saved model with non stateful ConvLSTM2D keras layer.'
        model = tf.keras.Sequential([tf.keras.layers.ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True, stateful=False, batch_input_shape=(1, 1, 10, 10, 1))])
        model.compile()
        saved_model_dir = os.path.join(self.get_temp_dir(), 'conv_lstm_2d')
        model.save(saved_model_dir, save_format='tf', include_optimizer=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)

    @test_util.run_v2_only
    def testKerasConvLSTM2DWithMoreThanOneDilationRate(self):
        if False:
            i = 10
            return i + 15
        input_tensor = tf.keras.layers.Input(batch_size=8, shape=[9, 10, 11, 12], name='input_tensor', dtype=tf.float32)
        output = tf.keras.layers.ConvLSTM2D(filters=3, kernel_size=3, strides=1, padding='VALID', dilation_rate=2, use_bias=False, bias_initializer='ones', data_format='channels_last')(input_tensor)
        model = tf.keras.Model(inputs=[input_tensor], outputs=output)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        saved_model_dir = os.path.join(self.get_temp_dir(), 'conv_lstm_2d_with_dilation_rate')
        model.save(saved_model_dir, save_format='tf', include_optimizer=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)

    @test_util.run_v2_only
    def testKerasFullyConnectedOutputShape3D(self):
        if False:
            i = 10
            return i + 15
        'Create a simple FullyConnected Model with an output of three dimensions.'
        input_tensor = tf.keras.layers.Input(batch_size=1, shape=[3, 3], name='input_tensor', dtype=tf.float32)
        x = tf.quantization.fake_quant_with_min_max_args(input_tensor, -3.0, 3.0)
        x = tf.keras.layers.Dense(3)(x)
        x = tf.quantization.fake_quant_with_min_max_args(x, -3.0, 3.0)
        model = tf.keras.Model(input_tensor, x)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        saved_model_dir = os.path.join(self.get_temp_dir(), 'fully_connected_output_3d')
        model.save(saved_model_dir, save_format='tf', include_optimizer=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        output_details = interpreter.get_output_details()
        input_details = interpreter.get_input_details()
        interpreter.allocate_tensors()
        input_data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        expected_value = model.predict(input_data)
        self.assertLen(output_details[0]['shape_signature'], 3)
        self.assertAllClose(expected_value, actual_value, atol=0.1)
        self.assertEqual(list(output_details[0]['shape_signature']), list(model.layers[-1].output_shape))

    @test_util.run_v2_only
    def testKerasConv2DTransposedWithMismatchQuantizedAxes(self):
        if False:
            print('Hello World!')

        class QuantConv2DTransposed(tf.keras.layers.Layer):

            def build(self, input_shape):
                if False:
                    while True:
                        i = 10
                self.kernel = self.add_weight('kernel', [3, 3, input_shape[-1], 24])

            def call(self, inputs):
                if False:
                    return 10
                filters = tf.quantization.fake_quant_with_min_max_vars_per_channel(self.kernel, -3.0 * tf.ones([24]), 3.0 * tf.ones([24]), narrow_range=True)
                filters = tf.transpose(filters, (0, 1, 3, 2))
                return tf.nn.conv2d_transpose(inputs, filters, [*inputs.shape[:-1], 24], 1)
        inp = tf.keras.Input(shape=(6, 8, 48), batch_size=1)
        x = tf.quantization.fake_quant_with_min_max_vars(inp, -3.0, 3.0, narrow_range=True)
        x = QuantConv2DTransposed()(x)
        x = tf.quantization.fake_quant_with_min_max_vars(x, -3.0, 3.0, narrow_range=True)
        model = tf.keras.Model(inp, x)
        saved_model_dir = os.path.join(self.get_temp_dir(), 'keras_conv2d_transpose')
        model.save(saved_model_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        with self.assertRaises(convert.ConverterError) as error:
            _ = converter.convert()
        self.assertIn('mismatched quantized axes of input and output', str(error.exception))

    def _createModelWithInputShape(self, shape):
        if False:
            return 10
        'Create a simple SavedModel with a certain shape.'
        saved_model_dir = os.path.join(self.get_temp_dir(), 'input_shape_model')
        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                unknown_shape = tf.TensorShape(shape)
                in_tensor = tf.compat.v1.placeholder(shape=unknown_shape, dtype=tf.float32, name='input')
                out_tensor = in_tensor + in_tensor
                inputs = {'input': in_tensor}
                outputs = {'output': out_tensor}
                saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
        return saved_model_dir

    @test_util.run_v2_only
    def testUnknownInputShapeModel(self):
        if False:
            return 10
        'Test a SavedModel with an unknown input shape.'
        saved_model_dir = self._createModelWithInputShape(None)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)
        tflite_model_obj = _convert_bytearray_to_object(tflite_model)
        for tensor in tflite_model_obj.subgraphs[0].tensors:
            self.assertEqual(False, tensor.hasRank)
            self.assertEqual([], tensor.shape.tolist())
        interpreter = Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        interpreter.resize_tensor_input(input_details[0]['index'], [3], strict=False)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual([2.0, 4.0, 6.0], list(actual_value))

    @test_util.run_v2_only
    def testScalarInputShapeModel(self):
        if False:
            print('Hello World!')
        'Test a SavedModel with a scalar input.'
        saved_model_dir = self._createModelWithInputShape([])
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)
        tflite_model_obj = _convert_bytearray_to_object(tflite_model)
        for tensor in tflite_model_obj.subgraphs[0].tensors:
            self.assertEqual(True, tensor.hasRank)
            self.assertEqual([], tensor.shape.tolist())

    @test_util.run_v2_only
    def testMatrixInputShapeModel(self):
        if False:
            return 10
        'Test a SavedModel with a matrix input.'
        saved_model_dir = self._createModelWithInputShape([2, 3])
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)
        tflite_model_obj = _convert_bytearray_to_object(tflite_model)
        for tensor in tflite_model_obj.subgraphs[0].tensors:
            self.assertEqual(True, tensor.hasRank)
            self.assertEqual([2, 3], tensor.shape.tolist())

    @parameterized.named_parameters(('_PerChannelQuant', False, False), ('_PerChannelMlirQuant', False, True), ('_PerTensorQuant', True, False), ('_PerTensorMlirQuant', True, True), ('_PerChannelDynamicRange', False, False, True), ('_PerTensorDynamicRange', True, False, True))
    @test_util.run_v2_only
    def testDisablePerChannelQuantization(self, disable_per_channel=False, enable_mlir_quantizer=False, representative_dataset=True):
        if False:
            i = 10
            return i + 15
        k_num_filters = 38
        model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(k_num_filters, (3, 3), activation='relu')])
        model.build(input_shape=(1, 5, 5, 3))
        saved_model_dir = os.path.join(self.get_temp_dir(), 'conv_saved_model')
        save(model, saved_model_dir)
        k_conv_name = 'sequential/conv2d/Conv2D'
        quantized_converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        if representative_dataset:

            def calib_gen():
                if False:
                    print('Hello World!')
                for _ in range(5):
                    yield [np.random.uniform(-1, 1, size=(1, 5, 5, 3)).astype(np.float32)]
            quantized_converter.representative_dataset = calib_gen
        quantized_converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS]
        quantized_converter.experimental_new_quantizer = enable_mlir_quantizer
        if disable_per_channel:
            quantized_converter._experimental_disable_per_channel = disable_per_channel
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        detail = next((d for d in interpreter.get_tensor_details() if d['name'].startswith(k_conv_name)))
        quant_params = detail['quantization_parameters']
        expected_num_params = k_num_filters
        if disable_per_channel:
            expected_num_params = 1
        self.assertLen(quant_params['scales'], expected_num_params)
        self.assertLen(quant_params['zero_points'], expected_num_params)

    @parameterized.named_parameters(('_INT8Quant_INT32Bias', False, False, dtypes.int32, True), ('_INT16Quant_INT64Bias', True, False, dtypes.int64, True), ('_INT8Quant_INT32Bias_Set', False, True, dtypes.int32, True), ('_INT8Quant_INT64Bias_Set', False, True, dtypes.int64, False), ('_INT16Quant_INT32Bias_Set', True, True, dtypes.int32, True), ('_INT16Quant_INT64Bias_Set', True, True, dtypes.int64, True), ('_INT16Quant_FLOAT32Bias_Set', True, True, dtypes.float32, False))
    @test_util.run_v2_only
    def testBiasQuantization(self, is_int16_quantize, explicitly_set_bias, bias_type, is_valid_bias_type):
        if False:
            i = 10
            return i + 15
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(1024, input_shape=[1024], activation=None, bias_initializer='ones')])
        saved_model_dir = os.path.join(self.get_temp_dir(), 'dense_saved_model')
        save(model, saved_model_dir)
        k_dense_bias_name = 'sequential/dense/BiasAdd/ReadVariableOp'
        quantized_converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        if explicitly_set_bias:
            quantized_converter._experimental_full_integer_quantization_bias_type = bias_type
        if is_int16_quantize:
            quantized_converter.target_spec.supported_ops = [lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        else:
            quantized_converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]

        def calibration_gen():
            if False:
                while True:
                    i = 10
            for _ in range(5):
                yield [np.random.randn(1, 1024).astype(np.float32)]
        quantized_converter.representative_dataset = calibration_gen
        if not is_valid_bias_type:
            with self.assertRaisesRegex(ValueError, 'Expected bias type to be'):
                quantized_converter.convert()
            return
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        dense_bias = next((d for d in interpreter.get_tensor_details() if d['name'].startswith(k_dense_bias_name)))
        self.assertEqual(bias_type, dense_bias['dtype'])

    @parameterized.named_parameters(('_Int8PerChannelMlirDynamicRangeQuant', True, False, False), ('_Int8PerChannelTocoDynamicRangeQuant', False, False, False), ('_Int8PerTensorMlirDynamicRangeQuant', True, True, False), ('_Int8PerTensorTocoDynamicRangeQuant', False, True, False), ('_Float16DynamicRangeQuant', True, False, True))
    @test_util.run_v2_only
    def testMlirDynamicRangeQuantization(self, enable_new_dynamic_range_quantizer, disable_per_channel, enable_float16_quant):
        if False:
            i = 10
            return i + 15
        num_filters = 1024
        conv_name = 'sequential/conv2d/Conv2D'
        model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu')])
        model.build(input_shape=(1, 32, 32, 3))
        saved_model_dir = self.create_tempdir()
        save(model, saved_model_dir.full_path)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir.full_path)
        converter.optimizations = [lite.Optimize.DEFAULT]
        converter.experimental_new_dynamic_range_quantizer = enable_new_dynamic_range_quantizer
        converter._experimental_disable_per_channel = disable_per_channel
        if enable_float16_quant:
            converter.target_spec.supported_types = [tf.float16]
        quantized_tflite_model = converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        quantized_weight = None
        quantized_weight_with_one_postfix = None
        quantized_weight_without_one_postfix = None
        for d in interpreter.get_tensor_details():
            if d['name'] == conv_name + '1':
                quantized_weight = d
                quantized_weight_with_one_postfix = d
                break
        for d in interpreter.get_tensor_details():
            if d['name'].startswith(conv_name):
                if quantized_weight is None:
                    quantized_weight = d
                quantized_weight_without_one_postfix = d
                break
        self.assertIsNotNone(quantized_weight)
        quant_params = quantized_weight['quantization_parameters']
        if enable_float16_quant:
            expected_num_params = 0
        else:
            expected_num_params = 1 if disable_per_channel else num_filters
        self.assertLen(quant_params['scales'], expected_num_params)
        self.assertLen(quant_params['zero_points'], expected_num_params)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        if enable_float16_quant:
            self.assertTrue(quantized_weight_with_one_postfix is not None and np.float16 == quantized_weight_with_one_postfix['dtype'] or (quantized_weight_without_one_postfix is not None and np.float16 == quantized_weight_without_one_postfix['dtype']))
        else:
            self.assertEqual(np.int8, quantized_weight['dtype'])
    'disable test for now '
    "@parameterized.named_parameters(\n      (\n          '_Float16Quantization',\n          _PresetQuantizationMethod.FLOAT16,\n      ),\n  )\n  @test_util.run_v2_only\n  def testMlirStableHLOPresetQuantizationMethod(\n      self, preset_quantization_method\n  ):\n    num_filters = 38\n    model = tf.keras.models.Sequential(\n        [tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu')]\n    )\n    model.build(input_shape=(1, 5, 5, 3))\n    saved_model_dir = os.path.join(self.get_temp_dir(), 'conv_saved_model')\n    save(model, saved_model_dir)\n\n    quantization_options = quant_opts_pb2.QuantizationOptions(\n        quantization_method=quant_opts_pb2.QuantizationMethod(\n            preset_quantization_method=quant_opts_pb2.PresetQuantizationMethod(\n                preset_method=preset_quantization_method\n            )\n        )\n    )\n\n    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)\n    converter._experimental_quantization_options = quantization_options\n\n    converter.target_spec.supported_ops = [\n        tf.lite.OpsSet.EXPERIMENTAL_STABLEHLO_OPS\n    ]\n    converter.exclude_conversion_metadata = True\n    converter.optimizations = [lite.Optimize.DEFAULT]\n    quantized_stablehlo_model = converter.convert()\n    self.assertIsNotNone(quantized_stablehlo_model)\n\n  @test_util.run_v2_only\n  def testMlirStableHLOCustomQuantizationMethod(self):\n    num_filters = 38\n    model = tf.keras.models.Sequential(\n        [tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu')]\n    )\n    model.build(input_shape=(1, 5, 5, 3))\n    saved_model_dir = os.path.join(self.get_temp_dir(), 'conv_saved_model')\n    save(model, saved_model_dir)\n\n    quantization_options = quant_opts_pb2.QuantizationOptions(\n        quantization_method=quant_opts_pb2.QuantizationMethod(\n            custom_quantization_method=quant_opts_pb2.CustomQuantizationMethod(\n                quantization_component_spec=[\n                    quant_opts_pb2.QuantizationComponentSpec(\n                        quantization_component=quant_opts_pb2.QuantizationComponentSpec.QuantizationComponent.COMPONENT_WEIGHT,\n                        bit_width=quant_opts_pb2.QuantizationComponentSpec.BitWidth.BIT_WIDTH_16,\n                        bit_type=quant_opts_pb2.QuantizationComponentSpec.BitType.BIT_TYPE_FLOAT,\n                    ),\n                    quant_opts_pb2.QuantizationComponentSpec(\n                        quantization_component=quant_opts_pb2.QuantizationComponentSpec.QuantizationComponent.COMPONENT_BIAS,\n                        bit_width=quant_opts_pb2.QuantizationComponentSpec.BitWidth.BIT_WIDTH_16,\n                    ),\n                ]\n            )\n        )\n    )\n\n    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)\n    converter._experimental_quantization_options = quantization_options\n\n    converter.target_spec.supported_ops = [\n        tf.lite.OpsSet.EXPERIMENTAL_STABLEHLO_OPS\n    ]\n    converter.exclude_conversion_metadata = True\n    converter.optimizations = [lite.Optimize.DEFAULT]\n    quantized_stablehlo_model = converter.convert()\n    self.assertIsNotNone(quantized_stablehlo_model)"

class FromKerasModelTest(lite_v2_test_util.ModelTest):

    @parameterized.named_parameters(('EnableMlirVariableQuantizationNumState1', True, 1), ('DisablMlirVariableQuantizationNumState1', False, 1), ('EnableMlirVariableQuantizationNumState2', True, 2), ('DisablMlirVariableQuantizationNumState2', False, 2))
    @test_util.run_v2_only
    def testVariableQuantization(self, variable_quantization, number_of_states):
        if False:
            while True:
                i = 10
        k_readvariable_name = 'model/read_assign/concat/ReadVariableOp'
        (model, calibration_gen) = self._createReadAssignModel(number_of_states)
        converter = lite.TFLiteConverterV2.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = calibration_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter._experimental_variable_quantization = variable_quantization
        quantized_tflite_model = converter.convert()
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        detail = next((d for d in interpreter.get_tensor_details() if d['name'].startswith(k_readvariable_name)))
        quant_params = detail['quantization_parameters']
        if variable_quantization:
            expected_num_params = 1
        else:
            expected_num_params = 0
        self.assertLen(quant_params['scales'], expected_num_params)
        self.assertLen(quant_params['zero_points'], expected_num_params)

    @parameterized.named_parameters(('EnableMlirVariableQuantizationNumState1', True, 1), ('DisablMlirVariableQuantizationNumState1', False, 1), ('EnableMlirVariableQuantizationNumState2', True, 2), ('DisablMlirVariableQuantizationNumState2', False, 2))
    def testVariableQuantizationInFloat16(self, variable_quantization, number_of_states):
        if False:
            for i in range(10):
                print('nop')
        (model, _) = self._createReadAssignModel(number_of_states)
        converter = lite.TFLiteConverterV2.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter._experimental_variable_quantization = variable_quantization
        if variable_quantization:
            with self.assertRaises(ValueError) as error:
                converter.convert()
            self.assertIn('`_experimental_variable_quantization` is only supported for full', str(error.exception))
        else:
            converter.convert()

    @test_util.run_v2_only
    def testSequentialModel(self):
        if False:
            while True:
                i = 10
        'Test a simple sequential tf.Keras model.'
        input_data = tf.constant(1.0, shape=[1, 1])
        x = np.array([[1.0], [2.0]])
        y = np.array([[2.0], [4.0]])
        model = tf.keras.models.Sequential([tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(units=1, input_shape=[1])])
        model.compile(optimizer='sgd', loss='mean_squared_error')
        model.fit(x, y, epochs=1)
        converter = lite.TFLiteConverterV2.from_keras_model(model)
        tflite_model = converter.convert()
        metadata = get_conversion_metadata(tflite_model)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.environment.modelType, metadata_fb.ModelType.KERAS_MODEL)
        expected_value = model.predict(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
        self.assertEqual(expected_value, actual_value)

    @test_util.run_v2_only
    def testSequentialMultiInputOutputModel(self):
        if False:
            return 10
        'Test a tf.Keras model with multiple inputs and outputs.'
        left_input_data = tf.constant(1.0, shape=[1, 3])
        right_input_data = tf.constant(1.0, shape=[1, 3])
        input_a_np = np.random.random((10, 3))
        input_b_np = np.random.random((10, 3))
        output_c_np = np.random.random((10, 3))
        output_d_np = np.random.random((10, 2))
        input_a = tf.keras.layers.Input(shape=(3,), name='input_a')
        input_b = tf.keras.layers.Input(shape=(3,), name='input_b')
        dense = tf.keras.layers.Dense(8, name='dense_1')
        interm_a = dense(input_a)
        interm_b = dense(input_b)
        merged = tf.keras.layers.concatenate([interm_a, interm_b], name='merge')
        output_c = tf.keras.layers.Dense(3, activation='softmax', name='dense_2')(merged)
        output_d = tf.keras.layers.Dense(2, activation='softmax', name='dense_3')(merged)
        model = tf.keras.models.Model(inputs=[input_a, input_b], outputs=[output_c, output_d])
        model.compile(optimizer='sgd', loss='mean_squared_error')
        model.fit([input_a_np, input_b_np], [output_c_np, output_d_np], epochs=1)
        converter = lite.TFLiteConverterV2.from_keras_model(model)
        tflite_model = converter.convert()
        input_data = [left_input_data, right_input_data]
        expected_value = model.predict(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, input_data)
        for (tf_result, tflite_result) in zip(expected_value, actual_value):
            self.assertAllClose(tf_result, tflite_result, atol=1e-05)

    @test_util.run_v2_only
    def testGraphDebugInfo(self):
        if False:
            return 10
        'Test a tf.Keras model has debug info captured.'
        x = [-1, 0, 1, 2, 3, 4]
        y = [-3, -1, 1, 3, 5, 7]
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
        model.compile(optimizer='sgd', loss='mean_squared_error')
        model.fit(x, y, epochs=1)
        converter = lite.TFLiteConverterV2.from_keras_model(model)
        converter.convert()
        self._assertValidDebugInfo(converter._debug_info)

    @test_util.run_v2_only
    def testKerasFallbackPath(self):
        if False:
            i = 10
            return i + 15
        'Test keras model which failed when exporting to the saved model.'
        input_data = tf.constant(np.array(np.random.random_sample(20), dtype=np.float32))

        class Model(tf.keras.Model):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super(Model, self).__init__()
                self.shared_weights = self.add_weight(name=None, shape=(20, 1), dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=300 ** (-0.5)))

            def call(self, x):
                if False:
                    return 10
                return tf.add(self.shared_weights, x)
        model = Model()
        model.compile(optimizer='sgd', loss='mean_squared_error')
        model.fit(input_data, input_data, epochs=1)
        converter = lite.TFLiteConverterV2.from_keras_model(model)
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)

    @test_util.run_v2_only
    def testSignatureDefs(self):
        if False:
            print('Hello World!')
        'Test converting SignatureDef is correct and uses SignatureDef API.'
        keras_model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3), name='tensor'), tf.keras.layers.Dense(10, name='output_tensor')])
        converter = lite.TFLiteConverterV2.from_keras_model(keras_model)
        tflite_model = converter.convert()
        input_data = tf.constant(np.random.uniform(-1, 1, size=(1, 32, 32, 3)).astype(np.float32))
        expected_value = keras_model(input_data)
        interpreter = Interpreter(model_content=tflite_model)
        signature_defs = interpreter.get_signature_list()
        results = self._evaluateTFLiteModelUsingSignatureDef(tflite_model, 'serving_default', {'tensor_input': input_data})
        self.assertEqual(list(results.keys()), ['output_tensor'])
        self.assertAllClose(expected_value.numpy(), results['output_tensor'])
        self.assertEqual(len(signature_defs), 1)
        self.assertEqual(list(signature_defs.keys()), ['serving_default'])
        self.assertEqual(len(signature_defs.values()), 1)
        self.assertEqual(list(signature_defs['serving_default'].keys()), ['inputs', 'outputs'])
        self.assertCountEqual(signature_defs['serving_default']['inputs'], ['tensor_input'])
        self.assertEqual(list(signature_defs['serving_default']['outputs']), ['output_tensor'])

    @parameterized.named_parameters(('_PerChannelMlirDynamicRangeQuant', True, False, False), ('_PerChannelTocoDynamicRangeQuant', False, False, False), ('_PerTensorMlirDynamicRangeQuant', True, True, False), ('_PerTensorTocoDynamicRangeQuant', False, True, False), ('_Float16DynamicRangeQuant', True, False, True))
    @test_util.run_v2_only
    def testMlirDynamicRangeQuantization(self, enable_new_dynamic_range_quantizer, disable_per_channel, enable_float16_quant):
        if False:
            i = 10
            return i + 15
        num_filters = 1024
        conv_name = 'sequential/conv2d/Conv2D'
        model = tf.keras.models.Sequential([tf.keras.Input(shape=(32, 32, 3)), tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu')])
        model.build()
        converter = lite.TFLiteConverterV2.from_keras_model(model)
        converter.optimizations = [lite.Optimize.DEFAULT]
        converter.experimental_new_dynamic_range_quantizer = enable_new_dynamic_range_quantizer
        converter._experimental_disable_per_channel = disable_per_channel
        if enable_float16_quant:
            converter.target_spec.supported_types = [tf.float16]
        quantized_tflite_model = converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()
        quantized_weight = None
        quantized_weight_with_one_postfix = None
        quantized_weight_without_one_postfix = None
        for d in interpreter.get_tensor_details():
            if d['name'] == conv_name + '1':
                quantized_weight = d
                quantized_weight_with_one_postfix = d
                break
        for d in interpreter.get_tensor_details():
            if d['name'].startswith(conv_name):
                if quantized_weight is None:
                    quantized_weight = d
                quantized_weight_without_one_postfix = d
                break
        self.assertIsNotNone(quantized_weight)
        quant_params = quantized_weight['quantization_parameters']
        if enable_float16_quant:
            expected_num_params = 0
        else:
            expected_num_params = 1 if disable_per_channel else num_filters
        self.assertLen(quant_params['scales'], expected_num_params)
        self.assertLen(quant_params['zero_points'], expected_num_params)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertEqual(np.float32, output_details[0]['dtype'])
        if enable_float16_quant:
            self.assertTrue(quantized_weight_with_one_postfix is not None and np.float16 == quantized_weight_with_one_postfix['dtype'] or (quantized_weight_without_one_postfix is not None and np.float16 == quantized_weight_without_one_postfix['dtype']))
        else:
            self.assertEqual(np.int8, quantized_weight['dtype'])

    @parameterized.named_parameters([('{}BitWeightOnly={}LowBit={}'.format(num_bits, weight_only, low_bit), num_bits, weight_only, low_bit) for (num_bits, weight_only, low_bit) in itertools.product((5, 7, 6), (True, False), (True, False))])
    @test_util.run_v2_only
    def testQATLowBitKerasModel(self, num_bits, weight_only, low_bit):
        if False:
            for i in range(10):
                print('nop')
        bit_max = (1 << num_bits - 1) - 1
        bit_min = -bit_max
        tf_input_shape = (5, 5, 3)
        tflite_input_shape = (1,) + tf_input_shape
        (model, input_name, output_name) = self._createV2QATLowBitKerasModel(tf_input_shape, weight_only, num_bits, bit_min, bit_max)
        input_data = np.linspace(0, 6, np.prod(tflite_input_shape)).reshape(tflite_input_shape)
        tf_result = model(input_data)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if low_bit:
            converter._experimental_low_bit_qat = True
        tflite_model = converter.convert()
        result = self._evaluateTFLiteModelUsingSignatureDef(tflite_model, 'serving_default', {input_name: input_data.astype(np.float32)})[output_name]
        self.assertAllClose([np.linalg.norm(result - tf_result.numpy().astype(np.float32))], [0.0])
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        num_8bit_activations = 0
        num_8bit_weights = 0
        kernel_name = 'model/conv_wrapper/Conv2D;model/conv_wrapper/FakeQuantWithMinMaxVarsPerChannel'
        for detail in interpreter.get_tensor_details():
            if detail['dtype'] == np.int8 and detail['name'] and (detail['name'] == kernel_name):
                num_8bit_weights += 1
                weights = interpreter.get_tensor(detail['index'])
                if low_bit:
                    self.assertFalse((bit_min > weights).any() or (weights > bit_max).any())
                else:
                    self.assertTrue((bit_min > weights).any() or (weights > bit_max).any())
                self.assertIn('scales', detail['quantization_parameters'])
                if low_bit and detail['quantization_parameters']['scales']:
                    self.assertAllClose(detail['quantization_parameters']['scales'], [1.0])
            elif detail['dtype'] == np.int8 and detail['name']:
                self.assertFalse(weight_only)
                self.assertIn('scales', detail['quantization_parameters'])
                if detail['quantization_parameters']['scales']:
                    self.assertAllClose(detail['quantization_parameters']['scales'], [6 / 255])
                num_8bit_activations += 1
        self.assertEqual(num_8bit_weights, 0 if weight_only and (not low_bit) else 1)
        self.assertEqual(num_8bit_activations, 0 if weight_only else 3)

    @test_util.run_v2_only
    def testKerasConv2DTransposedWithBiasAndActivation(self):
        if False:
            return 10

        class QuantConv2DTransposedWithBiasAndActivation(tf.keras.layers.Layer):

            def build(self, input_shape):
                if False:
                    i = 10
                    return i + 15
                self.kernel = self.add_weight('kernel', (3, 3, input_shape[-1], 3))
                self.bias = self.add_weight('bias', (3,))

            def call(self, inputs):
                if False:
                    print('Hello World!')
                filters = tf.quantization.fake_quant_with_min_max_vars(self.kernel, -3.0, 3.0, narrow_range=True)
                filters = tf.transpose(filters, (0, 1, 3, 2))
                result = tf.nn.conv2d_transpose(inputs, filters, [*inputs.shape[:-1], 3], 1)
                result = tf.nn.bias_add(result, self.bias)
                result = tf.nn.relu(result)
                return tf.quantization.fake_quant_with_min_max_vars(result, -3.0, 3.0, narrow_range=True)
        inp = tf.keras.Input(shape=(6, 8, 6), batch_size=1)
        x = tf.quantization.fake_quant_with_min_max_vars(inp, -3.0, 3.0, narrow_range=True)
        x = QuantConv2DTransposedWithBiasAndActivation()(x)
        model = tf.keras.Model(inp, x)
        tf_input_shape = (1, 6, 8, 6)
        input_data = np.linspace(0, 6, np.prod(tf_input_shape)).reshape(tf_input_shape)
        tf_result = model(input_data)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converted_model = converter.convert()
        tf.lite.experimental.Analyzer.analyze(model_content=converted_model)
        interpreter = tf.lite.Interpreter(model_content=converted_model)
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]['index']
        output_index = interpreter.get_output_details()[0]['index']
        interpreter.set_tensor(input_index, input_data.astype(np.float32))
        interpreter.invoke()
        tflite_result = interpreter.tensor(output_index)()
        self.assertAllClose([np.linalg.norm(tflite_result - tf_result.numpy().astype(np.float32))], [0.0])
        num_float32_tensor = 0
        for detail in interpreter.get_tensor_details():
            if detail['dtype'] == np.float32:
                num_float32_tensor += 1
        self.assertEqual(num_float32_tensor, 2)

class FromJaxModelTest(lite_v2_test_util.ModelTest):

    @test_util.run_v2_only
    def testInvalidInputsModel(self):
        if False:
            print('Hello World!')
        if DISABLE_JAX_TEST:
            return

        def simple_model(input1, input2):
            if False:
                for i in range(10):
                    print('nop')
            return jnp.sin(input1) + jnp.cos(input2)
        input_tensor = jnp.zeros([10, 10])
        converter = lite.TFLiteConverterV2.experimental_from_jax(None, [{'input1': input_tensor}])
        with self.assertRaisesRegex(ValueError, 'No serving func is specified.'):
            converter.convert()
        converter = lite.TFLiteConverterV2.experimental_from_jax([simple_model], None)
        with self.assertRaisesRegex(ValueError, 'Input tensors are not specified.'):
            converter.convert()
        converter = lite.TFLiteConverterV2.experimental_from_jax([simple_model], [])
        with self.assertRaisesRegex(ValueError, 'Input tensors are not specified.'):
            converter.convert()
        converter = lite.TFLiteConverterV2.experimental_from_jax([simple_model], input_tensor)
        with self.assertRaisesRegex(ValueError, 'The truth value of an array with more than one element is ambiguous.'):
            converter.convert()
        converter = lite.TFLiteConverterV2.experimental_from_jax([simple_model], [[('input1', input_tensor)]])
        with self.assertRaisesRegex(ValueError, 'Failed to convert the given Jax function to hlo.'):
            converter.convert()
        converter = lite.TFLiteConverterV2.experimental_from_jax([simple_model, simple_model], [[('input1', input_tensor), ('input2', input_tensor)]])
        with self.assertRaisesRegex(ValueError, 'Input tensor mapping len 1 does not match serving func len 2.'):
            converter.convert()
        converter = lite.TFLiteConverterV2.experimental_from_jax([simple_model, simple_model], [[('input1', input_tensor), ('input2', input_tensor)], [('input1', input_tensor), ('input2', input_tensor)]])
        with self.assertRaisesRegex(ValueError, 'Currently only support single serving function.'):
            converter.convert()

    @test_util.run_v2_only
    def testSingleInputModel(self):
        if False:
            print('Hello World!')
        if DISABLE_JAX_TEST:
            return

        def single_input(input_tensor):
            if False:
                return 10
            return jnp.sin(input_tensor)
        input_tensor = jnp.zeros([10, 10])
        converter = lite.TFLiteConverterV2.experimental_from_jax([single_input], [[('input_tensor', input_tensor)]])
        tflite_model = converter.convert()
        metadata = get_conversion_metadata(tflite_model)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.environment.modelType, metadata_fb.ModelType.JAX)
        input_data = np.random.random_sample((10, 10))
        tf_input_data = tf.constant(input_data, dtype=np.float32)
        actual_value = self._evaluateTFLiteModel(tflite_model, [tf_input_data])[0]
        expected_value = single_input(input_data)
        self.assertAllClose(expected_value, actual_value, atol=1e-05)

    @test_util.run_v2_only
    def testMultipleInputsModel(self):
        if False:
            i = 10
            return i + 15
        if DISABLE_JAX_TEST:
            return

        def multiple_inputs(input1, input2):
            if False:
                print('Hello World!')
            return input1 + input2
        input1 = jnp.zeros([10, 10])
        input2 = jnp.zeros([10, 1])
        converter = lite.TFLiteConverterV2.experimental_from_jax([multiple_inputs], [[('input1', input1), ('input2', input2)]])
        tflite_model = converter.convert()
        input1_data = np.random.random_sample((10, 10))
        tf_input1_data = tf.constant(input1_data, dtype=np.float32)
        input2_data = np.random.random_sample((10, 1))
        tf_input2_data = tf.constant(input2_data, dtype=np.float32)
        actual_value = self._evaluateTFLiteModel(tflite_model, [tf_input1_data, tf_input2_data])[0]
        expected_value = multiple_inputs(input1_data, input2_data)
        self.assertAllClose(expected_value, actual_value, atol=1e-05)

    @test_util.run_v2_only
    def testInputSignaturesModel(self):
        if False:
            print('Hello World!')
        if DISABLE_JAX_TEST:
            return

        def multiple_inputs(input1, input2):
            if False:
                print('Hello World!')
            return input1 + input2
        input1 = jnp.zeros([10, 10])
        input2 = jnp.zeros([10, 1])
        converter = lite.TFLiteConverterV2.experimental_from_jax([multiple_inputs], [[('input1', input1), ('input2', input2)]])
        tflite_model = converter.convert()
        input1_data = np.random.random_sample((10, 10))
        tf_input1_data = tf.constant(input1_data, dtype=np.float32)
        input2_data = np.random.random_sample((10, 1))
        tf_input2_data = tf.constant(input2_data, dtype=np.float32)
        actual_value = self._evaluateTFLiteModel(tflite_model, [tf_input1_data, tf_input2_data])[0]
        expected_value = multiple_inputs(input1_data, input2_data)
        self.assertAllClose(expected_value, actual_value, atol=1e-05)

    @test_util.run_v2_only
    def testModelWithParams(self):
        if False:
            while True:
                i = 10
        if DISABLE_JAX_TEST:
            return

        def model(inputs, weights):
            if False:
                while True:
                    i = 10
            return jnp.matmul(weights, inputs)
        weights = np.random.random_sample((10, 10))
        serving_func = functools.partial(model, weights=weights)
        input_tensor = jnp.zeros([10, 10])
        converter = lite.TFLiteConverterV2.experimental_from_jax([serving_func], [[('inputs', input_tensor)]])
        tflite_model = converter.convert()
        input_data = np.random.random_sample((10, 10))
        tf_input_data = tf.constant(input_data, dtype=np.float32)
        actual_value = self._evaluateTFLiteModel(tflite_model, [tf_input_data])[0]
        expected_value = serving_func(input_data)
        self.assertAllClose(expected_value, actual_value, atol=1e-05)

    @test_util.run_v2_only
    def testWhileLoop(self):
        if False:
            i = 10
            return i + 15
        if DISABLE_JAX_TEST:
            return

        def condition(x):
            if False:
                print('Hello World!')
            return jnp.sum(x, keepdims=False) < 100

        def body(x):
            if False:
                i = 10
                return i + 15
            return jnp.add(x, 2.0)

        def model(x):
            if False:
                return 10
            result = jax.lax.while_loop(condition, body, x)
            return result[0]
        input_tensor = jnp.zeros([3, 3])
        converter = lite.TFLiteConverterV2.experimental_from_jax([model], [[('x', input_tensor)]])
        tflite_model = converter.convert()
        input_data = np.random.random_sample((3, 3))
        tf_input_data = tf.constant(input_data, dtype=np.float32)
        actual_value = self._evaluateTFLiteModel(tflite_model, [tf_input_data])[0]
        expected_value = model(input_data)
        self.assertAllClose(expected_value, actual_value, atol=1e-05)

class ControlFlowTest(lite_v2_test_util.ModelTest):

    @test_util.run_v2_only
    def testCond(self):
        if False:
            print('Hello World!')
        input_data = {'x': tf.constant([1.0, 2.0], shape=[1, 2]), 'b': tf.constant(True)}
        weights = tf.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float32)

        def true_fn(x):
            if False:
                i = 10
                return i + 15
            return tf.matmul(x, weights)

        def false_fn(x):
            if False:
                print('Hello World!')
            return tf.add(x, weights)

        @tf.function(input_signature=[tf.TensorSpec(shape=[1, 2], dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.bool)])
        def model(x, b):
            if False:
                return 10
            return tf.cond(b, true_fn=lambda : true_fn(x), false_fn=lambda : false_fn(x))
        concrete_func = model.get_concrete_function()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], model)
        tflite_model = converter.convert()
        expected_value = concrete_func(**input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data['x'], input_data['b']])[0]
        self.assertAllClose(expected_value, actual_value)

    @test_util.run_v2_only
    def testCondWithFullIntegerQuantization(self):
        if False:
            for i in range(10):
                print('nop')
        weights = tf.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float32)

        def true_fn(x):
            if False:
                i = 10
                return i + 15
            return tf.matmul(x, weights)

        def false_fn(x):
            if False:
                while True:
                    i = 10
            return tf.add(x, weights)

        @tf.function(input_signature=[tf.TensorSpec(shape=[1, 2], dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.bool)])
        def model(x, b):
            if False:
                print('Hello World!')
            return tf.cond(b, true_fn=lambda : true_fn(x), false_fn=lambda : false_fn(x))

        def calibration_gen():
            if False:
                for i in range(10):
                    print('nop')
            for _ in range(5):
                yield [np.random.uniform(-1, 1, size=(1, 2)).astype(np.float32), tf.constant(True)]
            for _ in range(5):
                yield [np.random.uniform(-1, 1, size=(1, 2)).astype(np.float32), tf.constant(False)]
        concrete_func = model.get_concrete_function()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = calibration_gen
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)

    @test_util.run_v2_only
    def testConverterErrorOnControlFlowV1Ops(self):
        if False:
            for i in range(10):
                print('nop')
        filename = resource_loader.get_path_to_datafile('testdata/control_flow_v1_saved_model')
        converter = lite.TFLiteConverterV2.from_saved_model(filename)
        with self.assertRaises(convert.ConverterError) as error:
            converter.convert()
        self.assertIn('Failed to functionalize Control Flow V1 ops. Consider using Control Flow V2 ops instead. See https://www.tensorflow.org/api_docs/python/tf/compat/v1/enable_control_flow_v2.', str(error.exception))

    @test_util.run_v2_only
    def testStaticRnn(self):
        if False:
            while True:
                i = 10
        input_data = tf.constant(np.array(np.random.random_sample((3, 10)), dtype=np.float32))
        cell = tf.keras.layers.LSTMCell(10)

        @tf.function(input_signature=[tf.TensorSpec(shape=[3, 10], dtype=tf.float32)])
        def model(x):
            if False:
                return 10
            seq = tf.split(x, 3, 0)
            return rnn.static_rnn(cell, seq, dtype=tf.float32, sequence_length=[1])
        concrete_func = model.get_concrete_function()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], model)
        tflite_model = converter.convert()
        expected_value = concrete_func(input_data)[0]
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
        for (expected, actual) in zip(expected_value, actual_value):
            self.assertAllClose(expected, actual)

    @test_util.run_v2_only
    def testWhileLoop(self):
        if False:
            while True:
                i = 10
        input_data = tf.constant([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
        weights = tf.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float32)

        def condition(x):
            if False:
                while True:
                    i = 10
            return tf.reduce_sum(x) < 100

        def body(x):
            if False:
                for i in range(10):
                    print('nop')
            return tf.add(x, weights)

        @tf.function(input_signature=[tf.TensorSpec(shape=[2, 2], dtype=tf.float32)])
        def model(x):
            if False:
                return 10
            return tf.while_loop(condition, body, [x])
        concrete_func = model.get_concrete_function()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], model)
        tflite_model = converter.convert()
        expected_value = concrete_func(input_data)[0]
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]
        self.assertAllClose(expected_value, actual_value)

    @test_util.run_v2_only
    def testDynamicRnn(self):
        if False:
            print('Hello World!')
        input_data = tf.constant(np.array(np.random.random_sample((3, 10, 10)), dtype=np.float32))
        cell = tf.keras.layers.LSTMCell(10)

        @tf.function(input_signature=[tf.TensorSpec(shape=[3, 10, 10], dtype=tf.float32)])
        def model(x):
            if False:
                while True:
                    i = 10
            rnn_layer = tf.keras.layers.RNN([cell], return_sequences=True)
            return rnn_layer(x)
        concrete_func = model.get_concrete_function()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], model)
        tflite_model = converter.convert()
        expected_value = concrete_func(input_data)
        lite_outputs = self._evaluateTFLiteModel(tflite_model, [input_data])
        self.assertLen(lite_outputs, 1)
        actual_value = lite_outputs[0]
        for (expected, actual) in zip(expected_value, actual_value):
            self.assertAllClose(expected, actual)

    @parameterized.named_parameters(('LSTMBatchSizeOne', tf.keras.layers.LSTM, True), ('LSTM', tf.keras.layers.LSTM, False), ('SimpleRNNBatchSizeOne', tf.keras.layers.SimpleRNN, True), ('SimpleRNN', tf.keras.layers.SimpleRNN, False), ('GRUBatchSizeOne', tf.keras.layers.GRU, True), ('GRU', tf.keras.layers.GRU, False))
    @test_util.run_v2_only
    def testKerasRNN(self, rnn_layer, default_to_single_batch):
        if False:
            return 10
        input_data = tf.constant(np.array(np.random.random_sample((1, 10, 10)), dtype=np.float32))
        rnn_obj = rnn_layer(units=10, input_shape=(10, 10))
        model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(10, 10), name='input'), rnn_obj])
        converter = lite.TFLiteConverterV2.from_keras_model(model)
        converter._experimental_default_to_single_batch_in_tensor_list_ops = default_to_single_batch
        if not default_to_single_batch:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]
        expected_value = model.predict(input_data)
        self.assertAllClose(expected_value, actual_value, atol=1e-05)

    @parameterized.named_parameters(('LSTM', tf.keras.layers.LSTM), ('SimpleRNN', tf.keras.layers.SimpleRNN), ('GRU', tf.keras.layers.GRU))
    @test_util.run_v2_only
    def testKerasRNNMultiBatches(self, rnn_layer):
        if False:
            return 10
        input_data = tf.constant(np.array(np.random.random_sample((4, 10, 10)), dtype=np.float32))
        x = tf.keras.layers.Input(batch_shape=(4, 10, 10))
        y = rnn_layer(units=10, input_shape=(10, 10))(x)
        model = tf.keras.Model(inputs=[x], outputs=[y])
        converter = lite.TFLiteConverterV2.from_keras_model(model)
        tflite_model = converter.convert()
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]
        expected_value = model.predict(input_data)
        self.assertAllClose(expected_value, actual_value, atol=1e-05)

    @parameterized.named_parameters(('ForceToUseBatchSizeOne', True), ('DontForceToUseBatchSizeOne', False))
    @test_util.run_v2_only
    def testKerasBidirectionalRNNReturnSequence(self, default_to_single_batch):
        if False:
            print('Hello World!')
        input_data = tf.constant(np.array(np.random.random_sample((1, 10, 10)), dtype=np.float32))
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(10, 10), name='input'))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=10, return_sequences=True), input_shape=(10, 10)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(5))
        model.add(tf.keras.layers.Activation('softmax'))
        converter = lite.TFLiteConverterV2.from_keras_model(model)
        converter._experimental_default_to_single_batch_in_tensor_list_ops = default_to_single_batch
        if not default_to_single_batch:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]
        expected_value = model.predict(input_data)
        self.assertAllClose(expected_value, actual_value, atol=1e-05)

    @parameterized.named_parameters(('ForceToUseBatchSizeOne', True), ('DontForceToUseBatchSizeOne', False))
    @test_util.run_v2_only
    def testKerasBidirectionalRNN(self, default_to_single_batch):
        if False:
            return 10
        input_data = tf.constant(np.array(np.random.random_sample((1, 10, 10)), dtype=np.float32))
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(10, 10), name='input'))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=10)))
        model.add(tf.keras.layers.Dense(5))
        model.add(tf.keras.layers.Activation('softmax'))
        converter = lite.TFLiteConverterV2.from_keras_model(model)
        converter._experimental_default_to_single_batch_in_tensor_list_ops = default_to_single_batch
        if not default_to_single_batch:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]
        expected_value = model.predict(input_data)
        self.assertAllClose(expected_value, actual_value, atol=1e-05)

class StridedSliceTest(lite_v2_test_util.ModelTest):

    @test_util.run_v2_only
    def testStridedSlice(self):
        if False:
            while True:
                i = 10
        input_data = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6], shape=[6], dtype=tf.float32)
        begin = tf.Variable([1], dtype=tf.int32)

        @tf.function(input_signature=[tf.TensorSpec(shape=[6], dtype=tf.float32), tf.TensorSpec(shape=[1], dtype=tf.int32)])
        def model(a, begin):
            if False:
                print('Hello World!')
            return tf.strided_slice(a, begin, begin + 3)
        concrete_func = model.get_concrete_function()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], model)
        tflite_model = converter.convert()
        expected_value = concrete_func(input_data, begin)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data, begin])[0]
        self.assertAllClose(expected_value, actual_value)

class GrapplerTest(lite_v2_test_util.ModelTest):

    @test_util.run_v2_only
    def testConstantFolding(self):
        if False:
            print('Hello World!')
        input_data = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], shape=[3, 3])

        @tf.function
        def func(x):
            if False:
                return 10
            y_const = tf.constant([1.0, 2.0, 3.0])
            y_broadcast = tf.broadcast_to(y_const, [3, 3])
            return tf.matmul(x, y_broadcast)
        root = autotrackable.AutoTrackable()
        root.f = func
        concrete_func = root.f.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        tflite_model = converter.convert()
        expected_value = root.f(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]
        self.assertAllClose(expected_value, actual_value)
        converter.optimizations = [lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])[0]
        self.assertAllClose(expected_value, actual_value)

class UnknownShapes(lite_v2_test_util.ModelTest):

    @test_util.run_v2_only
    def testMatMul(self):
        if False:
            i = 10
            return i + 15
        input_data = tf.constant(np.array(np.random.random_sample((10, 4)), dtype=np.float32))

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 4], dtype=tf.float32)])
        def model(in_tensor):
            if False:
                for i in range(10):
                    print('nop')
            shape = tf.shape(in_tensor)
            fill = tf.transpose(tf.fill(shape, 1.0))
            return tf.matmul(fill, in_tensor)
        concrete_func = model.get_concrete_function()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], model)
        tflite_model = converter.convert()
        expected_value = concrete_func(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data], input_shapes=[([-1, 4], [10, 4])])[0]
        self.assertAllClose(expected_value, actual_value, atol=1e-06)

    def _getIntegerQuantizeModelWithUnknownShapes(self):
        if False:
            print('Hello World!')
        np.random.seed(0)

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 33], dtype=tf.float32)])
        def model(input_tensor):
            if False:
                i = 10
                return i + 15
            'Define a model with tf.MatMul and unknown shapes.'
            const_tensor = tf.constant(np.random.uniform(low=-10.0, high=10.0, size=[33, 33]), shape=[33, 33], dtype=tf.float32, name='inputB')
            shape = tf.shape(input_tensor)
            fill = tf.transpose(tf.fill(shape, 1.0))
            mult = tf.matmul(fill, input_tensor)
            return tf.matmul(mult, const_tensor)
        root = autotrackable.AutoTrackable()
        root.f = model
        concrete_func = root.f.get_concrete_function()

        def calibration_gen():
            if False:
                for i in range(10):
                    print('nop')
            for batch in range(5, 20, 5):
                for _ in range(5):
                    yield [np.random.uniform(-1, 1, size=(batch, 33)).astype(np.float32)]
        return (root, concrete_func, calibration_gen)

    @test_util.run_v2_only
    def testMatMulQuantize(self):
        if False:
            for i in range(10):
                print('nop')
        (root, concrete_func, _) = self._getIntegerQuantizeModelWithUnknownShapes()
        float_converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        float_tflite_model = float_converter.convert()
        quantized_converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_tflite_model = quantized_converter.convert()
        quantized_interpreter = Interpreter(model_content=quantized_tflite_model)
        quantized_interpreter.allocate_tensors()
        input_details = quantized_interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([-1, 33], input_details[0]['shape_signature'])
        self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

    @test_util.run_v2_only
    def testMatMulCalibrateAndQuantize(self):
        if False:
            while True:
                i = 10
        (root, concrete_func, calibration_gen) = self._getIntegerQuantizeModelWithUnknownShapes()
        float_converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        float_tflite_model = float_converter.convert()
        quantized_converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        quantized_converter.optimizations = [lite.Optimize.DEFAULT]
        quantized_converter.representative_dataset = calibration_gen
        quantized_tflite_model = quantized_converter.convert()
        quantized_interpreter = Interpreter(model_content=quantized_tflite_model)
        quantized_interpreter.allocate_tensors()
        input_details = quantized_interpreter.get_input_details()
        self.assertLen(input_details, 1)
        self.assertEqual(np.float32, input_details[0]['dtype'])
        self.assertAllEqual([-1, 33], input_details[0]['shape_signature'])
        self.assertLess(len(quantized_tflite_model), len(float_tflite_model))

    def testBatchMatMul(self):
        if False:
            return 10
        input_data_1 = tf.constant(np.array(np.random.random_sample((1, 256, 256)), dtype=np.float32))
        input_data_2 = tf.constant(np.array(np.random.random_sample((1, 256, 256)), dtype=np.float32))

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 256, 256], dtype=tf.float32), tf.TensorSpec(shape=[None, 256, 256], dtype=tf.float32)])
        def model(in_tensor_1, in_tensor_2):
            if False:
                i = 10
                return i + 15
            return tf.matmul(in_tensor_1, in_tensor_2)
        concrete_func = model.get_concrete_function()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], model)
        tflite_model = converter.convert()
        expected_value = concrete_func(input_data_1, input_data_2)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data_1, input_data_2], input_shapes=[([-1, 256, 256], [1, 256, 256])])[0]
        self.assertAllClose(expected_value, actual_value, atol=4)

    def testBatchMatMulInputInt8Int8OutputInt32(self):
        if False:
            while True:
                i = 10
        input_data_1 = tf.constant(np.array(np.random.random_integers(-128, high=127, size=(1, 20, 30)), dtype=np.int8))
        input_data_2 = tf.constant(np.array(np.random.random_integers(-128, high=127, size=(1, 30, 10)), dtype=np.int8))

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 20, 30], dtype=tf.int8), tf.TensorSpec(shape=[None, 30, 10], dtype=tf.int8)])
        def model(in_tensor_1, in_tensor_2):
            if False:
                return 10
            return tf.matmul(in_tensor_1, in_tensor_2, output_type=tf.int32)
        concrete_func = model.get_concrete_function()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], model)
        tflite_model = converter.convert()
        expected_value = concrete_func(input_data_1, input_data_2)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data_1, input_data_2], input_shapes=[([-1, 20, 30], [1, 20, 30]), ([-1, 30, 10], [1, 30, 10])])[0]
        self.assertAllEqual(expected_value, actual_value)

    def testBatchMatMulHybrid(self):
        if False:
            return 10
        input_data = tf.constant(np.array(np.random.random_sample((1, 256, 128)), dtype=np.float32))

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 256, 128], dtype=tf.float32)])
        def model(in_tensor):
            if False:
                print('Hello World!')
            rhs = tf.constant(np.array(np.random.random_sample((1, 128, 256)), dtype=np.float32))
            return tf.matmul(in_tensor, rhs)
        concrete_func = model.get_concrete_function()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        expected_value = concrete_func(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data], input_shapes=[([-1, 256, 128], [1, 256, 128])])[0]
        self.assertAllClose(expected_value, actual_value, atol=4)

    def testSizeInvalid(self):
        if False:
            for i in range(10):
                print('nop')

        @tf.function(input_signature=[tf.TensorSpec(shape=[1, None, 16, 3], dtype=tf.float32)])
        def model(in_tensor):
            if False:
                for i in range(10):
                    print('nop')
            return in_tensor + in_tensor
        concrete_func = model.get_concrete_function()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], model)
        converter.experimental_new_converter = False
        with self.assertRaises(ValueError) as error:
            converter.convert()
        self.assertEqual("None is only supported in the 1st dimension. Tensor 'in_tensor' has invalid shape '[1, None, 16, 3]'.", str(error.exception))

class ResourceAndVariantTypes(lite_v2_test_util.ModelTest):

    @test_util.run_v2_only
    def testVariants(self):
        if False:
            return 10

        @tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.float32)])
        def model(v):
            if False:
                while True:
                    i = 10
            m = map_ops.empty_tensor_map()
            k = tf.constant(1.0)
            p = tf.add(k, v)
            with ops.control_dependencies([m]):
                m2 = map_ops.tensor_map_insert(m, p, v)
                with ops.control_dependencies([m2]):
                    return map_ops.tensor_map_size(m2)
        concrete_func = model.get_concrete_function()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        input_data = np.array([1.0], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(1, actual_value)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(1, actual_value)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(1, actual_value)

    @test_util.run_v2_only
    def testVariantsWithCond(self):
        if False:
            while True:
                i = 10

        def create_v1_saved_model():
            if False:
                return 10
            saved_model_dir = os.path.join(self.get_temp_dir(), 'variants_with_cond')
            with tf.Graph().as_default():
                with tf.compat.v1.Session() as sess:
                    m = map_ops.empty_tensor_map()

                    def body(i, m):
                        if False:
                            i = 10
                            return i + 15
                        m = map_ops.tensor_map_insert(m, i, i)
                        return (i + 1, m)
                    in_tensor = tf.compat.v1.placeholder(shape=[1], dtype=tf.int32, name='input')
                    (_, result_m) = tf.cond(in_tensor < 10, lambda : body(in_tensor, m), lambda : body(in_tensor + 1, m))
                    out_tensor = in_tensor + map_ops.tensor_map_size(result_m)
                    inputs = {'x': in_tensor}
                    outputs = {'z': out_tensor}
                    saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
            return saved_model_dir
        saved_model_dir = create_v1_saved_model()
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        input_data = np.array([0], dtype=np.int32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        expected_value = np.array([1], dtype=np.int32)
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(expected_value, actual_value)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(expected_value, actual_value)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(expected_value, actual_value)

    @test_util.run_v2_only
    def testVariantsWithWhile(self):
        if False:
            for i in range(10):
                print('nop')

        def create_v1_saved_model():
            if False:
                for i in range(10):
                    print('nop')
            saved_model_dir = os.path.join(self.get_temp_dir(), 'variants_with_while')
            with tf.Graph().as_default():
                with tf.compat.v1.Session() as sess:
                    m = map_ops.empty_tensor_map()

                    def cond(i, m):
                        if False:
                            for i in range(10):
                                print('nop')
                        del m
                        return i < 10

                    def body(i, m):
                        if False:
                            i = 10
                            return i + 15
                        m = map_ops.tensor_map_insert(m, i, i)
                        return (i + 1, m)
                    (_, result_m) = tf.while_loop(cond, body, [0, m])
                    in_tensor = tf.compat.v1.placeholder(shape=[1], dtype=tf.int32, name='input')
                    out_tensor = in_tensor + map_ops.tensor_map_size(result_m)
                    inputs = {'x': in_tensor}
                    outputs = {'z': out_tensor}
                    saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
            return saved_model_dir
        saved_model_dir = create_v1_saved_model()
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        input_data = np.array([0], dtype=np.int32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(10, actual_value)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(10, actual_value)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(10, actual_value)

    @test_util.run_v2_only
    def testResources(self):
        if False:
            while True:
                i = 10

        def create_v1_saved_model():
            if False:
                i = 10
                return i + 15
            saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_resources')
            with tf.Graph().as_default():
                with tf.compat.v1.Session() as sess:
                    in_tensor = tf.compat.v1.placeholder(shape=[1], dtype=tf.float32, name='input')
                    stack = tf.raw_ops.StackV2(max_size=10, elem_type=tf.float32)
                    w = tf.raw_ops.StackPushV2(handle=stack, elem=in_tensor)
                    with ops.control_dependencies([w]):
                        a = in_tensor + in_tensor
                        with ops.control_dependencies([a]):
                            out_tensor = a + tf.raw_ops.StackPopV2(handle=stack, elem_type=tf.float32)
                    inputs = {'x': in_tensor}
                    outputs = {'z': out_tensor}
                    saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
            return saved_model_dir
        saved_model_dir = create_v1_saved_model()
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        input_data = np.array([1.0], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(3.0, actual_value)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(3.0, actual_value)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(3.0, actual_value)

    @test_util.run_v2_only
    def testResourcesWithCond(self):
        if False:
            return 10

        def create_v1_saved_model():
            if False:
                print('Hello World!')
            saved_model_dir = os.path.join(self.get_temp_dir(), 'resources_with_cond')
            with tf.Graph().as_default():
                with tf.compat.v1.Session() as sess:
                    in_tensor = tf.compat.v1.placeholder(shape=[1], dtype=tf.float32, name='input')

                    def body(i, arr):
                        if False:
                            return 10
                        n = tf.raw_ops.StackPushV2(handle=arr, elem=tf.cast(i, dtype=tf.float32))
                        return (n, arr)
                    arr = tf.raw_ops.StackV2(max_size=10, elem_type=tf.float32)
                    (n, result_arr) = tf.cond(in_tensor < 10, lambda : body(0, arr), lambda : body(1, arr))
                    with ops.control_dependencies([result_arr, n]):
                        out_tensor = tf.raw_ops.StackPopV2(handle=result_arr, elem_type=tf.float32)
                    inputs = {'x': in_tensor}
                    outputs = {'a': out_tensor}
                    saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
            return saved_model_dir
        saved_model_dir = create_v1_saved_model()
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        input_data = np.array([1.0], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(0.0, actual_value)

    @test_util.run_v2_only
    def testResourcesWithWhile(self):
        if False:
            i = 10
            return i + 15

        def create_v1_saved_model():
            if False:
                while True:
                    i = 10
            saved_model_dir = os.path.join(self.get_temp_dir(), 'resources_with_while')
            with tf.Graph().as_default():
                with tf.compat.v1.Session() as sess:
                    in_tensor = tf.compat.v1.placeholder(shape=[1], dtype=tf.float32, name='input')

                    def cond(i, arr, m):
                        if False:
                            i = 10
                            return i + 15
                        del arr
                        del m
                        return i < 10

                    def body(i, arr, m):
                        if False:
                            print('Hello World!')
                        del m
                        n = tf.raw_ops.StackPushV2(handle=arr, elem=tf.cast(i, dtype=tf.float32))
                        return (i + 1, arr, n)
                    arr = tf.raw_ops.StackV2(max_size=10, elem_type=tf.float32)
                    (_, result_arr, n) = tf.while_loop(cond, body, [0, arr, 0.0])
                    with ops.control_dependencies([result_arr, n]):
                        out_tensor = tf.raw_ops.StackPopV2(handle=result_arr, elem_type=tf.float32)
                    inputs = {'x': in_tensor}
                    outputs = {'a': out_tensor}
                    saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
            return saved_model_dir
        saved_model_dir = create_v1_saved_model()
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        input_data = np.array([1.0], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(9.0, actual_value)

    @parameterized.named_parameters(('EnableLoweringTensorListOps', True), ('DisableLoweringTensorListOps', False))
    @test_util.run_v2_only
    def testTensorListWithStaticSize(self, lower_tensor_list_ops):
        if False:
            while True:
                i = 10

        def create_v1_saved_model():
            if False:
                i = 10
                return i + 15
            saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_mutable_variable')
            with tf.Graph().as_default():
                with tf.compat.v1.Session() as sess:
                    in_tensor = tf.compat.v1.placeholder(shape=[1], dtype=tf.float32, name='input')
                    ta = tf.TensorArray(tf.float32, size=3, dynamic_size=False, clear_after_read=False)
                    ta = ta.write(0, 10.0)
                    ta = ta.write(1, 20.0)
                    ta = ta.write(2, 30.0)
                    out_tensor = ta.read(0) + ta.read(2)
                    inputs = {'x': in_tensor}
                    outputs = {'z': out_tensor}
                    saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
            return saved_model_dir
        saved_model_dir = create_v1_saved_model()
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        if not lower_tensor_list_ops:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_lower_tensor_list_ops = lower_tensor_list_ops
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        input_data = np.array([1.0], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(40.0, actual_value)

    @parameterized.named_parameters(('EnableLoweringTensorListOps', True), ('DisableLoweringTensorListOps', False))
    @test_util.run_v2_only
    def testTensorListWithDynamicSize(self, lower_tensor_list_ops):
        if False:
            for i in range(10):
                print('nop')

        def create_v1_saved_model():
            if False:
                i = 10
                return i + 15
            saved_model_dir = os.path.join(self.get_temp_dir(), 'simple_mutable_variable')
            with tf.Graph().as_default():
                with tf.compat.v1.Session() as sess:
                    in_tensor = tf.compat.v1.placeholder(shape=[1], dtype=tf.float32, name='input')
                    ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
                    ta = ta.write(0, 10.0)
                    ta = ta.write(1, 20.0)
                    ta = ta.write(2, 30.0)
                    out_tensor = ta.read(0) + ta.read(2)
                    inputs = {'x': in_tensor}
                    outputs = {'z': out_tensor}
                    saved_model.simple_save(sess, saved_model_dir, inputs, outputs)
            return saved_model_dir
        saved_model_dir = create_v1_saved_model()
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        if lower_tensor_list_ops:
            with self.assertRaises(convert.ConverterError) as error:
                converter.convert()
            self.assertIn('Lowering tensor list ops is failed. Please consider using Select TF ops and disabling `_experimental_lower_tensor_list_ops` flag in the TFLite converter object.', str(error.exception))
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        input_data = np.array([1.0], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(40.0, actual_value)

class CalibrateAndQuantizeWithCustomOpTest(lite_v2_test_util.ModelTest):

    def _createGraphWithCustomOp(self):
        if False:
            return 10
        np.random.seed(0)
        saved_model_dir = os.path.join(self.get_temp_dir(), 'double_model')
        with ops.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                in_tensor = tf.compat.v1.placeholder(shape=[1, 4], dtype=dtypes.float32, name='input')
                out_tensor = double_op.double(in_tensor)
                inputs = {'x': in_tensor}
                outputs = {'z': out_tensor}
                saved_model.simple_save(sess, saved_model_dir, inputs, outputs)

        def calibration_gen():
            if False:
                while True:
                    i = 10
            for _ in range(100):
                yield [np.random.uniform(-1, 1, size=(1, 4)).astype(np.float32)]
        return (saved_model_dir, calibration_gen)

    def testCustomOpRegistererByName(self):
        if False:
            return 10
        'Test a calibration with custom op registered by name.'
        (saved_model_dir, calibration_gen) = self._createGraphWithCustomOp()
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        converter.optimizations = [lite.Optimize.DEFAULT]
        converter.representative_dataset = calibration_gen
        converter.allow_custom_ops = True
        converter.target_spec._experimental_custom_op_registerers = ['TF_TestRegisterer']
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)
        self.assertGreater(test_registerer.get_num_test_registerer_calls(), 0)
        self.assertIn('Double', tflite_test_util.get_ops_list(tflite_model))
        metadata = get_conversion_metadata(tflite_model)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.options.allowCustomOps, True)
        interpreter = InterpreterWithCustomOps(model_content=tflite_model, custom_op_registerers=['TF_TestRegisterer'])
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        test_input = np.array([[0.0, 0.1, 0.2, 0.3]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output_details = interpreter.get_output_details()
        expected_output = np.array([[0.0, 0.2, 0.4, 0.6]], dtype=np.float32)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        self.assertArrayNear(expected_output[0], output_data[0], err=0.01)

    def testCustomOpRegistererByFunc(self):
        if False:
            print('Hello World!')
        'Test a calibration with custom op registered by function.'
        (saved_model_dir, calibration_gen) = self._createGraphWithCustomOp()
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        converter.optimizations = [lite.Optimize.DEFAULT]
        converter.representative_dataset = calibration_gen
        converter.allow_custom_ops = True
        converter.target_spec._experimental_custom_op_registerers = [test_registerer.TF_TestRegisterer]
        tflite_model = converter.convert()
        self.assertTrue(tflite_model)
        self.assertGreater(test_registerer.get_num_test_registerer_calls(), 0)
        self.assertIn('Double', tflite_test_util.get_ops_list(tflite_model))
        interpreter = InterpreterWithCustomOps(model_content=tflite_model, custom_op_registerers=[test_registerer.TF_TestRegisterer])
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        test_input = np.array([[0.0, 0.1, 0.2, 0.3]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output_details = interpreter.get_output_details()
        expected_output = np.array([[0.0, 0.2, 0.4, 0.6]], dtype=np.float32)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        self.assertArrayNear(expected_output[0], output_data[0], err=0.01)

    def testCustomOpRegistererFailure(self):
        if False:
            return 10
        'Test a calibration with wrong custom op registerer.'
        (saved_model_dir, calibration_gen) = self._createGraphWithCustomOp()
        bogus_name = 'CompletelyBogusRegistererName'
        converter = lite.TFLiteConverterV2.from_saved_model(saved_model_dir)
        converter.optimizations = [lite.Optimize.DEFAULT]
        converter.representative_dataset = calibration_gen
        converter.allow_custom_ops = True
        converter.target_spec._experimental_custom_op_registerers = [bogus_name]
        with self.assertRaisesRegex(ValueError, "Looking up symbol '" + bogus_name + "' failed"):
            converter.convert()

class IntermediatesTest(lite_v2_test_util.ModelTest):

    def _run(self, experimental_preserve_all_tensors):
        if False:
            for i in range(10):
                print('nop')

        @tf.function
        def f(x):
            if False:
                print('Hello World!')
            y = tf.add(x, x, name='y')
            z = tf.add(y, y, name='z')
            w = tf.add(z, z, name='w')
            return w
        input_data = np.array(2.0, np.float32)
        concrete_func = f.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], f)
        tflite_model = converter.convert()
        interpreter = Interpreter(model_content=tflite_model, experimental_preserve_all_tensors=experimental_preserve_all_tensors)
        interpreter.allocate_tensors()
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
        interpreter.invoke()
        out = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        tensors = {}
        for t in interpreter.get_tensor_details():
            val = None
            try:
                val = interpreter.get_tensor(t['index'])
            except ValueError:
                pass
            tensors.update({t['name']: val})
        return (tensors, out)

    def testPreserve(self):
        if False:
            while True:
                i = 10
        (tensors, result) = self._run(experimental_preserve_all_tensors=True)
        self.assertAllClose(tensors['x'], 2.0)
        self.assertAllClose(tensors['y'], 4.0)
        self.assertAllClose(tensors['z'], 8.0)
        self.assertAllClose(result, 16.0)

    def testNoPreserve(self):
        if False:
            for i in range(10):
                print('nop')
        (tensors, result) = self._run(experimental_preserve_all_tensors=False)
        self.assertAllClose(tensors['x'], 2.0)
        self.assertTrue(tensors['y'] != 4.0 or tensors['z'] != 8.0)
        self.assertAllClose(result, 16.0)

class DatasetOpsTest(lite_v2_test_util.ModelTest):

    @test_util.run_v2_only
    def testReduceDataset(self):
        if False:
            while True:
                i = 10

        @tf.function
        def model():
            if False:
                while True:
                    i = 10
            dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])
            output = dataset.reduce(np.int32(0), lambda x, y: x + y)
            return output
        concrete_func = model.get_concrete_function()
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        interpreter = Interpreter(model_content=tflite_model)
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(10, actual_value)

class SparsityTest(lite_v2_test_util.ModelTest):

    def _getSparsificableModel(self, matrix_b_values):
        if False:
            return 10
        np.random.seed(0)
        root = autotrackable.AutoTrackable()

        @tf.function(input_signature=[tf.TensorSpec(shape=[16, 4], dtype=tf.float32)])
        def func(inp):
            if False:
                for i in range(10):
                    print('nop')
            matrix_b = tf.constant(matrix_b_values, dtype=tf.float32)
            matrix_b = tf.reshape(matrix_b, [4, 8])
            matmul = tf.matmul(inp, matrix_b, transpose_a=False, transpose_b=False)
            output = tf.nn.relu(matmul, name='output')
            return output
        root.f = func
        to_save = root.f.get_concrete_function()
        return (root, to_save)

    def testRandomSparsity(self):
        if False:
            return 10
        matrix_b_values = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        (root, func) = self._getSparsificableModel(matrix_b_values)
        float_converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
        float_converter.optimizations = [lite.Optimize.EXPERIMENTAL_SPARSITY]
        float_tflite_model = float_converter.convert()
        self.assertIsNotNone(float_tflite_model)
        metadata = get_conversion_metadata(float_tflite_model)
        self.assertIsNotNone(metadata)
        self.assertAllEqual([metadata_fb.ModelOptimizationMode.RANDOM_SPARSITY], metadata.options.modelOptimizationModes)

    def testBlockSparsity(self):
        if False:
            while True:
                i = 10
        matrix_b_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        (root, func) = self._getSparsificableModel(matrix_b_values)
        float_converter = lite.TFLiteConverterV2.from_concrete_functions([func], root)
        float_converter.optimizations = [lite.Optimize.EXPERIMENTAL_SPARSITY]
        float_tflite_model = float_converter.convert()
        self.assertIsNotNone(float_tflite_model)
        metadata = get_conversion_metadata(float_tflite_model)
        self.assertIsNotNone(metadata)
        self.assertAllEqual([metadata_fb.ModelOptimizationMode.BLOCK_SPARSITY], metadata.options.modelOptimizationModes)

    def testQuantizedBlockSparsity(self):
        if False:
            i = 10
            return i + 15
        weight_values = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 0, 7, 0, 0, 0, -6, -2, 0, 0, 0, 0, 0, -2, 0, 6]])
        custom_init = tf.constant_initializer(weight_values.transpose())
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(8, kernel_initializer=custom_init, input_shape=[16])])

        def calibration_gen():
            if False:
                return 10
            for _ in range(10):
                yield [np.random.uniform(-1, 1, size=(1, 16)).astype(np.float32) * 16]
        quantized_converter = lite.TFLiteConverterV2.from_keras_model(model)
        quantized_converter.optimizations = [lite.Optimize.EXPERIMENTAL_SPARSITY, lite.Optimize.DEFAULT]
        quantized_converter.representative_dataset = calibration_gen
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        metadata = get_conversion_metadata(quantized_tflite_model)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.environment.tensorflowVersion.decode('utf-8'), versions.__version__)
        self.assertEqual(metadata.environment.apiVersion, 2)
        self.assertAllEqual([metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER, metadata_fb.ModelOptimizationMode.BLOCK_SPARSITY], metadata.options.modelOptimizationModes)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        input_data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertArrayNear(np.array([0, 87, 0, 0, 0, 0, 0, 34], dtype=np.float32), actual_value.flatten(), err=1)

    def testQuantizedButNotEnoughBlockSparsity(self):
        if False:
            return 10
        weight_values = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [4, 4, -3, 4, 4, 1, -2, -2, 1, 3, 4, 1, 1, 1, -4, -5], [1, 1, 5, -1, 3, -1, 1, -3, 4, -3, 2, -3, 3, -1, 3, -4], [0, -3, -2, 5, 4, 2, 1, 4, -4, 4, 1, -2, 3, -2, -2, -1]])
        custom_init = tf.constant_initializer(weight_values.transpose())
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(4, kernel_initializer=custom_init, input_shape=[16])])

        def calibration_gen():
            if False:
                i = 10
                return i + 15
            for _ in range(10):
                yield [np.random.uniform(-1, 1, size=(1, 16)).astype(np.float32) * 16]
        quantized_converter = lite.TFLiteConverterV2.from_keras_model(model)
        quantized_converter.optimizations = [lite.Optimize.EXPERIMENTAL_SPARSITY, lite.Optimize.DEFAULT]
        quantized_converter.representative_dataset = calibration_gen
        quantized_tflite_model = quantized_converter.convert()
        self.assertIsNotNone(quantized_tflite_model)
        metadata = get_conversion_metadata(quantized_tflite_model)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.environment.tensorflowVersion.decode('utf-8'), versions.__version__)
        self.assertEqual(metadata.environment.apiVersion, 2)
        self.assertAllEqual([metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER], metadata.options.modelOptimizationModes)
        self.assertNotIn(metadata_fb.ModelOptimizationMode.RANDOM_SPARSITY, metadata.options.modelOptimizationModes)
        self.assertNotIn(metadata_fb.ModelOptimizationMode.BLOCK_SPARSITY, metadata.options.modelOptimizationModes)
        interpreter = Interpreter(model_content=quantized_tflite_model)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()
        input_data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        actual_value = interpreter.get_tensor(output_details[0]['index'])
        self.assertArrayNear(np.array([0, -3, 4, 35], dtype=np.float32), actual_value.flatten(), err=1)

class BufferOffsetTest(lite_v2_test_util.ModelTest):

    @test_util.run_v2_only
    def testCOncreteFunctionFloat(self):
        if False:
            i = 10
            return i + 15
        root = self._getSimpleVariableModel()
        input_data = tf.constant(1.0, shape=[1])
        concrete_func = root.f.get_concrete_function(input_data)
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        converter._experimental_use_buffer_offset = True
        tflite_model = converter.convert()
        expected_value = root.f(input_data)
        actual_value = self._evaluateTFLiteModel(tflite_model, [input_data])
        self.assertEqual(expected_value.numpy(), actual_value)

    @test_util.run_v2_only
    def testConcreteFunctionStringInput(self):
        if False:
            while True:
                i = 10

        class Model(tf.Module):

            @tf.function
            def __call__(self, x):
                if False:
                    while True:
                        i = 10
                return x
        root = Model()
        concrete_func = root.__call__.get_concrete_function(tf.constant([str(x) for x in range(11)]))
        converter = lite.TFLiteConverterV2.from_concrete_functions([concrete_func], root)
        converter._experimental_use_buffer_offset = True
        tflite_model = converter.convert()
        input_data = tf.constant([str(x) for x in range(11)], shape=(11,), dtype=tf.dtypes.string)
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        my_signature = interpreter.get_signature_runner()
        with self.assertRaises(ValueError) as error:
            _ = my_signature(x=input_data)
        self.assertIn('Passed in value type is not a numpy array, got type ', str(error.exception))

    @test_util.run_v2_only
    def testSavedModelSignatureDefs(self):
        if False:
            i = 10
            return i + 15
        'Test converting SignatureDef is correct and uses SignatureDef API.'
        root = self._getMultiFunctionModel()
        input_data_0 = tf.constant(1.0, shape=[1])
        input_data_1 = tf.constant(3.0, shape=[1])
        mul_add_func = root.mul_add.get_concrete_function(input_data_1, input_data_0)
        save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
        save(root, save_dir, {'mul_add': mul_add_func})
        converter = lite.TFLiteConverterV2.from_saved_model(save_dir, signature_keys=['mul_add'])
        converter._experimental_use_buffer_offset = True
        tflite_model = converter.convert()
        expected_value = root.mul_add(input_data_1, input_data_0)
        interpreter = Interpreter(model_content=tflite_model)
        signature_defs = interpreter.get_signature_list()
        results = self._evaluateTFLiteModelUsingSignatureDef(tflite_model, 'mul_add', {'y': input_data_0, 'x': input_data_1})
        self.assertEqual(list(results.keys()), ['output_0'])
        self.assertEqual(expected_value.numpy(), results['output_0'])
        self.assertEqual(len(signature_defs), 1)
        self.assertEqual(list(signature_defs.keys()), ['mul_add'])
        self.assertEqual(len(signature_defs.values()), 1)
        self.assertEqual(list(signature_defs['mul_add'].keys()), ['inputs', 'outputs'])
        self.assertCountEqual(signature_defs['mul_add']['inputs'], ['x', 'y'])
        self.assertEqual(list(signature_defs['mul_add']['outputs']), ['output_0'])
if __name__ == '__main__':
    test.main()