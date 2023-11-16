"""Tests for lite.py functionality related to TensorFlow 2.0."""
import os
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable

class ModelTest(test_util.TensorFlowTestCase, parameterized.TestCase):
    """Base test class for TensorFlow Lite 2.x model tests."""

    def _evaluateTFLiteModel(self, tflite_model, input_data, input_shapes=None):
        if False:
            print('Hello World!')
        'Evaluates the model on the `input_data`.\n\n    Args:\n      tflite_model: TensorFlow Lite model.\n      input_data: List of EagerTensor const ops containing the input data for\n        each input tensor.\n      input_shapes: List of tuples representing the `shape_signature` and the\n        new shape of each input tensor that has unknown dimensions.\n\n    Returns:\n      [np.ndarray]\n    '
        interpreter = Interpreter(model_content=tflite_model)
        input_details = interpreter.get_input_details()
        if input_shapes:
            for (idx, (shape_signature, final_shape)) in enumerate(input_shapes):
                self.assertTrue((input_details[idx]['shape_signature'] == shape_signature).all())
                index = input_details[idx]['index']
                interpreter.resize_tensor_input(index, final_shape, strict=True)
        interpreter.allocate_tensors()
        output_details = interpreter.get_output_details()
        input_details = interpreter.get_input_details()
        for (input_tensor, tensor_data) in zip(input_details, input_data):
            interpreter.set_tensor(input_tensor['index'], tensor_data.numpy())
        interpreter.invoke()
        return [interpreter.get_tensor(details['index']) for details in output_details]

    def _evaluateTFLiteModelUsingSignatureDef(self, tflite_model, signature_key, inputs):
        if False:
            print('Hello World!')
        "Evaluates the model on the `inputs`.\n\n    Args:\n      tflite_model: TensorFlow Lite model.\n      signature_key: Signature key.\n      inputs: Map from input tensor names in the SignatureDef to tensor value.\n\n    Returns:\n      Dictionary of outputs.\n      Key is the output name in the SignatureDef 'signature_key'\n      Value is the output value\n    "
        interpreter = Interpreter(model_content=tflite_model)
        signature_runner = interpreter.get_signature_runner(signature_key)
        return signature_runner(**inputs)

    def _getSimpleVariableModel(self):
        if False:
            print('Hello World!')
        root = autotrackable.AutoTrackable()
        root.v1 = variables.Variable(3.0)
        root.v2 = variables.Variable(2.0)
        root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
        return root

    def _getSimpleModelWithVariables(self):
        if False:
            print('Hello World!')

        class SimpleModelWithOneVariable(autotrackable.AutoTrackable):
            """Basic model with 1 variable."""

            def __init__(self):
                if False:
                    return 10
                super(SimpleModelWithOneVariable, self).__init__()
                self.var = variables.Variable(array_ops.zeros((1, 10), name='var'))

            @def_function.function
            def assign_add(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                self.var.assign_add(x)
                return self.var
        return SimpleModelWithOneVariable()

    def _getMultiFunctionModel(self):
        if False:
            i = 10
            return i + 15

        class BasicModel(autotrackable.AutoTrackable):
            """Basic model with multiple functions."""

            def __init__(self):
                if False:
                    print('Hello World!')
                self.y = None
                self.z = None

            @def_function.function
            def add(self, x):
                if False:
                    print('Hello World!')
                if self.y is None:
                    self.y = variables.Variable(2.0)
                return x + self.y

            @def_function.function
            def sub(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                if self.z is None:
                    self.z = variables.Variable(3.0)
                return x - self.z

            @def_function.function
            def mul_add(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                if self.z is None:
                    self.z = variables.Variable(3.0)
                return x * self.z + y
        return BasicModel()

    def _getMultiFunctionModelWithSharedWeight(self):
        if False:
            i = 10
            return i + 15

        class BasicModelWithSharedWeight(autotrackable.AutoTrackable):
            """Model with multiple functions and a shared weight."""

            def __init__(self):
                if False:
                    print('Hello World!')
                self.weight = constant_op.constant([1.0], shape=(1, 512, 512, 1), dtype=dtypes.float32)

            @def_function.function
            def add(self, x):
                if False:
                    print('Hello World!')
                return x + self.weight

            @def_function.function
            def sub(self, x):
                if False:
                    while True:
                        i = 10
                return x - self.weight

            @def_function.function
            def mul(self, x):
                if False:
                    while True:
                        i = 10
                return x * self.weight
        return BasicModelWithSharedWeight()

    def _getMatMulModelWithSmallWeights(self):
        if False:
            i = 10
            return i + 15

        class MatMulModelWithSmallWeights(autotrackable.AutoTrackable):
            """MatMul model with small weights and relatively large biases."""

            def __init__(self):
                if False:
                    return 10
                self.weight = constant_op.constant([[0.001, -0.001], [-0.0002, 0.0002]], shape=(2, 2), dtype=dtypes.float32)
                self.bias = constant_op.constant([1.28, 2.55], shape=(2,), dtype=dtypes.float32)

            @def_function.function
            def matmul(self, x):
                if False:
                    while True:
                        i = 10
                return x @ self.weight + self.bias
        return MatMulModelWithSmallWeights()

    def _getSqrtModel(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a model with only one sqrt op, to test non-quantizable op.'

        @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=(1, 10), dtype=dtypes.float32)])
        def sqrt(x):
            if False:
                i = 10
                return i + 15
            return math_ops.sqrt(x)

        def calibration_gen():
            if False:
                print('Hello World!')
            for _ in range(5):
                yield [np.random.uniform(0, 16, size=(1, 10)).astype(np.float32)]
        return (sqrt, calibration_gen)

    def _assertValidDebugInfo(self, debug_info):
        if False:
            for i in range(10):
                print('nop')
        'Verify the DebugInfo is valid.'
        file_names = set()
        for file_path in debug_info.files:
            file_names.add(os.path.basename(file_path))
        self.assertIn('lite_v2_test.py', file_names)
        self.assertNotIn('lite_test.py', file_names)

    def _createV2QATLowBitKerasModel(self, shape, weight_only, num_bits, bit_min, bit_max):
        if False:
            i = 10
            return i + 15
        'Creates a simple QAT num_bits-Weight Keras Model.'
        input_name = 'input'
        output_name = 'scores'

        class ConvWrapper(tf.keras.layers.Wrapper):
            """A Wrapper for simulating QAT on Conv2D layers."""

            def build(self, input_shape):
                if False:
                    for i in range(10):
                        print('nop')
                if not self.layer.built:
                    self.layer.build(input_shape)
                self.quantized_weights = self.layer.kernel

            def call(self, inputs):
                if False:
                    i = 10
                    return i + 15
                self.layer.kernel = tf.quantization.fake_quant_with_min_max_vars_per_channel(self.quantized_weights, min=[bit_min], max=[bit_max], num_bits=num_bits, narrow_range=True)
                if not weight_only:
                    quant_inputs = tf.quantization.fake_quant_with_min_max_vars(inputs, min=0, max=6, num_bits=8)
                    outputs = self.layer.call(quant_inputs)
                    return tf.quantization.fake_quant_with_min_max_vars(outputs, min=0, max=6, num_bits=8)
                return self.layer.call(inputs)
        input_tensor = tf.keras.layers.Input(shape, name=input_name)
        kernel_shape = (shape[-1], 3, 3, 1)
        initial_weights = np.linspace(bit_min, bit_max, np.prod(kernel_shape)).reshape(kernel_shape)
        test_initializer = tf.constant_initializer(initial_weights)
        x = ConvWrapper(tf.keras.layers.Conv2D(1, (3, 3), kernel_initializer=test_initializer, activation='relu6'))(input_tensor)
        scores = tf.keras.layers.Flatten(name=output_name)(x)
        model = tf.keras.Model(input_tensor, scores)
        return (model, input_name, output_name)

    def _createReadAssignModel(self, number_of_states=2):
        if False:
            i = 10
            return i + 15
        dtype = float

        class ReadAssign(tf.keras.layers.Layer):
            """ReadAssign model for the variable quantization test."""

            def __init__(self, number_of_states=2, **kwargs):
                if False:
                    while True:
                        i = 10
                super().__init__(**kwargs)
                self.number_of_states = number_of_states

            def build(self, input_shape):
                if False:
                    print('Hello World!')
                super().build(input_shape)
                state_shape = (1, 2, 3)
                self.states = [None] * self.number_of_states
                for i in range(self.number_of_states):
                    self.states[i] = self.add_weight(name=f'states{i}', shape=state_shape, trainable=False, initializer=tf.zeros_initializer, dtype=dtype)

            def call(self, inputs):
                if False:
                    i = 10
                    return i + 15
                for state in self.states:
                    memory = tf.keras.backend.concatenate([state, inputs], 1)
                    new_state = memory[:, :state.shape[1], :]
                    state.assign(new_state)
                return inputs

        def calibration_gen():
            if False:
                print('Hello World!')
            for _ in range(5):
                yield [np.random.uniform(-1, 1, size=(1, 2, 3)).astype(np.float32)]
        inputs = tf.keras.layers.Input(shape=(2, 3), batch_size=1, dtype=dtype)
        outputs = ReadAssign(number_of_states)(inputs)
        model = tf.keras.Model(inputs, outputs)
        return (model, calibration_gen)