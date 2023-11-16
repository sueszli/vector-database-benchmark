"""Tests for SignatureDef utils."""
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils_impl
from tensorflow.python.saved_model import utils
_STRING = meta_graph_pb2.TensorInfo(name='foobar', dtype=dtypes.string.as_datatype_enum)
_FLOAT = meta_graph_pb2.TensorInfo(name='foobar', dtype=dtypes.float32.as_datatype_enum)

def _make_signature(inputs, outputs, name=None):
    if False:
        return 10
    input_info = {input_name: utils.build_tensor_info(tensor) for (input_name, tensor) in inputs.items()}
    output_info = {output_name: utils.build_tensor_info(tensor) for (output_name, tensor) in outputs.items()}
    return signature_def_utils_impl.build_signature_def(input_info, output_info, name)

class SignatureDefUtilsTest(test.TestCase):

    def testBuildSignatureDef(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            x = array_ops.placeholder(dtypes.float32, 1, name='x')
            x_tensor_info = utils.build_tensor_info(x)
            inputs = {}
            inputs['foo-input'] = x_tensor_info
            y = array_ops.placeholder(dtypes.float32, name='y')
            y_tensor_info = utils.build_tensor_info(y)
            outputs = {}
            outputs['foo-output'] = y_tensor_info
            default_tensor = constant_op.constant(1.0, name='w')
            defaults = {}
            defaults['w'] = default_tensor
        signature_def = signature_def_utils_impl.build_signature_def(inputs, outputs, 'foo-method-name', defaults)
        self.assertEqual('foo-method-name', signature_def.method_name)
        self.assertEqual(1, len(signature_def.inputs))
        x_tensor_info_actual = signature_def.inputs['foo-input']
        self.assertEqual('x:0', x_tensor_info_actual.name)
        self.assertEqual(types_pb2.DT_FLOAT, x_tensor_info_actual.dtype)
        self.assertEqual(1, len(x_tensor_info_actual.tensor_shape.dim))
        self.assertEqual(1, x_tensor_info_actual.tensor_shape.dim[0].size)
        self.assertEqual(1, len(signature_def.outputs))
        y_tensor_info_actual = signature_def.outputs['foo-output']
        self.assertEqual('y:0', y_tensor_info_actual.name)
        self.assertEqual(types_pb2.DT_FLOAT, y_tensor_info_actual.dtype)
        self.assertEqual(0, len(y_tensor_info_actual.tensor_shape.dim))
        self.assertEqual(1, len(signature_def.defaults))
        self.assertEqual(types_pb2.DT_FLOAT, signature_def.defaults['w'].dtype)
        self.assertEqual(1.0, signature_def.defaults['w'].float_val[0])

    def testRegressionSignatureDef(self):
        if False:
            return 10
        with ops.Graph().as_default():
            input1 = constant_op.constant('a', name='input-1')
            output1 = constant_op.constant(2.2, name='output-1')
            signature_def = signature_def_utils_impl.regression_signature_def(input1, output1)
        self.assertEqual(signature_constants.REGRESS_METHOD_NAME, signature_def.method_name)
        self.assertEqual(1, len(signature_def.inputs))
        x_tensor_info_actual = signature_def.inputs[signature_constants.REGRESS_INPUTS]
        self.assertEqual('input-1:0', x_tensor_info_actual.name)
        self.assertEqual(types_pb2.DT_STRING, x_tensor_info_actual.dtype)
        self.assertEqual(0, len(x_tensor_info_actual.tensor_shape.dim))
        self.assertEqual(1, len(signature_def.outputs))
        y_tensor_info_actual = signature_def.outputs[signature_constants.REGRESS_OUTPUTS]
        self.assertEqual('output-1:0', y_tensor_info_actual.name)
        self.assertEqual(types_pb2.DT_FLOAT, y_tensor_info_actual.dtype)
        self.assertEqual(0, len(y_tensor_info_actual.tensor_shape.dim))

    def testClassificationSignatureDef(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            input1 = constant_op.constant('a', name='input-1')
            output1 = constant_op.constant('b', name='output-1')
            output2 = constant_op.constant(3.3, name='output-2')
            signature_def = signature_def_utils_impl.classification_signature_def(input1, output1, output2)
        self.assertEqual(signature_constants.CLASSIFY_METHOD_NAME, signature_def.method_name)
        self.assertEqual(1, len(signature_def.inputs))
        x_tensor_info_actual = signature_def.inputs[signature_constants.CLASSIFY_INPUTS]
        self.assertEqual('input-1:0', x_tensor_info_actual.name)
        self.assertEqual(types_pb2.DT_STRING, x_tensor_info_actual.dtype)
        self.assertEqual(0, len(x_tensor_info_actual.tensor_shape.dim))
        self.assertEqual(2, len(signature_def.outputs))
        classes_tensor_info_actual = signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_CLASSES]
        self.assertEqual('output-1:0', classes_tensor_info_actual.name)
        self.assertEqual(types_pb2.DT_STRING, classes_tensor_info_actual.dtype)
        self.assertEqual(0, len(classes_tensor_info_actual.tensor_shape.dim))
        scores_tensor_info_actual = signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_SCORES]
        self.assertEqual('output-2:0', scores_tensor_info_actual.name)
        self.assertEqual(types_pb2.DT_FLOAT, scores_tensor_info_actual.dtype)
        self.assertEqual(0, len(scores_tensor_info_actual.tensor_shape.dim))

    def testPredictionSignatureDef(self):
        if False:
            return 10
        with ops.Graph().as_default():
            input1 = constant_op.constant('a', name='input-1')
            input2 = constant_op.constant('b', name='input-2')
            output1 = constant_op.constant('c', name='output-1')
            output2 = constant_op.constant('d', name='output-2')
            signature_def = signature_def_utils_impl.predict_signature_def({'input-1': input1, 'input-2': input2}, {'output-1': output1, 'output-2': output2})
        self.assertEqual(signature_constants.PREDICT_METHOD_NAME, signature_def.method_name)
        self.assertEqual(2, len(signature_def.inputs))
        input1_tensor_info_actual = signature_def.inputs['input-1']
        self.assertEqual('input-1:0', input1_tensor_info_actual.name)
        self.assertEqual(types_pb2.DT_STRING, input1_tensor_info_actual.dtype)
        self.assertEqual(0, len(input1_tensor_info_actual.tensor_shape.dim))
        input2_tensor_info_actual = signature_def.inputs['input-2']
        self.assertEqual('input-2:0', input2_tensor_info_actual.name)
        self.assertEqual(types_pb2.DT_STRING, input2_tensor_info_actual.dtype)
        self.assertEqual(0, len(input2_tensor_info_actual.tensor_shape.dim))
        self.assertEqual(2, len(signature_def.outputs))
        output1_tensor_info_actual = signature_def.outputs['output-1']
        self.assertEqual('output-1:0', output1_tensor_info_actual.name)
        self.assertEqual(types_pb2.DT_STRING, output1_tensor_info_actual.dtype)
        self.assertEqual(0, len(output1_tensor_info_actual.tensor_shape.dim))
        output2_tensor_info_actual = signature_def.outputs['output-2']
        self.assertEqual('output-2:0', output2_tensor_info_actual.name)
        self.assertEqual(types_pb2.DT_STRING, output2_tensor_info_actual.dtype)
        self.assertEqual(0, len(output2_tensor_info_actual.tensor_shape.dim))

    def testTrainSignatureDef(self):
        if False:
            while True:
                i = 10
        self._testSupervisedSignatureDef(signature_def_utils_impl.supervised_train_signature_def, signature_constants.SUPERVISED_TRAIN_METHOD_NAME)

    def testEvalSignatureDef(self):
        if False:
            for i in range(10):
                print('nop')
        self._testSupervisedSignatureDef(signature_def_utils_impl.supervised_eval_signature_def, signature_constants.SUPERVISED_EVAL_METHOD_NAME)

    def _testSupervisedSignatureDef(self, fn_to_test, method_name):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            inputs = {'input-1': constant_op.constant('a', name='input-1'), 'input-2': constant_op.constant('b', name='input-2')}
            loss = {'loss-1': constant_op.constant(0.45, name='loss-1')}
            predictions = {'classes': constant_op.constant([100], name='classes')}
            metrics_val = constant_op.constant(100.0, name='metrics_val')
            metrics = {'metrics/value': metrics_val, 'metrics/update_op': array_ops.identity(metrics_val, name='metrics_op')}
            signature_def = fn_to_test(inputs, loss, predictions, metrics)
        self.assertEqual(method_name, signature_def.method_name)
        self.assertEqual(2, len(signature_def.inputs))
        input1_tensor_info_actual = signature_def.inputs['input-1']
        self.assertEqual('input-1:0', input1_tensor_info_actual.name)
        self.assertEqual(types_pb2.DT_STRING, input1_tensor_info_actual.dtype)
        self.assertEqual(0, len(input1_tensor_info_actual.tensor_shape.dim))
        input2_tensor_info_actual = signature_def.inputs['input-2']
        self.assertEqual('input-2:0', input2_tensor_info_actual.name)
        self.assertEqual(types_pb2.DT_STRING, input2_tensor_info_actual.dtype)
        self.assertEqual(0, len(input2_tensor_info_actual.tensor_shape.dim))
        self.assertEqual(4, len(signature_def.outputs))
        self.assertEqual('loss-1:0', signature_def.outputs['loss-1'].name)
        self.assertEqual(types_pb2.DT_FLOAT, signature_def.outputs['loss-1'].dtype)
        self.assertEqual('classes:0', signature_def.outputs['classes'].name)
        self.assertEqual(1, len(signature_def.outputs['classes'].tensor_shape.dim))
        self.assertEqual('metrics_val:0', signature_def.outputs['metrics/value'].name)
        self.assertEqual(types_pb2.DT_FLOAT, signature_def.outputs['metrics/value'].dtype)
        self.assertEqual('metrics_op:0', signature_def.outputs['metrics/update_op'].name)
        self.assertEqual(types_pb2.DT_FLOAT, signature_def.outputs['metrics/value'].dtype)

    def testTrainSignatureDefMissingInputs(self):
        if False:
            for i in range(10):
                print('nop')
        self._testSupervisedSignatureDefMissingInputs(signature_def_utils_impl.supervised_train_signature_def, signature_constants.SUPERVISED_TRAIN_METHOD_NAME)

    def testEvalSignatureDefMissingInputs(self):
        if False:
            while True:
                i = 10
        self._testSupervisedSignatureDefMissingInputs(signature_def_utils_impl.supervised_eval_signature_def, signature_constants.SUPERVISED_EVAL_METHOD_NAME)

    def _testSupervisedSignatureDefMissingInputs(self, fn_to_test, method_name):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            inputs = {'input-1': constant_op.constant('a', name='input-1'), 'input-2': constant_op.constant('b', name='input-2')}
            loss = {'loss-1': constant_op.constant(0.45, name='loss-1')}
            predictions = {'classes': constant_op.constant([100], name='classes')}
            metrics_val = constant_op.constant(100, name='metrics_val')
            metrics = {'metrics/value': metrics_val, 'metrics/update_op': array_ops.identity(metrics_val, name='metrics_op')}
            with self.assertRaises(ValueError):
                signature_def = fn_to_test({}, loss=loss, predictions=predictions, metrics=metrics)
            signature_def = fn_to_test(inputs, loss=loss)
            self.assertEqual(method_name, signature_def.method_name)
            self.assertEqual(1, len(signature_def.outputs))
            signature_def = fn_to_test(inputs, metrics=metrics, loss=loss)
            self.assertEqual(method_name, signature_def.method_name)
            self.assertEqual(3, len(signature_def.outputs))

    def _assertValidSignature(self, inputs, outputs, method_name):
        if False:
            return 10
        signature_def = signature_def_utils_impl.build_signature_def(inputs, outputs, method_name)
        self.assertTrue(signature_def_utils_impl.is_valid_signature(signature_def))

    def _assertInvalidSignature(self, inputs, outputs, method_name):
        if False:
            for i in range(10):
                print('nop')
        signature_def = signature_def_utils_impl.build_signature_def(inputs, outputs, method_name)
        self.assertFalse(signature_def_utils_impl.is_valid_signature(signature_def))

    def testValidSignaturesAreAccepted(self):
        if False:
            while True:
                i = 10
        self._assertValidSignature({'inputs': _STRING}, {'classes': _STRING, 'scores': _FLOAT}, signature_constants.CLASSIFY_METHOD_NAME)
        self._assertValidSignature({'inputs': _STRING}, {'classes': _STRING}, signature_constants.CLASSIFY_METHOD_NAME)
        self._assertValidSignature({'inputs': _STRING}, {'scores': _FLOAT}, signature_constants.CLASSIFY_METHOD_NAME)
        self._assertValidSignature({'inputs': _STRING}, {'outputs': _FLOAT}, signature_constants.REGRESS_METHOD_NAME)
        self._assertValidSignature({'foo': _STRING, 'bar': _FLOAT}, {'baz': _STRING, 'qux': _FLOAT}, signature_constants.PREDICT_METHOD_NAME)

    def testInvalidMethodNameSignatureIsRejected(self):
        if False:
            while True:
                i = 10
        self._assertInvalidSignature({'inputs': _STRING}, {'classes': _STRING, 'scores': _FLOAT}, 'WRONG method name')

    def testInvalidClassificationSignaturesAreRejected(self):
        if False:
            return 10
        self._assertInvalidSignature({'inputs': _FLOAT}, {'classes': _STRING, 'scores': _FLOAT}, signature_constants.CLASSIFY_METHOD_NAME)
        self._assertInvalidSignature({'inputs': _STRING}, {'classes': _FLOAT, 'scores': _FLOAT}, signature_constants.CLASSIFY_METHOD_NAME)
        self._assertInvalidSignature({'inputs': _STRING}, {'classes': _STRING, 'scores': _STRING}, signature_constants.CLASSIFY_METHOD_NAME)
        self._assertInvalidSignature({}, {'classes': _STRING, 'scores': _FLOAT}, signature_constants.CLASSIFY_METHOD_NAME)
        self._assertInvalidSignature({'inputs_WRONG': _STRING}, {'classes': _STRING, 'scores': _FLOAT}, signature_constants.CLASSIFY_METHOD_NAME)
        self._assertInvalidSignature({'inputs': _STRING}, {'classes_WRONG': _STRING, 'scores': _FLOAT}, signature_constants.CLASSIFY_METHOD_NAME)
        self._assertInvalidSignature({'inputs': _STRING}, {}, signature_constants.CLASSIFY_METHOD_NAME)
        self._assertInvalidSignature({'inputs': _STRING}, {'classes': _STRING, 'scores': _FLOAT, 'extra_WRONG': _STRING}, signature_constants.CLASSIFY_METHOD_NAME)

    def testInvalidRegressionSignaturesAreRejected(self):
        if False:
            i = 10
            return i + 15
        self._assertInvalidSignature({'inputs': _FLOAT}, {'outputs': _FLOAT}, signature_constants.REGRESS_METHOD_NAME)
        self._assertInvalidSignature({'inputs': _STRING}, {'outputs': _STRING}, signature_constants.REGRESS_METHOD_NAME)
        self._assertInvalidSignature({}, {'outputs': _FLOAT}, signature_constants.REGRESS_METHOD_NAME)
        self._assertInvalidSignature({'inputs_WRONG': _STRING}, {'outputs': _FLOAT}, signature_constants.REGRESS_METHOD_NAME)
        self._assertInvalidSignature({'inputs': _STRING}, {'outputs_WRONG': _FLOAT}, signature_constants.REGRESS_METHOD_NAME)
        self._assertInvalidSignature({'inputs': _STRING}, {}, signature_constants.REGRESS_METHOD_NAME)
        self._assertInvalidSignature({'inputs': _STRING}, {'outputs': _FLOAT, 'extra_WRONG': _STRING}, signature_constants.REGRESS_METHOD_NAME)

    def testInvalidPredictSignaturesAreRejected(self):
        if False:
            return 10
        self._assertInvalidSignature({}, {'baz': _STRING, 'qux': _FLOAT}, signature_constants.PREDICT_METHOD_NAME)
        self._assertInvalidSignature({'foo': _STRING, 'bar': _FLOAT}, {}, signature_constants.PREDICT_METHOD_NAME)

    def testOpSignatureDef(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            key = 'adding_1_and_2_key'
            add_op = math_ops.add(1, 2, name='adding_1_and_2')
            signature_def = signature_def_utils_impl.op_signature_def(add_op, key)
        self.assertIn(key, signature_def.outputs)
        self.assertEqual(add_op.name, signature_def.outputs[key].name)

    def testLoadOpFromSignatureDef(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            key = 'adding_1_and_2_key'
            add_op = math_ops.add(1, 2, name='adding_1_and_2')
            signature_def = signature_def_utils_impl.op_signature_def(add_op, key)
            self.assertEqual(add_op, signature_def_utils_impl.load_op_from_signature_def(signature_def, key))
if __name__ == '__main__':
    test.main()