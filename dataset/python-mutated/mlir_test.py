"""Tests for python.compiler.mlir."""
import os
from tensorflow.python.compiler.mlir import mlir
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.pywrap_mlir import experimental_tflite_to_tosa_bytecode
from tensorflow.python.pywrap_mlir import import_graphdef

class MLIRGraphDefImportTest(test.TestCase):

    def testImport(self):
        if False:
            i = 10
            return i + 15
        'Tests the basic flow of `tf.mlir.experimental.convert_graph_def`.'
        mlir_module = mlir.convert_graph_def('')
        self.assertIn('func @main', mlir_module)

    def testInvalidPbtxt(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(errors.InvalidArgumentError, 'Could not parse input proto'):
            mlir.convert_graph_def('some invalid proto')

    def testGraphDefToTf(self):
        if False:
            while True:
                i = 10
        'Tests the basic flow of `tf.mlir.experimental.convert_graph_def`\n\n        with tf-standard-pipeline converting all the way to the TF dialect.\n    '
        tensor_shape = (10, 10)

        @def_function.function(input_signature=(tensor_spec.TensorSpec(shape=tensor_shape, dtype=dtypes.float32), tensor_spec.TensorSpec(shape=tensor_shape, dtype=dtypes.float32)))
        def add_func(lhs, rhs):
            if False:
                print('Hello World!')
            return math_ops.add(lhs, rhs)
        tf_graph_def = add_func.get_concrete_function().graph.as_graph_def()
        mlir_tf = import_graphdef(tf_graph_def, 'tf-standard-pipeline', False, input_names=['lhs', 'rhs'], input_data_types=['DT_FLOAT', 'DT_FLOAT'], input_data_shapes=['10,10', '10,10'], output_names=['Add'])
        self.assertRegex(mlir_tf, 'func @main\\(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>')
        self.assertRegex(mlir_tf, 'inputs = "lhs,rhs"')
        self.assertRegex(mlir_tf, 'outputs = "Add"')
        mlir_tf = import_graphdef(tf_graph_def, 'tf-standard-pipeline', False, input_names=['lhs', 'rhs'], input_data_types=['DT_FLOAT', 'DT_FLOAT'], input_data_shapes=['', ''], output_names=['Add'])
        self.assertRegex(mlir_tf, 'func @main\\(%arg0: tensor<f32>, %arg1: tensor<f32>')
        with self.assertRaisesRegex(errors.InvalidArgumentError, "Length of input node array and data type doesn't match"):
            import_graphdef(tf_graph_def, 'tf-standard-pipeline', False, input_names=['lhs'], input_data_types=['DT_FLOAT', 'DT_FLOAT'], input_data_shapes=['10,10', '10,10'], output_names=['Add'])
        with self.assertRaisesRegex(errors.InvalidArgumentError, 'Dimensions must be equal'):
            import_graphdef(tf_graph_def, 'tf-standard-pipeline', False, input_names=['lhs', 'rhs'], input_data_types=['DT_FLOAT', 'DT_FLOAT'], input_data_shapes=['10,11', '10,10'], output_names=['Add'])

class MLIRConcreteFunctionImportTest(test.TestCase):

    @test_util.run_v2_only
    def testImport(self):
        if False:
            print('Hello World!')

        @def_function.function
        def sqr(i):
            if False:
                return 10
            return i * i
        concrete_function = sqr.get_concrete_function(tensor_spec.TensorSpec(None, dtypes.float32))
        mlir_module = mlir.convert_function(concrete_function, show_debug_info=True)
        self.assertRegex(mlir_module, 'func @.*sqr.*\\(')
        self.assertRegex(mlir_module, 'loc\\(".*mlir_test.py":.*:1\\)')

    @test_util.run_v2_only
    def testImportWithCall(self):
        if False:
            i = 10
            return i + 15

        @def_function.function
        def callee(i):
            if False:
                return 10
            return i

        @def_function.function
        def caller(i):
            if False:
                while True:
                    i = 10
            return callee(i)
        concrete_function = caller.get_concrete_function(tensor_spec.TensorSpec(None, dtypes.float32))
        mlir_module = mlir.convert_function(concrete_function)
        self.assertRegex(mlir_module, 'func @.*caller.*\\(')
        self.assertRegex(mlir_module, 'func private @.*callee.*\\(')

    @test_util.run_v2_only
    def testImportWithControlRet(self):
        if False:
            i = 10
            return i + 15

        @def_function.function
        def logging():
            if False:
                while True:
                    i = 10
            logging_ops.print_v2('some message')
        concrete_function = logging.get_concrete_function()
        mlir_module = mlir.convert_function(concrete_function, pass_pipeline='')
        self.assertRegex(mlir_module, 'tf\\.PrintV2')
        self.assertRegex(mlir_module, 'tf_executor.fetch.*: !tf_executor.control')

class MLIRFlatbufferImportTest(test.TestCase):

    def testImport(self):
        if False:
            i = 10
            return i + 15
        'Tests the basic flow of `experimental_tflite_to_tosa_bytecode`.'
        filename = os.path.join(self.get_temp_dir(), 'multi_add_tosa.mlirbc')
        experimental_tflite_to_tosa_bytecode(resource_loader.get_path_to_datafile('multi_add.tflite'), filename)
        with open(filename, mode='rb') as f:
            chunk = f.read(4)
        self.assertEqual(b'ML\xefR', chunk)
if __name__ == '__main__':
    test.main()