"""Tests for py_function_lib."""
from tensorflow.compiler.mlir.quantization.tensorflow import exported_model_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import py_function_lib
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.platform import test

class PyFunctionLibTest(test.TestCase):

    def test_assign_ids_to_custom_aggregator_ops(self):
        if False:
            i = 10
            return i + 15
        func_lib = py_function_lib.PyFunctionLibrary()
        exported_model = exported_model_pb2.ExportedModel()
        function_def: function_pb2.FunctionDef = exported_model.graph_def.library.function.add()
        node_def_1: node_def_pb2.NodeDef = function_def.node_def.add()
        node_def_1.op = 'CustomAggregator'
        node_def_2: node_def_pb2.NodeDef = function_def.node_def.add()
        node_def_2.op = 'Identity'
        result_exported_model = exported_model_pb2.ExportedModel.FromString(func_lib.assign_ids_to_custom_aggregator_ops(exported_model.SerializeToString()))
        result_function_def = result_exported_model.graph_def.library.function[0]
        result_node_def_1 = result_function_def.node_def[0]
        self.assertEqual(result_node_def_1.op, 'CustomAggregator')
        self.assertIn('id', result_node_def_1.attr)
        self.assertLen(result_node_def_1.attr, 1)
        result_node_def_2 = result_function_def.node_def[1]
        self.assertEqual(result_node_def_2.op, 'Identity')
        self.assertNotIn('id', result_node_def_2.attr)
if __name__ == '__main__':
    test.main()