"""Tests for tensorflow.python.client.graph_util."""
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import test
from tensorflow.python.util import compat

def TestDeviceFuncPinVariableToCpu(op):
    if False:
        for i in range(10):
            print('nop')
    if op.device:
        return op.device
    return '/cpu:0' if op.node_def.op in ['Variable', 'VariableV2'] else op.device

class GraphUtilTest(test.TestCase):

    def testTwoDeviceFunctions(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default() as g:
            var_0 = gen_state_ops.variable(shape=[1], dtype=dtypes.float32, name='var_0', container='', shared_name='')
            with g.device(TestDeviceFuncPinVariableToCpu):
                var_1 = gen_state_ops.variable(shape=[1], dtype=dtypes.float32, name='var_1', container='', shared_name='')
            var_2 = gen_state_ops.variable(shape=[1], dtype=dtypes.float32, name='var_2', container='', shared_name='')
            var_3 = gen_state_ops.variable(shape=[1], dtype=dtypes.float32, name='var_3', container='', shared_name='')
            with g.device(TestDeviceFuncPinVariableToCpu):
                var_4 = gen_state_ops.variable(shape=[1], dtype=dtypes.float32, name='var_4', container='', shared_name='')
                with g.device('/device:GPU:0'):
                    var_5 = gen_state_ops.variable(shape=[1], dtype=dtypes.float32, name='var_5', container='', shared_name='')
                var_6 = gen_state_ops.variable(shape=[1], dtype=dtypes.float32, name='var_6', container='', shared_name='')
        self.assertDeviceEqual(var_0.device, None)
        self.assertDeviceEqual(var_1.device, '/device:CPU:0')
        self.assertDeviceEqual(var_2.device, None)
        self.assertDeviceEqual(var_3.device, None)
        self.assertDeviceEqual(var_4.device, '/device:CPU:0')
        self.assertDeviceEqual(var_5.device, '/device:GPU:0')
        self.assertDeviceEqual(var_6.device, '/device:CPU:0')

    @test_util.run_v1_only('b/120545219')
    def testNestedDeviceFunctions(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            var_0 = variable_v1.VariableV1(0)
            with ops.device(TestDeviceFuncPinVariableToCpu):
                var_1 = variable_v1.VariableV1(1)
                with ops.device(lambda op: '/device:GPU:0'):
                    var_2 = variable_v1.VariableV1(2)
                with ops.device('/device:GPU:0'):
                    var_3 = variable_v1.VariableV1(3)
        self.assertDeviceEqual(var_0.device, None)
        self.assertDeviceEqual(var_1.device, '/device:CPU:0')
        self.assertDeviceEqual(var_2.device, '/device:GPU:0')
        self.assertDeviceEqual(var_3.device, '/device:GPU:0')

    def testExplicitDevice(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default() as g:
            const_0 = constant_op.constant(5.0)
            with g.device('/device:GPU:0'):
                const_1 = constant_op.constant(5.0)
            with g.device('/device:GPU:1'):
                const_2 = constant_op.constant(5.0)
            with g.device('/device:CPU:0'):
                const_3 = constant_op.constant(5.0)
            with g.device('/device:CPU:1'):
                const_4 = constant_op.constant(5.0)
            with g.device('/job:ps'):
                const_5 = constant_op.constant(5.0)
        self.assertDeviceEqual(const_0.device, None)
        self.assertDeviceEqual(const_1.device, '/device:GPU:0')
        self.assertDeviceEqual(const_2.device, '/device:GPU:1')
        self.assertDeviceEqual(const_3.device, '/device:CPU:0')
        self.assertDeviceEqual(const_4.device, '/device:CPU:1')
        self.assertDeviceEqual(const_5.device, '/job:ps')

    def testDefaultDevice(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default() as g, g.device(TestDeviceFuncPinVariableToCpu):
            with g.device('/job:ps'):
                const_0 = constant_op.constant(5.0)
            with g.device('/device:GPU:0'):
                const_1 = constant_op.constant(5.0)
            with g.device('/device:GPU:1'):
                const_2 = constant_op.constant(5.0)
            with g.device('/device:CPU:0'):
                const_3 = constant_op.constant(5.0)
            with g.device('/device:CPU:1'):
                const_4 = constant_op.constant(5.0)
            with g.device('/replica:0'):
                const_5 = constant_op.constant(5.0)
        self.assertDeviceEqual(const_0.device, '/job:ps')
        self.assertDeviceEqual(const_1.device, '/device:GPU:0')
        self.assertDeviceEqual(const_2.device, '/device:GPU:1')
        self.assertDeviceEqual(const_3.device, '/device:CPU:0')
        self.assertDeviceEqual(const_4.device, '/device:CPU:1')
        self.assertDeviceEqual(const_5.device, '/replica:0')

    def testExtractSubGraph(self):
        if False:
            for i in range(10):
                print('nop')
        graph_def = graph_pb2.GraphDef()
        n1 = graph_def.node.add()
        n1.name = 'n1'
        n1.input.extend(['n5'])
        n2 = graph_def.node.add()
        n2.name = 'n2'
        n2.input.extend(['n1:0'])
        n3 = graph_def.node.add()
        n3.name = 'n3'
        n3.input.extend(['^n2'])
        n4 = graph_def.node.add()
        n4.name = 'n4'
        n5 = graph_def.node.add()
        n5.name = 'n5'
        n5.input.extend(['n1'])
        sub_graph = graph_util.extract_sub_graph(graph_def, ['n3'])
        self.assertEqual('n1', sub_graph.node[0].name)
        self.assertEqual('n2', sub_graph.node[1].name)
        self.assertEqual('n3', sub_graph.node[2].name)
        self.assertEqual('n5', sub_graph.node[3].name)

    def testExtractSubGraphWithInvalidDestNodes(self):
        if False:
            return 10
        graph_def = graph_pb2.GraphDef()
        n1 = graph_def.node.add()
        n1.name = 'n1'
        with self.assertRaisesRegex(TypeError, 'must be an iterable'):
            graph_util.extract_sub_graph(graph_def, 'n1')

    def create_node_def(self, op, name, inputs):
        if False:
            print('Hello World!')
        new_node = node_def_pb2.NodeDef()
        new_node.op = op
        new_node.name = name
        new_node.input.extend(inputs)
        return new_node

    def create_constant_node_def(self, name, value, dtype, shape=None, inputs=None):
        if False:
            while True:
                i = 10
        node = self.create_node_def('Const', name, inputs or [])
        self.set_attr_dtype(node, 'dtype', dtype)
        self.set_attr_tensor(node, 'value', value, dtype, shape)
        return node

    def set_attr_dtype(self, node, key, value):
        if False:
            i = 10
            return i + 15
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(type=value.as_datatype_enum))

    def set_attr_list(self, node, key, value_list):
        if False:
            print('Hello World!')
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(s=value_list)))

    def set_attr_tensor(self, node, key, value, dtype, shape=None):
        if False:
            print('Hello World!')
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(value, dtype=dtype, shape=shape)))

    def testRemoveTrainingNodes(self):
        if False:
            i = 10
            return i + 15
        a_constant_name = 'a_constant'
        b_constant_name = 'b_constant'
        c_constant_name = 'c_constant'
        a_check_name = 'a_check'
        b_check_name = 'b_check'
        a_identity_name = 'a_identity'
        b_identity_name = 'b_identity'
        c_identity_name = 'c_identity'
        add_name = 'add'
        sub_name = 'sub'
        graph_def = graph_pb2.GraphDef()
        a_constant = self.create_constant_node_def(a_constant_name, value=1, dtype=dtypes.float32, shape=[])
        graph_def.node.extend([a_constant])
        a_check_node = self.create_node_def('CheckNumerics', a_check_name, [a_constant_name])
        graph_def.node.extend([a_check_node])
        a_identity_node = self.create_node_def('Identity', a_identity_name, [a_constant_name, '^' + a_check_name])
        graph_def.node.extend([a_identity_node])
        b_constant = self.create_constant_node_def(b_constant_name, value=1, dtype=dtypes.float32, shape=[])
        graph_def.node.extend([b_constant])
        b_check_node = self.create_node_def('CheckNumerics', b_check_name, [b_constant_name])
        graph_def.node.extend([b_check_node])
        b_identity_node = self.create_node_def('Identity', b_identity_name, [b_constant_name, '^' + b_check_name])
        graph_def.node.extend([b_identity_node])
        add_node = self.create_node_def('Add', add_name, [a_identity_name, b_identity_name])
        self.set_attr_dtype(add_node, 'T', dtypes.float32)
        graph_def.node.extend([add_node])
        c_constant = self.create_constant_node_def(c_constant_name, value=1, dtype=dtypes.float32, shape=[])
        graph_def.node.extend([c_constant])
        c_identity_node = self.create_node_def('Identity', c_identity_name, [c_constant_name])
        graph_def.node.extend([c_identity_node])
        sub_node = self.create_node_def('Sub', sub_name, [c_constant_name, c_identity_name])
        self.set_attr_list(sub_node, '_class', [compat.as_bytes(c_identity_name)])
        graph_def.node.extend([sub_node])
        expected_output = graph_pb2.GraphDef()
        a_constant = self.create_constant_node_def(a_constant_name, value=1, dtype=dtypes.float32, shape=[])
        expected_output.node.extend([a_constant])
        b_constant = self.create_constant_node_def(b_constant_name, value=1, dtype=dtypes.float32, shape=[])
        expected_output.node.extend([b_constant])
        add_node = self.create_node_def('Add', add_name, [a_constant_name, b_constant_name])
        self.set_attr_dtype(add_node, 'T', dtypes.float32)
        expected_output.node.extend([add_node])
        c_constant = self.create_constant_node_def(c_constant_name, value=1, dtype=dtypes.float32, shape=[])
        expected_output.node.extend([c_constant])
        c_identity_node = self.create_node_def('Identity', c_identity_name, [c_constant_name])
        expected_output.node.extend([c_identity_node])
        sub_node = self.create_node_def('Sub', sub_name, [c_constant_name, c_identity_name])
        self.set_attr_list(sub_node, '_class', [compat.as_bytes(c_identity_name)])
        expected_output.node.extend([sub_node])
        output = graph_util.remove_training_nodes(graph_def)
        self.assertProtoEquals(expected_output, output)

    def testRemoveIdentityChains(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that chains of Identity nodes are correctly pruned.\n\n    Create a chain of four nodes, A, B, C, and D where A inputs B, B inputs C,\n    and C inputs D. Nodes B and C are "Identity" and should be pruned, resulting\n    in the nodes A and D, where A inputs D.\n    '
        graph_def = graph_pb2.GraphDef()
        graph_def.node.extend([self.create_node_def('Aop', 'A', ['B']), self.create_node_def('Identity', 'B', ['C']), self.create_node_def('Identity', 'C', ['D']), self.create_node_def('Dop', 'D', [])])
        expected_graph_def = graph_pb2.GraphDef()
        expected_graph_def.node.extend([self.create_node_def('Aop', 'A', ['D']), self.create_node_def('Dop', 'D', [])])
        self.assertProtoEquals(expected_graph_def, graph_util.remove_training_nodes(graph_def))

    def testRemoveIdentityUsedAsControlInputInConst(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that Identity nodes used as control inputs are not removed.'
        graph_def = graph_pb2.GraphDef()
        graph_def.node.extend([self.create_constant_node_def('C', 1, dtypes.float32, inputs=['^I']), self.create_node_def('Identity', 'I', ['Base']), self.create_node_def('BaseOp', 'Base', [])])
        self.assertProtoEquals(graph_def, graph_util.remove_training_nodes(graph_def))

    def testSimpleGraphdefsCompareEqual(self):
        if False:
            i = 10
            return i + 15
        graph_def1 = graph_pb2.GraphDef()
        graph_def1.node.extend([self.create_constant_node_def('C', 1, dtypes.float32, inputs=['^I']), self.create_node_def('Identity', 'I', ['Base']), self.create_node_def('BaseOp', 'Base', [])])
        graph_def2 = graph_pb2.GraphDef()
        graph_def2.node.extend([self.create_constant_node_def('C', 1, dtypes.float32, inputs=['^I']), self.create_node_def('Identity', 'I', ['Base']), self.create_node_def('BaseOp', 'Base', [])])
        self.assertTrue(graph_util.graph_defs_equal(graph_def1, graph_def2))

    def testNodeDefsInDifferentOrderCompareEqual(self):
        if False:
            return 10
        graph_def1 = graph_pb2.GraphDef()
        graph_def1.node.extend([self.create_node_def('Identity', 'I', ['Base']), self.create_node_def('BaseOp', 'Base', []), self.create_constant_node_def('C', 1, dtypes.float32, inputs=['^I'])])
        graph_def2 = graph_pb2.GraphDef()
        graph_def2.node.extend([self.create_constant_node_def('C', 1, dtypes.float32, inputs=['^I']), self.create_node_def('Identity', 'I', ['Base']), self.create_node_def('BaseOp', 'Base', [])])
        self.assertTrue(graph_util.graph_defs_equal(graph_def1, graph_def2))

    def testDifferentGraphDefsCompareNotEqual(self):
        if False:
            return 10
        graph_def1 = graph_pb2.GraphDef()
        graph_def1.node.extend([self.create_constant_node_def('C', 1, dtypes.float32, inputs=['^I']), self.create_node_def('Identity', 'I', ['Base']), self.create_node_def('BaseOp', 'Base', [])])
        graph_def2 = graph_pb2.GraphDef()
        graph_def2.node.extend([self.create_constant_node_def('C', 2, dtypes.float32, inputs=['^I']), self.create_node_def('Identity', 'I', ['Base']), self.create_node_def('BaseOp', 'Base', [])])
        self.assertFalse(graph_util.graph_defs_equal(graph_def1, graph_def2))

    def testGraphdefsWithNanCompareNonEqual(self):
        if False:
            while True:
                i = 10
        graph_def1 = graph_pb2.GraphDef()
        graph_def1.node.extend([self.create_constant_node_def('C', float('nan'), dtypes.float32, inputs=['^I']), self.create_node_def('Identity', 'I', ['Base']), self.create_node_def('BaseOp', 'Base', [])])
        graph_def2 = graph_pb2.GraphDef()
        graph_def2.node.extend([self.create_constant_node_def('C', float('nan'), dtypes.float32, inputs=['^I']), self.create_node_def('Identity', 'I', ['Base']), self.create_node_def('BaseOp', 'Base', [])])
        self.assertFalse(graph_util.graph_defs_equal(graph_def1, graph_def2))

    def testSimpleGraphdefEqualityWithNansEqual(self):
        if False:
            for i in range(10):
                print('nop')
        graph_def1 = graph_pb2.GraphDef()
        graph_def1.node.extend([self.create_constant_node_def('C', float('nan'), dtypes.float32, inputs=['^I']), self.create_node_def('Identity', 'I', ['Base']), self.create_node_def('BaseOp', 'Base', [])])
        graph_def2 = graph_pb2.GraphDef()
        graph_def2.node.extend([self.create_constant_node_def('C', float('nan'), dtypes.float32, inputs=['^I']), self.create_node_def('Identity', 'I', ['Base']), self.create_node_def('BaseOp', 'Base', [])])
        self.assertTrue(graph_util.graph_defs_equal(graph_def1, graph_def2, treat_nan_as_equal=True))

    def testGraphDefsWithFunctionLibsCompareEqual(self):
        if False:
            while True:
                i = 10

        @function.Defun(dtypes.float32)
        def F1(x):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.exp(x) - math_ops.exp(-x)
        library = function_pb2.FunctionDefLibrary()
        library.function.extend([F1.definition])
        graph_def1 = graph_pb2.GraphDef()
        graph_def1.library.CopyFrom(library)
        graph_def2 = graph_pb2.GraphDef()
        graph_def2.library.CopyFrom(library)
        self.assertTrue(graph_util.graph_defs_equal(graph_def1, graph_def2))

    def testGraphDefsWithPermutedFunctionsCompareEqual(self):
        if False:
            return 10

        @function.Defun(dtypes.float32)
        def F1(x):
            if False:
                return 10
            return math_ops.exp(x) - math_ops.exp(-x)

        @function.Defun(dtypes.float32)
        def F2(x):
            if False:
                i = 10
                return i + 15
            return math_ops.exp(x)
        definition_1 = F1.definition
        definition_2 = F2.definition
        library = function_pb2.FunctionDefLibrary()
        library.function.extend([definition_1, definition_2])
        graph_def1 = graph_pb2.GraphDef()
        graph_def1.library.CopyFrom(library)
        reversed_library = function_pb2.FunctionDefLibrary()
        reversed_library.function.extend([definition_2, definition_1])
        graph_def2 = graph_pb2.GraphDef()
        graph_def2.library.CopyFrom(reversed_library)
        self.assertTrue(graph_util.graph_defs_equal(graph_def1, graph_def2))

    def testGraphDefsWithPermutedNodesInFunctionsCompareEqual(self):
        if False:
            return 10

        @function.Defun(dtypes.float32)
        def F1(x):
            if False:
                i = 10
                return i + 15
            return math_ops.exp(x) - math_ops.exp(-x)
        f1_def = F1.definition
        library = function_pb2.FunctionDefLibrary()
        library.function.extend([f1_def])
        graph_def1 = graph_pb2.GraphDef()
        graph_def1.library.CopyFrom(library)
        reversed_function = function_pb2.FunctionDef()
        reversed_function.CopyFrom(f1_def)
        del reversed_function.node_def[:]
        reversed_function.node_def.extend(reversed(f1_def.node_def))
        reversed_library = function_pb2.FunctionDefLibrary()
        reversed_library.function.extend([reversed_function])
        graph_def2 = graph_pb2.GraphDef()
        graph_def2.library.CopyFrom(reversed_library)
        self.assertTrue(graph_util.graph_defs_equal(graph_def1, graph_def2))
if __name__ == '__main__':
    test.main()