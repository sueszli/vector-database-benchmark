"""Tests for StatSummarizer Python wrapper."""
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import test
from tensorflow.tools import graph_transforms

class TransformGraphTest(test.TestCase):

    def testTransformGraph(self):
        if False:
            print('Hello World!')
        input_graph_def = graph_pb2.GraphDef()
        const_op1 = input_graph_def.node.add()
        const_op1.op = 'Const'
        const_op1.name = 'const_op1'
        const_op1.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
        const_op1.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto([1, 2], dtypes.float32, [1, 2])))
        const_op2 = input_graph_def.node.add()
        const_op2.op = 'Const'
        const_op2.name = 'const_op2'
        const_op2.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
        const_op2.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto([3, 4], dtypes.float32, [1, 2])))
        add_op = input_graph_def.node.add()
        add_op.op = 'Add'
        add_op.attr['T'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
        add_op.name = 'add_op'
        add_op.input.extend(['const_op1', 'const_op2'])
        relu_op = input_graph_def.node.add()
        relu_op.op = 'Relu'
        relu_op.attr['T'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
        relu_op.name = 'relu_op'
        relu_op.input.extend(['add_op'])
        input_names = []
        output_names = ['add_op']
        transforms = ['strip_unused_nodes']
        transformed_graph_def = graph_transforms.TransformGraph(input_graph_def, input_names, output_names, transforms)
        for node in transformed_graph_def.node:
            self.assertNotEqual('Relu', node.op)
if __name__ == '__main__':
    test.main()