"""Tests the node stripping tool."""
import os
from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.tools import strip_unused_lib

class StripUnusedTest(test_util.TensorFlowTestCase):

    def testStripUnused(self):
        if False:
            for i in range(10):
                print('nop')
        input_graph_name = 'input_graph.pb'
        output_graph_name = 'output_graph.pb'
        with ops.Graph().as_default():
            constant_node = constant_op.constant(1.0, name='constant_node')
            wanted_input_node = math_ops.subtract(constant_node, 3.0, name='wanted_input_node')
            output_node = math_ops.multiply(wanted_input_node, 2.0, name='output_node')
            math_ops.add(output_node, 2.0, name='later_node')
            sess = session.Session()
            output = self.evaluate(output_node)
            self.assertNear(-4.0, output, 1e-05)
            graph_io.write_graph(sess.graph, self.get_temp_dir(), input_graph_name)
        input_graph_path = os.path.join(self.get_temp_dir(), input_graph_name)
        input_binary = False
        output_binary = True
        output_node_names = 'output_node'
        output_graph_path = os.path.join(self.get_temp_dir(), output_graph_name)

        def strip(input_node_names):
            if False:
                for i in range(10):
                    print('nop')
            strip_unused_lib.strip_unused_from_files(input_graph_path, input_binary, output_graph_path, output_binary, input_node_names, output_node_names, dtypes.float32.as_datatype_enum)
        with self.assertRaises(KeyError):
            strip('does_not_exist')
        with self.assertRaises(ValueError):
            strip('wanted_input_node:0')
        input_node_names = 'wanted_input_node'
        strip(input_node_names)
        with ops.Graph().as_default():
            output_graph_def = graph_pb2.GraphDef()
            with open(output_graph_path, 'rb') as f:
                output_graph_def.ParseFromString(f.read())
                _ = importer.import_graph_def(output_graph_def, name='')
            self.assertEqual(3, len(output_graph_def.node))
            for node in output_graph_def.node:
                self.assertNotEqual('Add', node.op)
                self.assertNotEqual('Sub', node.op)
                if node.name == input_node_names:
                    self.assertTrue('shape' in node.attr)
            with session.Session() as sess:
                input_node = sess.graph.get_tensor_by_name('wanted_input_node:0')
                output_node = sess.graph.get_tensor_by_name('output_node:0')
                output = sess.run(output_node, feed_dict={input_node: [10.0]})
                self.assertNear(20.0, output, 1e-05)

    def testStripUnusedMultipleInputs(self):
        if False:
            while True:
                i = 10
        input_graph_name = 'input_graph.pb'
        output_graph_name = 'output_graph.pb'
        with ops.Graph().as_default():
            constant_node1 = constant_op.constant(1.0, name='constant_node1')
            constant_node2 = constant_op.constant(2.0, name='constant_node2')
            input_node1 = math_ops.subtract(constant_node1, 3.0, name='input_node1')
            input_node2 = math_ops.subtract(constant_node2, 5.0, name='input_node2')
            output_node = math_ops.multiply(input_node1, input_node2, name='output_node')
            math_ops.add(output_node, 2.0, name='later_node')
            sess = session.Session()
            output = self.evaluate(output_node)
            self.assertNear(6.0, output, 1e-05)
            graph_io.write_graph(sess.graph, self.get_temp_dir(), input_graph_name)
        input_graph_path = os.path.join(self.get_temp_dir(), input_graph_name)
        input_binary = False
        input_node_names = 'input_node1,input_node2'
        input_node_types = [dtypes.float32.as_datatype_enum, dtypes.float32.as_datatype_enum]
        output_binary = True
        output_node_names = 'output_node'
        output_graph_path = os.path.join(self.get_temp_dir(), output_graph_name)
        strip_unused_lib.strip_unused_from_files(input_graph_path, input_binary, output_graph_path, output_binary, input_node_names, output_node_names, input_node_types)
        with ops.Graph().as_default():
            output_graph_def = graph_pb2.GraphDef()
            with open(output_graph_path, 'rb') as f:
                output_graph_def.ParseFromString(f.read())
                _ = importer.import_graph_def(output_graph_def, name='')
            self.assertEqual(3, len(output_graph_def.node))
            for node in output_graph_def.node:
                self.assertNotEqual('Add', node.op)
                self.assertNotEqual('Sub', node.op)
                if node.name == input_node_names:
                    self.assertTrue('shape' in node.attr)
            with session.Session() as sess:
                input_node1 = sess.graph.get_tensor_by_name('input_node1:0')
                input_node2 = sess.graph.get_tensor_by_name('input_node2:0')
                output_node = sess.graph.get_tensor_by_name('output_node:0')
                output = sess.run(output_node, feed_dict={input_node1: [10.0], input_node2: [-5.0]})
                self.assertNear(-50.0, output, 1e-05)
if __name__ == '__main__':
    test.main()