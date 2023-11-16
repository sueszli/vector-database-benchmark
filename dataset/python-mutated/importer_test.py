"""Tests for tensorflow.python.framework.importer."""
import numpy as np
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
import tensorflow.python.ops.nn_grad
from tensorflow.python.platform import test

class ImportGraphDefTest(test.TestCase):

    def _MakeGraphDef(self, text, producer=versions.GRAPH_DEF_VERSION, min_consumer=versions.GRAPH_DEF_VERSION_MIN_CONSUMER):
        if False:
            while True:
                i = 10
        text = 'versions: { producer: %d min_consumer: %d };\n%s' % (producer, min_consumer, text)
        ret = graph_pb2.GraphDef()
        text_format.Merge(text, ret)
        return ret

    def testBasic(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            (a, b, c, d) = importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A' op: 'IntOutputFloatOutput' }\n          node { name: 'B' op: 'ListOutput'\n                 attr { key: 'T'\n                        value { list { type: DT_INT32 type: DT_FLOAT } } } }\n          node { name: 'C' op: 'ListInput'\n                 attr { key: 'N' value { i: 2 } }\n                 attr { key: 'T' value { type: DT_INT32 } }\n                 input: 'A:0' input: 'B:0' }\n          node { name: 'D' op: 'ListInput'\n                 attr { key: 'N' value { i: 2 } }\n                 attr { key: 'T' value { type: DT_FLOAT } }\n                 input: 'A:1' input: 'B:1' }\n          "), return_elements=['A', 'B', 'C', 'D'], name='import')
            self.assertNotEqual(a.outputs[0].name, a.outputs[1].name)
            self.assertNotEqual(b.outputs[0].name, b.outputs[1].name)
            self.assertNotEqual(a.outputs[0].name, b.outputs[0].name)
            self.assertNotEqual(a.outputs[0].name, b.outputs[1].name)
            self.assertNotEqual(a.outputs[1].name, b.outputs[0].name)
            self.assertNotEqual(a.outputs[1].name, b.outputs[1].name)
            self.assertEqual(c.inputs[0], a.outputs[0])
            self.assertEqual(c.inputs[1], b.outputs[0])
            self.assertEqual(d.inputs[0], a.outputs[1])
            self.assertEqual(d.inputs[1], b.outputs[1])
            self.assertEqual(a.type, 'IntOutputFloatOutput')
            self.assertEqual(b.type, 'ListOutput')
            self.assertEqual(c.type, 'ListInput')
            self.assertEqual(d.type, 'ListInput')
            self.assertEqual(a.outputs[0].dtype, dtypes.int32)
            self.assertEqual(a.outputs[1].dtype, dtypes.float32)
            self.assertEqual(b.outputs[0].dtype, dtypes.int32)
            self.assertEqual(b.outputs[1].dtype, dtypes.float32)
            self.assertEqual(a.name, 'import/A')
            self.assertEqual(b.name, 'import/B')
            self.assertEqual(c.name, 'import/C')
            self.assertEqual(d.name, 'import/D')
            self.assertNotEqual(None, a.op_def)

    def testMultipleImport(self):
        if False:
            return 10
        graph_def = self._MakeGraphDef("\n    node { name: 'A' op: 'IntOutput' }\n    node { name: 'B' op: 'IntInput' input: 'A:0' }\n    ")
        with ops.Graph().as_default():
            (a, b) = importer.import_graph_def(graph_def, return_elements=['A', 'B'], name='')
            self.assertEqual(a.name, 'A')
            self.assertEqual(b.name, 'B')
            self.assertEqual(list(b.inputs), [a.outputs[0]])
            (a1, b1) = importer.import_graph_def(graph_def, return_elements=['A', 'B'], name='')
            self.assertEqual(a1.name, 'A_1')
            self.assertEqual(b1.name, 'B_1')
            self.assertEqual(list(b1.inputs), [a1.outputs[0]])
            (a2, b2) = importer.import_graph_def(graph_def, return_elements=['A', 'B'], name='')
            self.assertEqual(a2.name, 'A_2')
            self.assertEqual(b2.name, 'B_2')
            self.assertEqual(list(b2.inputs), [a2.outputs[0]])
            (a3, b3) = importer.import_graph_def(graph_def, return_elements=['A', 'B'], name='A')
            self.assertEqual(a3.name, 'A_3/A')
            self.assertEqual(b3.name, 'A_3/B')
            self.assertEqual(list(b3.inputs), [a3.outputs[0]])
            (a_a, a_b) = importer.import_graph_def(graph_def, return_elements=['A', 'B'], name='A/')
            self.assertEqual(a_a.name, 'A/A')
            self.assertEqual(a_b.name, 'A/B')
            self.assertEqual(list(a_b.inputs), [a_a.outputs[0]])
            (a_a1, a_b1) = importer.import_graph_def(graph_def, return_elements=['A', 'B'], name='A/')
            self.assertEqual(a_a1.name, 'A/A_1')
            self.assertEqual(a_b1.name, 'A/B_1')
            self.assertEqual(list(a_b1.inputs), [a_a1.outputs[0]])
            (a1_1, b1_1) = importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A_1' op: 'IntOutput' }\n          node { name: 'B_1' op: 'IntInput' input: 'A_1:0' }\n          "), return_elements=['A_1', 'B_1'], name='')
            self.assertEqual(a1_1.name, 'A_1_1')
            self.assertEqual(b1_1.name, 'B_1_1')
            self.assertEqual(list(b1_1.inputs), [a1_1.outputs[0]])
            with ops.name_scope('foo'):
                constant_op.constant(1)
            (foo,) = importer.import_graph_def(self._MakeGraphDef("node { name: 'foo' op: 'IntOutput' }"), return_elements=['foo'], name='')
            self.assertEqual(foo.name, 'foo_1')
            with ops.name_scope('outer'):
                with ops.name_scope('inner'):
                    c = constant_op.constant(1, name='c')
                    self.assertEqual(c.op.name, 'outer/inner/c')
            (outer, inner, new_c, outer_inner, outer_inner_c) = importer.import_graph_def(self._MakeGraphDef("node { name: 'outer' op: 'IntOutput' }node { name: 'inner' op: 'IntOutput' }node { name: 'c' op: 'IntOutput' }node { name: 'outer/inner' op: 'IntOutput' }node { name: 'outer/inner/c' op: 'IntOutput' }"), return_elements=['outer', 'inner', 'c', 'outer/inner', 'outer/inner/c'], name='')
            self.assertEqual(outer.name, 'outer_1')
            self.assertEqual(inner.name, 'inner')
            self.assertEqual(new_c.name, 'c')
            self.assertEqual(outer_inner.name, 'outer/inner_1')
            self.assertEqual(outer_inner_c.name, 'outer/inner/c_1')

    def testEmptyNameScope(self):
        if False:
            return 10
        with ops.Graph().as_default():
            with ops.name_scope('foo'):
                pass
            (op,) = importer.import_graph_def(self._MakeGraphDef("node { name: 'foo' op: 'IntOutput' }"), return_elements=['foo'], name='')
            self.assertEqual(op.name, 'foo')

    def testInputMap(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            feed_a_0 = constant_op.constant(0, dtype=dtypes.int32)
            feed_b_1 = constant_op.constant(1, dtype=dtypes.int32)
            (a, b, c, d) = importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A' op: 'TwoIntOutputs' }\n          node { name: 'B' op: 'TwoIntOutputs' }\n          node { name: 'C' op: 'ListInput'\n                 attr { key: 'N' value { i: 2 } }\n                 attr { key: 'T' value { type: DT_INT32 } }\n                 input: 'A:0' input: 'B:0' }\n          node { name: 'D' op: 'ListInput'\n                 attr { key: 'N' value { i: 2 } }\n                 attr { key: 'T' value { type: DT_INT32 } }\n                 input: 'A:1' input: 'B:1' }\n          "), input_map={'A:0': feed_a_0, 'B:1': feed_b_1}, return_elements=['A', 'B', 'C', 'D'])
            self.assertEqual(c.inputs[0], feed_a_0)
            self.assertEqual(c.inputs[1], b.outputs[0])
            self.assertEqual(d.inputs[0], a.outputs[1])
            self.assertEqual(d.inputs[1], feed_b_1)

    def testInputMapBytes(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            feed_a_0 = constant_op.constant(0, dtype=dtypes.int32)
            feed_b_1 = constant_op.constant(1, dtype=dtypes.int32)
            (a, b, c, d) = importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A' op: 'TwoIntOutputs' }\n          node { name: 'B' op: 'TwoIntOutputs' }\n          node { name: 'C' op: 'ListInput'\n                 attr { key: 'N' value { i: 2 } }\n                 attr { key: 'T' value { type: DT_INT32 } }\n                 input: 'A:0' input: 'B:0' }\n          node { name: 'D' op: 'ListInput'\n                 attr { key: 'N' value { i: 2 } }\n                 attr { key: 'T' value { type: DT_INT32 } }\n                 input: 'A:1' input: 'B:1' }\n          "), input_map={b'A:0': feed_a_0, b'B:1': feed_b_1}, return_elements=[b'A', b'B', b'C', b'D'])
            self.assertEqual(c.inputs[0], feed_a_0)
            self.assertEqual(c.inputs[1], b.outputs[0])
            self.assertEqual(d.inputs[0], a.outputs[1])
            self.assertEqual(d.inputs[1], feed_b_1)

    def testInputMapUnicode(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            feed_a_0 = constant_op.constant(0, dtype=dtypes.int32)
            feed_b_1 = constant_op.constant(1, dtype=dtypes.int32)
            (a, b, c, d) = importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A' op: 'TwoIntOutputs' }\n          node { name: 'B' op: 'TwoIntOutputs' }\n          node { name: 'C' op: 'ListInput'\n                 attr { key: 'N' value { i: 2 } }\n                 attr { key: 'T' value { type: DT_INT32 } }\n                 input: 'A:0' input: 'B:0' }\n          node { name: 'D' op: 'ListInput'\n                 attr { key: 'N' value { i: 2 } }\n                 attr { key: 'T' value { type: DT_INT32 } }\n                 input: 'A:1' input: 'B:1' }\n          "), input_map={u'A:0': feed_a_0, u'B:1': feed_b_1}, return_elements=[u'A', u'B', u'C', u'D'])
            self.assertEqual(c.inputs[0], feed_a_0)
            self.assertEqual(c.inputs[1], b.outputs[0])
            self.assertEqual(d.inputs[0], a.outputs[1])
            self.assertEqual(d.inputs[1], feed_b_1)

    def testImplicitZerothOutput(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            (a, b) = importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A' op: 'TwoIntOutputs' }\n          node { name: 'B' op: 'IntInput' input: 'A' }\n          "), return_elements=['A', 'B'])
            self.assertEqual(b.inputs[0], a.outputs[0])

    def testInputMapImplicitZerothOutput(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            feed_a_0 = constant_op.constant(0, dtype=dtypes.int32)
            (b,) = importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A' op: 'TwoIntOutputs' }\n          node { name: 'B' op: 'IntInput' input: 'A:0' }\n          "), input_map={'A': feed_a_0}, return_elements=['B'])
            self.assertEqual(b.inputs[0], feed_a_0)

    def testWithControlDependency(self):
        if False:
            return 10
        with ops.Graph().as_default():
            (a, b) = importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A' op: 'None' }\n          node { name: 'B' op: 'None' input: '^A' }\n          "), return_elements=['A', 'B'])
            self.assertEqual(b.control_inputs, [a])

    def testWithRefs(self):
        if False:
            return 10
        with ops.Graph().as_default():
            (a, b, c, d) = importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A' op: 'RefOutput' }\n          node { name: 'B' op: 'IntOutput' }\n          node { name: 'C' op: 'TwoIntInputs' input: 'A:0' input: 'B:0' }\n          node { name: 'D' op: 'RefInputIntInput' input: 'A:0' input: 'B:0' }\n          "), return_elements=['A', 'B', 'C', 'D'])
            self.assertEqual(c.inputs[0], a.outputs[0])
            self.assertEqual(c.inputs[1], b.outputs[0])
            self.assertEqual(d.inputs[0], a.outputs[0])
            self.assertEqual(d.inputs[1], b.outputs[0])
            self.assertEqual(a.outputs[0].dtype, dtypes.int32_ref)
            self.assertEqual(c._input_types, [dtypes.int32, dtypes.int32])
            self.assertEqual(c.outputs, [])
            self.assertEqual(d._input_types, [dtypes.int32_ref, dtypes.int32])
            self.assertEqual(d.outputs, [])

    def testResources(self):
        if False:
            while True:
                i = 10
        graph = ops.Graph()
        with graph.as_default():
            var = resource_variable_ops.ResourceVariable(1.0)
            var_assign = var.assign(2.0)
            var_shape = resource_variable_ops.variable_shape(var.handle)
            init = variables.global_variables_initializer()
        graph_def = graph.as_graph_def()
        with ops.Graph().as_default():
            (imported_var, imported_assign, imported_shape, imported_init) = importer.import_graph_def(graph_def, return_elements=[var.name, var_assign.name, var_shape.name, init.name])
            new_var_shape = resource_variable_ops.variable_shape(imported_var)

    def testWhileLoop(self):
        if False:
            return 10
        graph = ops.Graph()
        with graph.as_default():
            r = while_loop.while_loop(lambda i: i < 10, lambda i: i + 1, [0])
            math_ops.add(r, 1)
        graph_def = graph.as_graph_def()
        with ops.Graph().as_default():
            (imported_r,) = importer.import_graph_def(graph_def, return_elements=[r.name])
            self.assertEqual(imported_r.name, 'import/' + r.name)
            with self.cached_session() as sess:
                self.assertEqual(self.evaluate(imported_r), 10)

    def testImportWhileLoopInCond(self):
        if False:
            return 10
        graph = ops.Graph()
        with graph.as_default():
            r = while_loop.while_loop(lambda i: i < 10, lambda i: i + 1, [0])
        graph_def = graph.as_graph_def()
        with ops.Graph().as_default():

            def ImportFn():
                if False:
                    for i in range(10):
                        print('nop')
                return importer.import_graph_def(graph_def, return_elements=[r.name])[0]
            pred = array_ops.placeholder(dtypes.bool)
            out = cond.cond(pred, ImportFn, lambda : constant_op.constant(1))
            with self.cached_session() as sess:
                self.assertEqual(sess.run(out, {pred: True}), 10)
                self.assertEqual(sess.run(out, {pred: False}), 1)

    def testImportWhileLoopInWhileLoop(self):
        if False:
            i = 10
            return i + 15
        self.skipTest('b/111757448')
        graph = ops.Graph()
        with graph.as_default():
            r = while_loop.while_loop(lambda i: i < 10, lambda i: i + 1, [0])
        graph_def = graph.as_graph_def()
        with ops.Graph().as_default():

            def ImportFn(_):
                if False:
                    i = 10
                    return i + 15
                return importer.import_graph_def(graph_def, return_elements=[r.name])[0]
            out = while_loop.while_loop(lambda i: i < 2, ImportFn, [0], shape_invariants=[tensor_shape.TensorShape(None)])
            with self.cached_session() as sess:
                self.assertEqual(self.evaluate(out), 10)

    def testTypeMismatchInGraphDef(self):
        if False:
            print('Hello World!')
        error_msg = 'Input 0 of node import/B was passed int32 from import/A:0 incompatible with expected float.'
        with ops.Graph().as_default():
            with self.assertRaisesRegex(ValueError, error_msg):
                importer.import_graph_def(self._MakeGraphDef("\n            node { name: 'A' op: 'IntOutput' }\n            node { name: 'B' op: 'FloatInput' input: 'A:0' }\n            "))

    def testShapeAllowlistViolation(self):
        if False:
            return 10
        with ops.Graph().as_default():
            with self.assertRaises(ValueError) as e:
                _ = importer.import_graph_def(self._MakeGraphDef("\n              node { name: 'A' op: 'FloatOutput' }\n              node { name: 'B' op: 'L2Loss'\n                     input: 'A:0'\n                     attr { key: 'T' value { type: DT_FLOAT } }\n                     attr { key: '_output_shapes'\n                            value { list { shape { dim { size: 43 } } } } } }\n            "), return_elements=['B'], name='import')
                self.assertTrue('Shapes () and (43,) are not compatible' in str(e.exception))

    def testInvalidSignatureTooManyInputsInGraphDef(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            with self.assertRaisesRegex(ValueError, "NodeDef expected inputs '' do not match 1 inputs specified"):
                importer.import_graph_def(self._MakeGraphDef("\n            node { name: 'A' op: 'IntOutput' }\n            node { name: 'B' op: 'None' input: 'A:0' }\n            "))

    def testInvalidSignatureNotEnoughInputsInGraphDef(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            with self.assertRaisesRegex(ValueError, "NodeDef expected inputs 'int32, float' do not match 1 inputs specified"):
                importer.import_graph_def(self._MakeGraphDef("\n            node { name: 'A' op: 'IntOutput' }\n            node { name: 'B' op: 'IntInputFloatInput' input: 'A:0' }\n            "))

    def testMissingInputOpInGraphDef(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            with self.assertRaisesRegex(ValueError, "Node 'B': Unknown input node 'A:0'"):
                importer.import_graph_def(self._MakeGraphDef("\n            node { name: 'B' op: 'FloatInput' input: 'A:0' }\n            "))

    def testMissingInputOpInGraphDefButAppearsInInputMap(self):
        if False:
            return 10
        with ops.Graph().as_default():
            feed_a_0 = constant_op.constant(5.0)
            (b,) = importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'B' op: 'FloatInput' input: 'A:0' }\n          "), input_map={'A:0': feed_a_0}, return_elements=['B'])
            self.assertEqual(b.inputs[0], feed_a_0)

    def testMissingInputTensorInGraphDef(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            with self.assertRaisesRegex(ValueError, "Node 'B': Connecting to invalid output 1 of source node A which has 1 outputs"):
                importer.import_graph_def(self._MakeGraphDef("\n            node { name: 'A' op: 'FloatOutput' }\n            node { name: 'B' op: 'FloatInput' input: 'A:1' }\n            "))

    def testMissingControlInputInGraphDef(self):
        if False:
            return 10
        with ops.Graph().as_default():
            with self.assertRaisesRegex(ValueError, "Node 'B': Unknown input node '\\^A'"):
                importer.import_graph_def(self._MakeGraphDef("\n            node { name: 'B' op: 'None' input: '^A' }\n            "))

    def testInvalidTensorNameOutputIndexInGraphDef(self):
        if False:
            return 10
        with ops.Graph().as_default():
            with self.assertRaisesRegex(ValueError, "Node 'B': Unknown input node 'A:B'"):
                importer.import_graph_def(self._MakeGraphDef("\n            node { name: 'B' op: 'None' input: 'A:B' }\n            "))

    def testInvalidTensorNameInGraphDef(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            with self.assertRaisesRegex(ValueError, "Node 'B': Unknown input node 'A:B:0'"):
                importer.import_graph_def(self._MakeGraphDef("\n            node { name: 'B' op: 'None' input: 'A:B:0' }\n            "))

    def testMissingReturnOperation(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            with self.assertRaisesRegex(ValueError, "Requested return node 'B' not found in graph def"):
                importer.import_graph_def(self._MakeGraphDef("\n            node { name: 'A' op: 'None' }\n            "), return_elements=['B'])

    def testMissingReturnTensor(self):
        if False:
            return 10
        with ops.Graph().as_default():
            with self.assertRaisesRegex(ValueError, "Invalid return output 1 of node 'A', which has 1 output\\(s\\)"):
                importer.import_graph_def(self._MakeGraphDef("\n            node { name: 'A' op: 'IntOutput' }\n            "), return_elements=['A:1'])
            with self.assertRaisesRegex(ValueError, "Requested return tensor 'B:0' not found in graph def"):
                importer.import_graph_def(self._MakeGraphDef("\n            node { name: 'A' op: 'IntOutput' }\n            "), return_elements=['B:0'])
            with self.assertRaisesRegex(ValueError, "Cannot convert 'A:B:0' to a tensor name."):
                importer.import_graph_def(self._MakeGraphDef("\n            node { name: 'A' op: 'IntOutput' }\n            "), return_elements=['A:B:0'])

    def testMissingInputMap(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            with self.assertRaisesRegex(ValueError, 'Attempted to map inputs that were not found in graph_def: \\[B:0\\]'):
                importer.import_graph_def(self._MakeGraphDef("\n            node { name: 'A' op: 'None' }\n            "), input_map={'B:0': constant_op.constant(5.0)})

    def testInputMapUnusedAsInput(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A' op: 'IntOutput' }\n          "), input_map={'A:0': constant_op.constant(5.0)})
            with self.assertRaisesRegex(ValueError, 'Attempted to map inputs that were not found in graph_def: \\[A:2\\]'):
                importer.import_graph_def(self._MakeGraphDef("\n            node { name: 'A' op: 'IntOutput' }\n            "), input_map={'A:2': constant_op.constant(5.0)})

    def testInputMapTypeMismatch(self):
        if False:
            return 10
        with ops.Graph().as_default():
            with self.assertRaisesRegex(ValueError, 'Input 0 of node import/B was passed float from Const:0 incompatible with expected int32.'):
                importer.import_graph_def(self._MakeGraphDef("\n            node { name: 'A' op: 'IntOutput' }\n            node { name: 'B' op: 'IntInput' input: 'A:0' }\n            "), input_map={'A:0': constant_op.constant(5.0)})

    def testNoReturns(self):
        if False:
            return 10
        with ops.Graph().as_default() as g:
            ret = importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A' op: 'None' }\n          "))
            self.assertEqual(ret, None)
            a = g.get_operation_by_name('import/A')
            self.assertEqual(a.type, 'None')

    def testOverrideNamePrefix(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            (a,) = importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A' op: 'None' }\n          "), return_elements=['A'], name='imported_graph')
            self.assertEqual(a.name, 'imported_graph/A')

    def testDefaultNamePrefix(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            (a,) = importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A' op: 'None' }\n          "), return_elements=['A'], name=None)
            self.assertEqual(a.name, 'import/A')

    def testNamePrefixColocationAttrs(self):
        if False:
            while True:
                i = 10
        original_graph_def = self._MakeGraphDef("\n          node { name: 'A' op: 'None' }\n          node { name: 'B' op: 'None'  attr {\n            key: '_class'\n            value { list { s: 'loc:@A' } }\n          } }")
        with ops.Graph().as_default():
            (b,) = importer.import_graph_def(original_graph_def, return_elements=['B'], name='imported_graph')
            self.assertTrue('_class' in b.node_def.attr)
            self.assertProtoEquals("list { s: 'loc:@imported_graph/A' }", b.node_def.attr['_class'])

    def testColocationAndDevice(self):
        if False:
            i = 10
            return i + 15
        original_graph_def = self._MakeGraphDef("\n          node { name: 'A' op: 'None' device: '/device:CPU:0' attr {\n            key: '_class'\n            value { list { s: 'loc:@A' } }\n          } }\n          node { name: 'B' op: 'None'  attr {\n            key: '_class'\n            value { list { s: 'loc:@A' } }\n          } }")
        with ops.Graph().as_default():
            (a, b) = importer.import_graph_def(original_graph_def, return_elements=['A', 'B'], name='')
            self.assertEqual(a.device, '/device:CPU:0')
            self.assertEqual(b.device, '/device:CPU:0')
            self.assertEqual(a.colocation_groups(), [b'loc:@A'])
            self.assertEqual(b.colocation_groups(), [b'loc:@A'])
        original_graph_def = self._MakeGraphDef("\n          node { name: 'A' op: 'None' attr {\n            key: '_class'\n            value { list { s: 'loc:@A' } }\n          } }\n          node { name: 'B' op: 'None' device: '/device:CPU:0' attr {\n            key: '_class'\n            value { list { s: 'loc:@A' } }\n          } }")
        with ops.Graph().as_default():
            (a, b) = importer.import_graph_def(original_graph_def, return_elements=['A', 'B'], name='')
            self.assertEqual(a.device, '')
            self.assertEqual(b.device, '')
            self.assertEqual(a.colocation_groups(), [b'loc:@A'])
            self.assertEqual(b.colocation_groups(), [b'loc:@A'])

    def testColocationWithDeviceFn(self):
        if False:
            for i in range(10):
                print('nop')
        original_graph_def = self._MakeGraphDef("\n          node { name: 'A' op: 'None' attr {\n            key: '_class'\n            value { list { s: 'loc:@A' } }\n          } }\n          node { name: 'B' op: 'None'  attr {\n            key: '_class'\n            value { list { s: 'loc:@A' } }\n          } }")

        def CustomDeviceFn(op):
            if False:
                while True:
                    i = 10
            if 'A' in op.name:
                return '/device:A:0'
            else:
                return '/device:B:0'
        with ops.Graph().as_default():
            with ops.device(CustomDeviceFn):
                (a, b) = importer.import_graph_def(original_graph_def, return_elements=['A', 'B'], name='imported_graph')
            self.assertEqual(a.device, '/device:A:0')
            self.assertEqual(b.device, '/device:A:0')
            self.assertEqual(a.colocation_groups(), [b'loc:@imported_graph/A'])
            self.assertEqual(b.colocation_groups(), [b'loc:@imported_graph/A'])

        def BDeviceFn(op):
            if False:
                i = 10
                return i + 15
            if 'B' in op.name:
                return '/device:B:0'
            return ''
        with ops.Graph().as_default():
            with ops.device(BDeviceFn):
                (a, b) = importer.import_graph_def(original_graph_def, return_elements=['A', 'B'], name='imported_graph')
            self.assertEqual(a.device, '')
            self.assertEqual(b.device, '')
            self.assertEqual(a.colocation_groups(), [b'loc:@imported_graph/A'])
            self.assertEqual(b.colocation_groups(), [b'loc:@imported_graph/A'])

        def ADeviceFn(op):
            if False:
                while True:
                    i = 10
            if 'A' in op.name:
                return '/device:A:0'
            return ''
        with ops.Graph().as_default():
            with ops.device(ADeviceFn):
                (a, b) = importer.import_graph_def(original_graph_def, return_elements=['A', 'B'], name='imported_graph')
            self.assertEqual(a.device, '/device:A:0')
            self.assertEqual(b.device, '/device:A:0')
            self.assertEqual(a.colocation_groups(), [b'loc:@imported_graph/A'])
            self.assertEqual(b.colocation_groups(), [b'loc:@imported_graph/A'])

    def testMultipleColocationWithDeviceFn(self):
        if False:
            for i in range(10):
                print('nop')
        original_graph_def = self._MakeGraphDef("\n          node { name: 'A' op: 'None'}\n          node { name: 'B' op: 'None'}\n          node { name: 'C' op: 'None'  attr {\n            key: '_class'\n            value { list { s: 'loc:@A' s: 'loc:@B' } }\n          } }")

        def CustomDeviceFn(op):
            if False:
                while True:
                    i = 10
            if 'B' in op.name:
                return '/device:B:0'
            return ''
        with ops.Graph().as_default():
            with ops.device(CustomDeviceFn):
                (a, b, c) = importer.import_graph_def(original_graph_def, return_elements=['A', 'B', 'C'], name='imported_graph')
            self.assertEqual(a.device, '')
            self.assertEqual(b.device, '/device:B:0')
            self.assertEqual(c.device, '/device:B:0')
            self.assertEqual(a.colocation_groups(), [b'loc:@imported_graph/A'])
            self.assertEqual(b.colocation_groups(), [b'loc:@imported_graph/B'])
            self.assertEqual(c.colocation_groups(), [b'loc:@imported_graph/A', b'loc:@imported_graph/B'])

    def testNamePrefixColocationAttrsMultipleImport(self):
        if False:
            return 10
        original_graph_def = self._MakeGraphDef("\n          node { name: 'A' op: 'None' }\n          node { name: 'B' op: 'None'  attr {\n            key: '_class'\n            value { list { s: 'loc:@A' } }\n          } }")
        with ops.Graph().as_default():
            (a, b) = importer.import_graph_def(original_graph_def, return_elements=['A', 'B'], name='')
            (a_1, b_1) = importer.import_graph_def(original_graph_def, return_elements=['A', 'B'], name='')
            self.assertEqual(a.name, 'A')
            self.assertEqual(b.name, 'B')
            self.assertEqual(b.colocation_groups(), [b'loc:@A'])
            self.assertEqual(a_1.name, 'A_1')
            self.assertEqual(b_1.name, 'B_1')
            self.assertEqual(b_1.colocation_groups(), [b'loc:@A_1'])

    def testNamePrefixColocationAttrsNotFound(self):
        if False:
            i = 10
            return i + 15
        original_graph_def = self._MakeGraphDef("\n          node { name: 'B' op: 'None'  attr {\n            key: '_class'\n            value { list { s: 'loc:@A' } }\n          } }")
        with ops.Graph().as_default():
            with self.assertRaisesRegex(ValueError, "Node 'B' expects to be colocated with unknown node 'A'"):
                importer.import_graph_def(original_graph_def, return_elements=['B'], name='imported_graph')

    def testEmptyGraph(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default() as g:
            init_version = g.version
            importer.import_graph_def(self._MakeGraphDef(''))
            self.assertEqual(init_version, g.version)

    def testInvalidInputForGraphDef(self):
        if False:
            return 10
        with ops.Graph().as_default():
            with self.assertRaisesRegex(TypeError, 'Argument `graph_def` must be a GraphDef proto.'):
                importer.import_graph_def('')

    def testInvalidInputForInputMap(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            with self.assertRaisesRegex(TypeError, 'Argument `input_map` must be a dictionary. Obtained list'):
                importer.import_graph_def(self._MakeGraphDef(''), input_map=[constant_op.constant(5.0)])
        graph_def = self._MakeGraphDef("\n         node { name: 'a' op: 'Placeholder'\n                attr { key: 'dtype' value { type: DT_FLOAT } }}\n         node { name: 'id' op: 'Identity' input: 'a:0'\n                attr { key: 'T' value { type: DT_FLOAT } }}")
        with ops.Graph().as_default():
            with self.assertRaises(ValueError) as e:
                importer.import_graph_def(graph_def, input_map={'a:0': variables.Variable(5.0)}, name='')
            self.assertStartsWith(str(e.exception), 'tf.import_graph_def() requires a non-empty `name` if `input_map` contains non-Tensor values.')
        with ops.Graph().as_default():
            (t,) = importer.import_graph_def(graph_def, input_map={'a:0': constant_op.constant(5.0)}, name='', return_elements=['id:0'])
            with self.cached_session():
                self.assertEqual(5.0, self.evaluate(t))

    def testInvalidInputForReturnOperations(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            with self.assertRaisesRegex(TypeError, 'Argument `return_elements` must be a list of strings.'):
                importer.import_graph_def(self._MakeGraphDef(''), return_elements=[7])
            with self.assertRaisesRegex(ValueError, "Cannot convert 'a:b:c' to a tensor name."):
                importer.import_graph_def(self._MakeGraphDef(''), return_elements=['a:b:c'])

    def testDuplicateOperationNames(self):
        if False:
            return 10
        with self.assertRaisesRegex(ValueError, "Node 'A' is not unique"):
            importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A' op: 'IntOutput' }\n          node { name: 'B' op: 'IntOutput' }\n          node { name: 'A' op: 'IntOutput' }\n          "))

    @test_util.run_v1_only("v1 Tensor doesn't have attribute 'numpy'")
    def testWithExtensionAndAttr(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default() as g:
            c = constant_op.constant(5.0, dtype=dtypes.float32, name='c')
            array_ops_stack.stack([c, c], name='pack')
        gdef = g.as_graph_def()
        with self.cached_session():
            (pack,) = importer.import_graph_def(gdef, return_elements=['pack'])
            self.assertAllEqual(pack.outputs[0], [5.0, 5.0])

    def testWithDevice(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default() as g:
            a = constant_op.constant(3.0, name='a')
            with ops.device('/cpu:0'):
                b = constant_op.constant(4.0, name='b')
            with ops.device('/job:worker'):
                c = constant_op.constant(5.0, name='c')
        gdef = g.as_graph_def()
        with ops.Graph().as_default():
            (a2, b2, c2) = importer.import_graph_def(gdef, return_elements=['a', 'b', 'c'])
            self.assertEqual(a.device, a2.device)
            self.assertEqual(b.device, b2.device)
            self.assertEqual(c.device, c2.device)
        with ops.Graph().as_default():
            with ops.device(device.merge_device('/task:0')):
                (a3, b3, c3) = importer.import_graph_def(gdef, return_elements=['a', 'b', 'c'])
                self.assertEqual('/task:0', a3.device)
                self.assertEqual('/task:0/device:CPU:0', b3.device)
                self.assertEqual(c.device + '/task:0', c3.device)
        with ops.Graph().as_default():
            with ops.device(device.merge_device('/job:ps')):
                (a4, b4, c4) = importer.import_graph_def(gdef, return_elements=['a', 'b', 'c'])
                self.assertEqual('/job:ps', a4.device)
                self.assertEqual('/job:ps/device:CPU:0', b4.device)
                self.assertEqual(c.device, c4.device)
        with ops.Graph().as_default():
            with ops.device(device.merge_device('/device:GPU:0')):
                (a5, b5, c5) = importer.import_graph_def(gdef, return_elements=['a', 'b', 'c'])
                self.assertEqual('/device:GPU:0', a5.device)
                self.assertEqual('/device:CPU:0', b5.device)
                self.assertEqual(c.device + '/device:GPU:0', c5.device)

    def testWithDeviceFunctionDependingOnInputs(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default() as g:
            with ops.device('/job:ps'):
                v1 = constant_op.constant(1.0)
                v2 = constant_op.constant(1.0)
            _ = v1 + v2
            _ = v1 - v2
            _ = array_ops.identity(v1)
        gdef = g.as_graph_def()
        ops_with_two_inputs = []

        def InputCounter(op):
            if False:
                while True:
                    i = 10
            if len(op.inputs) == 2:
                ops_with_two_inputs.append(op)
            return ''
        with ops.Graph().as_default() as g:
            with ops.device(InputCounter):
                importer.import_graph_def(gdef)
        self.assertEqual(2, len(ops_with_two_inputs))

    def testGradient(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default() as g:
            inputs = array_ops.placeholder(dtypes.float32, shape=[None, 100], name='input')
            weights = array_ops.placeholder(dtypes.float32, shape=[100, 10], name='weights')
            biases = array_ops.placeholder(dtypes.float32, shape=[10], name='biases')
            activations = nn_ops.relu(math_ops.matmul(inputs, weights) + biases, name='activations')
            loss = math_ops.reduce_mean(activations, name='loss')
        gdef = g.as_graph_def()
        with ops.Graph().as_default() as g:
            input_placeholder = array_ops.placeholder(dtypes.float32, shape=[32, 100])
            weights_var = variables.Variable(random_ops.truncated_normal([100, 10]), name='weights')
            biases_var = variables.Variable(array_ops.zeros([10]), name='biases')
            (activations, loss) = importer.import_graph_def(gdef, input_map={'input:0': input_placeholder, 'weights:0': weights_var, 'biases:0': biases_var}, return_elements=['activations:0', 'loss:0'])
            self.assertEqual([32, 10], activations.get_shape())
            self.assertEqual([], loss.get_shape())
            (weights_grad, biases_grad) = gradients_impl.gradients(loss, [weights_var, biases_var])
            self.assertEqual([100, 10], weights_grad.get_shape())
            self.assertEqual([10], biases_grad.get_shape())

    def testLargeGraph(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            input_shape = [130, 1000, 1000]
            tensor_input = np.ones(input_shape, dtype=np.float32)
            t = constant_op.constant(tensor_input, shape=input_shape)
            g = array_ops.identity(t)
            self.evaluate(g)

    def testVersion(self):
        if False:
            i = 10
            return i + 15
        v0 = versions.GRAPH_DEF_VERSION_MIN_CONSUMER
        v2 = versions.GRAPH_DEF_VERSION
        v1 = (v0 + v2) // 2
        for producer in (v0, v1, v2):
            for min_consumer in (v0, v1, v2):
                with ops.Graph().as_default():
                    (a,) = importer.import_graph_def(self._MakeGraphDef("node { name: 'A' op: 'TwoIntOutputs' }", producer=producer, min_consumer=min_consumer), return_elements=['A'])
                    self.assertEqual(a.graph.graph_def_versions.producer, producer)
                    self.assertEqual(a.graph.graph_def_versions.min_consumer, min_consumer)

    def testVersionLow(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            with self.assertRaisesRegex(Exception, 'GraphDef producer version -1 below min producer %d supported by TensorFlow \\S+\\.  Please regenerate your graph.$' % versions.GRAPH_DEF_VERSION_MIN_PRODUCER):
                importer.import_graph_def(self._MakeGraphDef('', producer=-1))

    def testVersionHigh(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            with self.assertRaisesRegex(ValueError, 'GraphDef min consumer version %d above current version %d for TensorFlow \\S+\\.  Please upgrade TensorFlow\\.$' % (1 << 30, versions.GRAPH_DEF_VERSION)):
                importer.import_graph_def(self._MakeGraphDef('', min_consumer=1 << 30))

    def testVersionAppliesToOpConstruction(self):
        if False:
            for i in range(10):
                print('nop')
        'These tests rely on shape fns in test_ops.cc.'
        with ops.Graph().as_default():
            importer.import_graph_def(self._MakeGraphDef("node { name: 'A' op: 'RequiresOlderGraphVersion' }", producer=versions.GRAPH_DEF_VERSION - 1), return_elements=['A'])
        with ops.Graph().as_default():
            with self.assertRaisesWithPredicateMatch(ValueError, 'Wrong graph version.*'):
                importer.import_graph_def(self._MakeGraphDef("node { name: 'A' op: 'RequiresOlderGraphVersion' }", producer=versions.GRAPH_DEF_VERSION), return_elements=['A'])

    def testDefaultAttrsAdded(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            a = importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A' op: 'OpWithDefaultAttr' }\n          "), return_elements=['A'])
            self.assertEqual(123.0, a[0].get_attr('default_float'))

    def testDefaultAttrsRemoved(self):
        if False:
            i = 10
            return i + 15
        producer_op_list = op_def_pb2.OpList()
        text_format.Merge("\n      op {\n        name: 'OpWithFutureDefaultAttr'\n        attr { name: 'default_int' type: 'int' default_value { i: 456 } }\n      }\n    ", producer_op_list)
        with ops.Graph().as_default():
            a = importer.import_graph_def(self._MakeGraphDef("\n          node { name: 'A' op: 'OpWithFutureDefaultAttr'\n                 attr { key: 'default_int' value { i: 456 } } }\n          "), return_elements=['A'], producer_op_list=producer_op_list)
            with self.assertRaisesRegex(ValueError, "Operation 'import/A' has no attr named 'default_int'."):
                a[0].get_attr('default_int')

    def testFunctions(self):
        if False:
            while True:
                i = 10
        dtype = dtypes.float32

        @function.Defun(dtype, dtype, dtype, dtype)
        def Grad(x, y, dout1, dout2):
            if False:
                return 10
            return (x, y)

        @function.Defun(dtype, dtype, grad_func=Grad)
        def FuncWithGrad(x, y):
            if False:
                while True:
                    i = 10
            return (x + y, x - y)

        @function.Defun(dtypes.int32)
        def ExternalTensorFunc(x):
            if False:
                return 10
            return x + c

        @function.Defun(dtypes.int32, dtypes.int32)
        def OuterFunc(x, y):
            if False:
                i = 10
                return i + 15

            @function.Defun(dtypes.int32)
            def InnerFunc(x):
                if False:
                    i = 10
                    return i + 15
                return x + x
            return InnerFunc(x) + y
        with ops.Graph().as_default() as g1:
            p1 = array_ops.placeholder(dtype, name='p1')
            p2 = array_ops.placeholder(dtype, name='p2')
            (a, b) = FuncWithGrad(p1, p2, name='f')
            c = constant_op.constant(10, dtype=dtypes.int32)
            ExternalTensorFunc(1, name='external')
            OuterFunc(10, 1, name='outer')
        gdef = g1.as_graph_def()
        with ops.Graph().as_default() as g2:
            (p1, p2, a, b) = importer.import_graph_def(gdef, return_elements=['p1:0', 'p2:0', 'f:0', 'f:1'], name='')
            grad = gradients_impl.gradients([a], [p1, p2])
            with self.session(graph=g2) as sess:
                feed_dict = {p1: 1, p2: 2}
                (a_val, b_val, grad_val) = sess.run([a, b, grad], feed_dict=feed_dict)
                self.assertEqual(a_val, 3.0)
                self.assertEqual(b_val, -1.0)
                self.assertEqual(grad_val, [1.0, 2.0])
                self.assertEqual(sess.run('external:0'), 11)
                self.assertEqual(sess.run('outer:0'), 21)
        gdef = g2.as_graph_def()
        with ops.Graph().as_default() as g3:
            (p1, p2, a, b) = importer.import_graph_def(gdef, return_elements=['p1:0', 'p2:0', 'f:0', 'f:1'], name='')
            grad = gradients_impl.gradients([a], [p1, p2])
            with self.session(graph=g3) as sess:
                feed_dict = {p1: 1, p2: 2}
                (a_val, b_val, grad_val) = sess.run([a, b, grad], feed_dict=feed_dict)
                self.assertEqual(a_val, 3.0)
                self.assertEqual(b_val, -1.0)
                self.assertEqual(grad_val, [1.0, 2.0])
                self.assertEqual(sess.run('external:0'), 11)
                self.assertEqual(sess.run('outer:0'), 21)

    @test_util.run_v1_only('import inside defun not supported when eager execution is enabled.')
    def testImportInsideDefun(self):
        if False:
            print('Hello World!')
        g = ops.Graph()
        with g.as_default():

            @function.Defun()
            def Add2(x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return math_ops.add(x, y)
            x = constant_op.constant(3.0, dtype=dtypes.float32)
            y = constant_op.constant(-5.0, dtype=dtypes.float32)
            z = Add2(x, y, name='z')
        gdef = g.as_graph_def()

        @function.Defun()
        def TestFunc():
            if False:
                print('Hello World!')
            return importer.import_graph_def(gdef, return_elements=['z:0'])[0]
        z = TestFunc()
        with self.cached_session():
            z_val = self.evaluate(z)
            self.assertEqual(z_val, -2.0)

    @test_util.run_v1_only('_as_tf_output not supported when eager execution is enabled.')
    def testImportGraphWithFunctionTwice(self):
        if False:
            return 10
        g = ops.Graph()
        with g.as_default():

            @function.Defun()
            def Add2(x, y):
                if False:
                    while True:
                        i = 10
                return math_ops.add(x, y)
            x = array_ops.placeholder(dtype=dtypes.float32, name='x')
            y = array_ops.placeholder(dtype=dtypes.float32, name='y')
            _ = Add2(x, y, name='z')
        gdef = g.as_graph_def()
        x = random_ops.random_uniform(dtype=dtypes.float32, shape=())
        y = random_ops.random_uniform(dtype=dtypes.float32, shape=())
        input_map = {'x:0': x, 'y:0': y}
        with ops.name_scope('first'):
            z1 = importer.import_graph_def(gdef, return_elements=['z:0'], input_map=input_map)[0]
        with ops.name_scope('second'):
            z2 = importer.import_graph_def(gdef, return_elements=['z:0'], input_map=input_map)[0]
        with self.cached_session() as sess:
            (z1_val, z2_val) = sess.run((z1, z2))
            self.assertAllEqual(z1_val, z2_val)
if __name__ == '__main__':
    test.main()