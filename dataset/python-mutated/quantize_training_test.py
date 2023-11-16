"""Tests for the SWIG-wrapped quantize training rewriting."""
import os
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import quantize_training
from tensorflow.python.training import saver as saver_module

class PywrapQuantizeTrainingTest(test.TestCase):

    def testQuantizeTraining(self):
        if False:
            for i in range(10):
                print('nop')
        with session.Session() as sess:
            a = constant_op.constant(6.0, shape=[1, 1])
            b = constant_op.constant(7.0, shape=[1, 1])
            c = math_ops.matmul(a, b, name='matmul')
            self.assertEqual(c.eval(), 42.0)
            self.assertEqual(len(sess.graph_def.node), 3)
            result = quantize_training.do_quantize_training_on_graphdef(sess.graph_def, 8)
            self.assertGreater(len(result.node), 3)

    @test_util.run_v1_only('The API is only expect to work with v1 session mode.')
    def testQuantizedSaveRestore(self):
        if False:
            for i in range(10):
                print('nop')
        save_path = os.path.join(self.get_temp_dir(), 'quantized_save_restore')
        g = ops.Graph()
        with session.Session(graph=g) as sess:
            a = constant_op.constant(6.0, shape=[1, 1], name='a')
            b = variable_v1.VariableV1(constant_op.constant(7.0, shape=[1, 1]), name='b')
            c = math_ops.matmul(a, b, name='matmul')
            init_op = variables.global_variables_initializer()
            saver = saver_module.Saver({'b': b})
            result = quantize_training.do_quantize_training_on_graphdef(sess.graph_def, 8)
        with ops.Graph().as_default() as g, session.Session(graph=g) as sess:
            _ = importer.import_graph_def(result, name='')
            self.evaluate(g.get_operation_by_name(init_op.name))
            self.evaluate(g.get_tensor_by_name(c.name))
            saver.save(sess, save_path)
        with ops.Graph().as_default() as g, session.Session(graph=g) as sess:
            _ = importer.import_graph_def(result, name='')
            saver.restore(sess, save_path)
            self.assertEqual(7.0, sess.run(g.get_tensor_by_name('b:0')))
            self.assertEqual(6.0, sess.run(g.get_tensor_by_name('a/Min/Variable:0')))
            self.assertEqual(6.0, sess.run(g.get_tensor_by_name('a/Max/Variable:0')))
            self.assertEqual(7.0, sess.run(g.get_tensor_by_name('b/read/Min/Variable:0')))
            self.assertEqual(7.0, sess.run(g.get_tensor_by_name('b/read/Max/Variable:0')))
if __name__ == '__main__':
    test.main()