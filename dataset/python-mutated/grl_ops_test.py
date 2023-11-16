"""Tests for grl_ops."""
import tensorflow as tf
import grl_op_grads
import grl_ops
FLAGS = tf.app.flags.FLAGS

class GRLOpsTest(tf.test.TestCase):

    def testGradientReversalOp(self):
        if False:
            while True:
                i = 10
        with tf.Graph().as_default():
            with self.test_session():
                examples = tf.constant([5.0, 4.0, 3.0, 2.0, 1.0])
                output = grl_ops.gradient_reversal(examples)
                expected_output = examples
                self.assertAllEqual(output.eval(), expected_output.eval())
                self.assertAllEqual(output.get_shape(), expected_output.get_shape())
                examples = tf.constant([[1.0]])
                w = tf.get_variable(name='w', shape=[1, 1])
                b = tf.get_variable(name='b', shape=[1])
                init_op = tf.global_variables_initializer()
                init_op.run()
                features = tf.nn.xw_plus_b(examples, w, b)
                output1 = features
                output2 = grl_ops.gradient_reversal(features)
                gold = tf.constant([1.0])
                loss1 = gold - output1
                loss2 = gold - output2
                opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
                grads_and_vars_1 = opt.compute_gradients(loss1, tf.trainable_variables())
                grads_and_vars_2 = opt.compute_gradients(loss2, tf.trainable_variables())
                self.assertAllEqual(len(grads_and_vars_1), len(grads_and_vars_2))
                for i in range(len(grads_and_vars_1)):
                    g1 = grads_and_vars_1[i][0]
                    g2 = grads_and_vars_2[i][0]
                    self.assertAllEqual(tf.negative(g1).eval(), g2.eval())
if __name__ == '__main__':
    tf.test.main()