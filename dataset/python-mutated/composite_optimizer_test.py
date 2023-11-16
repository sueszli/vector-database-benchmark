"""Tests for CompositeOptimizer."""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging
from dragnn.python import composite_optimizer

class MockAdamOptimizer(tf.train.AdamOptimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam'):
        if False:
            return 10
        super(MockAdamOptimizer, self).__init__(learning_rate, beta1, beta2, epsilon, use_locking, name)

    def _create_slots(self, var_list):
        if False:
            print('Hello World!')
        super(MockAdamOptimizer, self)._create_slots(var_list)
        for v in var_list:
            self._zeros_slot(v, 'adam_counter', self._name)

    def _apply_dense(self, grad, var):
        if False:
            print('Hello World!')
        train_op = super(MockAdamOptimizer, self)._apply_dense(grad, var)
        counter = self.get_slot(var, 'adam_counter')
        return tf.group(train_op, tf.assign_add(counter, [1.0]))

class MockMomentumOptimizer(tf.train.MomentumOptimizer):

    def __init__(self, learning_rate, momentum, use_locking=False, name='Momentum', use_nesterov=False):
        if False:
            print('Hello World!')
        super(MockMomentumOptimizer, self).__init__(learning_rate, momentum, use_locking, name, use_nesterov)

    def _create_slots(self, var_list):
        if False:
            while True:
                i = 10
        super(MockMomentumOptimizer, self)._create_slots(var_list)
        for v in var_list:
            self._zeros_slot(v, 'momentum_counter', self._name)

    def _apply_dense(self, grad, var):
        if False:
            for i in range(10):
                print('nop')
        train_op = super(MockMomentumOptimizer, self)._apply_dense(grad, var)
        counter = self.get_slot(var, 'momentum_counter')
        return tf.group(train_op, tf.assign_add(counter, [1.0]))

class CompositeOptimizerTest(test_util.TensorFlowTestCase):

    def test_switching(self):
        if False:
            return 10
        with self.test_session() as sess:
            x_data = np.random.rand(100).astype(np.float32)
            y_data = x_data * 0.1 + 0.3
            w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
            b = tf.Variable(tf.zeros([1]))
            y = w * x_data + b
            loss = tf.reduce_mean(tf.square(y - y_data))
            step = tf.get_variable('step', shape=[], initializer=tf.zeros_initializer(), trainable=False, dtype=tf.int32)
            optimizer1 = MockAdamOptimizer(0.05)
            optimizer2 = MockMomentumOptimizer(0.05, 0.5)
            switch = tf.less(step, 100)
            optimizer = composite_optimizer.CompositeOptimizer(optimizer1, optimizer2, switch)
            train_op = optimizer.minimize(loss)
            sess.run(tf.global_variables_initializer())
            for iteration in range(201):
                self.assertEqual(sess.run(switch), iteration < 100)
                sess.run(train_op)
                sess.run(tf.assign_add(step, 1))
                slot_names = optimizer.get_slot_names()
                adam_slots = ['c1-m', 'c1-v', 'c1-adam_counter']
                momentum_slots = ['c2-momentum', 'c2-momentum_counter']
                self.assertItemsEqual(slot_names, adam_slots + momentum_slots)
                adam_counter = sess.run(optimizer.get_slot(w, 'c1-adam_counter'))
                momentum_counter = sess.run(optimizer.get_slot(w, 'c2-momentum_counter'))
                self.assertEqual(adam_counter, min(iteration + 1, 100))
                self.assertEqual(momentum_counter, max(iteration - 99, 0))
                if iteration % 20 == 0:
                    logging.info('%d %s %d %d', iteration, sess.run([switch, step, w, b]), adam_counter, momentum_counter)
if __name__ == '__main__':
    googletest.main()