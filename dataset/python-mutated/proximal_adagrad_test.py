"""Tests for Proximal Adagrad optimizer."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad
from tensorflow.python.training import proximal_adagrad

class ProximalAdagradOptimizerTest(xla_test.XLATestCase):

    def testResourceProximalAdagradwithoutRegularization(self):
        if False:
            i = 10
            return i + 15
        with self.session(), self.test_scope():
            var0 = resource_variable_ops.ResourceVariable([0.0, 0.0])
            var1 = resource_variable_ops.ResourceVariable([0.0, 0.0])
            grads0 = constant_op.constant([0.1, 0.2])
            grads1 = constant_op.constant([0.01, 0.02])
            opt = proximal_adagrad.ProximalAdagradOptimizer(3.0, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
            update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose([0.0, 0.0], self.evaluate(var0))
            self.assertAllClose([0.0, 0.0], self.evaluate(var1))
            for _ in range(3):
                update.run()
            self.assertAllClose(np.array([-2.60260963, -4.29698515]), self.evaluate(var0))
            self.assertAllClose(np.array([-0.28432083, -0.56694895]), self.evaluate(var1))
            opt_vars = opt.variables()
            self.assertStartsWith(opt_vars[0].name, var0._shared_name)
            self.assertStartsWith(opt_vars[1].name, var1._shared_name)
            self.assertEqual(2, len(opt_vars))

    def testProximalAdagradwithoutRegularization2(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session(), self.test_scope():
            var0 = resource_variable_ops.ResourceVariable([1.0, 2.0])
            var1 = resource_variable_ops.ResourceVariable([4.0, 3.0])
            grads0 = constant_op.constant([0.1, 0.2])
            grads1 = constant_op.constant([0.01, 0.02])
            opt = proximal_adagrad.ProximalAdagradOptimizer(3.0, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
            update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose([1.0, 2.0], self.evaluate(var0))
            self.assertAllClose([4.0, 3.0], self.evaluate(var1))
            for _ in range(3):
                update.run()
            self.assertAllClose(np.array([-1.60261, -2.296985]), self.evaluate(var0))
            self.assertAllClose(np.array([3.715679, 2.433051]), self.evaluate(var1))

    def testProximalAdagradWithL1(self):
        if False:
            i = 10
            return i + 15
        with self.session(), self.test_scope():
            var0 = resource_variable_ops.ResourceVariable([1.0, 2.0])
            var1 = resource_variable_ops.ResourceVariable([4.0, 3.0])
            grads0 = constant_op.constant([0.1, 0.2])
            grads1 = constant_op.constant([0.01, 0.02])
            opt = proximal_adagrad.ProximalAdagradOptimizer(3.0, initial_accumulator_value=0.1, l1_regularization_strength=0.001, l2_regularization_strength=0.0)
            update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose([1.0, 2.0], self.evaluate(var0))
            self.assertAllClose([4.0, 3.0], self.evaluate(var1))
            for _ in range(10):
                update.run()
            self.assertAllClose(np.array([-6.663634, -9.190331]), self.evaluate(var0))
            self.assertAllClose(np.array([2.959304, 1.029232]), self.evaluate(var1))

    def testProximalAdagradWithL1_L2(self):
        if False:
            while True:
                i = 10
        with self.session(), self.test_scope():
            var0 = resource_variable_ops.ResourceVariable([1.0, 2.0])
            var1 = resource_variable_ops.ResourceVariable([4.0, 3.0])
            grads0 = constant_op.constant([0.1, 0.2])
            grads1 = constant_op.constant([0.01, 0.02])
            opt = proximal_adagrad.ProximalAdagradOptimizer(3.0, initial_accumulator_value=0.1, l1_regularization_strength=0.001, l2_regularization_strength=2.0)
            update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose([1.0, 2.0], self.evaluate(var0))
            self.assertAllClose([4.0, 3.0], self.evaluate(var1))
            for _ in range(10):
                update.run()
            self.assertAllClose(np.array([-0.0495, -0.0995]), self.evaluate(var0))
            self.assertAllClose(np.array([-0.0045, -0.0095]), self.evaluate(var1))

    def applyOptimizer(self, opt, steps=5):
        if False:
            print('Hello World!')
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0])
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0])
        grads0 = constant_op.constant([0.1, 0.2])
        grads1 = constant_op.constant([0.01, 0.02])
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        for _ in range(steps):
            update.run()
        return (self.evaluate(var0), self.evaluate(var1))

    def testEquivAdagradwithoutRegularization(self):
        if False:
            print('Hello World!')
        with self.session(), self.test_scope():
            (val0, val1) = self.applyOptimizer(proximal_adagrad.ProximalAdagradOptimizer(3.0, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0))
        with self.session(), self.test_scope():
            (val2, val3) = self.applyOptimizer(adagrad.AdagradOptimizer(3.0, initial_accumulator_value=0.1))
        self.assertAllClose(val0, val2)
        self.assertAllClose(val1, val3)
if __name__ == '__main__':
    test.main()