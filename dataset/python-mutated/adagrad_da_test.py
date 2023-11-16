"""Tests for AdagradDA optimizer."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad_da

class AdagradDAOptimizerTest(xla_test.XLATestCase):

    def testAdagradDAWithoutRegularizationBasic1(self):
        if False:
            print('Hello World!')
        for dtype in self.float_types:
            with self.session(), self.test_scope():
                global_step = resource_variable_ops.ResourceVariable(0, dtype=dtypes.int64)
                var0 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
                var1 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
                grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
                grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
                opt = adagrad_da.AdagradDAOptimizer(3.0, global_step, initial_gradient_squared_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
                update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]), global_step=global_step)
                self.evaluate(variables.global_variables_initializer())
                self.assertAllClose([0.0, 0.0], self.evaluate(var0))
                self.assertAllClose([0.0, 0.0], self.evaluate(var1))
                update.run()
                self.assertAllCloseAccordingToType(np.array([-0.904534, -1.603567]), self.evaluate(var0))
                self.assertAllCloseAccordingToType(np.array([-0.094821, -0.189358]), self.evaluate(var1))

    def testAdagradDAwithoutRegularizationBasic2(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.float_types:
            with self.session(), self.test_scope():
                global_step = resource_variable_ops.ResourceVariable(0, dtype=dtypes.int64)
                var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
                var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
                grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
                grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
                opt = adagrad_da.AdagradDAOptimizer(3.0, global_step, initial_gradient_squared_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
                update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]), global_step=global_step)
                self.evaluate(variables.global_variables_initializer())
                self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
                self.assertAllCloseAccordingToType([4.0, 3.0], self.evaluate(var1))
                update.run()
                self.assertAllCloseAccordingToType(np.array([-0.904534, -1.603567]), self.evaluate(var0))
                self.assertAllCloseAccordingToType(np.array([-0.094821, -0.189358]), self.evaluate(var1))

    def testAdagradDAWithL1(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.float_types:
            with self.session(), self.test_scope():
                global_step = resource_variable_ops.ResourceVariable(0, dtype=dtypes.int64)
                var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
                var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
                grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
                grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
                opt = adagrad_da.AdagradDAOptimizer(3.0, global_step, initial_gradient_squared_accumulator_value=0.1, l1_regularization_strength=0.001, l2_regularization_strength=0.0)
                update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]), global_step=global_step)
                self.evaluate(variables.global_variables_initializer())
                self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
                self.assertAllCloseAccordingToType([4.0, 3.0], self.evaluate(var1))
                update.run()
                self.assertAllCloseAccordingToType(np.array([-0.895489, -1.59555]), self.evaluate(var0))
                self.assertAllCloseAccordingToType(np.array([-0.085339, -0.17989]), self.evaluate(var1))

    def testAdagradDAWithL1_L2(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self.float_types:
            with self.session(), self.test_scope():
                global_step = resource_variable_ops.ResourceVariable(0, dtype=dtypes.int64)
                var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
                var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
                grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
                grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
                opt = adagrad_da.AdagradDAOptimizer(3.0, global_step, initial_gradient_squared_accumulator_value=0.1, l1_regularization_strength=0.001, l2_regularization_strength=2.0)
                update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]), global_step=global_step)
                self.evaluate(variables.global_variables_initializer())
                self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
                self.assertAllCloseAccordingToType([4.0, 3.0], self.evaluate(var1))
                update.run()
                self.assertAllCloseAccordingToType(np.array([-0.046907, -0.093659]), self.evaluate(var0))
                self.assertAllCloseAccordingToType(np.array([-0.004275, -0.009023]), self.evaluate(var1))
if __name__ == '__main__':
    test.main()