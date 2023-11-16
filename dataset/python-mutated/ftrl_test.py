"""Tests for Ftrl optimizer."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad
from tensorflow.python.training import ftrl
from tensorflow.python.training import gradient_descent

class FtrlOptimizerTest(xla_test.XLATestCase):

    def initVariableAndGradient(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        var0 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.02, 0.04], dtype=dtype)
        return (var0, var1, grads0, grads1)

    def equivAdagradTest_FtrlPart(self, steps, dtype):
        if False:
            print('Hello World!')
        (var0, var1, grads0, grads1) = self.initVariableAndGradient(dtype)
        opt = ftrl.FtrlOptimizer(3.0, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
        ftrl_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose([0.0, 0.0], self.evaluate(var0))
        self.assertAllClose([0.0, 0.0], self.evaluate(var1))
        for _ in range(steps):
            ftrl_update.run()
        return (self.evaluate(var0), self.evaluate(var1))

    def equivAdagradTest_AdagradPart(self, steps, dtype):
        if False:
            while True:
                i = 10
        (var0, var1, grads0, grads1) = self.initVariableAndGradient(dtype)
        opt = adagrad.AdagradOptimizer(3.0, initial_accumulator_value=0.1)
        adagrad_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose([0.0, 0.0], self.evaluate(var0))
        self.assertAllClose([0.0, 0.0], self.evaluate(var1))
        for _ in range(steps):
            adagrad_update.run()
        return (self.evaluate(var0), self.evaluate(var1))

    def equivGradientDescentTest_FtrlPart(self, steps, dtype):
        if False:
            i = 10
            return i + 15
        (var0, var1, grads0, grads1) = self.initVariableAndGradient(dtype)
        opt = ftrl.FtrlOptimizer(3.0, learning_rate_power=-0.0, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
        ftrl_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose([0.0, 0.0], self.evaluate(var0))
        self.assertAllClose([0.0, 0.0], self.evaluate(var1))
        for _ in range(steps):
            ftrl_update.run()
        return (self.evaluate(var0), self.evaluate(var1))

    def equivGradientDescentTest_GradientDescentPart(self, steps, dtype):
        if False:
            for i in range(10):
                print('nop')
        (var0, var1, grads0, grads1) = self.initVariableAndGradient(dtype)
        opt = gradient_descent.GradientDescentOptimizer(3.0, name='sgd')
        sgd_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        self.assertAllClose([0.0, 0.0], self.evaluate(var0))
        self.assertAllClose([0.0, 0.0], self.evaluate(var1))
        for _ in range(steps):
            sgd_update.run()
        return (self.evaluate(var0), self.evaluate(var1))

    def testFtrlwithoutRegularization(self):
        if False:
            while True:
                i = 10
        for dtype in self.float_types:
            with self.session(), self.test_scope():
                var0 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
                var1 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
                grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
                grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
                opt = ftrl.FtrlOptimizer(3.0, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
                ftrl_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                self.evaluate(variables.global_variables_initializer())
                self.assertAllClose([0.0, 0.0], self.evaluate(var0))
                self.assertAllClose([0.0, 0.0], self.evaluate(var1))
                for _ in range(3):
                    ftrl_update.run()
                self.assertAllCloseAccordingToType(np.array([-2.60260963, -4.29698515]), self.evaluate(var0), float_rtol=0.0001, half_rtol=0.01)
                self.assertAllCloseAccordingToType(np.array([-0.28432083, -0.56694895]), self.evaluate(var1), float_rtol=1e-05, half_rtol=0.01)

    def testFtrlwithoutRegularization2(self):
        if False:
            return 10
        for dtype in self.float_types:
            with self.session(), self.test_scope():
                var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
                var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
                grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
                grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
                opt = ftrl.FtrlOptimizer(3.0, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0)
                ftrl_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                self.evaluate(variables.global_variables_initializer())
                self.assertAllClose([1.0, 2.0], self.evaluate(var0))
                self.assertAllClose([4.0, 3.0], self.evaluate(var1))
                for _ in range(3):
                    ftrl_update.run()
                self.assertAllCloseAccordingToType(np.array([-2.55607247, -3.98729396]), self.evaluate(var0), 1e-05, 1e-05, float_rtol=0.0001)
                self.assertAllCloseAccordingToType(np.array([-0.28232238, -0.56096673]), self.evaluate(var1), 1e-05, 1e-05)

    def testFtrlWithL1(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self.float_types:
            with self.session(), self.test_scope():
                var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
                var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
                grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
                grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
                opt = ftrl.FtrlOptimizer(3.0, initial_accumulator_value=0.1, l1_regularization_strength=0.001, l2_regularization_strength=0.0)
                ftrl_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                self.evaluate(variables.global_variables_initializer())
                self.assertAllClose([1.0, 2.0], self.evaluate(var0))
                self.assertAllClose([4.0, 3.0], self.evaluate(var1))
                for _ in range(10):
                    ftrl_update.run()
                self.assertAllCloseAccordingToType(np.array([-7.66718769, -10.91273689]), self.evaluate(var0), rtol=0.0001, bfloat16_rtol=0.1, bfloat16_atol=0.1)
                self.assertAllCloseAccordingToType(np.array([-0.93460727, -1.86147261]), self.evaluate(var1), rtol=0.0001)

    def testFtrlWithL1_L2(self):
        if False:
            while True:
                i = 10
        for dtype in self.float_types:
            with self.session(), self.test_scope():
                var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
                var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
                grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
                grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
                opt = ftrl.FtrlOptimizer(3.0, initial_accumulator_value=0.1, l1_regularization_strength=0.001, l2_regularization_strength=2.0)
                ftrl_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                self.evaluate(variables.global_variables_initializer())
                self.assertAllClose([1.0, 2.0], self.evaluate(var0))
                self.assertAllClose([4.0, 3.0], self.evaluate(var1))
                for _ in range(10):
                    ftrl_update.run()
                self.assertAllCloseAccordingToType(np.array([-0.24059935, -0.46829352]), self.evaluate(var0), rtol=1e-05)
                self.assertAllCloseAccordingToType(np.array([-0.02406147, -0.04830509]), self.evaluate(var1), rtol=1e-05)

    def testFtrlWithL1_L2_L2Shrinkage(self):
        if False:
            i = 10
            return i + 15
        'Test the new FTRL op with support for l2 shrinkage.\n\n    The addition of this parameter which places a constant pressure on weights\n    towards the origin causes the gradient descent trajectory to differ. The\n    weights will tend to have smaller magnitudes with this parameter set.\n    '
        for dtype in self.float_types:
            with self.session(), self.test_scope():
                var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
                var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
                grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
                grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
                opt = ftrl.FtrlOptimizer(3.0, initial_accumulator_value=0.1, l1_regularization_strength=0.001, l2_regularization_strength=2.0, l2_shrinkage_regularization_strength=0.1)
                ftrl_update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                self.evaluate(variables.global_variables_initializer())
                self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
                self.assertAllCloseAccordingToType([4.0, 3.0], self.evaluate(var1))
                for _ in range(10):
                    ftrl_update.run()
                self.assertAllCloseAccordingToType(np.array([-0.22578996, -0.44345799]), self.evaluate(var0), rtol=0.0001)
                self.assertAllCloseAccordingToType(np.array([-0.14378493, -0.13229476]), self.evaluate(var1), rtol=0.0001)

    def testFtrlWithL2ShrinkageDoesNotChangeLrSchedule(self):
        if False:
            for i in range(10):
                print('nop')
        'Verifies that l2 shrinkage in FTRL does not change lr schedule.'
        for dtype in self.float_types:
            with self.session(), self.test_scope():
                var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
                var1 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
                grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
                grads1 = constant_op.constant([0.1, 0.2], dtype=dtype)
                opt0 = ftrl.FtrlOptimizer(3.0, initial_accumulator_value=0.1, l1_regularization_strength=0.001, l2_regularization_strength=2.0, l2_shrinkage_regularization_strength=0.1)
                opt1 = ftrl.FtrlOptimizer(3.0, initial_accumulator_value=0.1, l1_regularization_strength=0.001, l2_regularization_strength=2.0)
                update0 = opt0.apply_gradients([(grads0, var0)])
                update1 = opt1.apply_gradients([(grads1, var1)])
                self.evaluate(variables.global_variables_initializer())
                self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
                self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var1))
                for _ in range(10):
                    update0.run()
                    update1.run()
                self.assertTrue((var0.eval() ** 2 < self.evaluate(var1) ** 2).all())
                accum0 = list(opt0._slots['accum'].values())[0].eval()
                accum1 = list(opt1._slots['accum'].values())[0].eval()
                self.assertAllCloseAccordingToType(accum0, accum1)

    def testEquivAdagradwithoutRegularization(self):
        if False:
            for i in range(10):
                print('nop')
        steps = 5
        for dtype in self.float_types:
            with self.session(), self.test_scope():
                (val0, val1) = self.equivAdagradTest_FtrlPart(steps, dtype)
            with self.session(), self.test_scope():
                (val2, val3) = self.equivAdagradTest_AdagradPart(steps, dtype)
        self.assertAllCloseAccordingToType(val0, val2, rtol=0.0001, half_rtol=0.01)
        self.assertAllCloseAccordingToType(val1, val3, rtol=0.0001, half_rtol=0.01)

    def testEquivGradientDescentwithoutRegularization(self):
        if False:
            for i in range(10):
                print('nop')
        steps = 5
        for dtype in self.float_types:
            with self.session(), self.test_scope():
                (val0, val1) = self.equivGradientDescentTest_FtrlPart(steps, dtype)
            with self.session(), self.test_scope():
                (val2, val3) = self.equivGradientDescentTest_GradientDescentPart(steps, dtype)
        self.assertAllCloseAccordingToType(val0, val2, rtol=1e-05)
        self.assertAllCloseAccordingToType(val1, val3, rtol=1e-05)
if __name__ == '__main__':
    test.main()