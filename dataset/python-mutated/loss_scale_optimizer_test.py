"""Tests for MixedPrecisionLossScaleOptimizer."""
import os
from absl.testing import parameterized
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import momentum
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import loss_scale_optimizer
default_strategy_fn = distribute_lib.get_strategy

def create_mirrored_strategy():
    if False:
        print('Hello World!')
    if context.num_gpus() >= 1:
        return mirrored_strategy.MirroredStrategy(['cpu:0', 'gpu:0'])
    else:
        return mirrored_strategy.MirroredStrategy(['cpu:0'])
TESTCASES = ({'testcase_name': 'Base', 'strategy_fn': default_strategy_fn}, {'testcase_name': 'Distribute', 'strategy_fn': create_mirrored_strategy})

def get_gradients(opt, loss, params):
    if False:
        for i in range(10):
            print('nop')
    grads_and_vars = opt.compute_gradients(loss, params)
    (grads, _) = zip(*grads_and_vars)
    return grads

def create_identity_with_grad_check_fn(expected_gradient, expected_dtype=None):
    if False:
        i = 10
        return i + 15
    "Returns a function that asserts it's gradient has a certain value.\n\n  This serves as a hook to assert intermediate gradients have a certain value.\n  This returns an identity function. The identity's gradient function is also\n  the identity function, except it asserts that the gradient equals\n  `expected_gradient` and has dtype `expected_dtype`.\n\n  Args:\n    expected_gradient: The gradient function asserts that the gradient is this\n      value.\n    expected_dtype: The gradient function asserts the gradient has this dtype.\n\n  Returns:\n    An identity function whose gradient function asserts the gradient has a\n    certain value.\n  "

    @custom_gradient.custom_gradient
    def _identity_with_grad_check(x):
        if False:
            while True:
                i = 10
        "Function that asserts it's gradient has a certain value."
        x = array_ops.identity(x)

        def grad(dx):
            if False:
                print('Hello World!')
            'Gradient function that asserts the gradient has a certain value.'
            if expected_dtype:
                assert dx.dtype == expected_dtype, 'dx.dtype should be %s but is: %s' % (expected_dtype, dx.dtype)
            expected_tensor = tensor_conversion.convert_to_tensor_v2(expected_gradient, dtype=dx.dtype, name='expected_gradient')
            with ops.control_dependencies([x]):
                assert_op = check_ops.assert_equal(dx, expected_tensor)
            with ops.control_dependencies([assert_op]):
                dx = array_ops.identity(dx)
            return dx
        return (x, grad)

    def identity_with_grad_check(x):
        if False:
            return 10
        return _identity_with_grad_check(x)
    return identity_with_grad_check

class MixedPrecisionLossScaleOptimizerTest(test.TestCase, parameterized.TestCase):

    def _run_if_in_graph_mode(self, val):
        if False:
            return 10
        if not context.executing_eagerly():
            self.evaluate(val)

    def _run_fn_with_grad_check(self, strategy, var, opt, expected_grad):
        if False:
            i = 10
            return i + 15
        grad_check_fn = create_identity_with_grad_check_fn(expected_grad)
        loss = lambda : grad_check_fn(var) / strategy.num_replicas_in_sync
        return lambda : opt.minimize(loss, var_list=[var])

    @parameterized.named_parameters(*TESTCASES)
    @test_util.run_in_graph_and_eager_modes
    def testFixedLossScaleAppliedToLossWithMinimize(self, strategy_fn):
        if False:
            i = 10
            return i + 15
        with strategy_fn().scope() as strategy:
            var = variables.Variable([5.0])
            opt = gradient_descent.GradientDescentOptimizer(2.0)
            loss_scale = 10.0
            opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(opt, loss_scale)
            self.assertEqual(loss_scale % strategy.num_replicas_in_sync, 0)
            run_fn = self._run_fn_with_grad_check(strategy, var, opt, loss_scale / strategy.num_replicas_in_sync)
            run_op = strategy.experimental_run(run_fn)
            self.evaluate(variables.global_variables_initializer())
            self._run_if_in_graph_mode(run_op)
            self.assertAllClose([3.0], self.evaluate(var))

    @test_util.deprecated_graph_mode_only
    def testFixedLossScaleAppliedToLossWithGetGradients(self):
        if False:
            print('Hello World!')
        var = variables.Variable([2.0])
        opt = gradient_descent.GradientDescentOptimizer(1.0)
        loss_scale = 10.0
        opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(opt, loss_scale)
        grad_check_fn = create_identity_with_grad_check_fn(loss_scale)
        loss = grad_check_fn(var)
        run_op = get_gradients(opt, loss, [var])
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(run_op)

    @parameterized.named_parameters(*TESTCASES)
    @test_util.run_in_graph_and_eager_modes
    def testDynamicLossScale(self, strategy_fn):
        if False:
            return 10
        strategy = strategy_fn()
        learning_rate = 2.0
        expected_gradient = resource_variable_ops.ResourceVariable(learning_rate / strategy.num_replicas_in_sync)
        with strategy.scope():
            var = variables.Variable([5.0])
            opt = gradient_descent.GradientDescentOptimizer(learning_rate)
            loss_scale = loss_scale_module.DynamicLossScale(initial_loss_scale=2, increment_period=1, multiplier=2)
            opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(opt, loss_scale)
            self.assertEqual(loss_scale.initial_loss_scale % strategy.num_replicas_in_sync, 0)
            run_fn = self._run_fn_with_grad_check(strategy, var, opt, expected_gradient)
            run_op = strategy.experimental_run(run_fn)
            self.evaluate(variables.global_variables_initializer())
            self._run_if_in_graph_mode(run_op)
            self.assertAllClose([3.0], self.evaluate(var))
            self.evaluate(expected_gradient.assign(2 * learning_rate / strategy.num_replicas_in_sync))
            run_op = strategy.experimental_run(run_fn)
            self._run_if_in_graph_mode(run_op)
            self.assertAllClose([1.0], self.evaluate(var))

    @parameterized.named_parameters(*TESTCASES)
    @test_util.run_in_graph_and_eager_modes
    def testDynamicUpdate(self, strategy_fn):
        if False:
            while True:
                i = 10
        with strategy_fn().scope() as strategy:
            var = variables.Variable([1.0, 2.0])
            opt = gradient_descent.GradientDescentOptimizer(1.0)
            loss_scale = loss_scale_module.DynamicLossScale(initial_loss_scale=2, increment_period=1, multiplier=2)
            opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(opt, loss_scale)
            loss = lambda : var * 2.0 / strategy.num_replicas_in_sync
            run_fn = lambda : opt.minimize(loss, var_list=[var])
            run_op = strategy.experimental_run(run_fn)
            self.evaluate(variables.global_variables_initializer())
            self._run_if_in_graph_mode(run_op)
            self.assertAllClose([-1.0, 0.0], self.evaluate(var))
            self.assertEqual(4.0, self.evaluate(opt._loss_scale()))
            loss = lambda : var * float('NaN')
            run_fn = lambda : opt.minimize(loss, var_list=[var])
            run_op = strategy.experimental_run(run_fn)
            self._run_if_in_graph_mode(run_op)
            self.assertAllClose(self.evaluate(var), [-1.0, 0.0])
            self.assertEqual(2.0, self.evaluate(opt._loss_scale()))

    @parameterized.named_parameters(*TESTCASES)
    @test_util.run_in_graph_and_eager_modes
    def testDynamicLossScaleWithSlots(self, strategy_fn):
        if False:
            for i in range(10):
                print('nop')
        with strategy_fn().scope() as strategy:
            var = variables.Variable([1.0, 2.0])
            opt = momentum.MomentumOptimizer(1.0, momentum=1.0)
            initial_loss_scale = 2.0
            loss_scale = loss_scale_module.DynamicLossScale(initial_loss_scale=initial_loss_scale, increment_period=1, multiplier=4)
            opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(opt, loss_scale)
            loss = lambda : var / strategy.num_replicas_in_sync
            run_fn = lambda : opt.minimize(loss, var_list=[var])
            run_op = strategy.experimental_run(run_fn)
            self.evaluate(variables.global_variables_initializer())
            self._run_if_in_graph_mode(run_op)
            self.assertAllClose([0.0, 1.0], self.evaluate(var))
            self.assertEqual(self.evaluate(opt._loss_scale()), initial_loss_scale * 4)
            run_op = strategy.experimental_run(run_fn)
            self._run_if_in_graph_mode(run_op)
            self.assertAllClose([-2.0, -1.0], self.evaluate(var))
            self.assertEqual(self.evaluate(opt._loss_scale()), initial_loss_scale * 16)

    @parameterized.named_parameters(*TESTCASES)
    @test_util.run_in_graph_and_eager_modes
    def testCheckpoint(self, strategy_fn):
        if False:
            while True:
                i = 10
        strategy = strategy_fn()
        if isinstance(strategy, mirrored_strategy.MirroredStrategy) and (not context.executing_eagerly()):
            return
        with self.test_session(), strategy.scope():
            var = variables.Variable([2.0])
            loss_scale = loss_scale_module.DynamicLossScale(initial_loss_scale=1.0, increment_period=2.0, multiplier=2.0)
            opt = momentum.MomentumOptimizer(1.0, momentum=1.0)
            opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(opt, loss_scale)
            run_fn = lambda : opt.minimize(lambda : var + 1.0, var_list=[var])
            opt_op = strategy.experimental_run(run_fn)
            self.evaluate(variables.global_variables_initializer())
            self.evaluate(opt_op)
            self.assertEqual(self.evaluate(loss_scale()), 1.0)
            self.assertEqual(self.evaluate(loss_scale._num_good_steps), 1)
            checkpoint = trackable_utils.Checkpoint(optimizer=opt)
            prefix = os.path.join(self.get_temp_dir(), 'ckpt')
            save_path = checkpoint.save(prefix)
            self.evaluate(strategy.experimental_run(run_fn))
            self.assertEqual(self.evaluate(loss_scale()), 2.0)
            self.assertEqual(self.evaluate(loss_scale._num_good_steps), 0)
            status = checkpoint.restore(save_path)
            status.assert_consumed()
            status.run_restore_ops()
            self.assertEqual(self.evaluate(loss_scale()), 1.0)
            self.assertEqual(self.evaluate(loss_scale._num_good_steps), 1)

    def testPassingNoneToLossScale(self):
        if False:
            for i in range(10):
                print('nop')
        opt = gradient_descent.GradientDescentOptimizer(1.0)
        with self.assertRaisesRegex(ValueError, 'loss_scale cannot be None'):
            loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(opt, None)
if __name__ == '__main__':
    test.main()