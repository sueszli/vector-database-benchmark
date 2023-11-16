import os
from absl.testing import parameterized
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import gradient_descent as gradient_descent_v1
from tensorflow.python.training.experimental import loss_scale_optimizer as loss_scale_optimizer_v1
from tensorflow.python.training.experimental import mixed_precision
from tensorflow.python.training.experimental import mixed_precision_global_state

class MixedPrecisionTest(test.TestCase, parameterized.TestCase):
    IGNORE_PERF_VAR = 'TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(MixedPrecisionTest, self).setUp()
        self._original_ignore_perf_value = os.getenv(self.IGNORE_PERF_VAR)
        os.environ[self.IGNORE_PERF_VAR] = '1'

    def tearDown(self):
        if False:
            return 10
        if self._original_ignore_perf_value is not None:
            os.environ[self.IGNORE_PERF_VAR] = self._original_ignore_perf_value
        else:
            del os.environ[self.IGNORE_PERF_VAR]
        mixed_precision.disable_mixed_precision_graph_rewrite_v1()
        super(MixedPrecisionTest, self).tearDown()

    @test_util.run_in_graph_and_eager_modes
    def test_wrap_optimizer(self):
        if False:
            return 10
        opt = gradient_descent_v1.GradientDescentOptimizer(1.0)
        opt = mixed_precision.enable_mixed_precision_graph_rewrite_v1(opt, 123.0)
        self.assertIsInstance(opt, loss_scale_optimizer_v1.MixedPrecisionLossScaleOptimizer)
        self.assertEqual(self.evaluate(opt._loss_scale()), 123.0)

    @test_util.run_in_graph_and_eager_modes
    def test_optimizer_errors(self):
        if False:
            for i in range(10):
                print('nop')
        opt = 1
        expected_regex = '"opt" must be an instance of a tf.train.Optimizer or a tf.keras.optimizers.Optimizer, but got'
        with self.assertRaisesRegex(ValueError, expected_regex):
            mixed_precision.enable_mixed_precision_graph_rewrite_v1(opt)
        self.assertFalse(config.get_optimizer_experimental_options().get('auto_mixed_precision', False))
        opt = gradient_descent_v1.GradientDescentOptimizer(1.0)
        opt = loss_scale_optimizer_v1.MixedPrecisionLossScaleOptimizer(opt, 'dynamic')
        with self.assertRaisesRegex(ValueError, '"opt" must not already be an instance of a MixedPrecisionLossScaleOptimizer.'):
            mixed_precision.enable_mixed_precision_graph_rewrite_v1(opt)
        self.assertFalse(config.get_optimizer_experimental_options().get('auto_mixed_precision', False))

    @test_util.run_in_graph_and_eager_modes()
    def test_register_loss_scale_wrapper_with_2_arguments(self):
        if False:
            print('Hello World!')

        class MyOptimizer:
            pass

        class MyLossScaleOptimizer(MyOptimizer):

            def __init__(self, inner_optimizer, loss_scale):
                if False:
                    for i in range(10):
                        print('nop')
                self.inner_optimizer = inner_optimizer
                self.loss_scale = loss_scale
        mixed_precision.register_loss_scale_wrapper(MyOptimizer, MyLossScaleOptimizer)
        opt = MyOptimizer()
        opt = mixed_precision.enable_mixed_precision_graph_rewrite_v1(opt, 123.0)
        self.assertIsInstance(opt, MyLossScaleOptimizer)
        self.assertEqual(opt.loss_scale, 123.0)

    @test_util.run_in_graph_and_eager_modes()
    def test_register_loss_scale_wrapper_with_3_arguments(self):
        if False:
            i = 10
            return i + 15

        class MyOptimizer:
            pass

        class MyLossScaleOptimizer(MyOptimizer):

            def __init__(self, inner_optimizer, loss_scale):
                if False:
                    while True:
                        i = 10
                self.inner_optimizer = inner_optimizer
                self.loss_scale = loss_scale
        is_called = False

        def create_lso(inner_optimizer, loss_scale):
            if False:
                print('Hello World!')
            nonlocal is_called
            is_called = True
            return MyLossScaleOptimizer(inner_optimizer, loss_scale)
        mixed_precision.register_loss_scale_wrapper(MyOptimizer, create_lso, MyLossScaleOptimizer)
        opt = MyOptimizer()
        opt = mixed_precision.enable_mixed_precision_graph_rewrite_v1(opt, 123.0)
        self.assertIsInstance(opt, MyLossScaleOptimizer)
        self.assertEqual(opt.loss_scale, 123.0)
        self.assertTrue(is_called)

    @test_util.run_gpu_only
    @test_util.run_in_graph_and_eager_modes
    @test_util.disable_tfrt("Grappler rewrite doesn't apply to tfrt.")
    def test_grappler_pass_enabled(self):
        if False:
            while True:
                i = 10
        opt = gradient_descent_v1.GradientDescentOptimizer(1.0)
        mixed_precision.enable_mixed_precision_graph_rewrite_v1(opt, 123.0)
        var = variables.Variable([[1.0]])

        def overflow_in_float16():
            if False:
                print('Hello World!')
            out = var * 2 ** 10
            out = math_ops.matmul(out, out)
            return array_ops.reshape(out, ())
        if context.executing_eagerly():
            f = def_function.function(overflow_in_float16)
            self.assertEqual(f().numpy(), float('Inf'))
            self.assertAlmostEqual(overflow_in_float16().numpy(), 2 ** 20)
            mixed_precision.disable_mixed_precision_graph_rewrite_v1()
            self.assertEqual(f().numpy(), 2 ** 20)
        else:
            with session.Session() as sess:
                out = overflow_in_float16()
                sess.run(var.initializer)
                self.assertEqual(sess.run(out), float('Inf'))
            with session.Session(config=config_pb2.ConfigProto()) as sess:
                out = overflow_in_float16()
                sess.run(var.initializer)
                self.assertEqual(sess.run(out), float('Inf'))
            mixed_precision.disable_mixed_precision_graph_rewrite_v1()
            with session.Session() as sess:
                out = overflow_in_float16()
                sess.run(var.initializer)
                self.assertAlmostEqual(sess.run(out), 2 ** 20)

    @test.mock.patch.object(tf_logging, 'warn')
    def test_warn_if_session_already_exists(self, mock_warn):
        if False:
            i = 10
            return i + 15
        mixed_precision_global_state.set_non_mixed_precision_session_created(False)
        with session.Session():
            mixed_precision.enable_mixed_precision_graph_rewrite_v1(gradient_descent_v1.GradientDescentOptimizer(1.0))
            mock_warn.assert_any_call('You already have existing Sessions that do not use mixed precision. enable_mixed_precision_graph_rewrite() will not affect these Sessions.')

    @test.mock.patch.object(tf_logging, 'warn')
    def test_do_not_warn_if_session_does_not_already_exist(self, mock_warn):
        if False:
            i = 10
            return i + 15
        mixed_precision_global_state.set_non_mixed_precision_session_created(False)
        mixed_precision.enable_mixed_precision_graph_rewrite_v1(gradient_descent_v1.GradientDescentOptimizer(1.0))
        with session.Session():
            for call_arg in mock_warn.call_args_list:
                msg = call_arg[0][0]
                self.assertNotIn('You already have existing Sessions that do not use mixed precision', msg)
if __name__ == '__main__':
    test.main()