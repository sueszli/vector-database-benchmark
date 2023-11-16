"""Functional tests for fused batch norm operations."""
from absl.testing import parameterized
import numpy as np
from tensorflow.compiler.tests import test_utils
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn
from tensorflow.python.platform import test
DATA_FORMATS = (('_data_format_NHWC', 'NHWC'), ('_data_format_NCHW', 'NCHW'))
DATA_FORMATS_AND_AVG_FACTORS = (('_data_format_NHWC_no_averaging', 'NHWC', 1.0), ('_data_format_NHWC_averaging', 'NHWC', 0.6), ('_data_format_NCHW_no_averaging', 'NCHW', 1.0), ('_data_format_NCHW_averaging', 'NCHW', 0.6))

class FusedBatchNormTest(xla_test.XLATestCase, parameterized.TestCase):

    def _reference_training(self, x, scale, offset, old_mean, old_var, epsilon, exponential_avg_factor, data_format):
        if False:
            while True:
                i = 10
        if data_format != 'NHWC':
            raise ValueError('data_format must be NHWC, got %s.' % data_format)
        x_square = x * x
        x_square_sum = np.sum(x_square, (0, 1, 2))
        x_sum = np.sum(x, axis=(0, 1, 2))
        element_count = np.size(x) / int(np.shape(x)[-1])
        mean = x_sum / element_count
        var = x_square_sum / element_count - mean * mean
        factor = element_count / max(element_count - 1, 1)
        corrected_var = var * factor
        normalized = (x - mean) / np.sqrt(var + epsilon)
        if exponential_avg_factor != 1.0:
            mean = (1.0 - exponential_avg_factor) * old_mean + exponential_avg_factor * mean
            corrected_var = (1.0 - exponential_avg_factor) * old_var + exponential_avg_factor * corrected_var
        return (normalized * scale + offset, mean, var, corrected_var)

    def _reference_grad(self, x, grad_y, scale, mean, var, epsilon, data_format):
        if False:
            for i in range(10):
                print('nop')
        if data_format != 'NHWC':
            raise ValueError('data_format must be NHWC, got %s.' % data_format)
        grad_x = scale * (grad_y - np.mean(grad_y, axis=(0, 1, 2)) - (x - mean) * np.mean(grad_y * (x - mean), axis=(0, 1, 2)) / (var + epsilon)) / np.sqrt(var + epsilon)
        grad_scale = np.sum(grad_y * (x - mean) / np.sqrt(var + epsilon), axis=(0, 1, 2))
        grad_offset = np.sum(grad_y, axis=(0, 1, 2))
        return (grad_x, grad_scale, grad_offset)

    @parameterized.named_parameters(*DATA_FORMATS)
    def testInference(self, data_format):
        if False:
            while True:
                i = 10
        channel = 3
        x_shape = [2, 2, 6, channel]
        scale_shape = [channel]
        x_val = np.random.random_sample(x_shape).astype(np.float32)
        scale_val = np.random.random_sample(scale_shape).astype(np.float32)
        offset_val = np.random.random_sample(scale_shape).astype(np.float32)
        epsilon = 0.001
        exponential_avg_factor = 1.0
        data_format_src = 'NHWC'
        (y_ref, mean_ref, var_ref, _) = self._reference_training(x_val, scale_val, offset_val, None, None, epsilon, exponential_avg_factor, data_format_src)
        with self.session() as sess, self.test_scope():
            x_val_converted = test_utils.ConvertBetweenDataFormats(x_val, data_format_src, data_format)
            y_ref_converted = test_utils.ConvertBetweenDataFormats(y_ref, data_format_src, data_format)
            t_val = array_ops.placeholder(np.float32, shape=x_val_converted.shape, name='x')
            scale = array_ops.placeholder(np.float32, shape=scale_shape, name='scale')
            offset = array_ops.placeholder(np.float32, shape=scale_shape, name='offset')
            (y, mean, variance) = nn.fused_batch_norm(t_val, scale, offset, mean=mean_ref, variance=var_ref, epsilon=epsilon, data_format=data_format, is_training=False)
            (y_val, _, _) = sess.run([y, mean, variance], {t_val: x_val_converted, scale: scale_val, offset: offset_val})
            self.assertAllClose(y_val, y_ref_converted, atol=0.001)

    def _testLearning(self, use_gradient_checker, data_format, exponential_avg_factor):
        if False:
            print('Hello World!')
        channel = 3
        x_shape = [2, 2, 6, channel]
        scale_shape = [channel]
        x_val = np.random.random_sample(x_shape).astype(np.float32)
        scale_val = np.random.random_sample(scale_shape).astype(np.float32)
        offset_val = np.random.random_sample(scale_shape).astype(np.float32)
        mean_val = np.random.random_sample(scale_shape).astype(np.float32)
        var_val_corr = np.random.random_sample(scale_shape).astype(np.float32)
        epsilon = 0.001
        data_format_src = 'NHWC'
        (y_ref, mean_ref, _, var_ref_corr) = self._reference_training(x_val, scale_val, offset_val, mean_val, var_val_corr, epsilon, exponential_avg_factor, data_format_src)
        with self.session() as sess, self.test_scope():
            x_val_converted = test_utils.ConvertBetweenDataFormats(x_val, data_format_src, data_format)
            y_ref_converted = test_utils.ConvertBetweenDataFormats(y_ref, data_format_src, data_format)
            t_val = array_ops.placeholder(np.float32, shape=x_val_converted.shape, name='x')
            scale = array_ops.placeholder(np.float32, shape=scale_shape, name='scale')
            offset = array_ops.placeholder(np.float32, shape=scale_shape, name='offset')
            if exponential_avg_factor == 1.0:
                old_mean = None
                old_var = None
            else:
                old_mean = array_ops.placeholder(np.float32, shape=scale_shape, name='old_mean')
                old_var = array_ops.placeholder(np.float32, shape=scale_shape, name='old_var')
            (y, mean, var) = nn.fused_batch_norm(t_val, scale, offset, mean=old_mean, variance=old_var, epsilon=epsilon, exponential_avg_factor=exponential_avg_factor, data_format=data_format, is_training=True)
            if exponential_avg_factor == 1.0:
                feed_dict = {t_val: x_val_converted, scale: scale_val, offset: offset_val}
            else:
                feed_dict = {t_val: x_val_converted, scale: scale_val, offset: offset_val, old_mean: mean_val, old_var: var_val_corr}
            if use_gradient_checker:
                err = gradient_checker.compute_gradient_error(t_val, x_val_converted.shape, y, x_val_converted.shape, extra_feed_dict=feed_dict)
                self.assertLess(err, 0.001)
            (y_tf, mean_tf, var_tf) = sess.run([y, mean, var], feed_dict)
            self.assertAllClose(y_tf, y_ref_converted, atol=0.001)
            self.assertAllClose(mean_tf, mean_ref, atol=0.001)
            self.assertAllClose(var_tf, var_ref_corr, atol=0.001)

    @parameterized.named_parameters(*DATA_FORMATS_AND_AVG_FACTORS)
    def testLearning(self, data_format, exponential_avg_factor):
        if False:
            while True:
                i = 10
        self._testLearning(False, data_format, exponential_avg_factor)

    @parameterized.named_parameters(*DATA_FORMATS_AND_AVG_FACTORS)
    def testLearningWithGradientChecker(self, data_format, exponential_avg_factor):
        if False:
            return 10
        self._testLearning(True, data_format, exponential_avg_factor)

    @parameterized.named_parameters(*DATA_FORMATS)
    def testGradientTraining(self, data_format):
        if False:
            return 10
        if test_util.is_mlir_bridge_enabled() and self.device == 'XLA_GPU':
            self.skipTest('b/189039456')
        channel = 3
        x_shape = [2, 2, 6, channel]
        scale_shape = [channel]
        grad_val = np.random.random_sample(x_shape).astype(np.float32)
        x_val = np.random.random_sample(x_shape).astype(np.float32)
        scale_val = np.random.random_sample(scale_shape).astype(np.float32)
        mean_val = np.random.random_sample(scale_shape).astype(np.float32)
        var_val = np.random.random_sample(scale_shape).astype(np.float32)
        epsilon = 0.001
        reserve_space_1_val = mean_val
        if self.device == 'XLA_GPU':
            reserve_space_2_val = np.reciprocal(np.sqrt(var_val + epsilon))
        else:
            reserve_space_2_val = var_val
        data_format_src = 'NHWC'
        (grad_x_ref, grad_scale_ref, grad_offset_ref) = self._reference_grad(x_val, grad_val, scale_val, mean_val, var_val, epsilon, data_format_src)
        with self.session() as sess, self.test_scope():
            grad_val_converted = test_utils.ConvertBetweenDataFormats(grad_val, data_format_src, data_format)
            x_val_converted = test_utils.ConvertBetweenDataFormats(x_val, data_format_src, data_format)
            grad_x_ref_converted = test_utils.ConvertBetweenDataFormats(grad_x_ref, data_format_src, data_format)
            grad = array_ops.placeholder(np.float32, shape=x_val_converted.shape, name='grad')
            x = array_ops.placeholder(np.float32, shape=x_val_converted.shape, name='x')
            reserve_space_1 = array_ops.placeholder(np.float32, shape=scale_shape, name='reserve_space_1')
            reserve_space_2 = array_ops.placeholder(np.float32, shape=scale_shape, name='reserve_space_2')
            scale = array_ops.placeholder(np.float32, shape=scale_shape, name='scale')
            (grad_x, grad_scale, grad_offset, _, _) = gen_nn_ops.fused_batch_norm_grad(grad, x, scale, reserve_space_1, reserve_space_2, data_format=data_format, is_training=True)
            (grad_x_val, grad_scale_val, grad_offset_val) = sess.run([grad_x, grad_scale, grad_offset], {grad: grad_val_converted, x: x_val_converted, reserve_space_1: reserve_space_1_val, reserve_space_2: reserve_space_2_val, scale: scale_val})
            self.assertAllClose(grad_x_val, grad_x_ref_converted, atol=0.01)
            self.assertAllClose(grad_scale_val, grad_scale_ref, atol=0.01)
            self.assertAllClose(grad_offset_val, grad_offset_ref, atol=0.001)

    @parameterized.named_parameters(*DATA_FORMATS)
    def testGradientInference(self, data_format):
        if False:
            return 10
        channel = 3
        x_shape = [2, 2, 6, channel]
        scale_shape = [channel]
        grad_val = np.random.random_sample(x_shape).astype(np.float32)
        x_val = np.random.random_sample(x_shape).astype(np.float32)
        scale_val = np.random.random_sample(scale_shape).astype(np.float32)
        mean_val = np.random.random_sample(scale_shape).astype(np.float32)
        var_val = np.random.random_sample(scale_shape).astype(np.float32)
        data_format_src = 'NHWC'
        with self.session() as sess, self.test_scope():
            grad_val_converted = test_utils.ConvertBetweenDataFormats(grad_val, data_format_src, data_format)
            x_val_converted = test_utils.ConvertBetweenDataFormats(x_val, data_format_src, data_format)
            grad = array_ops.placeholder(np.float32, shape=x_val_converted.shape, name='grad')
            x = array_ops.placeholder(np.float32, shape=x_val_converted.shape, name='x')
            mean = array_ops.placeholder(np.float32, shape=scale_shape, name='mean')
            var = array_ops.placeholder(np.float32, shape=scale_shape, name='var')
            scale = array_ops.placeholder(np.float32, shape=scale_shape, name='scale')
            with self.test_scope():
                out = gen_nn_ops.fused_batch_norm_grad(grad, x, scale, mean, var, data_format=data_format, is_training=False)
                (grad_x, grad_scale, grad_offset, _, _) = out
            (ref_x, ref_scale, ref_offset, _, _) = gen_nn_ops.fused_batch_norm_grad(grad, x, scale, mean, var, data_format=data_format, is_training=False)
            (grad_x_val, grad_scale_val, grad_offset_val) = sess.run([grad_x, grad_scale, grad_offset], {grad: grad_val_converted, x: x_val_converted, mean: mean_val, var: var_val, scale: scale_val})
            (grad_x_ref, grad_scale_ref, grad_offset_ref) = sess.run([ref_x, ref_scale, ref_offset], {grad: grad_val_converted, x: x_val_converted, mean: mean_val, var: var_val, scale: scale_val})
            self.assertAllClose(grad_x_val, grad_x_ref, atol=0.01)
            self.assertAllClose(grad_scale_val, grad_scale_ref, atol=0.01)
            self.assertAllClose(grad_offset_val, grad_offset_ref, atol=0.001)
if __name__ == '__main__':
    test.main()