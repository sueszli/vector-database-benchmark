import logging
import unittest
import numpy as np
import paddle
from paddle import sparse
from paddle.base import core
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSparseConv(unittest.TestCase):

    def test_conv2d(self):
        if False:
            while True:
                i = 10
        kernel = [[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]]
        dense_kernel = paddle.to_tensor(kernel, dtype='float32', stop_gradient=False)
        dense_kernel = paddle.reshape(dense_kernel, [3, 3, 1, 1])
        paddings = [0, 0]
        strides = [1, 1]
        dilations = [1, 1]
        bias = [1]
        indices = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [1, 2, 3, 4]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 3, 4, 1]
        correct_out_values = [[5], [11]]
        sparse_input = core.eager.sparse_coo_tensor(indices, values, dense_shape, False)
        out = paddle.sparse.nn.functional.conv2d(sparse_input, dense_kernel, bias=paddle.to_tensor(bias, dtype='float32'), stride=strides, padding=paddings, dilation=dilations, groups=1, data_format='NHWC')
        out.backward(out)
        out = paddle.sparse.coalesce(out)
        np.testing.assert_array_equal(correct_out_values, out.values().numpy())

    def test_conv3d(self):
        if False:
            print('Hello World!')
        kernel = [[[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]]]
        dense_kernel = paddle.to_tensor(kernel, dtype='float32', stop_gradient=False)
        dense_kernel = paddle.reshape(dense_kernel, [1, 3, 3, 1, 1])
        paddings = [0, 0, 0]
        strides = [1, 1, 1]
        dilations = [1, 1, 1]
        bias = [1]
        indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [1, 2, 3, 4]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 1, 3, 4, 1]
        correct_out_values = [[5], [11]]
        sparse_input = core.eager.sparse_coo_tensor(indices, values, dense_shape, False)
        out = paddle.sparse.nn.functional.conv3d(sparse_input, dense_kernel, bias=paddle.to_tensor(bias, dtype='float32'), stride=strides, padding=paddings, dilation=dilations, groups=1, data_format='NDHWC')
        out.backward(out)
        out = paddle.sparse.coalesce(out)
        np.testing.assert_array_equal(correct_out_values, out.values().numpy())

    def test_subm_conv2d(self):
        if False:
            i = 10
            return i + 15
        indices = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 3, 4, 1]
        sparse_x = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, stop_gradient=True)
        weight = paddle.randn((1, 3, 3, 1), dtype='float32')
        y = paddle.sparse.nn.functional.subm_conv2d(sparse_x, weight, key='subm_conv')
        np.testing.assert_array_equal(sparse_x.indices().numpy(), y.indices().numpy())

    def test_subm_conv3d(self):
        if False:
            for i in range(10):
                print('nop')
        indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 1, 3, 4, 1]
        sparse_x = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, stop_gradient=True)
        weight = paddle.randn((1, 3, 3, 1, 1), dtype='float32')
        y = paddle.sparse.nn.functional.subm_conv3d(sparse_x, weight, key='subm_conv')
        np.testing.assert_array_equal(sparse_x.indices().numpy(), y.indices().numpy())

    def test_Conv2D(self):
        if False:
            return 10
        indices = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 3, 4, 1]
        correct_out_values = [[4], [10]]
        sparse_input = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, False)
        sparse_conv2d = paddle.sparse.nn.Conv2D(1, 1, (3, 3), data_format='NHWC')
        sparse_out = sparse_conv2d(sparse_input)
        with self.assertRaises(ValueError):
            conv2d = paddle.sparse.nn.SubmConv2D(1, 1, (3, 3), data_format='NCHW', key='subm_conv')

    def test_Conv3D(self):
        if False:
            while True:
                i = 10
        indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 1, 3, 4, 1]
        correct_out_values = [[4], [10]]
        sparse_input = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, False)
        sparse_conv3d = paddle.sparse.nn.Conv3D(1, 1, (1, 3, 3), data_format='NDHWC')
        sparse_out = sparse_conv3d(sparse_input)
        with self.assertRaises(ValueError):
            conv3d = paddle.sparse.nn.SubmConv3D(1, 1, (1, 3, 3), data_format='NCDHW', key='subm_conv')

    def test_SubmConv2D(self):
        if False:
            print('Hello World!')
        indices = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 3, 4, 1]
        correct_out_values = [[4], [10]]
        sparse_input = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, False)
        subm_conv2d = paddle.sparse.nn.SubmConv2D(1, 1, (3, 3), data_format='NHWC', key='subm_conv')
        logger.info(subm_conv2d.extra_repr())
        sparse_out = subm_conv2d(sparse_input)
        np.testing.assert_array_equal(indices, sparse_out.indices().numpy())
        with self.assertRaises(ValueError):
            conv2d = paddle.sparse.nn.SubmConv2D(1, 1, (3, 3), data_format='NCHW', key='subm_conv')

    def test_SubmConv3D(self):
        if False:
            for i in range(10):
                print('nop')
        indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
        values = [[1], [2], [3], [4]]
        indices = paddle.to_tensor(indices, dtype='int32')
        values = paddle.to_tensor(values, dtype='float32')
        dense_shape = [1, 1, 3, 4, 1]
        correct_out_values = [[4], [10]]
        sparse_input = paddle.sparse.sparse_coo_tensor(indices, values, dense_shape, False)
        subm_conv3d = paddle.sparse.nn.SubmConv3D(1, 1, (1, 3, 3), data_format='NDHWC', key='subm_conv')
        print(subm_conv3d.extra_repr())
        sparse_out = subm_conv3d(sparse_input)
        np.testing.assert_array_equal(indices, sparse_out.indices().numpy())
        with self.assertRaises(ValueError):
            conv3d = paddle.sparse.nn.SubmConv3D(1, 1, (1, 3, 3), data_format='NCDHW', key='subm_conv')

    def test_Conv2D_bias(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.seed(0)
        shape = [1, 4, 4, 3]
        x = paddle.randn(shape)
        sp_x = x.to_sparse_coo(3)
        conv2d = paddle.nn.Conv2D(3, 2, 3, data_format='NHWC')
        sp_conv2d = paddle.sparse.nn.Conv2D(3, 2, 3, data_format='NHWC')
        sp_conv2d.weight.set_value(paddle.to_tensor(conv2d.weight.numpy().transpose(2, 3, 1, 0)))
        sp_conv2d.bias.set_value(paddle.to_tensor(conv2d.bias.numpy()))
        x.stop_gradient = False
        out = conv2d(x)
        loss = out.mean()
        loss.backward()
        sp_x.stop_gradient = False
        sp_out = sp_conv2d(sp_x)
        dense_out = sp_out.to_dense()
        sp_loss = dense_out.mean()
        sp_loss.backward()
        np.testing.assert_allclose(out.numpy(), dense_out.numpy(), atol=0.001, rtol=0.001)
        np.testing.assert_allclose(conv2d.weight.grad.numpy().transpose(2, 3, 1, 0), sp_conv2d.weight.grad.numpy(), atol=0.001, rtol=0.001)
        np.testing.assert_allclose(conv2d.bias.grad.numpy(), sp_conv2d.bias.grad.numpy(), atol=1e-05, rtol=1e-05)

    def test_Conv3D_bias(self):
        if False:
            i = 10
            return i + 15
        paddle.seed(0)
        shape = [1, 4, 4, 4, 3]
        x = paddle.randn(shape)
        sp_x = x.to_sparse_coo(4)
        conv3d = paddle.nn.Conv3D(3, 2, 3, data_format='NDHWC')
        sp_conv3d = paddle.sparse.nn.Conv3D(3, 2, 3, data_format='NDHWC')
        sp_conv3d.weight.set_value(paddle.to_tensor(conv3d.weight.numpy().transpose(2, 3, 4, 1, 0)))
        sp_conv3d.bias.set_value(paddle.to_tensor(conv3d.bias.numpy()))
        x.stop_gradient = False
        out = conv3d(x)
        loss = out.mean()
        loss.backward()
        sp_x.stop_gradient = False
        sp_out = sp_conv3d(sp_x)
        dense_out = sp_out.to_dense()
        sp_loss = dense_out.mean()
        sp_loss.backward()
        np.testing.assert_allclose(out.numpy(), dense_out.numpy(), atol=0.001, rtol=0.001)
        np.testing.assert_allclose(conv3d.weight.grad.numpy().transpose(2, 3, 4, 1, 0), sp_conv3d.weight.grad.numpy(), atol=0.001, rtol=0.001)
        np.testing.assert_allclose(conv3d.bias.grad.numpy(), sp_conv3d.bias.grad.numpy(), atol=1e-05, rtol=1e-05)

class TestStatic(unittest.TestCase):

    def test(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        main = paddle.static.Program()
        with paddle.static.program_guard(main):
            indices = paddle.static.data(name='indices', shape=[4, 4], dtype='int32')
            values = paddle.static.data(name='values', shape=[4, 1], dtype='float32')
            dense_shape = [1, 1, 3, 4, 1]
            sp_x = sparse.sparse_coo_tensor(indices, values, dense_shape)
            weight_shape = [1, 3, 3, 1, 1]
            weight = paddle.static.data(name='weight', shape=weight_shape, dtype='float32')
            bias_shape = [1]
            bias = paddle.static.data(name='bias', shape=bias_shape, dtype='float32')
            out = sparse.nn.functional.conv3d(sp_x, weight, bias, stride=1, padding=0, dilation=1, groups=1, data_format='NDHWC')
            sp_out = sparse.nn.functional.relu(out)
            out_indices = sp_out.indices()
            out_values = sp_out.values()
            out = sp_out.to_dense()
            exe = paddle.static.Executor()
            indices_data = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
            values_data = [[1.0], [2.0], [3.0], [4.0]]
            weight_data = np.array([[[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]]]).astype('float32')
            weight_data = weight_data.reshape(weight_shape)
            bias_data = np.array([1]).astype('float32')
            fetch = exe.run(feed={'indices': indices_data, 'values': values_data, 'weight': weight_data, 'bias': bias_data}, fetch_list=[out, out_indices, out_values], return_numpy=True)
            correct_out = np.array([[[[[5.0], [11.0]]]]]).astype('float64')
            correct_out_values = [[5.0], [11.0]]
            np.testing.assert_array_equal(correct_out, fetch[0])
            np.testing.assert_array_equal(correct_out_values, fetch[2])
            self.assertTrue(out_indices.dtype == paddle.int32)
        paddle.disable_static()

    def test_cpu(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        main = paddle.static.Program()
        with paddle.static.program_guard(main):
            indices = paddle.static.data(name='indices', shape=[4, 4], dtype='int32')
            values = paddle.static.data(name='values', shape=[4, 1], dtype='float32')
            dense_shape = [1, 1, 3, 4, 1]
            sp_x = sparse.sparse_coo_tensor(indices, values, dense_shape)
            weight_shape = [1, 3, 3, 1, 1]
            weight = paddle.static.data(name='weight', shape=weight_shape, dtype='float32')
            bias_shape = [1]
            bias = paddle.static.data(name='bias', shape=bias_shape, dtype='float32')
            out = sparse.nn.functional.conv3d(sp_x, weight, bias, stride=1, padding=0, dilation=1, groups=1, data_format='NDHWC')
            sp_out = sparse.nn.functional.relu(out)
            out_indices = sp_out.indices()
            out_values = sp_out.values()
            out = sp_out.to_dense()
            place = paddle.CPUPlace()
            exe = paddle.static.Executor()
            indices_data = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
            values_data = [[1.0], [2.0], [3.0], [4.0]]
            weight_data = np.array([[[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]]]).astype('float32')
            weight_data = weight_data.reshape(weight_shape)
            bias_data = np.array([1]).astype('float32')
            fetch = exe.run(feed={'indices': indices_data, 'values': values_data, 'weight': weight_data, 'bias': bias_data}, fetch_list=[out, out_indices, out_values], return_numpy=True)
            correct_out = np.array([[[[[5.0], [11.0]]]]]).astype('float64')
            correct_out_values = [[5.0], [11.0]]
            np.testing.assert_array_equal(correct_out, fetch[0])
            np.testing.assert_array_equal(correct_out_values, fetch[2])
            self.assertTrue(out_indices.dtype == paddle.int32)
        paddle.disable_static()

    def test2D(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        main = paddle.static.Program()
        with paddle.static.program_guard(main):
            indices = paddle.static.data(name='indices', shape=[3, 4], dtype='int32')
            values = paddle.static.data(name='values', shape=[4, 1], dtype='float32')
            dense_shape = [1, 3, 4, 1]
            sp_x = sparse.sparse_coo_tensor(indices, values, dense_shape)
            weight_shape = [3, 3, 1, 1]
            weight = paddle.static.data(name='weight', shape=weight_shape, dtype='float32')
            bias_shape = [1]
            bias = paddle.static.data(name='bias', shape=bias_shape, dtype='float32')
            out = sparse.nn.functional.conv2d(sp_x, weight, bias, stride=1, padding=0, dilation=1, groups=1, data_format='NHWC')
            sp_out = sparse.nn.functional.relu(out)
            out_indices = sp_out.indices()
            out_values = sp_out.values()
            out = sp_out.to_dense()
            exe = paddle.static.Executor()
            indices_data = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
            values_data = [[1.0], [2.0], [3.0], [4.0]]
            weight_data = np.array([[[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]]]).astype('float32')
            weight_data = weight_data.reshape(weight_shape)
            bias_data = np.array([1]).astype('float32')
            fetch = exe.run(feed={'indices': indices_data, 'values': values_data, 'weight': weight_data, 'bias': bias_data}, fetch_list=[out, out_indices, out_values], return_numpy=True)
            correct_out = np.array([[[[5.0], [11.0]]]]).astype('float64')
            correct_out_values = [[5.0], [11.0]]
            np.testing.assert_array_equal(correct_out, fetch[0])
            np.testing.assert_array_equal(correct_out_values, fetch[2])
            self.assertTrue(out_indices.dtype == paddle.int32)
        paddle.disable_static()

    def test2D_cpu(self):
        if False:
            return 10
        paddle.enable_static()
        main = paddle.static.Program()
        with paddle.static.program_guard(main):
            indices = paddle.static.data(name='indices', shape=[3, 4], dtype='int32')
            values = paddle.static.data(name='values', shape=[4, 1], dtype='float32')
            dense_shape = [1, 3, 4, 1]
            sp_x = sparse.sparse_coo_tensor(indices, values, dense_shape)
            weight_shape = [3, 3, 1, 1]
            weight = paddle.static.data(name='weight', shape=weight_shape, dtype='float32')
            bias_shape = [1]
            bias = paddle.static.data(name='bias', shape=bias_shape, dtype='float32')
            out = sparse.nn.functional.conv2d(sp_x, weight, bias, stride=1, padding=0, dilation=1, groups=1, data_format='NHWC')
            sp_out = sparse.nn.functional.relu(out)
            out_indices = sp_out.indices()
            out_values = sp_out.values()
            out = sp_out.to_dense()
            place = paddle.CPUPlace()
            exe = paddle.static.Executor()
            indices_data = [[0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
            values_data = [[1.0], [2.0], [3.0], [4.0]]
            weight_data = np.array([[[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]]]).astype('float32')
            weight_data = weight_data.reshape(weight_shape)
            bias_data = np.array([1]).astype('float32')
            fetch = exe.run(feed={'indices': indices_data, 'values': values_data, 'weight': weight_data, 'bias': bias_data}, fetch_list=[out, out_indices, out_values], return_numpy=True)
            correct_out = np.array([[[[5.0], [11.0]]]]).astype('float64')
            correct_out_values = [[5.0], [11.0]]
            np.testing.assert_array_equal(correct_out, fetch[0])
            np.testing.assert_array_equal(correct_out_values, fetch[2])
            self.assertTrue(out_indices.dtype == paddle.int32)
        paddle.disable_static()
if __name__ == '__main__':
    unittest.main()