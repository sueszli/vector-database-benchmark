import os
import numpy as np
import paddle
import paddle.distributed as dist

class TestMatmulApiForSemiAutoParallel:

    def __init__(self):
        if False:
            return 10
        self._dtype = os.getenv('dtype')
        self._backend = os.getenv('backend')
        self._seed = eval(os.getenv('seed'))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=['x'])

    def check_tensor_eq(self, a, b):
        if False:
            return 10
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=0.0001, verbose=True)

    def test_body(self, x_shape, y_shape, x_specs, y_specs, trans_x=False, trans_y=False):
        if False:
            while True:
                i = 10
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        x_np = np.random.random(size=x_shape).astype(self._dtype)
        y_np = np.random.random(size=y_shape).astype(self._dtype)
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        x.stop_gradient = False
        y.stop_gradient = False
        x_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=x_specs)
        y_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=y_specs)
        dist_x = dist.shard_tensor(x_np, dist_attr=x_dist_attr)
        dist_y = dist.shard_tensor(y_np, dist_attr=y_dist_attr)
        dist_x.stop_gradient = False
        dist_y.stop_gradient = False
        out = paddle.matmul(x, y, transpose_x=trans_x, transpose_y=trans_y)
        dist_out = paddle.matmul(dist_x, dist_y, transpose_x=trans_x, transpose_y=trans_y)
        self.check_tensor_eq(out, dist_out)
        out.backward()
        dist_out.backward()
        self.check_tensor_eq(x.grad, dist_x.grad)
        self.check_tensor_eq(y.grad, dist_y.grad)
        return (dist_out, dist_x.grad, dist_y.grad)

    def test_matmul_x_row_shard(self):
        if False:
            print('Hello World!')
        (dist_out, dist_x_grad, dist_y_grad) = self.test_body(x_shape=[64, 32], y_shape=[32, 48], x_specs=['x', None], y_specs=[None, None])
        np.testing.assert_equal(dist_out._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(dist_out.dist_attr.dims_mapping, [0, -1], verbose=True)
        assert dist_out.dist_attr._is_partial() is False
        np.testing.assert_equal(dist_x_grad._local_shape, [32, 32], verbose=True)
        np.testing.assert_equal(dist_x_grad.dist_attr.dims_mapping, [0, -1], verbose=True)
        assert dist_x_grad.dist_attr._is_partial() is False
        np.testing.assert_equal(dist_y_grad._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(dist_y_grad.dist_attr.dims_mapping, [-1, -1], verbose=True)
        assert dist_y_grad.dist_attr._is_partial() is False

    def test_matmul_x_column_shard(self):
        if False:
            for i in range(10):
                print('nop')
        (dist_out, dist_x_grad, dist_y_grad) = self.test_body(x_shape=[64, 32], y_shape=[32, 48], x_specs=[None, 'x'], y_specs=[None, None])
        np.testing.assert_equal(dist_out._local_shape, [64, 48], verbose=True)
        np.testing.assert_equal(dist_out.dist_attr.dims_mapping, [-1, -1], verbose=True)
        np.testing.assert_equal(dist_x_grad._local_shape, [64, 16], verbose=True)
        np.testing.assert_equal(dist_x_grad.dist_attr.dims_mapping, [-1, 0], verbose=True)
        assert dist_x_grad.dist_attr._is_partial() is False
        np.testing.assert_equal(dist_y_grad._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(dist_y_grad.dist_attr.dims_mapping, [-1, -1], verbose=True)
        assert dist_y_grad.dist_attr._is_partial() is False

    def test_matmul_x_column_shard_trans_x_y(self):
        if False:
            print('Hello World!')
        (dist_out, dist_x_grad, dist_y_grad) = self.test_body(x_shape=[32, 64], y_shape=[48, 32], x_specs=[None, 'x'], y_specs=[None, None], trans_x=True, trans_y=True)
        np.testing.assert_equal(dist_out._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(dist_out.dist_attr.dims_mapping, [0, -1], verbose=True)
        assert dist_out.dist_attr._is_partial() is False
        np.testing.assert_equal(dist_x_grad._local_shape, [32, 32], verbose=True)
        np.testing.assert_equal(dist_x_grad.dist_attr.dims_mapping, [-1, 0], verbose=True)
        assert dist_x_grad.dist_attr._is_partial() is False
        np.testing.assert_equal(dist_y_grad._local_shape, [48, 32], verbose=True)
        np.testing.assert_equal(dist_y_grad.dist_attr.dims_mapping, [-1, -1], verbose=True)
        assert dist_y_grad.dist_attr._is_partial() is False

    def test_matmul_x_column_shard_trans_x(self):
        if False:
            return 10
        (dist_out, dist_x_grad, dist_y_grad) = self.test_body(x_shape=[32, 64], y_shape=[32, 48], x_specs=[None, 'x'], y_specs=[None, None], trans_x=True, trans_y=False)
        np.testing.assert_equal(dist_out._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(dist_out.dist_attr.dims_mapping, [0, -1], verbose=True)
        assert dist_out.dist_attr._is_partial() is False
        np.testing.assert_equal(dist_x_grad._local_shape, [32, 32], verbose=True)
        np.testing.assert_equal(dist_x_grad.dist_attr.dims_mapping, [-1, 0], verbose=True)
        assert dist_x_grad.dist_attr._is_partial() is False
        np.testing.assert_equal(dist_y_grad._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(dist_y_grad.dist_attr.dims_mapping, [-1, -1], verbose=True)
        assert dist_y_grad.dist_attr._is_partial() is False

    def test_matmul_x_row_shard_trans_y(self):
        if False:
            return 10
        (dist_out, dist_x_grad, dist_y_grad) = self.test_body(x_shape=[64, 32], y_shape=[48, 32], x_specs=['x', None], y_specs=[None, None], trans_x=False, trans_y=True)
        np.testing.assert_equal(dist_out._local_shape, [32, 48], verbose=True)
        np.testing.assert_equal(dist_out.dist_attr.dims_mapping, [0, -1], verbose=True)
        assert dist_out.dist_attr._is_partial() is False
        np.testing.assert_equal(dist_x_grad._local_shape, [32, 32], verbose=True)
        np.testing.assert_equal(dist_x_grad.dist_attr.dims_mapping, [0, -1], verbose=True)
        assert dist_x_grad.dist_attr._is_partial() is False
        np.testing.assert_equal(dist_y_grad._local_shape, [48, 32], verbose=True)
        np.testing.assert_equal(dist_y_grad.dist_attr.dims_mapping, [-1, -1], verbose=True)
        assert dist_y_grad.dist_attr._is_partial() is False

    def test_matmul_with_complex_type(self):
        if False:
            while True:
                i = 10
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        x_np = np.random.random(size=[64, 32]).astype(np.complex128)
        y_np = np.random.random(size=[32, 48]).astype(np.float32)
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        x.stop_gradient = False
        y.stop_gradient = False
        x_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=[None, None])
        y_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=[None, None])
        dist_x = dist.shard_tensor(x_np, dist_attr=x_dist_attr)
        dist_y = dist.shard_tensor(y_np, dist_attr=y_dist_attr)
        dist_x.stop_gradient = False
        dist_y.stop_gradient = False
        out = paddle.matmul(x, y, transpose_x=False, transpose_y=False)
        dist_out = paddle.matmul(dist_x, dist_y, transpose_x=False, transpose_y=False)
        self.check_tensor_eq(out, dist_out)
        out.backward()
        dist_out.backward()
        self.check_tensor_eq(x.grad, dist_x.grad)
        self.check_tensor_eq(y.grad, dist_y.grad)

    def run_test_case(self):
        if False:
            while True:
                i = 10
        if self._backend == 'cpu':
            paddle.set_device('cpu')
        elif self._backend == 'gpu':
            paddle.set_device('gpu:' + str(dist.get_rank()))
        else:
            raise ValueError('Only support cpu or gpu backend.')
        self.test_matmul_x_row_shard()
        self.test_matmul_x_column_shard()
        self.test_matmul_x_column_shard_trans_x_y()
        self.test_matmul_x_column_shard_trans_x()
        self.test_matmul_x_row_shard_trans_y()
        self.test_matmul_with_complex_type()
if __name__ == '__main__':
    TestMatmulApiForSemiAutoParallel().run_test_case()