import os
import numpy as np
import paddle
import paddle.distributed as dist

class TestBitwiseApiForSemiAutoParallel:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._dtype = os.getenv('dtype')
        self._backend = os.getenv('backend')
        self._seed = eval(os.getenv('seed'))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        self._check_grad = False
        self._rtol = 1e-06
        self._atol = 0.0
        paddle.seed(self._seed)
        np.random.seed(self._seed)

    def check_tensor_eq(self, a, b):
        if False:
            return 10
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=self._rtol, atol=self._atol, verbose=True)

    def test_unary_body(self, x_shape, out_shape, x_specs, unary_func):
        if False:
            print('Hello World!')
        x = paddle.randint(0, 100, x_shape, self._dtype)
        x.stop_gradient = False
        x_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=x_specs)
        dist_x = dist.shard_tensor(x, dist_attr=x_dist_attr)
        dist_x.stop_gradient = False
        dist_out = unary_func(dist_x)
        out = unary_func(x)
        self.check_tensor_eq(out, dist_out)
        if self._check_grad:
            dist_out.backward()
            out.backward()
            self.check_tensor_eq(x.grad, dist_x.grad)

    def test_binary_body(self, x_shape, y_shape, out_shape, x_specs, y_specs, binary_func):
        if False:
            i = 10
            return i + 15
        x = paddle.randint(0, 100, x_shape, self._dtype)
        y = paddle.randint(0, 100, y_shape, self._dtype)
        x.stop_gradient = False
        y.stop_gradient = False
        x_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=x_specs)
        y_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=y_specs)
        dist_x = dist.shard_tensor(x, dist_attr=x_dist_attr)
        dist_y = dist.shard_tensor(y, dist_attr=y_dist_attr)
        dist_x.stop_gradient = False
        dist_y.stop_gradient = False
        dist_out = binary_func(dist_x, dist_y)
        out = binary_func(x, y)
        self.check_tensor_eq(out, dist_out)
        if self._check_grad:
            dist_out.backward()
            out.backward()
            self.check_tensor_eq(x.grad, dist_x.grad)
            self.check_tensor_eq(y.grad, dist_y.grad)

    def test_bitwise_and_x_shard(self):
        if False:
            i = 10
            return i + 15
        self.test_binary_body(x_shape=[16, 32], y_shape=[16, 32], out_shape=[16, 32], x_specs=['x', None], y_specs=[None, None], binary_func=paddle.bitwise_and)

    def test_bitwise_and_x_shard_broadcast(self):
        if False:
            return 10
        self.test_binary_body(x_shape=[16, 32], y_shape=[2, 16, 32], out_shape=[2, 16, 32], x_specs=['x', None], y_specs=[None, None, None], binary_func=paddle.bitwise_and)

    def test_bitwise_and_x_y_shard(self):
        if False:
            while True:
                i = 10
        if self._backend == 'cpu':
            return
        self.test_binary_body(x_shape=[16, 32], y_shape=[16, 32], out_shape=[16, 32], x_specs=['x', None], y_specs=[None, 'x'], binary_func=paddle.bitwise_and)

    def test_bitwise_and_x_y_shard_broadcast(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_binary_body(x_shape=[4, 16, 32], y_shape=[16, 32], out_shape=[4, 16, 32], x_specs=['x', None, None], y_specs=[None, None], binary_func=paddle.bitwise_and)

    def test_bitwise_not_x_shard(self):
        if False:
            return 10
        self.test_unary_body(x_shape=[16, 32], out_shape=[16, 32], x_specs=['x', None], unary_func=paddle.bitwise_not)

    def test_bitwise_not_x_shard_broadcast(self):
        if False:
            print('Hello World!')
        self.test_binary_body(x_shape=[16, 32], y_shape=[2, 16, 32], out_shape=[2, 16, 32], x_specs=['x', None], y_specs=[None, None, None], binary_func=paddle.bitwise_not)

    def run_test_case(self):
        if False:
            return 10
        if self._backend == 'cpu':
            paddle.set_device('cpu')
        elif self._backend == 'gpu':
            paddle.set_device('gpu:' + str(dist.get_rank()))
        else:
            raise ValueError('Only support cpu or gpu backend.')
        self.test_bitwise_and_x_shard()
        self.test_bitwise_and_x_shard_broadcast()
        self.test_bitwise_and_x_y_shard()
        self.test_bitwise_and_x_y_shard_broadcast()
        self.test_bitwise_not_x_shard()
if __name__ == '__main__':
    TestBitwiseApiForSemiAutoParallel().run_test_case()