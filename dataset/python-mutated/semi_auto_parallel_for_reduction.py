import os
import numpy as np
import paddle
import paddle.distributed as dist

class TestReductionApiForSemiAutoParallel:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._dtype = os.getenv('dtype')
        self._backend = os.getenv('backend')
        self._seed = eval(os.getenv('seed'))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=['x'])

    def check_tensor_eq(self, a, b):
        if False:
            print('Hello World!')
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05, verbose=True)

    def test_body(self, x_shape, out_shape, x_specs, axis, keepdim, op_func):
        if False:
            print('Hello World!')
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        x = paddle.randn(x_shape, self._dtype)
        x.stop_gradient = False
        x_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=x_specs)
        dist_x = dist.shard_tensor(x, dist_attr=x_dist_attr)
        dist_x.stop_gradient = False
        dist_out = op_func(dist_x, axis=axis, keepdim=keepdim)
        out = op_func(x, axis=axis, keepdim=keepdim)
        self.check_tensor_eq(out, dist_out)
        np.testing.assert_equal(dist_out.shape, out_shape, verbose=True)
        dist_out.backward()
        out.backward()
        self.check_tensor_eq(x.grad, dist_x.grad)

    def test_sum_x_shard(self):
        if False:
            print('Hello World!')
        self.test_body(x_shape=[4, 8, 6], out_shape=[4, 6], x_specs=['x', None, None], axis=1, keepdim=False, op_func=paddle.sum)

    def test_sum_x_shard_on_axis(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_body(x_shape=[4, 8, 6], out_shape=[4], x_specs=[None, 'x', None], axis=[1, 2], keepdim=False, op_func=paddle.sum)

    def test_sum_x_shard_on_axis_keepdim(self):
        if False:
            while True:
                i = 10
        self.test_body(x_shape=[4, 8, 6], out_shape=[4, 1, 6], x_specs=[None, 'x', None], axis=1, keepdim=True, op_func=paddle.sum)

    def test_mean_x_shard(self):
        if False:
            return 10
        self.test_body(x_shape=[4, 8, 6], out_shape=[8, 6], x_specs=['x', None, None], axis=-3, keepdim=False, op_func=paddle.mean)

    def run_test_case(self):
        if False:
            return 10
        if self._backend == 'cpu':
            paddle.set_device('cpu')
        elif self._backend == 'gpu':
            paddle.set_device('gpu:' + str(dist.get_rank()))
        else:
            raise ValueError('Only support cpu or gpu backend.')
        self.test_sum_x_shard()
        self.test_sum_x_shard_on_axis()
        self.test_sum_x_shard_on_axis_keepdim()
        self.test_mean_x_shard()
if __name__ == '__main__':
    TestReductionApiForSemiAutoParallel().run_test_case()