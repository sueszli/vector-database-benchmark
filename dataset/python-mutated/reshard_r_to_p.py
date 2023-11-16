import os
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.framework import core

class TestReshardRToP:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._shape = eval(os.getenv('shape'))
        self._dtype = os.getenv('dtype')
        self._seeds = eval(os.getenv('seeds'))
        self._backend = os.getenv('backend')
        self._mesh = dist.ProcessMesh([0, 1], dim_names=['x'])

    def run_test_case(self):
        if False:
            while True:
                i = 10
        if self._backend == 'cpu':
            paddle.set_device('cpu')
            place = paddle.CPUPlace()
        elif self._backend == 'gpu':
            place = paddle.CUDAPlace(dist.get_rank())
        dev_ctx = core.DeviceContext.create(place)
        a = paddle.ones(self._shape)
        in_shard_specs = [None for i in range(len(self._shape))]
        out_shard_specs = [None for i in range(len(self._shape))]
        dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=in_shard_specs)
        out_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=out_shard_specs)
        out_dist_attr._set_partial_dims([0])
        input_tensor = dist.shard_tensor(a, dist_attr=dist_attr)
        reshard_func = core.RToPReshardFunction()
        assert reshard_func.is_suitable(input_tensor, out_dist_attr)
        out = reshard_func.eval(dev_ctx, input_tensor, out_dist_attr)
        if dist.get_rank() == 0:
            np.testing.assert_equal(out._local_value().numpy(), input_tensor.numpy())
        else:
            zeros = paddle.zeros(self._shape)
            np.testing.assert_equal(out._local_value().numpy(), zeros.numpy())
        assert np.equal(out.shape, input_tensor.shape).all()
        assert np.equal(out._local_shape, input_tensor._local_shape).all()
if __name__ == '__main__':
    TestReshardRToP().run_test_case()