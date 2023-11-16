import os
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.framework import core

class TestReshardSToS:

    def __init__(self):
        if False:
            return 10
        self._shape = eval(os.getenv('shape'))
        self._dtype = os.getenv('dtype')
        self._seeds = eval(os.getenv('seeds'))
        self._backend = os.getenv('backend')
        self._mesh = dist.ProcessMesh([0, 1], dim_names=['x'])

    def test_body(self, in_shard, out_shard):
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
        in_shard_specs[in_shard] = 'x'
        out_shard_specs = [None for i in range(len(self._shape))]
        out_shard_specs[out_shard] = 'x'
        dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=in_shard_specs)
        out_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=out_shard_specs)
        input_tensor = dist.shard_tensor(a, dist_attr=dist_attr)
        reshard_func = core.SToSReshardFunction()
        assert reshard_func.is_suitable(input_tensor, out_dist_attr)
        out = reshard_func.eval(dev_ctx, input_tensor, out_dist_attr)
        out_shape = list(self._shape)
        out_shape[out_shard] = out_shape[out_shard] // 2
        assert np.equal(out.shape, input_tensor.shape).all()
        assert np.equal(out._local_shape, out_shape).all()

    def test_case1(self):
        if False:
            return 10
        self.test_body(0, len(self._shape) - 1)

    def test_case2(self):
        if False:
            return 10
        self.test_body(len(self._shape) - 1, 0)

    def run_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_case1()
        self.test_case2()
if __name__ == '__main__':
    TestReshardSToS().run_test_case()