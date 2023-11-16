import os
import legacy_test.test_collective_api_base as test_collective_base
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed.communication.stream.reduce_scatter import _reduce_scatter_base

class StreamReduceScatterTestCase:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._sync_op = eval(os.getenv('sync_op'))
        self._use_calc_stream = eval(os.getenv('use_calc_stream'))
        self._backend = os.getenv('backend')
        self._shape = eval(os.getenv('shape'))
        self._dtype = os.getenv('dtype')
        self._seeds = eval(os.getenv('seeds'))
        if self._backend not in ['nccl', 'gloo']:
            raise NotImplementedError('Only support nccl and gloo as the backend for now.')
        os.environ['PADDLE_DISTRI_BACKEND'] = self._backend

    def run_test_case(self):
        if False:
            return 10
        dist.init_parallel_env()
        test_data_list = []
        for seed in self._seeds:
            test_data_list.append(test_collective_base.create_test_data(shape=self._shape, dtype=self._dtype, seed=seed))
        reduce_result = sum(test_data_list)
        result1 = reduce_result[0:reduce_result.shape[0] // 2]
        result2 = reduce_result[reduce_result.shape[0] // 2:]
        rank = dist.get_rank()
        tensor = paddle.to_tensor(test_data_list[rank])
        (t1, t2) = paddle.split(tensor, 2, axis=0)
        result_tensor = paddle.empty_like(t1)
        task = dist.stream.reduce_scatter(result_tensor, [t1, t2], sync_op=self._sync_op, use_calc_stream=self._use_calc_stream)
        if not self._sync_op:
            task.wait()
        if rank == 0:
            np.testing.assert_allclose(result_tensor, result1, rtol=1e-05, atol=1e-05)
        else:
            np.testing.assert_allclose(result_tensor, result2, rtol=1e-05, atol=1e-05)
        result_tensor = paddle.empty_like(t1)
        task = dist.stream.reduce_scatter(result_tensor, tensor, sync_op=self._sync_op, use_calc_stream=self._use_calc_stream)
        if not self._sync_op:
            task.wait()
        if rank == 0:
            np.testing.assert_allclose(result_tensor, result1, rtol=1e-05, atol=1e-05)
        else:
            np.testing.assert_allclose(result_tensor, result2, rtol=1e-05, atol=1e-05)
        result_tensor = paddle.empty_like(t1)
        task = _reduce_scatter_base(result_tensor, tensor, sync_op=self._sync_op, use_calc_stream=self._use_calc_stream)
        if not self._sync_op:
            task.wait()
        if rank == 0:
            np.testing.assert_allclose(result_tensor, result1, rtol=1e-05, atol=1e-05)
        else:
            np.testing.assert_allclose(result_tensor, result2, rtol=1e-05, atol=1e-05)
if __name__ == '__main__':
    StreamReduceScatterTestCase().run_test_case()