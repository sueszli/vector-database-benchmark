import os
import legacy_test.test_collective_api_base as test_collective_base
import numpy as np
import paddle
import paddle.distributed as dist

class StreamBroadcastTestCase:

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
            print('Hello World!')
        dist.init_parallel_env()
        src_rank = 1
        result = test_collective_base.create_test_data(shape=self._shape, dtype=self._dtype, seed=self._seeds[src_rank])
        tensor = paddle.to_tensor(result)
        task = dist.stream.broadcast(tensor, src=src_rank, sync_op=self._sync_op, use_calc_stream=self._use_calc_stream)
        if not self._sync_op:
            task.wait()
        np.testing.assert_allclose(tensor, result, rtol=1e-05, atol=1e-05)
if __name__ == '__main__':
    StreamBroadcastTestCase().run_test_case()