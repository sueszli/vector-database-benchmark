import os
import legacy_test.test_collective_api_base as test_collective_base
import numpy as np
import paddle
import paddle.distributed as dist

class StreamSendRecvTestCase:

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
        src_rank = 0
        dst_rank = 1
        rank = dist.get_rank()
        tensor = paddle.to_tensor(test_data_list[rank])
        if rank == 0:
            task = dist.stream.send(tensor, dst=dst_rank, sync_op=self._sync_op, use_calc_stream=self._use_calc_stream)
        else:
            task = dist.stream.recv(tensor, src=src_rank, sync_op=self._sync_op, use_calc_stream=self._use_calc_stream)
        if not self._sync_op:
            task.wait()
        result = test_data_list[src_rank]
        np.testing.assert_allclose(tensor, result, rtol=1e-05, atol=1e-05)
if __name__ == '__main__':
    StreamSendRecvTestCase().run_test_case()