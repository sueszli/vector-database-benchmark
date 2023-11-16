import os
import numpy as np
from legacy_test.test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main
import paddle
from paddle import base
from paddle.distributed.utils import moe_utils

class TestCollectiveGlobalGatherAPI(TestCollectiveAPIRunnerBase):

    def __init__(self):
        if False:
            return 10
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, indata=None):
        if False:
            i = 10
            return i + 15
        with base.program_guard(main_prog, startup_program):
            seed = os.getpid()
            np.random.seed(seed)
            in_feat = 2
            n_expert = 2
            world_size = 2
            tot_expert = n_expert * world_size
            local_expert_count = np.random.randint(1, 4, size=tot_expert).astype('int')
            local_expert_count = paddle.to_tensor(local_expert_count)
            global_expert_count = []
            paddle.distributed.alltoall(paddle.split(local_expert_count, 2, axis=0), global_expert_count)
            global_expert_count = paddle.concat(global_expert_count, axis=0)
            fwd_expert_count = sum(global_expert_count)
            np.random.seed(seed)
            local_input_buf = np.random.rand(fwd_expert_count, in_feat).astype('float32')
            local_input_buf = paddle.to_tensor(local_input_buf)
            local_input_buf.stop_gradient = False
            output = moe_utils.global_gather(local_input_buf, local_expert_count, global_expert_count)
            output.stop_gradient = False
            c = output * output
            c.stop_gradient = False
            c.backward()
            return [output.numpy(False), local_input_buf.grad.numpy(False)]
if __name__ == '__main__':
    runtime_main(TestCollectiveGlobalGatherAPI, 'global_gather')