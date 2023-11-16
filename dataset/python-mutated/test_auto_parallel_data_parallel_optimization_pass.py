import random
import sys
import unittest
import numpy as np
from auto_parallel_pass_test_base import AutoPallelPassTestBase
import paddle
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.static.dist_context import get_default_distributed_context
from paddle.distributed.auto_parallel.static.operators.common import is_data_parallel_reduce_op
from paddle.distributed.passes import PassContext, new_pass
sys.path.append('..')

class TestDataParallelPassWithScale1(AutoPallelPassTestBase):

    def init(self):
        if False:
            i = 10
            return i + 15
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
        self.rtol = 1e-05
        self.atol = 1e-08
        self._apply_pass = False
        rank = paddle.distributed.get_rank()
        paddle.seed(rank + 2021)
        random.seed(rank + 2021)
        np.random.seed(rank + 2021)

    def apply_passes(self):
        if False:
            print('Hello World!')
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)
        self._apply_pass = True

    def apply_no_passes(self):
        if False:
            while True:
                i = 10
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)
        self._apply_pass = False

    def test_bs_8(self):
        if False:
            return 10
        self.check_main(gpus=[0, 1], batch_size=8, sequence_len=512, vocab_size=1000)

    def get_model(self, place, batch_size, sequence_len, vocab_size):
        if False:
            while True:
                i = 10
        (dist_main_prog, dist_startup_prog, data_holder, [loss], gen_data) = self.get_gpt_model('dp', place, batch_size, sequence_len, vocab_size)
        if self._apply_pass:
            config = {}
            config['dist_context'] = get_default_distributed_context()
            config['global_rank'] = paddle.distributed.get_rank()
            dp_pass = new_pass('auto_parallel_data_parallel_optimization', config)
            dp_pass.apply([dist_main_prog], [dist_startup_prog], PassContext())
        return (dist_main_prog, dist_startup_prog, data_holder, [loss], gen_data)

class TestDataParallelPassWithScale2(TestDataParallelPassWithScale1):

    def get_model(self, place, batch_size, sequence_len, vocab_size):
        if False:
            return 10
        (dist_main_prog, dist_startup_prog, data_holder, [loss], gen_data) = self.get_gpt_model('dp', place, batch_size, sequence_len, vocab_size, optimizer='LarsMomentum')
        if self._apply_pass:
            config = {}
            config['dist_context'] = get_default_distributed_context()
            config['global_rank'] = paddle.distributed.get_rank()
            dp_pass = new_pass('auto_parallel_data_parallel_optimization', config)
            dp_pass.apply([dist_main_prog], [dist_startup_prog], PassContext())
        return (dist_main_prog, dist_startup_prog, data_holder, [loss], gen_data)

class TestDataParallelPassWithStandaloneEXE(TestDataParallelPassWithScale1):

    def init(self):
        if False:
            return 10
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
        self.rtol = 1e-05
        self.atol = 1e-08
        self._apply_pass = False
        rank = paddle.distributed.get_rank()
        paddle.seed(rank + 2021)
        random.seed(rank + 2021)
        np.random.seed(rank + 2021)

    def get_model(self, place, batch_size, sequence_len, vocab_size):
        if False:
            print('Hello World!')
        (dist_main_prog, dist_startup_prog, data_holder, [loss], gen_data) = self.get_gpt_model('dp', place, batch_size, sequence_len, vocab_size, optimizer='LarsMomentum')
        if self._apply_pass:
            config = {}
            config['dist_context'] = get_default_distributed_context()
            config['global_rank'] = paddle.distributed.get_rank()
            dp_pass = new_pass('auto_parallel_data_parallel_optimization', config)
            dp_pass.apply([dist_main_prog], [dist_startup_prog], PassContext())
            ops = dist_main_prog.global_block().ops
            allreduce_op_idx = -1
            for idx in range(len(ops)):
                if is_data_parallel_reduce_op(ops[idx]):
                    allreduce_op_idx = idx
                    break
            assert allreduce_op_idx > 0
            allreduce_op = ops[allreduce_op_idx]
            assert allreduce_op.attr('use_calc_stream') is True
            assert allreduce_op.dist_attr.execution_stream is not None
            assert ops[allreduce_op_idx - 1].type == 'nop'
            assert ops[allreduce_op_idx + 1].type == 'nop'
        return (dist_main_prog, dist_startup_prog, data_holder, [loss], gen_data)
if __name__ == '__main__':
    unittest.main()