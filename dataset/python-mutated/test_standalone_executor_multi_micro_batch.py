import platform
import unittest
import numpy as np
import paddle
from paddle.base import core
from paddle.base.core import Job, Plan
from paddle.base.executor import _add_feed_fetch_ops, _StandaloneExecutor
from paddle.distributed.passes.pass_utils import set_skip_gc_vars, split_program
from paddle.nn import TransformerEncoderLayer
paddle.enable_static()

class TestEncorderMulitMicroBatchRun(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.place_desc = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()
        self.place = core.Place()
        self.place.set_place(self.place_desc)
        self.batch_size = 2
        self.src_len = 4
        self.d_model = 128
        self.n_head = 2
        self.run_step = 3
        (self.enc_input_data, self.attn_mask_data) = self.get_random_data(self.batch_size, self.src_len, self.d_model, self.n_head, self.run_step)

    def get_random_data(self, batch_size, src_len, d_model, n_head, run_step):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(2022)
        enc_input_data = np.random.rand(run_step, batch_size, src_len, d_model).astype(np.float32)
        attn_mask_data = np.random.rand(run_step, batch_size, n_head, src_len, src_len).astype(np.float32)
        return (enc_input_data, attn_mask_data)

    def batch_generator_creator(self, micro_batch_size):
        if False:
            while True:
                i = 10

        def __reader__():
            if False:
                i = 10
                return i + 15
            for i in range(self.run_step):
                for offset in range(0, self.batch_size, micro_batch_size):
                    enc_input = self.enc_input_data[i][offset:offset + micro_batch_size]
                    attn_mask = self.attn_mask_data[i][offset:offset + micro_batch_size]
                    yield (enc_input, attn_mask)
        return __reader__

    def build_program(self, micro_batch_size, src_len, d_model, n_head):
        if False:
            return 10
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            enc_input = paddle.static.data(name='enc_input', shape=[micro_batch_size, src_len, d_model], dtype='float32')
            attn_mask = paddle.static.data(name='attn_mask', shape=[micro_batch_size, n_head, src_len, src_len], dtype='float32')
            loader = paddle.base.io.DataLoader.from_generator(feed_list=[enc_input, attn_mask], use_double_buffer=False, capacity=16, iterable=False)
            loader.set_batch_generator(self.batch_generator_creator(micro_batch_size))
            encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward=512)
            attn_mask = paddle.nn.layer.transformer._convert_attention_mask(attn_mask, enc_input.dtype)
            enc_output = encoder_layer(enc_input, attn_mask)
            split_op_indics = [len(main_program.block(0).ops)]
            enc_output = encoder_layer(enc_output, attn_mask)
            fetch_list = [enc_output.name]
            return (startup_program, main_program, split_op_indics, loader, fetch_list)

    def avoid_randomness(self, program):
        if False:
            while True:
                i = 10
        for op in program.block(0).ops:
            if op.type == 'dropout':
                op._set_attr('dropout_prob', 0)

    def run_train(self, split=False, micro_batch_num=1):
        if False:
            return 10
        paddle.seed(2022)
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            (startup_program, main_program, split_op_indics, loader, fetch_list) = self.build_program(self.batch_size // micro_batch_num, self.src_len, self.d_model, self.n_head)
        self.avoid_randomness(main_program)
        startup_exe = _StandaloneExecutor(self.place, Plan([Job('startup')], {'startup': startup_program.desc}), scope)
        startup_exe.run([])
        programs = [main_program]
        fetch_op_num = len(fetch_list)
        fetch_op_indics = []
        if split:
            (programs, _, _) = split_program(main_program, split_op_indics)
            programs[-1] = _add_feed_fetch_ops(programs[-1], [], fetch_list, 'feed', 'fetch')
            op_num = len(programs[-1].block(0).ops)
            fetch_op_indics = list(range(op_num - fetch_op_num, op_num))
        else:
            programs[0] = _add_feed_fetch_ops(programs[0], [], fetch_list, 'feed', 'fetch')
            op_num = len(programs[0].block(0).ops)
            fetch_op_indics = list(range(op_num - fetch_op_num, op_num))
        job_list = []
        program_num = len(programs)
        for micro_batch_id in range(micro_batch_num):
            for program_id in range(program_num):
                job = Job(f'P{program_id}')
                job.set_micro_batch_id(micro_batch_id)
                job_list.append(job)
        job_types = []
        for program_id in range(program_num):
            job_types.append(f'P{program_id}')
        type_to_program = set_skip_gc_vars(micro_batch_num, job_types, programs, job_list)
        for type in type_to_program.keys():
            type_to_program[type] = type_to_program[type].desc
        plan = Plan(job_list, type_to_program)
        main_exe = _StandaloneExecutor(self.place, plan, scope)
        loader.start()
        res = []
        for i in range(self.run_step):
            fetch_res = main_exe.run(feed_names=[])
            res.append(np.array(fetch_res).reshape(self.batch_size, self.src_len, self.d_model))
        return res

    def check_result(self, expected_result, actual_result):
        if False:
            while True:
                i = 10
        if self.place.is_cpu_place() or platform.system().lower() == 'windows':
            np.testing.assert_allclose(expected_result, actual_result, atol=1e-06, rtol=1e-06)
        else:
            np.testing.assert_equal(expected_result, actual_result)

    def test_multi_micro_batch_run(self):
        if False:
            for i in range(10):
                print('nop')
        last_res = None
        for split in [True, False]:
            for micro_batch_num in [1, 2]:
                res = self.run_train(split, micro_batch_num)
                if last_res:
                    for i in range(len(res)):
                        self.check_result(last_res[i], res[i])
                last_res = res
if __name__ == '__main__':
    unittest.main()