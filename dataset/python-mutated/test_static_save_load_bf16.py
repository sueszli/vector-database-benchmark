import os
import tempfile
import unittest
import numpy as np
from test_imperative_base import new_program_scope
from test_static_save_load import PtbModel
import paddle
from paddle import base
from paddle.base import core, framework

@unittest.skipIf(not core.supports_bfloat16(), 'place does not support BF16 evaluation')
class TestSaveLoadBF16(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir.cleanup()

    def set_place(self):
        if False:
            for i in range(10):
                print('nop')
        return base.CPUPlace()

    def test_ptb_rnn_cpu_bfloat16(self):
        if False:
            i = 10
            return i + 15
        seed = 90
        hidden_size = 10
        vocab_size = 500
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4
        batch_num = 100
        with new_program_scope():
            base.default_startup_program().random_seed = seed
            base.default_main_program().random_seed = seed
            ptb_model = PtbModel('ptb_model', hidden_size=hidden_size, vocab_size=vocab_size, num_layers=num_layers, num_steps=num_steps, init_scale=init_scale)
            place = self.set_place()
            exe = base.Executor(place)
            sgd = paddle.optimizer.SGD(learning_rate=0.001)
            x = paddle.static.data(name='x', shape=[-1, num_steps], dtype='int64')
            x.desc.set_need_check_feed(False)
            y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
            y.desc.set_need_check_feed(False)
            init_hidden = paddle.static.data(name='init_hidden', shape=[-1, 1], dtype='float32')
            init_hidden.desc.set_need_check_feed(False)
            init_cell = paddle.static.data(name='init_cell', shape=[-1, 1], dtype='float32')
            init_cell.desc.set_need_check_feed(False)
            (static_loss, static_last_hidden, static_last_cell) = ptb_model(x, y, init_hidden, init_cell)
            sgd = paddle.static.amp.bf16.decorate_bf16(sgd, amp_lists=paddle.static.amp.bf16.AutoMixedPrecisionListsBF16(custom_fp32_list={'transpose2', 'concat'}), use_bf16_guard=False, use_pure_bf16=True)
            sgd.minimize(static_loss, framework.default_startup_program())
            out = exe.run(framework.default_startup_program())
            for i in range(batch_num):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros((num_layers, batch_size, hidden_size), dtype='uint16')
                init_cell_data = np.zeros((num_layers, batch_size, hidden_size), dtype='uint16')
                fetch_list = [static_loss, static_last_hidden, static_last_cell]
                out = exe.run(base.default_main_program(), feed={'x': x_data, 'y': y_data, 'init_hidden': init_hidden_data, 'init_cell': init_cell_data}, fetch_list=fetch_list)
            main_program = framework.default_main_program()
            base_map = {}
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(base.global_scope().find_var(var.name).get_tensor())
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t
            save_dir = os.path.join(self.temp_dir.name, 'test_1')
            paddle.static.save(main_program, save_dir)
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = base.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)
                    new_t = np.array(base.global_scope().find_var(var.name).get_tensor())
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)
            paddle.static.load(main_program, os.path.join(self.temp_dir.name, 'test_1.pdparams'), exe)
            for var in main_program.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(base.global_scope().find_var(var.name).get_tensor())
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()