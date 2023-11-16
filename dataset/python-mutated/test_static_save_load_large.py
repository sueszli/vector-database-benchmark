import os
import tempfile
import unittest
import numpy as np
from test_imperative_base import new_program_scope
import paddle
from paddle import base
from paddle.base import framework
LARGE_PARAM = 2 ** 26

class TestStaticSaveLoadLargeParameters(unittest.TestCase):

    def test_large_parameters_static_save(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        with new_program_scope():
            x = paddle.static.data(name='static_save_load_large_x', shape=[None, 10], dtype='float32')
            z = paddle.static.nn.fc(x, LARGE_PARAM, bias_attr=False)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            prog = paddle.static.default_main_program()
            base_map = {}
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(base.global_scope().find_var(var.name).get_tensor())
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t
            temp_dir = tempfile.TemporaryDirectory()
            path = os.path.join(temp_dir.name, 'test_static_save_load_large_param')
            path = os.path.join(path, 'static_save')
            protocol = 4
            paddle.static.save(prog, path, pickle_protocol=protocol)
            load_prog1 = paddle.static.Program()
            paddle.static.load(load_prog1, path)
            for var in load_prog1.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(base.global_scope().find_var(var.name).get_tensor())
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)
            load_prog2 = paddle.static.Program()
            program_state = paddle.static.load_program_state(path)
            paddle.static.set_program_state(load_prog2, program_state)
            for var in load_prog2.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(base.global_scope().find_var(var.name).get_tensor())
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)
            temp_dir.cleanup()
if __name__ == '__main__':
    unittest.main()