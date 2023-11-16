import unittest
import paddle
from paddle import base
from paddle.base import core
from paddle.distributed.transpiler import collective
from paddle.incubate.layers.nn import _pull_box_sparse

class TestTranspile(unittest.TestCase):
    """TestCases for BoxPS Preload"""

    def get_transpile(self, mode, trainers='127.0.0.1:6174'):
        if False:
            for i in range(10):
                print('nop')
        config = paddle.distributed.transpiler.DistributeTranspilerConfig()
        config.mode = 'collective'
        config.collective_mode = mode
        t = paddle.distributed.transpiler.DistributeTranspiler(config=config)
        return t

    def test_transpile(self):
        if False:
            return 10
        main_program = base.Program()
        startup_program = base.Program()
        t = self.get_transpile('single_process_multi_thread')
        t.transpile(trainer_id=0, startup_program=startup_program, trainers='127.0.0.1:6174', program=main_program)
        t = self.get_transpile('grad_allreduce')
        try:
            t.transpile(trainer_id=0, startup_program=startup_program, trainers='127.0.0.1:6174', program=main_program)
        except ValueError as e:
            print(e)

    def test_single_trainers(self):
        if False:
            for i in range(10):
                print('nop')
        transpiler = collective.GradAllReduce(0)
        try:
            transpiler.transpile(startup_program=base.Program(), main_program=base.Program(), rank=1, endpoints='127.0.0.1:6174', current_endpoint='127.0.0.1:6174', wait_port='6174')
        except ValueError as e:
            print(e)
        transpiler = collective.LocalSGD(0)
        try:
            transpiler.transpile(startup_program=base.Program(), main_program=base.Program(), rank=1, endpoints='127.0.0.1:6174', current_endpoint='127.0.0.1:6174', wait_port='6174')
        except ValueError as e:
            print(e)

class TestRunCmd(unittest.TestCase):
    """TestCases for run_cmd"""

    def test_run_cmd(self):
        if False:
            i = 10
            return i + 15
        ret1 = int(core.run_cmd('ls; echo $?').strip().split('\n')[-1])
        ret2 = int(core.run_cmd('ls; echo $?', -1, -1).strip().split('\n')[-1])
        self.assertTrue(ret1 == 0)
        self.assertTrue(ret2 == 0)

class TestPullBoxSparseOP(unittest.TestCase):
    """TestCases for _pull_box_sparse op"""

    def test_pull_box_sparse_op(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        program = base.Program()
        with base.program_guard(program):
            x = paddle.static.data(name='x', shape=[-1, 1], dtype='int64', lod_level=0)
            y = paddle.static.data(name='y', shape=[-1, 1], dtype='int64', lod_level=0)
            (emb_x, emb_y) = _pull_box_sparse([x, y], size=1)
if __name__ == '__main__':
    unittest.main()