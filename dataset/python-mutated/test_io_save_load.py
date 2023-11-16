import os
import tempfile
import unittest
import paddle
from paddle import base, static
from paddle.base import core

class TestSaveLoadAPIError(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_dir = os.path.join(self.temp_dir.name, 'fake_dir')

    def tearDown(self):
        if False:
            print('Hello World!')
        self.temp_dir.cleanup()

    def test_get_valid_program_error(self):
        if False:
            i = 10
            return i + 15
        graph = core.Graph(core.ProgramDesc())
        compiled_program = base.CompiledProgram(graph)
        with self.assertRaises(TypeError):
            paddle.static.io._get_valid_program(compiled_program)
        with self.assertRaises(TypeError):
            paddle.static.io._get_valid_program('program')

    def test_load_vars_error(self):
        if False:
            i = 10
            return i + 15
        place = base.CPUPlace()
        exe = base.Executor(place)
        with self.assertRaises(TypeError):
            static.io.load_vars(executor=exe, dirname=self.save_dir, main_program='program')
        with self.assertRaises(TypeError):
            static.io.load_vars(executor=exe, dirname=self.save_dir, main_program='program', vars='vars')

class TestSaveInferenceModelAPIError(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.temp_dir.cleanup()

    def test_useless_feeded_var_names(self):
        if False:
            print('Hello World!')
        start_prog = base.Program()
        main_prog = base.Program()
        with base.program_guard(main_prog, start_prog):
            x = paddle.static.data(name='x', shape=[10, 16], dtype='float32')
            y = paddle.static.data(name='y', shape=[10, 16], dtype='float32')
            z = paddle.static.nn.fc(x, 4)
        exe = base.Executor(base.CPUPlace())
        exe.run(start_prog)
        with self.assertRaisesRegex(ValueError, 'not involved in the target_vars calculation'):
            paddle.static.io.save_inference_model(path_prefix=os.path.join(self.temp_dir.name, 'model'), feed_vars=[x, y], fetch_vars=[z], executor=exe, program=main_prog)

class TestWhenTrainWithNoGrad(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir.cleanup()

    def test_when_train_with_no_grad(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        net = paddle.nn.Linear(1024, 1)
        net = paddle.jit.to_static(net)
        x = paddle.rand([1024], 'float32')
        net(x)
        save_path = os.path.join(self.temp_dir.name, 'train_with_no_grad')
        paddle.jit.save(net, save_path)
        net = paddle.jit.load(save_path)
        net.train()
        with paddle.no_grad():
            x = paddle.rand([1024], 'float32')
            net(x)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()