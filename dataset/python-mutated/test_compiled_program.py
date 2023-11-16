import unittest
import numpy as np
from simple_nets import simple_fc_net
from test_imperative_base import new_program_scope
import paddle
from paddle import base
from paddle.base import core

class TestCompiledProgram(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.seed = 100
        self.img = np.random.random(size=(16, 784)).astype('float32')
        self.label = np.random.randint(low=0, high=10, size=[16, 1], dtype=np.int64)
        with new_program_scope():
            paddle.seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
            place = base.CUDAPlace(0) if core.is_compiled_with_cuda() else base.CPUPlace()
            exe = base.Executor(place)
            loss = simple_fc_net()
            exe.run(base.default_startup_program())
            (loss_data,) = exe.run(base.default_main_program(), feed={'image': self.img, 'label': self.label}, fetch_list=[loss.name])
            self.loss = float(loss_data)

    def test_compiled_program_base(self):
        if False:
            return 10
        with new_program_scope():
            paddle.seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
            place = base.CUDAPlace(0) if core.is_compiled_with_cuda() else base.CPUPlace()
            exe = base.Executor(place)
            loss = simple_fc_net()
            exe.run(base.default_startup_program())
            compiled_prog = base.CompiledProgram(base.default_main_program())
            (loss_data,) = exe.run(compiled_prog, feed={'image': self.img, 'label': self.label}, fetch_list=[loss.name])
            np.testing.assert_array_equal(float(loss_data), self.loss)

class TestCompiledProgramError(unittest.TestCase):

    def test_program_or_graph_error(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, base.CompiledProgram, 'program')

    def build_simple_model(self):
        if False:
            while True:
                i = 10
        img = paddle.static.data(name='image', shape=[-1, 1, 28, 28], dtype='float32')
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
        prediction = paddle.static.nn.fc(x=img, size=10, activation='softmax')
        loss = paddle.nn.functional.cross_entropy(input=prediction, label=label, reduction='none', use_softmax=False)
        avg_loss = paddle.mean(loss)

    def compile_program(self):
        if False:
            print('Hello World!')
        with base.program_guard(base.Program()):
            self.build_simple_model()
            program = base.default_main_program()
            compiled_program = base.CompiledProgram(program)
            scope = base.global_scope()
            place = base.CPUPlace()
            compiled_program._compile(scope, place)
            return (compiled_program, scope, place)

    def test_compile_scope_error(self):
        if False:
            print('Hello World!')
        (compiled_program, _, place) = self.compile_program()
        new_scope = core.Scope()
        with self.assertRaises(ValueError):
            compiled_program._compile(new_scope, place)

    def test_compile_place_error(self):
        if False:
            for i in range(10):
                print('nop')
        if core.is_compiled_with_cuda():
            (compiled_program, scope, _) = self.compile_program()
            new_place = base.CUDAPlace(0)
            with self.assertRaises(ValueError):
                compiled_program._compile(scope, new_place)
if __name__ == '__main__':
    unittest.main()