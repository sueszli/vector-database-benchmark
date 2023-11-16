import unittest
import numpy
import paddle
from paddle import base

class TestExecutor(unittest.TestCase):

    def net(self):
        if False:
            print('Hello World!')
        lr = 0.0
        x = paddle.static.data(name='x', shape=[None, 1], dtype='float32')
        y = paddle.static.data(name='y', shape=[None, 1], dtype='float32')
        y_predict = paddle.static.nn.fc(x, size=1)
        cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
        avg_cost = paddle.mean(cost)
        opt = paddle.optimizer.Adam(learning_rate=lr)
        opt.minimize(avg_cost)
        return (paddle.to_tensor(lr), avg_cost)

    def test_program_feed_float(self):
        if False:
            for i in range(10):
                print('nop')
        main_program = base.Program()
        startup_program = base.Program()
        scope = base.Scope()
        with base.program_guard(main_program, startup_program):
            with base.scope_guard(scope):
                cpu = base.CPUPlace()
                exe = base.Executor(cpu)
                (lr, cost) = self.net()
                exe.run(startup_program)
                train_data = numpy.array([[1.0], [2.0], [3.0], [4.0]]).astype('float32')
                y_true = numpy.array([[2.0], [4.0], [6.0], [8.0]]).astype('float32')
                a = 0.01
                (_lr, _) = exe.run(feed={'x': train_data, 'y': y_true, 'lr': a}, fetch_list=[lr, cost], return_numpy=False)
            self.assertEqual(_lr._dtype(), lr.dtype)
            self.assertEqual(_lr._dtype(), base.core.VarDesc.VarType.FP32)
            self.assertEqual(type(a), float)

    def test_program_feed_int(self):
        if False:
            print('Hello World!')
        main_program = base.Program()
        startup_program = base.Program()
        scope = base.Scope()
        with base.program_guard(main_program, startup_program):
            with base.scope_guard(scope):
                cpu = base.CPUPlace()
                exe = base.Executor(cpu)
                (lr, cost) = self.net()
                exe.run(startup_program)
                train_data = numpy.array([[1.0], [2.0], [3.0], [4.0]]).astype('float32')
                y_true = numpy.array([[2.0], [4.0], [6.0], [8.0]]).astype('float32')
                a = 0
                (_lr, _) = exe.run(feed={'x': train_data, 'y': y_true, 'lr': a}, fetch_list=[lr, cost], return_numpy=False)
            self.assertEqual(_lr._dtype(), lr.dtype)
            self.assertEqual(_lr._dtype(), base.core.VarDesc.VarType.FP32)
            self.assertEqual(type(a), int)

    def test_program_feed_list(self):
        if False:
            while True:
                i = 10
        main_program = base.Program()
        startup_program = base.Program()
        scope = base.Scope()
        with base.program_guard(main_program, startup_program):
            with base.scope_guard(scope):
                cpu = base.CPUPlace()
                exe = base.Executor(cpu)
                (lr, cost) = self.net()
                exe.run(startup_program)
                train_data = [[1.0], [2.0], [3.0], [4.0]]
                y_true = [[2.0], [4.0], [6.0], [8.0]]
                a = 0
                (_lr, _) = exe.run(feed={'x': train_data, 'y': y_true, 'lr': a}, fetch_list=[lr, cost], return_numpy=False)
            self.assertEqual(_lr._dtype(), lr.dtype)
            self.assertEqual(_lr._dtype(), base.core.VarDesc.VarType.FP32)
            self.assertEqual(type(y_true), list)

    def test_compiled_program_feed_scalar(self):
        if False:
            for i in range(10):
                print('nop')
        main_program = base.Program()
        startup_program = base.Program()
        scope = base.Scope()
        with base.program_guard(main_program, startup_program):
            with base.scope_guard(scope):
                (lr, cost) = self.net()
                cpu = base.CPUPlace()
                exe = base.Executor(cpu)
                exe.run(startup_program)
                compiled_prog = base.CompiledProgram(main_program)
                train_data = numpy.array([[1.0], [2.0], [3.0], [4.0]]).astype('float32')
                y_true = numpy.array([[2.0], [4.0], [6.0], [8.0]]).astype('float32')
                a = 0.01
                (_lr, _) = exe.run(compiled_prog, feed={'x': train_data, 'y': y_true, 'lr': a}, fetch_list=[lr, cost], return_numpy=False)
                self.assertEqual(_lr._dtype(), lr.dtype)
                self.assertEqual(_lr._dtype(), base.core.VarDesc.VarType.FP32)
                self.assertEqual(type(a), float)

class TestAsLodTensor(unittest.TestCase):

    def test_as_lodtensor_int32(self):
        if False:
            while True:
                i = 10
        cpu = base.CPUPlace()
        tensor = base.executor._as_lodtensor(1.0, cpu, base.core.VarDesc.VarType.INT32)
        self.assertEqual(tensor._dtype(), base.core.VarDesc.VarType.INT32)

    def test_as_lodtensor_fp64(self):
        if False:
            i = 10
            return i + 15
        cpu = base.CPUPlace()
        tensor = base.executor._as_lodtensor(1, cpu, base.core.VarDesc.VarType.FP64)
        self.assertEqual(tensor._dtype(), base.core.VarDesc.VarType.FP64)

    def test_as_lodtensor_assertion_error(self):
        if False:
            print('Hello World!')
        cpu = base.CPUPlace()
        self.assertRaises(AssertionError, base.executor._as_lodtensor, 1, cpu)

    def test_as_lodtensor_type_error(self):
        if False:
            i = 10
            return i + 15
        cpu = base.CPUPlace()
        self.assertRaises(TypeError, base.executor._as_lodtensor, {'a': 1}, cpu, base.core.VarDesc.VarType.INT32)

    def test_as_lodtensor_list(self):
        if False:
            i = 10
            return i + 15
        cpu = base.CPUPlace()
        tensor = base.executor._as_lodtensor([1, 2], cpu, base.core.VarDesc.VarType.FP64)
        self.assertEqual(tensor._dtype(), base.core.VarDesc.VarType.FP64)

    def test_as_lodtensor_tuple(self):
        if False:
            while True:
                i = 10
        cpu = base.CPUPlace()
        tensor = base.executor._as_lodtensor((1, 2), cpu, base.core.VarDesc.VarType.FP64)
        self.assertEqual(tensor._dtype(), base.core.VarDesc.VarType.FP64)

    def test_as_lodtensor_nested_list(self):
        if False:
            return 10
        cpu = base.CPUPlace()
        self.assertRaises(TypeError, base.executor._as_lodtensor, [{1.2, 1.2}, {1, 2}], cpu, base.core.VarDesc.VarType.INT32)
if __name__ == '__main__':
    unittest.main()