import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16, paddle_static_guard
import paddle
from paddle import base
from paddle.base import Program, core, program_guard
from paddle.pir_utils import test_with_pir_api

def accuracy_wrapper(infer, indices, label):
    if False:
        while True:
            i = 10
    return paddle._C_ops.accuracy(infer, indices, label)

class TestAccuracyOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'accuracy'
        self.python_api = accuracy_wrapper
        self.dtype = np.float32
        self.init_dtype()
        n = 8192
        infer = np.random.random((n, 1)).astype(self.dtype)
        indices = np.random.randint(0, 2, (n, 1)).astype('int64')
        label = np.random.randint(0, 2, (n, 1)).astype('int64')
        self.inputs = {'Out': infer, 'Indices': indices, 'Label': label}
        num_correct = 0
        for rowid in range(n):
            for ele in indices[rowid]:
                if ele == label[rowid]:
                    num_correct += 1
                    break
        self.outputs = {'Accuracy': np.array(num_correct / float(n)).astype(self.dtype), 'Correct': np.array(num_correct).astype('int32'), 'Total': np.array(n).astype('int32')}

    def init_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_pir=True)

class TestAccuracyOpFp16(TestAccuracyOp):

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float16

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(atol=0.001, check_pir=True)

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not compiled with CUDA and not support the bfloat16')
class TestAccuracyOpBf16(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'accuracy'
        self.python_api = accuracy_wrapper
        self.init_dtype()
        n = 8192
        infer = np.random.random((n, 1)).astype(np.float32)
        indices = np.random.randint(0, 2, (n, 1)).astype('int64')
        label = np.random.randint(0, 2, (n, 1)).astype('int64')
        self.inputs = {'Out': convert_float_to_uint16(infer), 'Indices': indices, 'Label': label}
        num_correct = 0
        for rowid in range(n):
            for ele in indices[rowid]:
                if ele == label[rowid]:
                    num_correct += 1
                    break
        self.outputs = {'Accuracy': convert_float_to_uint16(np.array(num_correct / float(n)).astype(np.float32)), 'Correct': np.array(num_correct).astype('int32'), 'Total': np.array(n).astype('int32')}

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = np.uint16

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=0.01, check_pir=True)

class TestAccuracyOpError(unittest.TestCase):

    def test_type_errors(self):
        if False:
            for i in range(10):
                print('nop')
        with paddle_static_guard():
            with program_guard(Program(), Program()):
                x1 = base.create_lod_tensor(np.array([[-1]]), [[1]], base.CPUPlace())
                label = paddle.static.data(name='label', shape=[-1, 1], dtype='int32')
                self.assertRaises(TypeError, paddle.static.accuracy, x1, label)
                self.assertRaises(TypeError, paddle.metric.accuracy, x1, label)
                x2 = paddle.static.data(name='x2', shape=[-1, 4], dtype='int32')
                self.assertRaises(TypeError, paddle.static.accuracy, x2, label)
                self.assertRaises(TypeError, paddle.metric.accuracy, x2, label)
                x3 = paddle.static.data(name='input', shape=[-1, 2], dtype='float16')
                paddle.static.accuracy(input=x3, label=label)
                paddle.metric.accuracy(input=x3, label=label)

    def test_value_errors(self):
        if False:
            return 10
        with program_guard(Program(), Program()):
            with self.assertRaises(ValueError):
                x3 = paddle.to_tensor([0.1], dtype='float32')
                label3 = paddle.to_tensor(np.reshape([0], [1, 1]), dtype='int32')
                paddle.metric.accuracy(x3, label3)

class TestAccuracyAPI1(unittest.TestCase):

    def run_api(self, accuracy_api):
        if False:
            i = 10
            return i + 15
        with paddle_static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                self.predictions = paddle.static.data(shape=[2, 5], name='predictions', dtype='float32')
                self.label = paddle.static.data(shape=[2, 1], name='labels', dtype='int64')
                self.result = accuracy_api(input=self.predictions, label=self.label, k=1)
                self.input_predictions = np.array([[0.2, 0.1, 0.4, 0.1, 0.1], [0.2, 0.3, 0.1, 0.15, 0.25]], dtype='float32')
                self.input_labels = np.array([[2], [0]], dtype='int64')
                self.expect_value = np.array([0.5], dtype='float32')
                exe = paddle.static.Executor()
                (result,) = exe.run(feed={'predictions': self.input_predictions, 'labels': self.input_labels}, fetch_list=[self.result])
                self.assertEqual((result == self.expect_value).all(), True)

    @test_with_pir_api
    def test_api(self):
        if False:
            while True:
                i = 10
        self.run_api(accuracy_api=paddle.static.accuracy)
        self.run_api(accuracy_api=paddle.metric.accuracy)

class TestAccuracyAPI2(unittest.TestCase):

    def test_api(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard():
            predictions = paddle.to_tensor([[0.2, 0.1, 0.4, 0.1, 0.1], [0.2, 0.3, 0.1, 0.15, 0.25]], dtype='float32')
            label = paddle.to_tensor([[2], [0]], dtype='int64')
            result = paddle.static.accuracy(input=predictions, label=label, k=1)
            expect_value = np.array([0.5], dtype='float32')
            self.assertEqual((result.numpy() == expect_value).all(), True)

class TestAccuracyAPI(unittest.TestCase):

    def test_api(self):
        if False:
            i = 10
            return i + 15
        with base.dygraph.guard():
            predictions = paddle.to_tensor([[0.2, 0.1, 0.4, 0.1, 0.1], [0.2, 0.3, 0.1, 0.15, 0.25]], dtype='float32')
            label = paddle.to_tensor([[2], [0]], dtype='int64')
            result = paddle.metric.accuracy(input=predictions, label=label, k=1)
            expect_value = np.array([0.5], dtype='float32')
            self.assertEqual((result.numpy() == expect_value).all(), True)
if __name__ == '__main__':
    unittest.main()