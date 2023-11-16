import unittest
import numpy as np
import paddle
from paddle.static import Program, program_guard
np.random.seed(42)

def calc_hinge_embedding_loss(input, label, margin=1.0, reduction='mean'):
    if False:
        i = 10
        return i + 15
    result = np.where(label == -1.0, np.maximum(0.0, margin - input), 0.0) + np.where(label == 1.0, input, 0.0)
    if reduction == 'none':
        return result
    elif reduction == 'sum':
        return np.sum(result)
    elif reduction == 'mean':
        return np.mean(result)

class TestFunctionalHingeEmbeddingLoss(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.margin = 1.0
        self.shape = (10, 10, 5)
        self.input_np = np.random.random(size=self.shape).astype(np.float64)
        self.label_np = 2 * np.random.randint(0, 2, size=self.shape) - 1.0

    def run_dynamic_check(self, place=paddle.CPUPlace()):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static(place=place)
        input = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np, dtype=paddle.float64)
        dy_result = paddle.nn.functional.hinge_embedding_loss(input, label)
        expected = calc_hinge_embedding_loss(self.input_np, self.label_np)
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [])
        dy_result = paddle.nn.functional.hinge_embedding_loss(input, label, reduction='sum')
        expected = calc_hinge_embedding_loss(self.input_np, self.label_np, reduction='sum')
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [])
        dy_result = paddle.nn.functional.hinge_embedding_loss(input, label, reduction='none')
        expected = calc_hinge_embedding_loss(self.input_np, self.label_np, reduction='none')
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, list(self.shape))

    def run_static_check(self, place=paddle.CPUPlace):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        for reduction in ['none', 'mean', 'sum']:
            expected = calc_hinge_embedding_loss(self.input_np, self.label_np, reduction=reduction)
            with program_guard(Program(), Program()):
                input = paddle.static.data(name='input', shape=self.shape, dtype=paddle.float64)
                label = paddle.static.data(name='label', shape=self.shape, dtype=paddle.float64)
                st_result = paddle.nn.functional.hinge_embedding_loss(input, label, reduction=reduction)
                exe = paddle.static.Executor(place)
                (result_numpy,) = exe.run(feed={'input': self.input_np, 'label': self.label_np}, fetch_list=[st_result])
                np.testing.assert_allclose(result_numpy, expected, rtol=1e-05)

    def test_cpu(self):
        if False:
            i = 10
            return i + 15
        self.run_dynamic_check(place=paddle.CPUPlace())
        self.run_static_check(place=paddle.CPUPlace())

    def test_gpu(self):
        if False:
            i = 10
            return i + 15
        if not paddle.is_compiled_with_cuda():
            return
        self.run_dynamic_check(place=paddle.CUDAPlace(0))
        self.run_static_check(place=paddle.CUDAPlace(0))

    def test_reduce_errors(self):
        if False:
            print('Hello World!')

        def test_value_error():
            if False:
                while True:
                    i = 10
            loss = paddle.nn.functional.hinge_embedding_loss(self.input_np, self.label_np, reduction='reduce_mean')
        self.assertRaises(ValueError, test_value_error)

class TestClassHingeEmbeddingLoss(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.margin = 1.0
        self.shape = (10, 10, 5)
        self.input_np = np.random.random(size=self.shape).astype(np.float64)
        self.label_np = 2 * np.random.randint(0, 2, size=self.shape) - 1.0

    def run_dynamic_check(self, place=paddle.CPUPlace()):
        if False:
            return 10
        paddle.disable_static(place=place)
        input = paddle.to_tensor(self.input_np)
        label = paddle.to_tensor(self.label_np, dtype=paddle.float64)
        hinge_embedding_loss = paddle.nn.loss.HingeEmbeddingLoss()
        dy_result = hinge_embedding_loss(input, label)
        expected = calc_hinge_embedding_loss(self.input_np, self.label_np)
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [])
        hinge_embedding_loss = paddle.nn.loss.HingeEmbeddingLoss(reduction='sum')
        dy_result = hinge_embedding_loss(input, label)
        expected = calc_hinge_embedding_loss(self.input_np, self.label_np, reduction='sum')
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertEqual(dy_result.shape, [])
        hinge_embedding_loss = paddle.nn.loss.HingeEmbeddingLoss(reduction='none')
        dy_result = hinge_embedding_loss(input, label)
        expected = calc_hinge_embedding_loss(self.input_np, self.label_np, reduction='none')
        np.testing.assert_allclose(dy_result.numpy(), expected, rtol=1e-05)
        self.assertTrue(dy_result.shape, list(self.shape))

    def run_static_check(self, place=paddle.CPUPlace):
        if False:
            print('Hello World!')
        paddle.enable_static()
        for reduction in ['none', 'mean', 'sum']:
            expected = calc_hinge_embedding_loss(self.input_np, self.label_np, reduction=reduction)
            with program_guard(Program(), Program()):
                input = paddle.static.data(name='input', shape=self.shape, dtype=paddle.float64)
                label = paddle.static.data(name='label', shape=self.shape, dtype=paddle.float64)
                hinge_embedding_loss = paddle.nn.loss.HingeEmbeddingLoss(reduction=reduction)
                st_result = hinge_embedding_loss(input, label)
                exe = paddle.static.Executor(place)
                (result_numpy,) = exe.run(feed={'input': self.input_np, 'label': self.label_np}, fetch_list=[st_result])
                np.testing.assert_allclose(result_numpy, expected, rtol=1e-05)

    def test_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_dynamic_check(place=paddle.CPUPlace())
        self.run_static_check(place=paddle.CPUPlace())

    def test_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        if not paddle.is_compiled_with_cuda():
            return
        self.run_dynamic_check(place=paddle.CUDAPlace(0))
        self.run_static_check(place=paddle.CUDAPlace(0))

    def test_reduce_errors(self):
        if False:
            return 10

        def test_value_error():
            if False:
                for i in range(10):
                    print('nop')
            hinge_embedding_loss = paddle.nn.loss.HingeEmbeddingLoss(reduction='reduce_mean')
            loss = hinge_embedding_loss(self.input_np, self.label_np)
        self.assertRaises(ValueError, test_value_error)
if __name__ == '__main__':
    unittest.main()