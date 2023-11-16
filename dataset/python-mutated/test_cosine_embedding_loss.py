import unittest
import numpy as np
import paddle
from paddle import static

def cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean'):
    if False:
        for i in range(10):
            print('nop')
    z = (input1 * input2).sum(axis=-1)
    mag_square1 = np.square(input1).sum(axis=-1) + 1e-11
    mag_square2 = np.square(input2).sum(axis=-1) + 1e-11
    denom = np.sqrt(mag_square1 * mag_square2)
    cos = z / denom
    zeros = np.zeros_like(cos)
    pos = 1 - cos
    neg = np.clip(cos - margin, a_min=0, a_max=np.inf)
    out_pos = np.where(label == 1, pos, zeros)
    out_neg = np.where(label == -1, neg, zeros)
    out = out_pos + out_neg
    if reduction == 'none':
        return out
    if reduction == 'mean':
        return np.mean(out)
    elif reduction == 'sum':
        return np.sum(out)

class TestFunctionCosineEmbeddingLoss(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.input1_np = np.random.random(size=(5, 3)).astype(np.float64)
        self.input2_np = np.random.random(size=(5, 3)).astype(np.float64)
        a = np.array([-1, -1, -1]).astype(np.int32)
        b = np.array([1, 1]).astype(np.int32)
        self.label_np = np.concatenate((a, b), axis=0)
        np.random.shuffle(self.label_np)

    def run_dynamic(self):
        if False:
            return 10
        input1 = paddle.to_tensor(self.input1_np)
        input2 = paddle.to_tensor(self.input2_np)
        label = paddle.to_tensor(self.label_np)
        dy_result = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')
        expected1 = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='mean')
        np.testing.assert_allclose(dy_result.numpy(), expected1, rtol=1e-05)
        self.assertEqual(dy_result.shape, [])
        dy_result = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='sum')
        expected2 = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='sum')
        np.testing.assert_allclose(dy_result.numpy(), expected2, rtol=1e-05)
        self.assertEqual(dy_result.shape, [])
        dy_result = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='none')
        expected3 = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='none')
        np.testing.assert_allclose(dy_result.numpy(), expected3, rtol=1e-05)
        self.assertEqual(dy_result.shape, [5])

    def run_static(self, use_gpu=False):
        if False:
            for i in range(10):
                print('nop')
        input1 = static.data(name='input1', shape=[5, 3], dtype='float64')
        input2 = static.data(name='input2', shape=[5, 3], dtype='float64')
        label = static.data(name='label', shape=[5], dtype='int32')
        result0 = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='none')
        result1 = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='sum')
        result2 = paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')
        place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
        exe = static.Executor(place)
        exe.run(static.default_startup_program())
        static_result = exe.run(feed={'input1': self.input1_np, 'input2': self.input2_np, 'label': self.label_np}, fetch_list=[result0, result1, result2])
        expected = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='none')
        np.testing.assert_allclose(static_result[0], expected, rtol=1e-05)
        expected = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='sum')
        np.testing.assert_allclose(static_result[1], expected, rtol=1e-05)
        expected = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='mean')
        np.testing.assert_allclose(static_result[2], expected, rtol=1e-05)

    def test_cpu(self):
        if False:
            return 10
        paddle.disable_static(place=paddle.CPUPlace())
        self.run_dynamic()
        paddle.enable_static()
        with static.program_guard(static.Program()):
            self.run_static()

    def test_gpu(self):
        if False:
            while True:
                i = 10
        if not paddle.is_compiled_with_cuda():
            return
        paddle.disable_static(place=paddle.CUDAPlace(0))
        self.run_dynamic()
        paddle.enable_static()
        with static.program_guard(static.Program()):
            self.run_static(use_gpu=True)

    def test_errors(self):
        if False:
            return 10
        paddle.disable_static()
        input1 = paddle.to_tensor(self.input1_np)
        input2 = paddle.to_tensor(self.input2_np)
        label = paddle.to_tensor(self.label_np)

        def test_label_shape_error():
            if False:
                i = 10
                return i + 15
            label = paddle.to_tensor(np.random.randint(low=0, high=2, size=(2, 3)))
            paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')
        self.assertRaises(ValueError, test_label_shape_error)

        def test_input_different_shape_error():
            if False:
                while True:
                    i = 10
            input1 = paddle.to_tensor(self.input1_np[0])
            label = paddle.to_tensor(np.ndarray([1]))
            paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')
        self.assertRaises(ValueError, test_input_different_shape_error)

        def test_input_shape2D_error():
            if False:
                print('Hello World!')
            input1 = paddle.to_tensor(np.random.random(size=(2, 3, 4)).astype(np.float64))
            input2 = paddle.to_tensor(np.random.random(size=(2, 3, 4)).astype(np.float64))
            paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')
        self.assertRaises(ValueError, test_input_shape2D_error)

        def test_label_value_error():
            if False:
                i = 10
                return i + 15
            label = paddle.to_tensor(np.ndarray([-1, -2]))
            paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')
        self.assertRaises(ValueError, test_label_value_error)

        def test_input_type_error():
            if False:
                while True:
                    i = 10
            input1 = paddle.to_tensor(self.input1_np.astype(np.int64))
            paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')
        self.assertRaises(ValueError, test_input_type_error)

        def test_label_type_error():
            if False:
                for i in range(10):
                    print('nop')
            label = paddle.to_tensor(self.label_np.astype(np.int16))
            paddle.nn.functional.cosine_embedding_loss(input1, input2, label, margin=0.5, reduction='mean')
        self.assertRaises(ValueError, test_label_type_error)

class TestClassCosineEmbeddingLoss(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.input1_np = np.random.random(size=(10, 3)).astype(np.float32)
        self.input2_np = np.random.random(size=(10, 3)).astype(np.float32)
        a = np.array([-1, -1, -1, -1, -1]).astype(np.int64)
        b = np.array([1, 1, 1, 1, 1]).astype(np.int64)
        self.label_np = np.concatenate((a, b), axis=0)
        np.random.shuffle(self.label_np)
        self.input1_np_1D = np.random.random(size=10).astype(np.float32)
        self.input2_np_1D = np.random.random(size=10).astype(np.float32)
        self.label_np_1D = np.array([1]).astype(np.int64)

    def run_dynamic(self):
        if False:
            return 10
        input1 = paddle.to_tensor(self.input1_np)
        input2 = paddle.to_tensor(self.input2_np)
        label = paddle.to_tensor(self.label_np)
        CosineEmbeddingLoss = paddle.nn.CosineEmbeddingLoss(margin=0.5, reduction='mean')
        dy_result = CosineEmbeddingLoss(input1, input2, label)
        expected1 = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='mean')
        np.testing.assert_allclose(dy_result.numpy(), expected1, rtol=1e-05)
        self.assertEqual(dy_result.shape, [])
        input1_1D = paddle.to_tensor(self.input1_np_1D)
        input2_1D = paddle.to_tensor(self.input2_np_1D)
        label_1D = paddle.to_tensor(self.label_np_1D)
        dy_result = CosineEmbeddingLoss(input1_1D, input2_1D, label_1D)
        expected2 = cosine_embedding_loss(self.input1_np_1D, self.input2_np_1D, self.label_np_1D, margin=0.5, reduction='mean')
        np.testing.assert_allclose(dy_result.numpy(), expected2, rtol=1e-05)

    def run_static(self):
        if False:
            for i in range(10):
                print('nop')
        input1 = static.data(name='input1', shape=[10, 3], dtype='float32')
        input2 = static.data(name='input2', shape=[10, 3], dtype='float32')
        label = static.data(name='label', shape=[10], dtype='int64')
        CosineEmbeddingLoss = paddle.nn.CosineEmbeddingLoss(margin=0.5, reduction='mean')
        result = CosineEmbeddingLoss(input1, input2, label)
        place = paddle.CPUPlace()
        exe = static.Executor(place)
        exe.run(static.default_startup_program())
        static_result = exe.run(feed={'input1': self.input1_np, 'input2': self.input2_np, 'label': self.label_np}, fetch_list=[result])
        expected = cosine_embedding_loss(self.input1_np, self.input2_np, self.label_np, margin=0.5, reduction='mean')
        np.testing.assert_allclose(static_result[0], expected, rtol=1e-05)

    def test_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static(place=paddle.CPUPlace())
        self.run_dynamic()
        paddle.enable_static()
        with static.program_guard(static.Program()):
            self.run_static()

    def test_errors(self):
        if False:
            print('Hello World!')

        def test_margin_error():
            if False:
                i = 10
                return i + 15
            CosineEmbeddingLoss = paddle.nn.CosineEmbeddingLoss(margin=2, reduction='mean')
        self.assertRaises(ValueError, test_margin_error)

        def test_reduction_error():
            if False:
                return 10
            CosineEmbeddingLoss = paddle.nn.CosineEmbeddingLoss(margin=2, reduction='reduce_mean')
        self.assertRaises(ValueError, test_reduction_error)
if __name__ == '__main__':
    unittest.main()