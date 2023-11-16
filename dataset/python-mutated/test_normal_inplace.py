import unittest
import numpy as np
import paddle
from paddle import base

def output_hist(out):
    if False:
        print('Hello World!')
    (hist, _) = np.histogram(out, range=(-1, 1))
    hist = hist.astype('float32')
    hist /= float(out.size)
    prob = 0.1 * np.ones(10)
    return (hist, prob)

class TestNormalRandomInplaceOpDtype(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.shape = (1000, 784)

    def test_normal_inplace_op_dtype(self):
        if False:
            i = 10
            return i + 15

        def test_fp32():
            if False:
                for i in range(10):
                    print('nop')
            tensor_fp32 = paddle.ones(self.shape, dtype=paddle.float32)
            tensor_fp32.normal_()
            self.assertEqual(tensor_fp32.dtype, paddle.float32)

        def test_fp64():
            if False:
                print('Hello World!')
            tensor_fp64 = paddle.ones(self.shape, paddle.float64)
            tensor_fp64.normal_()
            self.assertEqual(tensor_fp64.dtype, paddle.float64)
        places = ['cpu']
        if base.core.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            test_fp32()
            test_fp64()

class TestNormalRandomInplaceOpIsInplace(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.shape = (1000, 784)

    def test_normal_inplace_op_is_inplace(self):
        if False:
            for i in range(10):
                print('nop')
        tensor_a = paddle.ones(self.shape)
        tensor_b = tensor_a.normal_()
        self.assertTrue(tensor_a is tensor_b)

class TestNormalRandomInplaceOpSeedIsZero(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.shape = (1000, 784)

    def test_normal_inplace_op_not_equal(self):
        if False:
            for i in range(10):
                print('nop')
        tensor = paddle.ones(self.shape)
        tensor.normal_()
        tensor_data_first = tensor.numpy()
        tensor.normal_()
        tensor_data_second = tensor.numpy()
        self.assertFalse((tensor_data_first == tensor_data_second).all())

class TestNormalRandomInplaceOpShape(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.shape = (1000, 784)

    def test_normal_inplace_op_shape(self):
        if False:
            return 10
        tensor = paddle.ones(self.shape)
        tensor.normal_()
        tensor_shape_np = np.array(tensor.shape)
        origin_shape = np.array(self.shape)
        self.assertTrue((tensor_shape_np == origin_shape).all())

class TestNormalRandomInplaceOpDistribution(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.shape = (1000, 784)
        self.mean = -3
        self.std = 5

    def test_normal_inplace_op_distribution(self):
        if False:
            while True:
                i = 10
        tensor = paddle.ones(self.shape)
        tensor.normal_(self.mean, self.std)
        ones = paddle.ones(self.shape)
        zeros = paddle.zeros(self.shape)
        all_num = self.shape[0] * self.shape[1]
        std_probs = [0.68, 0.95, 0.997]
        for (index, prob) in enumerate(std_probs):
            left = self.mean - (index + 1) * self.std
            right = self.mean + (index + 1) * self.std
            cond = paddle.logical_and(tensor >= left, tensor <= right)
            c_sum = paddle.where(cond, ones, zeros).sum()
            np.testing.assert_allclose(c_sum / all_num, prob, 0.01)

class TestNormalRandomInplaceOpEmptyTensor(unittest.TestCase):

    def test_normal_inplace_op_empty_tensor(self):
        if False:
            print('Hello World!')
        places = ['cpu']
        if base.core.is_compiled_with_cuda():
            places.append('gpu')
        test_shapes = [(200, 0), (0, 200)]
        for place in places:
            paddle.set_device(place)
            for test_shape in test_shapes:
                tensor = paddle.empty(shape=test_shape)
                tensor.normal_()
                tensor_shape_np = np.array(tensor.shape)
                origin_shape = np.array(test_shape)
                self.assertTrue((tensor_shape_np == origin_shape).all())

class TestNormalRandomInplaceGrad(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.shape = (1000, 784)

    def run_(self):
        if False:
            while True:
                i = 10

        def test_grad():
            if False:
                i = 10
                return i + 15
            tensor_a = paddle.ones(self.shape)
            tensor_a.stop_gradient = False
            tensor_b = tensor_a * 0.5
            tensor_b.retain_grads()
            tensor_b.normal_(mean=-2, std=2)
            loss = tensor_b.sum()
            loss.backward()
            normal_grad = tensor_b.grad.numpy()
            self.assertTrue((normal_grad == 0).all())
        places = ['cpu']
        if base.core.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            test_grad()

    def test_normal_inplace_grad(self):
        if False:
            i = 10
            return i + 15
        self.run_()
if __name__ == '__main__':
    unittest.main()