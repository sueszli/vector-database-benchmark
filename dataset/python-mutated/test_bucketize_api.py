import unittest
import numpy as np
import paddle
from paddle.base import core
np.random.seed(10)

class TestBucketizeAPI(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.sorted_sequence = np.array([2, 4, 8, 16]).astype('float64')
        self.x = np.array([[0, 8, 4, 16], [-1, 2, 8, 4]]).astype('float64')
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_api_static(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()

        def run(place):
            if False:
                for i in range(10):
                    print('nop')
            with paddle.static.program_guard(paddle.static.Program()):
                sorted_sequence = paddle.static.data('SortedSequence', shape=self.sorted_sequence.shape, dtype='float64')
                x = paddle.static.data('x', shape=self.x.shape, dtype='float64')
                out1 = paddle.bucketize(x, sorted_sequence)
                out2 = paddle.bucketize(x, sorted_sequence, right=True)
                exe = paddle.static.Executor(place)
                res = exe.run(feed={'SortedSequence': self.sorted_sequence, 'x': self.x}, fetch_list=[out1, out2])
            out_ref = np.searchsorted(self.sorted_sequence, self.x)
            out_ref1 = np.searchsorted(self.sorted_sequence, self.x, side='right')
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)
            np.testing.assert_allclose(out_ref1, res[1], rtol=1e-05)
        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        if False:
            print('Hello World!')

        def run(place):
            if False:
                i = 10
                return i + 15
            paddle.disable_static(place)
            sorted_sequence = paddle.to_tensor(self.sorted_sequence)
            x = paddle.to_tensor(self.x)
            out1 = paddle.bucketize(x, sorted_sequence)
            out2 = paddle.bucketize(x, sorted_sequence, right=True)
            out_ref1 = np.searchsorted(self.sorted_sequence, self.x)
            out_ref2 = np.searchsorted(self.sorted_sequence, self.x, side='right')
            np.testing.assert_allclose(out_ref1, out1.numpy(), rtol=1e-05)
            np.testing.assert_allclose(out_ref2, out2.numpy(), rtol=1e-05)
            paddle.enable_static()
        for place in self.place:
            run(place)

    def test_out_int32(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        sorted_sequence = paddle.to_tensor(self.sorted_sequence)
        x = paddle.to_tensor(self.x)
        out = paddle.bucketize(x, sorted_sequence, out_int32=True)
        self.assertTrue(out.type, 'int32')

    def test_bucketize_dims_error(self):
        if False:
            i = 10
            return i + 15
        with paddle.static.program_guard(paddle.static.Program()):
            sorted_sequence = paddle.static.data('SortedSequence', shape=[2, 2], dtype='float64')
            x = paddle.static.data('x', shape=[2, 5], dtype='float64')
            self.assertRaises(ValueError, paddle.bucketize, x, sorted_sequence)

    def test_input_error(self):
        if False:
            for i in range(10):
                print('nop')
        for place in self.place:
            paddle.disable_static(place)
            sorted_sequence = paddle.to_tensor(self.sorted_sequence)
            self.assertRaises(ValueError, paddle.bucketize, self.x, sorted_sequence)

    def test_empty_input_error(self):
        if False:
            for i in range(10):
                print('nop')
        for place in self.place:
            paddle.disable_static(place)
            sorted_sequence = paddle.to_tensor(self.sorted_sequence)
            x = paddle.to_tensor(self.x)
            self.assertRaises(ValueError, paddle.bucketize, None, sorted_sequence)
            self.assertRaises(AttributeError, paddle.bucketize, x, None)
if __name__ == '__main__':
    unittest.main()