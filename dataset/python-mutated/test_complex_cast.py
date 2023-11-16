import unittest
import numpy as np
import paddle

class TestComplexCastOp(unittest.TestCase):

    def test_complex_to_real(self):
        if False:
            while True:
                i = 10
        r = np.random.random(size=[10, 10]) * 10
        i = np.random.random(size=[10, 10])
        c_t = paddle.to_tensor(r + i * 1j, dtype='complex64')
        self.assertEqual(c_t.cast('int64').dtype, paddle.int64)
        self.assertEqual(c_t.cast('int32').dtype, paddle.int32)
        self.assertEqual(c_t.cast('float32').dtype, paddle.float32)
        self.assertEqual(c_t.cast('float64').dtype, paddle.float64)
        self.assertEqual(c_t.cast('bool').dtype, paddle.bool)
        np.testing.assert_allclose(c_t.cast('int64').numpy(), r.astype('int64'), rtol=1e-05)
        np.testing.assert_allclose(c_t.cast('int32').numpy(), r.astype('int32'), rtol=1e-05)
        np.testing.assert_allclose(c_t.cast('float32').numpy(), r.astype('float32'), rtol=1e-05)
        np.testing.assert_allclose(c_t.cast('float64').numpy(), r.astype('float64'), rtol=1e-05)
        np.testing.assert_allclose(c_t.cast('bool').numpy(), r.astype('bool'), rtol=1e-05)

    def test_real_to_complex(self):
        if False:
            while True:
                i = 10
        r = np.random.random(size=[10, 10]) * 10
        r_t = paddle.to_tensor(r)
        self.assertEqual(r_t.cast('complex64').dtype, paddle.complex64)
        self.assertEqual(r_t.cast('complex128').dtype, paddle.complex128)
        np.testing.assert_allclose(r_t.cast('complex64').real().numpy(), r, rtol=1e-05)
        np.testing.assert_allclose(r_t.cast('complex128').real().numpy(), r, rtol=1e-05)

    def test_complex64_complex128(self):
        if False:
            for i in range(10):
                print('nop')
        r = np.random.random(size=[10, 10])
        i = np.random.random(size=[10, 10])
        c = r + i * 1j
        c_64 = paddle.to_tensor(c, dtype='complex64')
        c_128 = paddle.to_tensor(c, dtype='complex128')
        self.assertTrue(c_64.cast('complex128').dtype, paddle.complex128)
        self.assertTrue(c_128.cast('complex128').dtype, paddle.complex64)
        np.testing.assert_allclose(c_64.cast('complex128').numpy(), c_128.numpy(), rtol=1e-05)
        np.testing.assert_allclose(c_128.cast('complex128').numpy(), c_64.numpy(), rtol=1e-05)
if __name__ == '__main__':
    unittest.main()