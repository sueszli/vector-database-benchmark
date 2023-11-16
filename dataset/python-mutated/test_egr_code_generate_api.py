import unittest
import numpy as np
import paddle

class EagerOpAPIGenerateTestCase(unittest.TestCase):

    def test_elementwise_add(self):
        if False:
            while True:
                i = 10
        paddle.set_device('cpu')
        np_x = np.ones([4, 16, 16, 32]).astype('float32')
        np_y = np.ones([4, 16, 16, 32]).astype('float32')
        x = paddle.to_tensor(np_x)
        y = paddle.to_tensor(np_y)
        out = paddle.add(x, y)
        out_arr = out.numpy()
        out_arr_expected = np.add(np_x, np_y)
        np.testing.assert_array_equal(out_arr, out_arr_expected)

    def test_sum(self):
        if False:
            print('Hello World!')
        x_data = np.array([[0.2, 0.3, 0.5, 0.9], [0.1, 0.2, 0.6, 0.7]]).astype('float32')
        x = paddle.to_tensor(x_data, 'float32')
        out = paddle.sum(x, axis=0)
        out_arr = out.numpy()
        out_arr_expected = np.sum(x_data, axis=0)
        np.testing.assert_array_equal(out_arr, out_arr_expected)

    def test_mm(self):
        if False:
            i = 10
            return i + 15
        np_input = np.random.random([16, 32]).astype('float32')
        np_mat2 = np.random.random([32, 32]).astype('float32')
        input = paddle.to_tensor(np_input)
        mat2 = paddle.to_tensor(np_mat2)
        out = paddle.mm(input, mat2)
        out_arr = out.numpy()
        out_arr_expected = np.matmul(np_input, np_mat2)
        np.testing.assert_allclose(out_arr, out_arr_expected, rtol=1e-05)

    def test_sigmoid(self):
        if False:
            for i in range(10):
                print('nop')
        np_x = np.array([-0.4, -0.2, 0.1, 0.3]).astype('float32')
        x = paddle.to_tensor(np_x)
        out = paddle.nn.functional.sigmoid(x)
        out_arr = out.numpy()
        out_arr_expected = np.array([0.40131234, 0.450166, 0.52497919, 0.57444252]).astype('float32')
        np.testing.assert_allclose(out_arr, out_arr_expected, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()