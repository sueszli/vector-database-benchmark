import unittest
import numpy as np
import paddle

class TestDygraphViewReuseAllocation(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_shape()

    def init_shape(self):
        if False:
            print('Hello World!')
        self.input_shape = [2, 3, 1]
        self.output_shape = [2, 3]

    def view_api_processing(self, var):
        if False:
            print('Hello World!')
        return paddle.squeeze(var)

    def test_view_api(self):
        if False:
            while True:
                i = 10
        var = paddle.rand(self.input_shape)
        view_var = self.view_api_processing(var)
        view_var[0] = 2.0
        self.assertEqual(var.shape, self.input_shape)
        self.assertEqual(view_var.shape, self.output_shape)
        var_numpy = var.numpy().reshape(self.output_shape)
        view_var_numpy = view_var.numpy()
        np.testing.assert_array_equal(var_numpy, view_var_numpy)

    def test_forward_version(self):
        if False:
            i = 10
            return i + 15
        var = paddle.rand(self.input_shape)
        self.assertEqual(var.inplace_version, 0)
        view_var = self.view_api_processing(var)
        self.assertEqual(view_var.inplace_version, 0)
        var[0] = 2.0
        self.assertEqual(var.inplace_version, 1)
        self.assertEqual(view_var.inplace_version, 1)
        view_var_2 = self.view_api_processing(var)
        self.assertEqual(view_var_2.inplace_version, 1)
        var[0] = 3.0
        self.assertEqual(view_var.inplace_version, 2)
        self.assertEqual(view_var_2.inplace_version, 2)

    def test_backward_error(self):
        if False:
            i = 10
            return i + 15
        with paddle.base.dygraph.guard():
            var_a = paddle.ones(shape=self.input_shape, dtype='float32')
            var_a.stop_gradient = False
            var_b = var_a ** 2
            var_c = var_b ** 2
            view_var_b = self.view_api_processing(var_b)
            view_var_b[0] = 2.0
            loss = paddle.nn.functional.relu(var_c)
            with self.assertRaisesRegex(RuntimeError, f'received tensor_version:{1} != wrapper_version_snapshot:{0}'):
                loss.backward()

class TestUnsqueezeDygraphViewReuseAllocation(TestDygraphViewReuseAllocation):

    def init_shape(self):
        if False:
            i = 10
            return i + 15
        self.input_shape = [2, 3]
        self.output_shape = [2, 3, 1]

    def view_api_processing(self, var):
        if False:
            i = 10
            return i + 15
        return paddle.unsqueeze(var, -1)

class TestReshapeDygraphViewReuseAllocation(TestDygraphViewReuseAllocation):

    def init_shape(self):
        if False:
            return 10
        self.input_shape = [3, 4]
        self.output_shape = [2, 2, 3]

    def view_api_processing(self, var):
        if False:
            i = 10
            return i + 15
        return paddle.reshape(var, [2, 2, 3])

class TestFlattenDygraphViewReuseAllocation(TestDygraphViewReuseAllocation):

    def init_shape(self):
        if False:
            return 10
        self.input_shape = [3, 4]
        self.output_shape = [12]

    def view_api_processing(self, var):
        if False:
            i = 10
            return i + 15
        return paddle.flatten(var)
if __name__ == '__main__':
    unittest.main()