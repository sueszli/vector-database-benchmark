import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base.dygraph.base import to_variable
from paddle.nn import Linear

class Test_Detach(unittest.TestCase):

    def generate_Data(self):
        if False:
            return 10
        data = np.array([[1, 8, 3, 9], [7, 20, 9, 6], [4, 6, 8, 10]]).astype('float32')
        return data

    def no_detach_multi(self):
        if False:
            while True:
                i = 10
        data = self.generate_Data()
        with base.dygraph.guard():
            linear_w_param_attrs = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(5.0))
            linear_b_param_attrs = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(6.0))
            linear = Linear(4, 10, weight_attr=linear_w_param_attrs, bias_attr=linear_b_param_attrs)
            linear1_w_param_attrs = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(7.0))
            linear1_b_param_attrs = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(8.0))
            linear1 = Linear(10, 1, weight_attr=linear1_w_param_attrs, bias_attr=linear1_b_param_attrs)
            linear2_w_param_attrs = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(9.0))
            linear2_b_param_attrs = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(10.0))
            linear2 = Linear(10, 1, weight_attr=linear2_w_param_attrs, bias_attr=linear2_b_param_attrs)
            data = to_variable(data)
            x = linear(data)
            x1 = linear1(x)
            x2 = linear2(x)
            loss = x1 + x2
            loss.backward()
            return x.gradient()

    def no_detach_single(self):
        if False:
            i = 10
            return i + 15
        data = self.generate_Data()
        with base.dygraph.guard():
            linear_w_param_attrs = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(5.0))
            linear_b_param_attrs = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(6.0))
            linear = Linear(4, 10, weight_attr=linear_w_param_attrs, bias_attr=linear_b_param_attrs)
            linear1_w_param_attrs = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(7.0))
            linear1_b_param_attrs = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(8.0))
            linear1 = Linear(10, 1, weight_attr=linear1_w_param_attrs, bias_attr=linear1_b_param_attrs)
            data = to_variable(data)
            x = linear(data)
            x.retain_grads()
            x1 = linear1(x)
            loss = x1
            loss.backward()
            return x.gradient()

    def detach_multi(self):
        if False:
            print('Hello World!')
        data = self.generate_Data()
        with base.dygraph.guard():
            linear_w_param_attrs = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(5.0))
            linear_b_param_attrs = base.ParamAttr(initializer=paddle.nn.initializer.Constant(6.0))
            linear = Linear(4, 10, weight_attr=linear_w_param_attrs, bias_attr=linear_b_param_attrs)
            linear1_w_param_attrs = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(7.0))
            linear1_b_param_attrs = base.ParamAttr(initializer=paddle.nn.initializer.Constant(8.0))
            linear1 = Linear(10, 1, weight_attr=linear1_w_param_attrs, bias_attr=linear1_b_param_attrs)
            linear2_w_param_attrs = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(9.0))
            linear2_b_param_attrs = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(10.0))
            linear2 = Linear(10, 1, weight_attr=linear2_w_param_attrs, bias_attr=linear2_b_param_attrs)
            data = to_variable(data)
            x = linear(data)
            x.retain_grads()
            x_detach = x.detach()
            x1 = linear1(x)
            x2 = linear2(x_detach)
            loss = x1 + x2
            loss.backward()
            return x.gradient()

    def test_NoDetachMulti_DetachMulti(self):
        if False:
            while True:
                i = 10
        array_no_detach_multi = self.no_detach_multi()
        array_detach_multi = self.detach_multi()
        assert not np.array_equal(array_no_detach_multi, array_detach_multi)

    def test_NoDetachSingle_DetachMulti(self):
        if False:
            print('Hello World!')
        array_no_detach_single = self.no_detach_single()
        array_detach_multi = self.detach_multi()
        np.testing.assert_array_equal(array_no_detach_single, array_detach_multi)

class TestInplace(unittest.TestCase):

    def test_forward_version(self):
        if False:
            i = 10
            return i + 15
        with paddle.base.dygraph.guard():
            var = paddle.to_tensor(np.ones((4, 2, 3)).astype(np.float32))
            self.assertEqual(var.inplace_version, 0)
            detach_var_1 = var.detach()
            self.assertEqual(detach_var_1.inplace_version, 0)
            var[0] = 1.1
            self.assertEqual(var.inplace_version, 1)
            detach_var_2 = var.detach()
            self.assertEqual(detach_var_2.inplace_version, 1)
            var[0] = 3
            self.assertEqual(detach_var_1.inplace_version, 2)
            self.assertEqual(detach_var_2.inplace_version, 2)

    def test_backward_error(self):
        if False:
            for i in range(10):
                print('nop')
        with paddle.base.dygraph.guard():
            var_a = paddle.ones(shape=[4, 2, 3], dtype='float32')
            var_a.stop_gradient = False
            var_b = var_a ** 2
            var_c = var_b ** 2
            detach_var_b = var_b.detach()
            detach_var_b[1:2] = 3.3
            var_d = var_b ** 2
            loss = paddle.nn.functional.relu(var_c + var_d)
            with self.assertRaisesRegex(RuntimeError, f'received tensor_version:{1} != wrapper_version_snapshot:{0}'):
                loss.backward()
if __name__ == '__main__':
    unittest.main()