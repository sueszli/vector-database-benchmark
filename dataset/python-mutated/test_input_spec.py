import os
import tempfile
import unittest
import numpy as np
import paddle
from paddle.base import core
from paddle.base.framework import convert_np_dtype_to_dtype_
from paddle.jit.dy2static.utils import _compatible_non_tensor_spec
from paddle.static import InputSpec

class TestInputSpec(unittest.TestCase):

    def test_default(self):
        if False:
            for i in range(10):
                print('nop')
        tensor_spec = InputSpec([3, 4])
        self.assertEqual(tensor_spec.dtype, convert_np_dtype_to_dtype_('float32'))
        self.assertIsNone(tensor_spec.name)

    def test_from_tensor(self):
        if False:
            i = 10
            return i + 15
        x_bool = paddle.tensor.fill_constant(shape=[1], dtype='bool', value=True)
        bool_spec = InputSpec.from_tensor(x_bool)
        self.assertEqual(bool_spec.dtype, x_bool.dtype)
        self.assertEqual(list(bool_spec.shape), list(x_bool.shape))
        self.assertEqual(bool_spec.name, x_bool.name)
        bool_spec2 = InputSpec.from_tensor(x_bool, name='bool_spec')
        self.assertEqual(bool_spec2.name, bool_spec2.name)

    def test_from_numpy(self):
        if False:
            i = 10
            return i + 15
        x_numpy = np.ones([10, 12])
        x_np_spec = InputSpec.from_numpy(x_numpy)
        self.assertEqual(x_np_spec.dtype, convert_np_dtype_to_dtype_(x_numpy.dtype))
        self.assertEqual(x_np_spec.shape, x_numpy.shape)
        self.assertIsNone(x_np_spec.name)
        x_numpy2 = np.array([1, 2, 3, 4]).astype('int64')
        x_np_spec2 = InputSpec.from_numpy(x_numpy2, name='x_np_int64')
        self.assertEqual(x_np_spec2.dtype, convert_np_dtype_to_dtype_(x_numpy2.dtype))
        self.assertEqual(x_np_spec2.shape, x_numpy2.shape)
        self.assertEqual(x_np_spec2.name, 'x_np_int64')

    def test_shape_with_none(self):
        if False:
            i = 10
            return i + 15
        tensor_spec = InputSpec([None, 4, None], dtype='int8', name='x_spec')
        self.assertEqual(tensor_spec.dtype, convert_np_dtype_to_dtype_('int8'))
        self.assertEqual(tensor_spec.name, 'x_spec')
        self.assertEqual(tensor_spec.shape, (-1, 4, -1))

    def test_shape_raise_error(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            tensor_spec = InputSpec(['None', 4, None], dtype='int8')
        with self.assertRaises(TypeError):
            tensor_spec = InputSpec(4, dtype='int8')

    def test_batch_and_unbatch(self):
        if False:
            while True:
                i = 10
        tensor_spec = InputSpec([10])
        batch_tensor_spec = tensor_spec.batch(16)
        self.assertEqual(batch_tensor_spec.shape, (16, 10))
        unbatch_spec = batch_tensor_spec.unbatch()
        self.assertEqual(unbatch_spec.shape, (10,))
        with self.assertRaises(ValueError):
            tensor_spec.batch([16, 12])
        with self.assertRaises(TypeError):
            tensor_spec.batch('16')

    def test_eq_and_hash(self):
        if False:
            while True:
                i = 10
        tensor_spec_1 = InputSpec([10, 16], dtype='float32')
        tensor_spec_2 = InputSpec([10, 16], dtype='float32')
        tensor_spec_3 = InputSpec([10, 16], dtype='float32', name='x')
        tensor_spec_4 = InputSpec([16], dtype='float32', name='x')
        self.assertTrue(tensor_spec_1 == tensor_spec_2)
        self.assertTrue(tensor_spec_1 != tensor_spec_3)
        self.assertTrue(tensor_spec_3 != tensor_spec_4)
        self.assertTrue(hash(tensor_spec_1) == hash(tensor_spec_2))
        self.assertTrue(hash(tensor_spec_1) == hash(tensor_spec_3))
        self.assertTrue(hash(tensor_spec_3) != hash(tensor_spec_4))

class NetWithNonTensorSpec(paddle.nn.Layer):

    def __init__(self, in_num, out_num):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.linear_1 = paddle.nn.Linear(in_num, out_num)
        self.bn_1 = paddle.nn.BatchNorm1D(out_num)
        self.linear_2 = paddle.nn.Linear(in_num, out_num)
        self.bn_2 = paddle.nn.BatchNorm1D(out_num)
        self.linear_3 = paddle.nn.Linear(in_num, out_num)
        self.bn_3 = paddle.nn.BatchNorm1D(out_num)

    def forward(self, x, bool_v=False, str_v='bn', int_v=1, list_v=None):
        if False:
            return 10
        x = self.linear_1(x)
        if 'bn' in str_v:
            x = self.bn_1(x)
        if bool_v:
            x = self.linear_2(x)
            x = self.bn_2(x)
        config = {'int_v': int_v, 'other_key': 'value'}
        if list_v and list_v[-1] > 2:
            x = self.linear_3(x)
            x = self.another_func(x, config)
        out = paddle.mean(x)
        return out

    def another_func(self, x, config=None):
        if False:
            return 10
        use_bn = config['int_v'] > 0
        x = self.linear_1(x)
        if use_bn:
            x = self.bn_3(x)
        return x

class TestNetWithNonTensorSpec(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.in_num = 16
        self.out_num = 16
        self.x_spec = paddle.static.InputSpec([-1, 16], name='x')
        self.x = paddle.randn([4, 16])
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        if False:
            return 10
        self.temp_dir.cleanup()

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        paddle.disable_static()

    def test_non_tensor_bool(self):
        if False:
            i = 10
            return i + 15
        specs = [self.x_spec, False]
        self.check_result(specs, 'bool')

    def test_non_tensor_str(self):
        if False:
            while True:
                i = 10
        specs = [self.x_spec, True, 'xxx']
        self.check_result(specs, 'str')

    def test_non_tensor_int(self):
        if False:
            return 10
        specs = [self.x_spec, True, 'bn', 10]
        self.check_result(specs, 'int')

    def test_non_tensor_list(self):
        if False:
            i = 10
            return i + 15
        specs = [self.x_spec, False, 'bn', -10, [4]]
        self.check_result(specs, 'list')

    def check_result(self, specs, path):
        if False:
            i = 10
            return i + 15
        path = os.path.join(self.temp_dir.name, './net_non_tensor_', path)
        net = NetWithNonTensorSpec(self.in_num, self.out_num)
        net.eval()
        dy_out = net(self.x, *specs[1:])
        paddle.jit.save(net, path + '_direct', input_spec=specs)
        load_net = paddle.jit.load(path + '_direct')
        load_net.eval()
        pred_out = load_net(self.x)
        np.testing.assert_allclose(dy_out, pred_out, rtol=1e-05)
        net = paddle.jit.to_static(net, input_spec=specs, full_graph=True)
        st_out = net(self.x, *specs[1:])
        np.testing.assert_allclose(dy_out, st_out, rtol=1e-05)
        paddle.jit.save(net, path)
        load_net = paddle.jit.load(path)
        load_net.eval()
        load_out = load_net(self.x)
        np.testing.assert_allclose(st_out, load_out, rtol=1e-05)

    def test_spec_compatible(self):
        if False:
            i = 10
            return i + 15
        net = NetWithNonTensorSpec(self.in_num, self.out_num)
        specs = [self.x_spec, False, 'bn', -10]
        net = paddle.jit.to_static(net, input_spec=specs, full_graph=True)
        net.eval()
        path = os.path.join(self.temp_dir.name, './net_twice')
        new_specs = [self.x_spec, True, 'bn', 10]
        with self.assertRaises(ValueError):
            paddle.jit.save(net, path, input_spec=new_specs)
        dy_out = net(self.x)
        paddle.jit.save(net, path, [self.x_spec, False, 'bn'])
        load_net = paddle.jit.load(path)
        load_net.eval()
        pred_out = load_net(self.x)
        np.testing.assert_allclose(dy_out, pred_out, rtol=1e-05)

class NetWithNonTensorSpecPrune(paddle.nn.Layer):

    def __init__(self, in_num, out_num):
        if False:
            while True:
                i = 10
        super().__init__()
        self.linear_1 = paddle.nn.Linear(in_num, out_num)
        self.bn_1 = paddle.nn.BatchNorm1D(out_num)

    def forward(self, x, y, use_bn=False):
        if False:
            while True:
                i = 10
        x = self.linear_1(x)
        if use_bn:
            x = self.bn_1(x)
        out = paddle.mean(x)
        if y is not None:
            loss = paddle.mean(y) + out
        return (out, loss)

class TestNetWithNonTensorSpecWithPrune(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.in_num = 16
        self.out_num = 16
        self.x_spec = paddle.static.InputSpec([-1, 16], name='x')
        self.y_spec = paddle.static.InputSpec([16], name='y')
        self.x = paddle.randn([4, 16])
        self.y = paddle.randn([16])
        self.temp_dir = tempfile.TemporaryDirectory()

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()

    def test_non_tensor_with_prune(self):
        if False:
            print('Hello World!')
        specs = [self.x_spec, self.y_spec, True]
        path = os.path.join(self.temp_dir.name, './net_non_tensor_prune_')
        net = NetWithNonTensorSpecPrune(self.in_num, self.out_num)
        net.eval()
        (dy_out, _) = net(self.x, self.y, *specs[2:])
        paddle.jit.save(net, path + '_direct', input_spec=specs)
        load_net = paddle.jit.load(path + '_direct')
        load_net.eval()
        (pred_out, _) = load_net(self.x, self.y)
        np.testing.assert_allclose(dy_out, pred_out, rtol=1e-05)
        net = paddle.jit.to_static(net, input_spec=specs, full_graph=True)
        (st_out, _) = net(self.x, self.y, *specs[2:])
        np.testing.assert_allclose(dy_out, st_out, rtol=1e-05)
        prune_specs = [self.x_spec, True]
        paddle.jit.save(net, path, prune_specs, output_spec=[st_out], input_names_after_prune=[self.x_spec.name])
        load_net = paddle.jit.load(path)
        load_net.eval()
        load_out = load_net(self.x)
        np.testing.assert_allclose(st_out, load_out, rtol=1e-05)

class UnHashableObject:

    def __init__(self, val):
        if False:
            print('Hello World!')
        self.val = val

    def __hash__(self):
        if False:
            while True:
                i = 10
        raise TypeError('Unsupported to call hash()')

class TestCompatibleNonTensorSpec(unittest.TestCase):

    def test_case(self):
        if False:
            return 10
        self.assertTrue(_compatible_non_tensor_spec([1, 2, 3], [1, 2, 3]))
        self.assertFalse(_compatible_non_tensor_spec([1, 2, 3], [1, 2]))
        self.assertFalse(_compatible_non_tensor_spec([1, 2, 3], [1, 3, 2]))
        self.assertTrue(_compatible_non_tensor_spec(UnHashableObject(1), UnHashableObject(1)))

class NegSpecNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.linear = paddle.nn.Linear(10, 5)

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.linear(x)

class TestNegSpecWithPrim(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        core._set_prim_all_enabled(True)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        core._set_prim_all_enabled(False)

    def test_run(self):
        if False:
            print('Hello World!')
        net = NegSpecNet()
        net = paddle.jit.to_static(net, input_spec=[paddle.static.InputSpec(shape=[-1, 10])], full_graph=True)
        x = paddle.randn([2, 10])
        out = net(x)
        np.testing.assert_equal(net.forward._input_spec, None)
if __name__ == '__main__':
    unittest.main()