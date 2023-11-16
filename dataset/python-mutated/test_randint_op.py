import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle import base
from paddle.base import core
from paddle.static import Program, program_guard
paddle.enable_static()

def output_hist(out):
    if False:
        return 10
    (hist, _) = np.histogram(out, range=(-10, 10))
    hist = hist.astype('float32')
    hist /= float(out.size)
    prob = 0.1 * np.ones(10)
    return (hist, prob)

class TestRandintOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'randint'
        self.python_api = paddle.randint
        self.inputs = {}
        self.init_attrs()
        self.outputs = {'Out': np.zeros((10000, 784)).astype('float32')}

    def init_attrs(self):
        if False:
            return 10
        self.attrs = {'shape': [10000, 784], 'low': -10, 'high': 10, 'seed': 10}
        self.output_hist = output_hist

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        if False:
            for i in range(10):
                print('nop')
        (hist, prob) = self.output_hist(np.array(outs[0]))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.001)

class TestRandintOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        with program_guard(Program(), Program()):
            self.assertRaises(TypeError, paddle.randint, 5, shape=np.array([2]))
            self.assertRaises(TypeError, paddle.randint, 5, dtype='float32')
            self.assertRaises(ValueError, paddle.randint, 5, 5)
            self.assertRaises(ValueError, paddle.randint, -5)
            self.assertRaises(TypeError, paddle.randint, 5, shape=['2'])
            shape_tensor = paddle.static.data('X', [1])
            self.assertRaises(TypeError, paddle.randint, 5, shape=shape_tensor)
            self.assertRaises(TypeError, paddle.randint, 5, shape=[shape_tensor])

class TestRandintOp_attr_tensorlist(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'randint'
        self.python_api = paddle.randint
        self.new_shape = (10000, 784)
        shape_tensor = []
        for (index, ele) in enumerate(self.new_shape):
            shape_tensor.append(('x' + str(index), np.ones(1).astype('int64') * ele))
        self.inputs = {'ShapeTensorList': shape_tensor}
        self.init_attrs()
        self.outputs = {'Out': np.zeros((10000, 784)).astype('int32')}

    def init_attrs(self):
        if False:
            return 10
        self.attrs = {'low': -10, 'high': 10, 'seed': 10}
        self.output_hist = output_hist

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        if False:
            while True:
                i = 10
        (hist, prob) = self.output_hist(np.array(outs[0]))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.001)

class TestRandint_attr_tensor(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'randint'
        self.python_api = paddle.randint
        self.inputs = {'ShapeTensor': np.array([10000, 784]).astype('int64')}
        self.init_attrs()
        self.outputs = {'Out': np.zeros((10000, 784)).astype('int64')}

    def init_attrs(self):
        if False:
            return 10
        self.attrs = {'low': -10, 'high': 10, 'seed': 10}
        self.output_hist = output_hist

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        if False:
            i = 10
            return i + 15
        (hist, prob) = self.output_hist(np.array(outs[0]))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.001)

class TestRandintAPI(unittest.TestCase):

    def test_api(self):
        if False:
            for i in range(10):
                print('nop')
        with program_guard(Program(), Program()):
            out1 = paddle.randint(5)
            out2 = paddle.randint(low=-100, high=100, shape=[64, 64], dtype='int32')
            out3 = paddle.randint(low=-100, high=100, shape=(32, 32, 3), dtype='int64')
            dim_1 = paddle.tensor.fill_constant([1], 'int64', 32)
            dim_2 = paddle.tensor.fill_constant([1], 'int32', 50)
            out4 = paddle.randint(low=-100, high=100, shape=[dim_1, 5, dim_2], dtype='int32')
            var_shape = paddle.static.data(name='var_shape', shape=[2], dtype='int64')
            out5 = paddle.randint(low=1, high=1000, shape=var_shape, dtype='int64')
            place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            outs = exe.run(feed={'var_shape': np.array([100, 100]).astype('int64')}, fetch_list=[out1, out2, out3, out4, out5])

class TestRandintImperative(unittest.TestCase):

    def test_case(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        n = 10
        x1 = paddle.randint(n, shape=[10], dtype='int32')
        x2 = paddle.tensor.randint(n)
        x3 = paddle.tensor.random.randint(n)
        for i in [x1, x2, x3]:
            for j in i.numpy().tolist():
                self.assertTrue(j >= 0 and j < n)
        paddle.enable_static()

class TestRandomValue(unittest.TestCase):

    def test_fixed_random_number(self):
        if False:
            for i in range(10):
                print('nop')
        if not paddle.is_compiled_with_cuda():
            return
        if 'V100' not in paddle.device.cuda.get_device_name():
            return
        print('Test Fixed Random number on GPU------>')
        paddle.disable_static()
        self.run_test_case()
        paddle.enable_static()

    def run_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.set_device('gpu')
        paddle.seed(100)
        x = paddle.randint(-10000, 10000, [32, 3, 1024, 1024], dtype='int32').numpy()
        self.assertTrue(x.mean(), -0.7517569760481516)
        self.assertTrue(x.std(), 5773.696619107639)
        expect = [2535, 2109, 5916, -5011, -261]
        np.testing.assert_array_equal(x[10, 0, 100, 100:105], expect)
        expect = [3465, 7206, -8660, -9628, -6574]
        np.testing.assert_array_equal(x[20, 1, 600, 600:605], expect)
        expect = [881, 1560, 1100, 9664, 1669]
        np.testing.assert_array_equal(x[30, 2, 1000, 1000:1005], expect)
        x = paddle.randint(-10000, 10000, [32, 3, 1024, 1024], dtype='int64').numpy()
        self.assertTrue(x.mean(), -1.461287518342336)
        self.assertTrue(x.std(), 5773.023477548159)
        expect = [7213, -9597, 754, 8129, -1158]
        np.testing.assert_array_equal(x[10, 0, 100, 100:105], expect)
        expect = [-7159, 8054, 7675, 6980, 8506]
        np.testing.assert_array_equal(x[20, 1, 600, 600:605], expect)
        expect = [3581, 3420, -8027, -5237, -2436]
        np.testing.assert_array_equal(x[30, 2, 1000, 1000:1005], expect)

class TestRandintAPI_ZeroDim(unittest.TestCase):

    def test_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        x = paddle.randint(0, 2, [])
        self.assertEqual(x.shape, [])
        paddle.enable_static()

    def test_static(self):
        if False:
            while True:
                i = 10
        with base.program_guard(base.Program(), base.Program()):
            x = paddle.randint(-10, 10, [])
            self.assertEqual(x.shape, ())
            exe = base.Executor()
            result = exe.run(fetch_list=[x])
            self.assertEqual(result[0].shape, ())
        paddle.enable_static()
if __name__ == '__main__':
    unittest.main()