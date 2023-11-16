from __future__ import print_function
import threading
import unittest
import mock
import numpy
import pytest
import six
import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr
import chainer.testing.backend
from chainer import utils
from chainer.utils import type_check
import chainerx
if chainerx.is_available():
    import chainerx.testing

def make_array(start, shape, dtype, device):
    if False:
        return 10
    size = numpy.product(shape, dtype='i')
    a = numpy.arange(start, start + size)
    a = a.reshape(shape)
    a = a.astype(dtype, copy=False)
    return device.send(a)

@testing.parameterize(*testing.product({'y_shape': [(4,), (0,), (2, 3), ()], 'x_shape': [(3,), (0,), (4, 1), ()]}))
class TestFunctionNode(unittest.TestCase):

    def _get_method(self, prefix, gpu):
        if False:
            return 10
        suffix = 'gpu' if gpu else 'cpu'
        return getattr(self.f, prefix + '_' + suffix)

    def _setup(self, device):
        if False:
            i = 10
            return i + 15
        y_shape = self.y_shape
        x_shape = self.x_shape
        y1 = make_array(1, y_shape, numpy.float32, device)
        y2 = make_array(2, y_shape, numpy.float32, device)
        gx1 = chainer.Variable(make_array(1, x_shape, numpy.float32, device))
        gx2 = None
        gy1 = make_array(1, y_shape, numpy.float32, device)
        gy2 = make_array(1, y_shape, numpy.float32, device)
        f = chainer.FunctionNode()
        f.check_type_forward = mock.MagicMock()
        f.forward_cpu = mock.MagicMock()
        f.forward_gpu = mock.MagicMock()
        f.backward = mock.MagicMock(return_value=(gx1, gx2))
        self.f = f
        self.x1 = make_array(0, x_shape, numpy.float32, device)
        self.x2 = make_array(0, x_shape, numpy.int32, device)
        self.y1 = y1
        self.y2 = y2
        self.gx1 = gx1
        self.gx2 = gx2
        self.gx1_orig = chainer.Variable(make_array(3, x_shape, numpy.float32, device))
        self.gx2_orig = chainer.Variable(make_array(2, x_shape, numpy.float32, device))
        self.gx1_accum = gx1 + self.gx1_orig
        self.gy1 = gy1
        self.gy2 = gy2

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.f = None
        self.y1 = None
        self.y2 = None
        self.gx1 = None

    def setup_cpu(self):
        if False:
            return 10
        self._setup(backend.CpuDevice())
        self.f.forward_cpu = mock.MagicMock(return_value=(self.y1, self.y2))

    def setup_gpu(self):
        if False:
            while True:
                i = 10
        self._setup(backend.GpuDevice.from_device_id(0))
        self.f.forward_gpu = mock.MagicMock(return_value=(self.y1, self.y2))

    def setup_chainerx(self, device_name='native:0'):
        if False:
            return 10
        self._setup(chainer.get_device(device_name))
        self.f.forward = mock.MagicMock(side_effect=lambda inputs: (utils.force_array(inputs[0] * inputs[1]), utils.force_array(inputs[0] + inputs[1])))

    def check_forward(self, gpu):
        if False:
            return 10
        (y1, y2) = self.f.forward((self.x1, self.x2))
        self.assertEqual(self.f.check_type_forward.call_count, 0)
        self.assertEqual(self._get_method('forward', not gpu).call_count, 0)
        self._get_method('forward', gpu).assert_called_once_with((self.x1, self.x2))
        self.assertTrue((cuda.to_cpu(y1) == cuda.to_cpu(self.y1)).all())
        self.assertTrue((cuda.to_cpu(y2) == cuda.to_cpu(self.y2)).all())

    def test_forward_cpu(self):
        if False:
            i = 10
            return i + 15
        self.setup_cpu()
        self.check_forward(False)

    @attr.gpu
    def test_forward_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_gpu()
        self.check_forward(True)

    def check_check_type_forward(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.f.check_type_forward.call_count, 1)
        ts = self.f.check_type_forward.call_args[0][0]
        self.assertIsInstance(ts, type_check.LightTypeInfoTuple)
        self.assertEqual(len(ts), 2)
        t1 = ts[0]
        assert t1.shape == self.x_shape
        assert t1.dtype == numpy.float32
        t2 = ts[1]
        assert t2.shape == self.x_shape
        assert t2.dtype == numpy.int32

    def check_apply(self):
        if False:
            i = 10
            return i + 15
        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2)
        x1._node._rank = 1
        x2._node._rank = 3
        ys = self.f.apply((x1, x2))
        self.assertEqual(len(ys), 2)
        self.check_check_type_forward()
        for y in ys:
            self.assertIsInstance(y, chainer.Variable)
            self.assertEqual(y.rank, 4)
            self.assertIs(y.creator_node, self.f)
            self.assertTrue(y.requires_grad)
        self.assertIsInstance(y.creator_node.outputs, tuple)

    def check_apply_chainerx(self):
        if False:
            return 10
        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2, requires_grad=False)
        ys = self.f.apply((x1, x2))
        self.assertEqual(len(ys), 2)
        self.check_check_type_forward()
        for y in ys:
            self.assertIsInstance(y, chainer.Variable)
            self.assertIsInstance(y.data, chainerx.ndarray)
            self.assertIs(y.data.device, self.x1.device)
            self.assertTrue(y.requires_grad)

    def test_apply_cpu(self):
        if False:
            print('Hello World!')
        self.setup_cpu()
        self.check_apply()

    @attr.gpu
    def test_apply_gpu(self):
        if False:
            while True:
                i = 10
        self.setup_gpu()
        self.check_apply()

    @attr.chainerx
    def test_apply_chainerx_cpu(self):
        if False:
            return 10
        self.setup_chainerx()
        self.check_apply_chainerx()

    @attr.chainerx
    @attr.gpu
    def test_apply_chainerx_gpu(self):
        if False:
            while True:
                i = 10
        self.setup_chainerx('cuda:0')
        self.check_apply_chainerx()

    @attr.chainerx
    @attr.multi_gpu(2)
    def test_apply_chainerx_multi_gpu(self):
        if False:
            print('Hello World!')
        self.setup_chainerx('cuda:1')
        self.check_apply_chainerx()

    def check_apply_all_ndarray(self):
        if False:
            return 10
        x1 = self.x1
        x2 = self.x2
        ys = self.f.apply((x1, x2))
        self.assertEqual(len(ys), 2)
        self.check_check_type_forward()
        xp = backend.get_array_module(x1)
        for y in ys:
            self.assertIsInstance(y, chainer.Variable)
            self.assertIsInstance(y.data, xp.ndarray)
            self.assertFalse(y.requires_grad)

    def test_apply_all_ndarray_cpu(self):
        if False:
            print('Hello World!')
        self.setup_cpu()
        self.check_apply_all_ndarray()

    @attr.gpu
    def test_apply_all_ndarray_gpu(self):
        if False:
            while True:
                i = 10
        self.setup_gpu()
        self.check_apply_all_ndarray()

    @attr.chainerx
    def test_apply_all_ndarray_chainerx_cpu(self):
        if False:
            i = 10
            return i + 15
        self.setup_chainerx()
        self.check_apply_all_ndarray()

    @attr.chainerx
    @attr.gpu
    def test_apply_all_ndarray_chainerx_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_chainerx('cuda:0')
        self.check_apply_all_ndarray()

    def check_apply_ndarray(self):
        if False:
            print('Hello World!')
        x1 = chainer.Variable(self.x1)
        x2 = self.x2
        x1._node._rank = 1
        ys = self.f.apply((x1, x2))
        self.assertEqual(len(ys), 2)
        self.check_check_type_forward()
        for y in ys:
            self.assertIsInstance(y, chainer.Variable)
            self.assertEqual(y.rank, 2)
            self.assertIs(y.creator_node, self.f)
            self.assertTrue(y.requires_grad)
        self.assertIsInstance(y.creator_node.outputs, tuple)

    def check_apply_ndarray_chainerx(self):
        if False:
            for i in range(10):
                print('nop')
        x1 = chainer.Variable(self.x1)
        x2 = self.x2
        ys = self.f.apply((x1, x2))
        self.assertEqual(len(ys), 2)
        self.check_check_type_forward()
        for y in ys:
            self.assertIsInstance(y, chainer.Variable)
            self.assertIsInstance(y.data, chainerx.ndarray)
            self.assertIs(y.data.device, self.x1.device)
            self.assertTrue(y.requires_grad)

    def test_apply_ndarray_cpu(self):
        if False:
            i = 10
            return i + 15
        self.setup_cpu()
        self.check_apply_ndarray()

    @attr.gpu
    def test_apply_ndarray_gpu(self):
        if False:
            while True:
                i = 10
        self.setup_gpu()
        self.check_apply_ndarray()

    @attr.chainerx
    def test_apply_ndarray_chainerx_cpu(self):
        if False:
            while True:
                i = 10
        self.setup_chainerx()
        self.check_apply_ndarray_chainerx()

    @attr.chainerx
    @attr.gpu
    def test_apply_ndarray_chainerx_gpu(self):
        if False:
            print('Hello World!')
        self.setup_chainerx('cuda:0')
        self.check_apply_ndarray_chainerx()

    def check_apply_single_return_value(self):
        if False:
            while True:
                i = 10
        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2)
        (ret,) = self.f.apply((x1, x2))
        self.assertIsInstance(ret, chainer.Variable)

    def check_apply_single_return_value_chainerx(self):
        if False:
            i = 10
            return i + 15
        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2, requires_grad=False)
        (ret,) = self.f.apply((x1, x2))
        self.assertIsInstance(ret, chainer.Variable)
        self.assertIsInstance(ret.data, chainerx.ndarray)
        self.assertIs(ret.data.device, self.x1.device)

    def test_apply_single_return_value_cpu(self):
        if False:
            print('Hello World!')
        self.setup_cpu()
        self.f.forward_cpu.return_value = (self.y1,)
        self.check_apply_single_return_value()

    @attr.gpu
    def test_apply_single_return_value_gpu(self):
        if False:
            return 10
        self.setup_gpu()
        self.f.forward_gpu.return_value = (self.y1,)
        self.check_apply_single_return_value()

    @attr.chainerx
    def test_apply_single_return_value_chainerx_cpu(self):
        if False:
            print('Hello World!')
        self.setup_chainerx()
        self.f.forward.side_effect = lambda inputs: (utils.force_array(inputs[0] * inputs[1]),)
        self.check_apply_single_return_value_chainerx()

    @attr.chainerx
    @attr.gpu
    def test_apply_single_return_value_chainerx_gpu(self):
        if False:
            while True:
                i = 10
        self.setup_chainerx('cuda:0')
        self.f.forward.side_effect = lambda inputs: (utils.force_array(inputs[0] * inputs[1]),)
        self.check_apply_single_return_value_chainerx()

    def _get_f(self):
        if False:
            i = 10
            return i + 15
        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2)
        (y1, y2) = self.f.apply((x1, x2))
        f = y1.creator_node
        return (f, x1, y1)

    def test_unchain(self):
        if False:
            return 10
        self.setup_cpu()
        (f, _x1, _y1) = self._get_f()
        (y1, y2) = f.outputs
        f.unchain()
        y1_ref = y1()
        self.assertIsNotNone(y1_ref)
        self.assertIsNone(y1_ref.creator)
        y2_ref = y2()
        self.assertIsNone(y2_ref)
        self.assertIsNone(f.inputs)

    def test_label(self):
        if False:
            return 10
        self.setup_cpu()
        self.assertEqual(self.f.label, 'FunctionNode')

class TestFunctionNodeMixChainerxAndXpArrays(unittest.TestCase):

    class SimpleFunctionNode(chainer.FunctionNode):

        def __init__(self, xp):
            if False:
                i = 10
                return i + 15
            self.xp = xp

        def forward(self, inputs):
            if False:
                return 10
            (x1, x2) = inputs
            assert isinstance(x1, self.xp.ndarray)
            assert isinstance(x2, self.xp.ndarray)
            return (x1 * x2,)

    def check_mix_xp(self, xp):
        if False:
            return 10
        xp_x1 = xp.random.randn(2, 3).astype(numpy.float32)
        xp_x2 = xp.random.randn(2, 3).astype(numpy.float32)
        x2 = backend.to_chx(xp_x2)
        fnode = self.SimpleFunctionNode(xp)
        with self.assertRaises(TypeError):
            fnode.apply((xp_x1, x2))

    @attr.chainerx
    def test_mix_numpy(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_mix_xp(numpy)

    @attr.chainerx
    @attr.gpu
    def test_mix_cupy(self):
        if False:
            i = 10
            return i + 15
        self.check_mix_xp(cuda.cupy)

class TestFunctionNodeInvalidType(unittest.TestCase):

    def test_forward_invalid1(self):
        if False:
            return 10

        class FunctionNode(chainer.FunctionNode):

            def check_type_forward(self, in_types):
                if False:
                    for i in range(10):
                        print('nop')
                (x_type,) = in_types
                type_check.expect(x_type.dtype == numpy.float32, x_type.ndim >= 2)

            def forward(self, inputs):
                if False:
                    for i in range(10):
                        print('nop')
                return inputs
        f = FunctionNode()
        v = chainer.Variable(numpy.random.randn(1, 5).astype(numpy.float32))
        (result,) = f.apply((v,))
        assert isinstance(result, chainer.Variable)
        msg = "Invalid operation is performed in: FunctionNode \\(Forward\\)\n\nExpect: in_types\\[0\\]\\.dtype == <(type|class) 'numpy\\.float32'>\nActual: float64 \\!= <(type|class) 'numpy\\.float32'>"
        v = chainer.Variable(numpy.random.randn(1, 5))
        with six.assertRaisesRegex(self, chainer.utils.type_check.InvalidType, msg):
            f.apply((v,))
        msg = 'Invalid operation is performed in: FunctionNode \\(Forward\\)\n\nExpect: in_types\\[0\\]\\.ndim >= 2\nActual: 1 < 2'
        v = chainer.Variable(numpy.random.randn(5).astype(numpy.float32))
        with six.assertRaisesRegex(self, chainer.utils.type_check.InvalidType, msg):
            f.apply((v,))

class TestFunctionNodeForwardTypeCheck(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x1 = numpy.random.rand(2, 3).astype(numpy.float32)
        self.x2 = numpy.random.rand(2, 3).astype(numpy.float32)

    def test_invalid_output_type(self):
        if False:
            return 10

        class FunctionNode(chainer.FunctionNode):

            def forward(self, inputs):
                if False:
                    for i in range(10):
                        print('nop')
                return (object(),)
        f = FunctionNode()
        x1 = chainer.Variable(self.x1)
        with six.assertRaisesRegex(self, TypeError, 'forward output must be a tuple of ndarrays'):
            f.apply((x1,))

    @attr.gpu
    def test_inconsistent_input_backends(self):
        if False:
            print('Hello World!')

        class FunctionNode(chainer.FunctionNode):

            def forward(self, inputs):
                if False:
                    return 10
                return inputs
        f = FunctionNode()
        x1 = cuda.to_gpu(self.x1)
        x1 = chainer.Variable(x1)
        x2 = chainer.Variable(self.x2)
        with self.assertRaises(TypeError):
            f.apply((x1, x2))

    @attr.gpu
    def test_inconsistent_output_backends(self):
        if False:
            print('Hello World!')

        class FunctionNode(chainer.FunctionNode):

            def forward(self, inputs):
                if False:
                    return 10
                return (inputs[0], cuda.to_gpu(inputs[1]))
        f = FunctionNode()
        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2)
        with self.assertRaises(TypeError):
            f.apply((x1, x2))

@testing.parameterize({'return_value': (numpy.array([float('nan')], numpy.float32),), 'valid': False}, {'return_value': (numpy.array([1], numpy.int32),), 'valid': True})
class TestFunctionNodeForwardDebug(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.original_debug = chainer.is_debug()
        chainer.set_debug(True)
        self.one = numpy.array([1], numpy.float32)
        self.f = chainer.FunctionNode()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        chainer.set_debug(self.original_debug)

    def check_debug_forward(self, x_data):
        if False:
            print('Hello World!')
        x = chainer.Variable(x_data)
        if self.valid:
            self.f.apply((x,))
        else:
            with self.assertRaises(RuntimeError):
                self.f.apply((x,))

    def test_debug_forward_cpu(self):
        if False:
            return 10
        self.f.forward_cpu = mock.MagicMock(return_value=self.return_value)
        self.check_debug_forward(self.one)

    @attr.gpu
    def test_debug_forward_gpu(self):
        if False:
            return 10
        return_value = tuple((None if x is None else cuda.to_gpu(x) for x in self.return_value))
        self.f.forward_gpu = mock.MagicMock(return_value=return_value)
        self.check_debug_forward(cuda.to_gpu(self.one))

@testing.backend.inject_backend_tests(None, testing.product({'use_cuda': [True, False]}))
class TestFunctionNodeInvalidBackwardChecks(unittest.TestCase):
    """Tests FunctionNode.backward correctness checks"""

    def setUp(self):
        if False:
            while True:
                i = 10
        self.f = chainer.FunctionNode()

    def _dummy_func(self, bwd_return_data):
        if False:
            i = 10
            return i + 15

        def one(xp):
            if False:
                while True:
                    i = 10
            return xp.array(1, numpy.float32)

        class DummyFunc(chainer.FunctionNode):

            def forward_cpu(self, inputs):
                if False:
                    i = 10
                    return i + 15
                return (one(numpy),)

            def forward_gpu(self, inputs):
                if False:
                    while True:
                        i = 10
                return (one(cuda.cupy),)

            def backward(self, indexes, grad_outputs):
                if False:
                    return 10
                return bwd_return_data
        return DummyFunc()

    def check_debug_backward_accumulate(self, backend_config, f, xs_data, errors, initial_gxs=None):
        if False:
            print('Hello World!')
        for (debug_mode, error) in errors.items():

            def to_xp(arrs):
                if False:
                    i = 10
                    return i + 15
                if backend_config.use_cuda:
                    return cuda.to_gpu(arrs)
                else:
                    return arrs
            xs_data = to_xp(xs_data)
            if initial_gxs is not None:
                initial_gxs = to_xp(initial_gxs)
            xs = [chainer.Variable(x) for x in xs_data]
            (y,) = f.apply(xs)
            if initial_gxs is not None:
                assert len(xs) == len(initial_gxs)
                for (x, gx) in zip(xs, initial_gxs):
                    x.grad = gx
            with chainer.using_config('debug', debug_mode):
                if error is None:
                    y.backward()
                else:
                    (error_type, error_regex) = error
                    with pytest.raises(error_type, match=error_regex):
                        y.backward()

    def test_ok(self, backend_config):
        if False:
            print('Hello World!')
        self.check_debug_backward_accumulate(backend_config, f=self._dummy_func((chainer.Variable(numpy.array([2.0], numpy.float32)),)), xs_data=(numpy.array([1], numpy.float32),), errors={False: None, True: None})

    def test_gradients_has_nan(self, backend_config):
        if False:
            i = 10
            return i + 15
        self.check_debug_backward_accumulate(backend_config, f=self._dummy_func((chainer.Variable(numpy.array([float('nan')], numpy.float32)),)), xs_data=(numpy.array([1], numpy.float32),), errors={True: (RuntimeError, 'NaN is detected on backward computation')})

    def test_invalid_number_of_gradients(self, backend_config):
        if False:
            i = 10
            return i + 15
        self.check_debug_backward_accumulate(backend_config, f=self._dummy_func((chainer.Variable(numpy.array([2.0], numpy.float32)), chainer.Variable(numpy.array([1.0], numpy.float32)))), xs_data=(numpy.array([1], numpy.float32),), errors={True: (ValueError, 'number of gradients returned from backward is incorrect')})

    def test_invalid_zero_gradients(self, backend_config):
        if False:
            print('Hello World!')
        self.check_debug_backward_accumulate(backend_config, f=self._dummy_func(()), xs_data=(numpy.array([1], numpy.float32),), errors={True: (ValueError, 'number of gradients returned from backward is incorrect')})

    def test_invalid_gradient_shape(self, backend_config):
        if False:
            for i in range(10):
                print('nop')
        self.check_debug_backward_accumulate(backend_config, f=self._dummy_func((chainer.Variable(backend_config.xp.array([2, 3], numpy.float32)),)), xs_data=(numpy.array([1], numpy.float32),), errors={True: (ValueError, 'shape of gradients returned from backward is incorrect')})

    def test_invalid_gradient_type(self, backend_config):
        if False:
            while True:
                i = 10
        self.check_debug_backward_accumulate(backend_config, f=self._dummy_func(backend_config.xp.array([2.0], numpy.float32)), xs_data=(numpy.array([1], numpy.float32),), errors={True: (ValueError, 'type of gradients returned from backward is incorrect')})

    def test_invalid_gradient_dtype(self, backend_config):
        if False:
            i = 10
            return i + 15
        self.check_debug_backward_accumulate(backend_config, f=self._dummy_func((chainer.Variable(backend_config.xp.array([2.0], numpy.int64)),)), xs_data=(numpy.array([1], numpy.float32),), initial_gxs=(numpy.array([1], numpy.float32),), errors={True: (ValueError, 'dtype of gradients returned from backward is incorrect')})

class TestNoBackpropMode(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = chainer.Variable(numpy.array([1.0], 'f'))

    def test_no_backprop_mode(self):
        if False:
            i = 10
            return i + 15
        y = self.x + 1
        self.assertTrue(y.creator_node is not None)
        with chainer.no_backprop_mode():
            y = self.x + 1
        self.assertTrue(y.creator_node is None)
        y = self.x + 1
        self.assertTrue(y.creator_node is not None)

    def test_force_backprop_mode(self):
        if False:
            return 10
        with chainer.no_backprop_mode():
            with chainer.force_backprop_mode():
                y = self.x + 1
        self.assertTrue(y.creator_node is not None)
        y = self.x + 1
        self.assertTrue(y.creator_node is not None)
        with chainer.force_backprop_mode():
            y = self.x + 1
        self.assertTrue(y.creator_node is not None)

    @attr.chainerx
    def test_backprop_mode_affects_chainerx(self):
        if False:
            for i in range(10):
                print('nop')
        assert chainerx.is_backprop_required()
        with chainer.no_backprop_mode():
            assert not chainerx.is_backprop_required()
            with chainer.force_backprop_mode():
                assert chainerx.is_backprop_required()
            with chainer.no_backprop_mode():
                assert not chainerx.is_backprop_required()
        assert chainerx.is_backprop_required()
        with chainer.force_backprop_mode():
            assert chainerx.is_backprop_required()
            with chainer.force_backprop_mode():
                assert chainerx.is_backprop_required()
            with chainer.no_backprop_mode():
                assert not chainerx.is_backprop_required()
        assert chainerx.is_backprop_required()

class MyThread(threading.Thread):

    def run(self):
        if False:
            i = 10
            return i + 15
        x = chainer.Variable(numpy.array([1], dtype='f'))
        with chainer.no_backprop_mode():
            y = x + 1
        self.creator_is_none = y.creator_node is None

class TestBackpropModeMultiThread(unittest.TestCase):

    def test_multi_thread(self):
        if False:
            print('Hello World!')
        t = MyThread()
        t.start()
        t.join()
        self.assertTrue(t.creator_is_none)

class FunctionNodeWithRetaining(chainer.FunctionNode):

    def __init__(self, input_indices, output_indices):
        if False:
            print('Hello World!')
        self.input_indices = input_indices
        self.output_indices = output_indices

    def forward(self, inputs):
        if False:
            print('Hello World!')
        self.retain_inputs(self.input_indices)
        self.retain_outputs(self.output_indices)
        return inputs

    def backward(self, _, grad_outputs):
        if False:
            return 10
        self.retained_backward_inputs = self.get_retained_inputs()
        self.retained_backward_outputs = self.get_retained_outputs()
        return grad_outputs

@testing.backend.inject_backend_tests(None, [{}, {'use_cuda': True}, {'use_chainerx': True, 'chainerx_device': 'native:0'}])
class TestFunctionNodeRetaining(unittest.TestCase):

    def test_retain(self, backend_config):
        if False:
            return 10
        xp = backend_config.xp
        input_arrs = backend_config.get_array([numpy.array([2], dtype=numpy.float32), numpy.array([-1], dtype=numpy.float32)])
        inputs = [chainer.Variable(input_arrs[0]), chainer.Variable(input_arrs[1], requires_grad=False)]
        input_arrays = [x.array for x in inputs]
        if xp is not chainerx:
            input_nodes = [x.node for x in inputs]
        f = FunctionNodeWithRetaining([1], [0, 1])
        outputs = f.apply(inputs)
        outputs[0].grad = backend_config.get_array(numpy.array([1], dtype=numpy.float32))
        outputs[0].backward()
        output_arrays = [y.array for y in outputs]
        inputs = None
        assert len(f.retained_backward_inputs) == 1
        assert len(f.retained_backward_outputs) == 2
        if xp is not chainerx:
            assert f.retained_backward_inputs[0].node is input_nodes[1]
        xp.testing.assert_array_equal(f.retained_backward_inputs[0].array, input_arrays[1])
        xp.testing.assert_array_equal(f.retained_backward_outputs[0].array, output_arrays[0])
        xp.testing.assert_array_equal(f.retained_backward_outputs[1].array, output_arrays[1])

    def check_no_retain(self, backend_config, skip_call):
        if False:
            for i in range(10):
                print('nop')

        class MyFunc(chainer.FunctionNode):
            backward_called = 0

            def forward(self, inputs):
                if False:
                    print('Hello World!')
                (x,) = inputs
                if not skip_call:
                    self.retain_outputs(())
                    self.retain_inputs(())
                return (x * 3,)

            def backward(self, input_indices, grad_outputs):
                if False:
                    while True:
                        i = 10
                self.backward_called += 1
                assert self.get_retained_outputs() == ()
                assert self.get_retained_inputs() == ()
                (gy,) = grad_outputs
                return (gy * 3,)
        x_arr = backend_config.get_array(numpy.array([1, 2], numpy.float32))
        x = chainer.Variable(x_arr, requires_grad=True)
        func = MyFunc()
        (y,) = func.apply((x,))
        y.grad = backend_config.get_array(numpy.array([1, 1], numpy.float32))
        y.backward()
        assert func.backward_called == 1

    def test_no_retain(self, backend_config):
        if False:
            while True:
                i = 10
        self.check_no_retain(backend_config, False)
        self.check_no_retain(backend_config, True)

def _get_value(x):
    if False:
        return 10
    if isinstance(x, chainer.Variable):
        return x.data
    return x

class TestGradTypeCheck(unittest.TestCase):

    def test_type_check(self):
        if False:
            i = 10
            return i + 15
        x = chainer.Variable(numpy.random.uniform(-1, 1, (2, 3)).astype('f'))
        y = x * x
        gx = chainer.Variable(numpy.random.uniform(-1, 1, (2, 3)).astype('f'))
        gy = chainer.Variable(numpy.random.uniform(-1, 1, (2, 3)).astype('f'))
        chainer.grad([y], [x], [gx], [gy])
        chainer.grad((y,), (x,), (gx,), (gy,))
        with self.assertRaises(TypeError):
            chainer.grad(y, [x], [gx], [gy])
        with self.assertRaises(TypeError):
            chainer.grad([y], x, [gx], [gy])
        with self.assertRaises(TypeError):
            chainer.grad([y], [x], gx, [gy])
        with self.assertRaises(TypeError):
            chainer.grad([y], [x], [gx], gy)

class TestGradValueCheck(unittest.TestCase):

    def test_length_check(self):
        if False:
            i = 10
            return i + 15
        x = chainer.Variable(numpy.array(3, numpy.float32))
        y = chainer.functions.identity(x)
        with self.assertRaises(ValueError):
            chainer.grad([y], [x], [], [None])
        with self.assertRaises(ValueError):
            chainer.grad([y], [x], [None, None], [None])
        with self.assertRaises(ValueError):
            chainer.grad([y], [x], [None], [])
        with self.assertRaises(ValueError):
            chainer.grad([y], [x], [None], [None, None])

class GradTestBase(object):
    shape = (3,)
    x_names = ()
    y_names = ()
    loss_scale = None
    extend_graph_x = False
    extend_graph_y = False

    def _init_attrs(self, names):
        if False:
            for i in range(10):
                print('nop')
        ret = []
        for name in names:
            v = chainer.Variable(numpy.random.randint(-4, 6, self.shape).astype('f'), name=name)
            ret.append(v)
            setattr(self, name, v)
        return ret

    def _init_ones(self, names):
        if False:
            while True:
                i = 10
        ret = []
        for name in names:
            v = chainer.Variable(numpy.ones(self.shape, dtype='f'))
            ret.append(v)
            setattr(self, name, v)
        return ret

    @staticmethod
    def _get_value(x):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(x, chainer.Variable):
            return x.data
        return x

    @staticmethod
    def _to_grad_names(names):
        if False:
            while True:
                i = 10
        return ['g%s' % name for name in names]

    def setUp(self):
        if False:
            while True:
                i = 10
        self.xs = self._init_attrs(self.x_names)
        self.gxs = self._init_attrs(self._to_grad_names(self.x_names))
        self.gys = self._init_attrs(self._to_grad_names(self.y_names))
        if self.loss_scale is not None:
            self._init_ones(self._to_grad_names(self.y_names))
            self.gys = None

    def use_device(self, device):
        if False:
            while True:
                i = 10
        for value in six.itervalues(self.__dict__):
            if isinstance(value, chainer.Variable):
                value.to_device(device)

    def forward(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def expected_grad(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def expected_double_grad(self):
        if False:
            return 10
        raise NotImplementedError

    def _print_variables(self, name, vs):
        if False:
            return 10
        print('{}: '.format(name), end='')
        print(*(self._get_value(v) for v in vs), sep=', ')

    def _print_inputs(self):
        if False:
            i = 10
            return i + 15
        self._print_variables('xs  ', self.xs)
        self._print_variables('gxs ', self.gxs)
        self._print_variables('gys ', self.gys)

    def check_grad(self):
        if False:
            while True:
                i = 10
        self.forward()
        ys = [getattr(self, name) for name in self.y_names]
        if self.extend_graph_y:
            self._ys = [v * 1.0 for v in ys]
        if self.extend_graph_x:
            for v in self.xs:
                v *= 1.0
        gxs = chainer.grad(ys, self.xs, self.gys, self.gxs, loss_scale=self.loss_scale)
        expected = self.expected_grad()
        for (i, gx) in enumerate(self.gxs):
            expected[i] += gx
        self.assertEqual(len(gxs), len(expected))
        try:
            for (a, e) in zip(gxs, expected):
                testing.assert_allclose(self._get_value(a), self._get_value(e))
        except Exception:
            self._print_inputs()
            self._print_variables('gxs (actual)  ', gxs)
            self._print_variables('gxs (expected)', expected)
            raise

    def test_grad(self, backend_config):
        if False:
            i = 10
            return i + 15
        self.use_device(backend_config.device)
        self.check_grad()

    def check_double_grad(self):
        if False:
            while True:
                i = 10
        self.forward()
        ys = [getattr(self, name) for name in self.y_names]
        gxs = chainer.grad(ys, self.xs, self.gys, self.gxs, enable_double_backprop=True, loss_scale=self.loss_scale)
        y = sum(gxs)
        ggxs = chainer.grad([y], self.xs)
        expected = self.expected_double_grad()
        self.assertEqual(len(ggxs), len(expected))
        try:
            for (a, e) in zip(ggxs, expected):
                testing.assert_allclose(self._get_value(a), self._get_value(e))
        except Exception:
            self._print_inputs()
            self._print_variables('gxs            ', gxs)
            self._print_variables('ggxs (actual)  ', ggxs)
            self._print_variables('ggxs (expected)', expected)
            raise

    def test_double_grad(self, backend_config):
        if False:
            return 10
        self.use_device(backend_config.device)
        self.check_double_grad()

@testing.parameterize(*testing.product({'loss_scale': [None, 1, 10]}))
@testing.backend.inject_backend_tests(None, [{}, {'use_ideep': 'always'}, {'use_cuda': True, 'cuda_device': 0}, {'use_cuda': True, 'cuda_device': 1}])
class TestGradSimple(GradTestBase, unittest.TestCase):
    x_names = ('x',)
    y_names = ('y',)

    def forward(self):
        if False:
            i = 10
            return i + 15
        self.y = self.x * self.x

    def expected_grad(self):
        if False:
            i = 10
            return i + 15
        grad = 2 * self.x * self.gy
        if self.loss_scale is not None:
            grad *= self.loss_scale
        return [grad]

    def expected_double_grad(self):
        if False:
            while True:
                i = 10
        ggrad = 2 * self.gy
        if self.loss_scale is not None:
            ggrad *= self.loss_scale
        return [ggrad]

@testing.parameterize(*testing.product({'loss_scale': [None, 1, 1.5, 2.5, 10]}))
@testing.backend.inject_backend_tests(None, [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestGradSimpleChainerX(GradTestBase, unittest.TestCase):
    x_names = ('x',)
    y_names = ('y',)

    def forward(self):
        if False:
            i = 10
            return i + 15
        self.y = self.x * self.x

    def expected_grad(self):
        if False:
            for i in range(10):
                print('nop')
        grad = 2 * self.x * self.gy
        return [grad]

    def expected_double_grad(self):
        if False:
            print('Hello World!')
        ggrad = 2 * self.gy
        return [ggrad]

@testing.parameterize(*testing.product({'extend_graph_x': [False, True], 'extend_graph_y': [False, True]}))
@testing.backend.inject_backend_tests(None, [{}, {'use_ideep': 'always'}, {'use_cuda': True, 'cuda_device': 0}, {'use_cuda': True, 'cuda_device': 1}, {'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestGradComplex(GradTestBase, unittest.TestCase):
    x_names = ('x1', 'x2')
    y_names = ('y1', 'y2')

    def forward(self):
        if False:
            i = 10
            return i + 15
        self.z = self.x1 * self.x1
        self.y1 = self.z + self.x1 * self.x2 + self.x2
        self.y2 = self.z + self.y1

    def expected_grad(self):
        if False:
            for i in range(10):
                print('nop')
        dz_dx = 2 * self.x1
        dy1_dx = self.gy1 + self.gy2
        return [dy1_dx * (dz_dx + self.x2) + self.gy2 * dz_dx, dy1_dx * (self.x1 + 1)]

    def expected_double_grad(self):
        if False:
            for i in range(10):
                print('nop')
        dy1_dx = self.gy1 + self.gy2
        return [3 * dy1_dx + 2 * self.gy2, dy1_dx]

class ExpPair(chainer.FunctionNode):

    def forward(self, inputs):
        if False:
            print('Hello World!')
        (x,) = inputs
        xp = backend.get_array_module(x)
        self.retain_outputs((0, 1))
        return (xp.exp(x), xp.exp(x))

    def backward(self, target_input_indexes, grad_outputs):
        if False:
            return 10
        return (sum([g * exp for (g, exp) in zip(grad_outputs, self.get_retained_outputs()) if g is not None]),)

def exp_pair(x):
    if False:
        while True:
            i = 10
    return ExpPair().apply((x,))

@testing.parameterize(*testing.product({'keep_y2': [False, True]}))
@testing.backend.inject_backend_tests(None, [{}, {'use_ideep': 'always'}, {'use_cuda': True, 'cuda_device': 0}, {'use_cuda': True, 'cuda_device': 1}, {'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestGradDelRetainedOutput(GradTestBase, unittest.TestCase):
    x_names = ('x1',)
    y_names = ('y1',)

    def forward(self):
        if False:
            while True:
                i = 10
        (self.y1, y2) = exp_pair(self.x1)
        if self.keep_y2:
            self.y2 = y2

    def expected_grad(self):
        if False:
            while True:
                i = 10
        return [self.gy1 * self.y1]

    def expected_double_grad(self):
        if False:
            return 10
        return [self.gy1 * self.y1]

class ExpAndExpm1(chainer.FunctionNode):

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (x,) = inputs
        xp = backend.get_array_module()
        y0 = xp.exp(x)
        y1 = xp.expm1(x)
        self.retain_outputs((0,))
        return (y0, y1)

    def backward(self, target_input_indexes, grad_outputs):
        if False:
            while True:
                i = 10
        (g0, g1) = grad_outputs
        (y0,) = self.get_retained_outputs()
        gx = []
        if g0 is not None:
            gx.append(g0 * y0)
        if g1 is not None:
            gx.append(g1 * y0)
        return (chainer.functions.add(*gx),)

def exp_and_expm1(x):
    if False:
        for i in range(10):
            print('nop')
    return ExpAndExpm1().apply((x,))

class TestGradDelRetainedOutput2(unittest.TestCase):

    def test_retain_output(self):
        if False:
            for i in range(10):
                print('nop')
        xp = numpy
        x_array = xp.random.randn(3)
        y1_grad = xp.random.randn(3)
        x_grad_grad = xp.random.randn(3)
        x = chainer.Variable(x_array, name='x')
        (y0, y1) = exp_and_expm1(x)
        del y0
        (gx,) = chainer.grad([y1], [x], [y1_grad], enable_double_backprop=True)
        xp.testing.assert_allclose(gx.array, xp.exp(x.array) * y1_grad)
        (gx_,) = chainer.grad([gx], [x], [x_grad_grad])
        xp.testing.assert_allclose(gx_.array, gx.array * x_grad_grad)

class TestUnchainSplitGrad(unittest.TestCase):

    def test_unchain_split(self):
        if False:
            return 10
        x = chainer.Variable(numpy.arange(4).astype('f').reshape(2, 2))
        (h0, h1) = chainer.functions.split_axis(x, [1], axis=0)
        y = chainer.functions.sum(h0)
        z = chainer.functions.sum(h1)
        w = y + z
        h0.unchain()
        dy_dh0 = numpy.array([[1.0, 1.0]])
        dz_dh1 = numpy.array([[1.0, 1.0]])
        dy_dx = None
        dz_dx = numpy.array([[0.0, 0.0], [1.0, 1.0]])
        dw_dx = numpy.array([[0.0, 0.0], [1.0, 1.0]])
        testing.assert_allclose(chainer.grad([y], [h0])[0].array, dy_dh0)
        testing.assert_allclose(chainer.grad([z], [h1])[0].array, dz_dh1)
        assert chainer.grad([y], [x])[0] is dy_dx
        testing.assert_allclose(chainer.grad([z], [x])[0].array, dz_dx)
        testing.assert_allclose(chainer.grad([w], [x])[0].array, dw_dx)

class TestGradV3Compat1(unittest.TestCase):

    def _var(self, val):
        if False:
            return 10
        return chainer.Variable(numpy.array(val, numpy.float32))

    def check(self, option, grads_before, grads_after):
        if False:
            while True:
                i = 10
        vs = []
        v = self._var(0.5)
        for _ in range(4):
            vs.append(v)
            v += v
            vs.append(v)
            v *= 1.0
        (_, x1, _, x2, _, y1, _, y2) = vs
        gx1 = self._var(1000.0)
        gx2 = self._var(100.0)
        gy1 = self._var(10.0)
        gy2 = self._var(1.0)
        for (v, g) in zip(vs, grads_before):
            if g is not None:
                v.grad_var = self._var(g)
        grads = chainer.grad([y1, y2], [x1, x2], [gy1, gy2], [gx1, gx2], **option)
        numpy.testing.assert_allclose(grads[0].array, 1248.0)
        numpy.testing.assert_allclose(grads[1].array, 124.0)
        for (v, ans) in zip(vs, grads_after):
            if ans is None:
                self.assertIsNone(v.grad)
            else:
                numpy.testing.assert_allclose(v.grad, ans)

    def test_no_option(self):
        if False:
            return 10
        self.check({}, [None] * 8, [None] * 8)
        self.check({}, [-1.0] * 8, [-1.0] * 8)

    def test_set_grad(self):
        if False:
            while True:
                i = 10
        self.check({'set_grad': True}, [None] * 8, [None, 1248.0, None, 124.0, None, None, None, None])
        self.check({'set_grad': True}, [-1.0] * 8, [-1.0, 1248.0, -1.0, 124.0, -1.0, -1.0, -1.0, -1.0])

    def test_retain_grad(self):
        if False:
            while True:
                i = 10
        self.check({'retain_grad': True}, [None] * 8, [None, 1248.0, 248.0, 124.0, 24.0, 12.0, 2.0, 1.0])
        self.check({'retain_grad': True}, [-1.0] * 8, [-1.0, 1248.0, 248.0, 124.0, 24.0, 12.0, 2.0, 1.0])

@attr.chainerx
class TestFunctionNodeBackwardChainerx(unittest.TestCase):

    class SimpleFunctionNode(chainer.FunctionNode):

        def __init__(self, backward_call_callback):
            if False:
                for i in range(10):
                    print('nop')
            self.backward_call_callback = backward_call_callback

        def forward(self, inputs):
            if False:
                print('Hello World!')
            return tuple([2 * x for x in inputs])

        def backward(self, indexes, grad_outputs):
            if False:
                while True:
                    i = 10
            self.backward_call_callback({'indexes': indexes, 'grad_outputs': grad_outputs})
            gxs = []
            for i_in in indexes:
                gx = 2 * grad_outputs[i_in]
                gxs.append(gx)
            return gxs

    def test_backward(self):
        if False:
            while True:
                i = 10
        shape = (2, 3)
        dtype = numpy.float32
        x1 = chainerx.full(shape, 3, dtype)
        x2 = chainerx.full(shape, 5, dtype).require_grad()
        gx2_expected = numpy.full(shape, 2, dtype)
        backward_call_args = []

        def backward_call_callback(call_arg):
            if False:
                for i in range(10):
                    print('nop')
            backward_call_args.append(call_arg)
        func = self.SimpleFunctionNode(backward_call_callback)
        (y1, y2) = func.apply((x1, x2))
        del func
        assert y1.requires_grad
        assert y2.requires_grad
        y2.backward()
        assert len(backward_call_args) == 1
        (call_arg,) = backward_call_args
        assert isinstance(call_arg['indexes'], tuple)
        assert call_arg['indexes'] == (1,)
        assert isinstance(call_arg['grad_outputs'], tuple)
        assert len(call_arg['grad_outputs']) == 2
        assert call_arg['grad_outputs'][0] is None
        chainerx.testing.assert_array_equal_ex(call_arg['grad_outputs'][1].array, numpy.full(shape, 1, dtype), strides_check=False)
        chainerx.testing.assert_array_equal_ex(x2.grad, gx2_expected, strides_check=False)
        assert not x2.grad.is_backprop_required()
        with pytest.raises(chainerx.ChainerxError):
            x1.grad

    @attr.gpu
    def test_backward_default_device(self):
        if False:
            i = 10
            return i + 15
        device = chainerx.get_device('cuda:0')
        shape = (2, 3)
        dtype = numpy.float32
        x1 = chainerx.full(shape, 3, dtype, device=device)
        x2 = chainerx.full(shape, 5, dtype, device=device).require_grad()
        backward_call_new_array = []

        def backward_call_callback(call_arg):
            if False:
                print('Hello World!')
            backward_call_new_array.append(chainerx.empty(shape, dtype))
        with chainerx.using_device('native:0'):
            func = self.SimpleFunctionNode(backward_call_callback)
            (y1, y2) = func.apply((x1, x2))
            y2.backward()
        assert backward_call_new_array[0].device is device
testing.run_module(__name__, __file__)