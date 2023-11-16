from itertools import repeat
import os
import re
from typing import Union, get_args, get_origin
import unittest
import torch.testing._internal.common_utils as common
from torch.testing._internal.common_utils import IS_WINDOWS, skipIfTorchDynamo
from torch.testing._internal.common_cuda import TEST_CUDA
import torch
import torch.backends.cudnn
import torch.utils.cpp_extension
try:
    import pytest
    HAS_PYTEST = True
except ImportError as e:
    HAS_PYTEST = False
try:
    if HAS_PYTEST:
        cpp_extension = pytest.importorskip('torch_test_cpp_extension.cpp')
        ort_extension = pytest.importorskip('torch_test_cpp_extension.ort')
        rng_extension = pytest.importorskip('torch_test_cpp_extension.rng')
    else:
        import torch_test_cpp_extension.cpp as cpp_extension
        import torch_test_cpp_extension.ort as ort_extension
        import torch_test_cpp_extension.rng as rng_extension
except ImportError as e:
    raise RuntimeError('test_cpp_extensions_aot.py cannot be invoked directly. Run `python run_test.py -i test_cpp_extensions_aot_ninja` instead.') from e

class TestCppExtensionAOT(common.TestCase):
    """Tests ahead-of-time cpp extensions

    NOTE: run_test.py's test_cpp_extensions_aot_ninja target
    also runs this test case, but with ninja enabled. If you are debugging
    a test failure here from the CI, check the logs for which target
    (test_cpp_extensions_aot_no_ninja vs test_cpp_extensions_aot_ninja)
    failed.
    """

    def test_extension_function(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        z = cpp_extension.sigmoid_add(x, y)
        self.assertEqual(z, x.sigmoid() + y.sigmoid())

    def test_extension_module(self):
        if False:
            i = 10
            return i + 15
        mm = cpp_extension.MatrixMultiplier(4, 8)
        weights = torch.rand(8, 4, dtype=torch.double)
        expected = mm.get().mm(weights)
        result = mm.forward(weights)
        self.assertEqual(expected, result)

    def test_backward(self):
        if False:
            i = 10
            return i + 15
        mm = cpp_extension.MatrixMultiplier(4, 8)
        weights = torch.rand(8, 4, dtype=torch.double, requires_grad=True)
        result = mm.forward(weights)
        result.sum().backward()
        tensor = mm.get()
        expected_weights_grad = tensor.t().mm(torch.ones([4, 4], dtype=torch.double))
        self.assertEqual(weights.grad, expected_weights_grad)
        expected_tensor_grad = torch.ones([4, 4], dtype=torch.double).mm(weights.t())
        self.assertEqual(tensor.grad, expected_tensor_grad)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not found')
    def test_cuda_extension(self):
        if False:
            print('Hello World!')
        import torch_test_cpp_extension.cuda as cuda_extension
        x = torch.zeros(100, device='cuda', dtype=torch.float32)
        y = torch.zeros(100, device='cuda', dtype=torch.float32)
        z = cuda_extension.sigmoid_add(x, y).cpu()
        self.assertEqual(z, torch.ones_like(z))

    @unittest.skipIf(not torch.backends.mps.is_available(), 'MPS not found')
    def test_mps_extension(self):
        if False:
            for i in range(10):
                print('nop')
        import torch_test_cpp_extension.mps as mps_extension
        tensor_length = 100000
        x = torch.randn(tensor_length, device='cpu', dtype=torch.float32)
        y = torch.randn(tensor_length, device='cpu', dtype=torch.float32)
        cpu_output = mps_extension.get_cpu_add_output(x, y)
        mps_output = mps_extension.get_mps_add_output(x.to('mps'), y.to('mps'))
        self.assertEqual(cpu_output, mps_output.to('cpu'))

    @common.skipIfRocm
    @unittest.skipIf(common.IS_WINDOWS, 'Windows not supported')
    @unittest.skipIf(not TEST_CUDA, 'CUDA not found')
    def test_cublas_extension(self):
        if False:
            print('Hello World!')
        from torch_test_cpp_extension import cublas_extension
        x = torch.zeros(100, device='cuda', dtype=torch.float32)
        z = cublas_extension.noop_cublas_function(x)
        self.assertEqual(z, x)

    @common.skipIfRocm
    @unittest.skipIf(common.IS_WINDOWS, 'Windows not supported')
    @unittest.skipIf(not TEST_CUDA, 'CUDA not found')
    def test_cusolver_extension(self):
        if False:
            print('Hello World!')
        from torch_test_cpp_extension import cusolver_extension
        x = torch.zeros(100, device='cuda', dtype=torch.float32)
        z = cusolver_extension.noop_cusolver_function(x)
        self.assertEqual(z, x)

    @unittest.skipIf(IS_WINDOWS, 'Not available on Windows')
    def test_no_python_abi_suffix_sets_the_correct_library_name(self):
        if False:
            while True:
                i = 10
        root = os.path.join('cpp_extensions', 'no_python_abi_suffix_test', 'build')
        matches = [f for (_, _, fs) in os.walk(root) for f in fs if f.endswith('so')]
        self.assertEqual(len(matches), 1, msg=str(matches))
        self.assertEqual(matches[0], 'no_python_abi_suffix_test.so', msg=str(matches))

    def test_optional(self):
        if False:
            return 10
        has_value = cpp_extension.function_taking_optional(torch.ones(5))
        self.assertTrue(has_value)
        has_value = cpp_extension.function_taking_optional(None)
        self.assertFalse(has_value)

    @common.skipIfRocm
    @unittest.skipIf(common.IS_WINDOWS, 'Windows not supported')
    @unittest.skipIf(not TEST_CUDA, 'CUDA not found')
    @unittest.skipIf(os.getenv('USE_NINJA', '0') == '0', 'cuda extension with dlink requires ninja to build')
    def test_cuda_dlink_libs(self):
        if False:
            print('Hello World!')
        from torch_test_cpp_extension import cuda_dlink
        a = torch.randn(8, dtype=torch.float, device='cuda')
        b = torch.randn(8, dtype=torch.float, device='cuda')
        ref = a + b
        test = cuda_dlink.add(a, b)
        self.assertEqual(test, ref)

class TestPybindTypeCasters(common.TestCase):
    """Pybind tests for ahead-of-time cpp extensions

    These tests verify the types returned from cpp code using custom type
    casters. By exercising pybind, we also verify that the type casters work
    properly.

    For each type caster in `torch/csrc/utils/pybind.h` we create a pybind
    function that takes no arguments and returns the type_caster type. The
    second argument to `PYBIND11_TYPE_CASTER` should be the type we expect to
    receive in python, in these tests we verify this at run-time.
    """

    @staticmethod
    def expected_return_type(func):
        if False:
            i = 10
            return i + 15
        '\n        Our Pybind functions have a signature of the form `() -> return_type`.\n        '
        from typing import List, Tuple
        return eval(re.search('-> (.*)\n', func.__doc__).group(1))

    def check(self, func):
        if False:
            print('Hello World!')
        val = func()
        expected = self.expected_return_type(func)
        origin = get_origin(expected)
        if origin is list:
            self.check_list(val, expected)
        elif origin is tuple:
            self.check_tuple(val, expected)
        else:
            self.assertIsInstance(val, expected)

    def check_list(self, vals, expected):
        if False:
            return 10
        self.assertIsInstance(vals, list)
        list_type = get_args(expected)[0]
        for val in vals:
            self.assertIsInstance(val, list_type)

    def check_tuple(self, vals, expected):
        if False:
            i = 10
            return i + 15
        self.assertIsInstance(vals, tuple)
        tuple_types = get_args(expected)
        if tuple_types[1] is ...:
            tuple_types = repeat(tuple_types[0])
        for (val, tuple_type) in zip(vals, tuple_types):
            self.assertIsInstance(val, tuple_type)

    def check_union(self, funcs):
        if False:
            while True:
                i = 10
        'Special handling for Union type casters.\n\n        A single cpp type can sometimes be cast to different types in python.\n        In these cases we expect to get exactly one function per python type.\n        '
        union_type = {self.expected_return_type(f) for f in funcs}
        assert len(union_type) == 1
        union_type = union_type.pop()
        self.assertIs(Union, get_origin(union_type))
        expected_types = set(get_args(union_type)) - {torch.SymInt}
        for func in funcs:
            val = func()
            for tp in expected_types:
                if isinstance(val, tp):
                    expected_types.remove(tp)
                    break
            else:
                raise AssertionError(f'{val} is not an instance of {expected_types}')
        self.assertFalse(expected_types, f'Missing functions for types {expected_types}')

    def test_pybind_return_types(self):
        if False:
            while True:
                i = 10
        functions = [cpp_extension.get_complex, cpp_extension.get_device, cpp_extension.get_generator, cpp_extension.get_intarrayref, cpp_extension.get_memory_format, cpp_extension.get_storage, cpp_extension.get_symfloat, cpp_extension.get_symintarrayref, cpp_extension.get_tensor]
        union_functions = [[cpp_extension.get_symint]]
        for func in functions:
            with self.subTest(msg=f'check {func.__name__}'):
                self.check(func)
        for funcs in union_functions:
            with self.subTest(msg=f'check {[f.__name__ for f in funcs]}'):
                self.check_union(funcs)

class TestORTTensor(common.TestCase):

    def test_unregistered(self):
        if False:
            while True:
                i = 10
        a = torch.arange(0, 10, device='cpu')
        with self.assertRaisesRegex(RuntimeError, 'Could not run'):
            b = torch.arange(0, 10, device='ort')

    @skipIfTorchDynamo('dynamo cannot model ort device')
    def test_zeros(self):
        if False:
            print('Hello World!')
        a = torch.empty(5, 5, device='cpu')
        self.assertEqual(a.device, torch.device('cpu'))
        b = torch.empty(5, 5, device='ort')
        self.assertEqual(b.device, torch.device('ort', 0))
        self.assertEqual(ort_extension.get_test_int(), 0)
        self.assertEqual(torch.get_default_dtype(), b.dtype)
        c = torch.empty((5, 5), dtype=torch.int64, device='ort')
        self.assertEqual(ort_extension.get_test_int(), 0)
        self.assertEqual(torch.int64, c.dtype)

    def test_add(self):
        if False:
            return 10
        a = torch.empty(5, 5, device='ort', requires_grad=True)
        self.assertEqual(ort_extension.get_test_int(), 0)
        b = torch.empty(5, 5, device='ort')
        self.assertEqual(ort_extension.get_test_int(), 0)
        c = a + b
        self.assertEqual(ort_extension.get_test_int(), 1)

    def test_conv_backend_override(self):
        if False:
            while True:
                i = 10
        input = torch.empty(2, 4, 10, 2, device='ort', requires_grad=True)
        weight = torch.empty(6, 4, 2, 2, device='ort', requires_grad=True)
        bias = torch.empty(6, device='ort')
        out = torch.nn.functional.conv2d(input, weight, bias, 2, 0, 1, 1)
        self.assertEqual(ort_extension.get_test_int(), 2)
        self.assertEqual(out.shape[0], input.shape[0])
        self.assertEqual(out.shape[1], weight.shape[0])
        grad = torch.autograd.grad(out, input, out, create_graph=True)
        self.assertEqual(ort_extension.get_test_int(), 3)
        self.assertEqual(grad[0].shape, input.shape)

class TestRNGExtension(common.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()

    @skipIfTorchDynamo('https://github.com/pytorch/torchdynamo/issues/1991')
    def test_rng(self):
        if False:
            return 10
        fourty_two = torch.full((10,), 42, dtype=torch.int64)
        t = torch.empty(10, dtype=torch.int64).random_()
        self.assertNotEqual(t, fourty_two)
        gen = torch.Generator(device='cpu')
        t = torch.empty(10, dtype=torch.int64).random_(generator=gen)
        self.assertNotEqual(t, fourty_two)
        self.assertEqual(rng_extension.getInstanceCount(), 0)
        gen = rng_extension.createTestCPUGenerator(42)
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        copy = gen
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        self.assertEqual(gen, copy)
        copy2 = rng_extension.identity(copy)
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        self.assertEqual(gen, copy2)
        t = torch.empty(10, dtype=torch.int64).random_(generator=gen)
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        self.assertEqual(t, fourty_two)
        del gen
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        del copy
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        del copy2
        self.assertEqual(rng_extension.getInstanceCount(), 0)

@unittest.skipIf(not TEST_CUDA, 'CUDA not found')
class TestTorchLibrary(common.TestCase):

    def test_torch_library(self):
        if False:
            i = 10
            return i + 15
        import torch_test_cpp_extension.torch_library

        def f(a: bool, b: bool):
            if False:
                for i in range(10):
                    print('nop')
            return torch.ops.torch_library.logical_and(a, b)
        self.assertTrue(f(True, True))
        self.assertFalse(f(True, False))
        self.assertFalse(f(False, True))
        self.assertFalse(f(False, False))
        s = torch.jit.script(f)
        self.assertTrue(s(True, True))
        self.assertFalse(s(True, False))
        self.assertFalse(s(False, True))
        self.assertFalse(s(False, False))
        self.assertIn('torch_library::logical_and', str(s.graph))
if __name__ == '__main__':
    common.run_tests()