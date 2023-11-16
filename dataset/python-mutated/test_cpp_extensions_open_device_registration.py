import os
import shutil
import sys
from typing import Union
import tempfile
import unittest
import torch.testing._internal.common_utils as common
from torch.testing._internal.common_utils import IS_ARM64, TEST_CUDA
import torch
import torch.utils.cpp_extension
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME
TEST_CUDA = TEST_CUDA and CUDA_HOME is not None
TEST_ROCM = TEST_CUDA and torch.version.hip is not None and (ROCM_HOME is not None)

def remove_build_path():
    if False:
        return 10
    if sys.platform == 'win32':
        return
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    if os.path.exists(default_build_root):
        shutil.rmtree(default_build_root, ignore_errors=True)

class DummyModule:

    @staticmethod
    def device_count() -> int:
        if False:
            print('Hello World!')
        return 1

    @staticmethod
    def get_rng_state(device: Union[int, str, torch.device]='foo') -> torch.Tensor:
        if False:
            print('Hello World!')
        return torch.empty(4, 4, device='foo')

    @staticmethod
    def set_rng_state(new_state: torch.Tensor, device: Union[int, str, torch.device]='foo') -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    @staticmethod
    def is_available():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def current_device():
        if False:
            i = 10
            return i + 15
        return 0

@unittest.skipIf(IS_ARM64, 'Does not work on arm')
class TestCppExtensionOpenRgistration(common.TestCase):
    """Tests Open Device Registration with C++ extensions.
    """
    module = None

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        assert self.module is not None

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        os.chdir(self.old_working_dir)

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        remove_build_path()
        cls.module = torch.utils.cpp_extension.load(name='custom_device_extension', sources=['cpp_extensions/open_registration_extension.cpp'], extra_include_paths=['cpp_extensions'], extra_cflags=['-g'], verbose=True)

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        remove_build_path()

    def test_open_device_registration(self):
        if False:
            for i in range(10):
                print('nop')

        def test_base_device_registration():
            if False:
                for i in range(10):
                    print('nop')
            torch.utils.rename_privateuse1_backend('foo')
            self.assertFalse(self.module.custom_add_called())
            device = self.module.custom_device()
            x = torch.empty(4, 4, device=device)
            y = torch.empty(4, 4, device=device)
            self.assertTrue(x.device == device)
            self.assertFalse(x.is_cpu)
            self.assertFalse(self.module.custom_add_called())
            z = x + y
            self.assertTrue(self.module.custom_add_called())
            z_cpu = z.to(device='cpu')
            self.assertTrue(z_cpu.is_cpu)
            self.assertFalse(z.is_cpu)
            self.assertTrue(z.device == device)
            self.assertEqual(z, z_cpu)
            z2 = z_cpu + z_cpu

        def test_before_common_registration():
            if False:
                for i in range(10):
                    print('nop')
            with self.assertRaisesRegex(RuntimeError, 'Expected one of cpu'):
                torch._register_device_module('xxx', DummyModule)
            torch.utils.rename_privateuse1_backend('foo')
            with self.assertRaisesRegex(RuntimeError, 'torch has no module of'):
                with torch.random.fork_rng(device_type='foo'):
                    pass
            self.assertFalse(hasattr(torch.Tensor, 'is_foo'))
            self.assertFalse(hasattr(torch.Tensor, 'foo'))
            self.assertFalse(hasattr(torch.TypedStorage, 'is_foo'))
            self.assertFalse(hasattr(torch.TypedStorage, 'foo'))
            self.assertFalse(hasattr(torch.UntypedStorage, 'is_foo'))
            self.assertFalse(hasattr(torch.UntypedStorage, 'foo'))
            self.assertFalse(hasattr(torch.nn.Module, 'foo'))

        def test_after_common_registration():
            if False:
                print('Hello World!')
            self.assertTrue(hasattr(torch.Tensor, 'is_foo'))
            self.assertTrue(hasattr(torch.Tensor, 'foo'))
            self.assertTrue(hasattr(torch.TypedStorage, 'is_foo'))
            self.assertTrue(hasattr(torch.TypedStorage, 'foo'))
            self.assertTrue(hasattr(torch.UntypedStorage, 'is_foo'))
            self.assertTrue(hasattr(torch.UntypedStorage, 'foo'))
            self.assertTrue(hasattr(torch.nn.Module, 'foo'))

        def test_common_registration():
            if False:
                i = 10
                return i + 15
            torch.utils.rename_privateuse1_backend('foo')
            with self.assertRaisesRegex(RuntimeError, 'torch.register_privateuse1_backend()'):
                torch.utils.rename_privateuse1_backend('xxx')
            torch._register_device_module('foo', DummyModule)
            self.assertTrue(torch.utils.backend_registration._get_custom_mod_func('device_count')() == 1)
            with self.assertRaisesRegex(RuntimeError, 'Try to call torch.foo'):
                torch.utils.backend_registration._get_custom_mod_func('func_name_')
            torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
            with self.assertRaisesRegex(RuntimeError, 'The custom device module of'):
                torch.utils.generate_methods_for_privateuse1_backend()

        def test_open_device_generator_registration_and_hooks():
            if False:
                i = 10
                return i + 15
            device = self.module.custom_device()
            self.assertFalse(self.module.custom_add_called())
            with self.assertRaisesRegex(RuntimeError, 'Please register a generator to the PrivateUse1 dispatch key'):
                gen_ = torch.Generator(device=device)
            self.module.register_generator_first()
            gen = torch.Generator(device=device)
            self.assertTrue(gen.device == device)
            with self.assertRaisesRegex(RuntimeError, 'Only can register a generator to the PrivateUse1 dispatch key once'):
                self.module.register_generator_second()
            self.module.register_hook()
            default_gen = self.module.default_generator(0)
            self.assertTrue(default_gen.device.type == torch._C._get_privateuse1_backend_name())

        def test_open_device_dispatchstub():
            if False:
                i = 10
                return i + 15
            torch.utils.rename_privateuse1_backend('foo')
            input_data = torch.randn(3, 4, 5, dtype=torch.float32, device='cpu')
            foo_input_data = input_data.to('foo')
            self.assertFalse(self.module.custom_abs_called())
            torch.abs(foo_input_data)
            self.assertTrue(self.module.custom_abs_called())

        def test_open_device_quantized():
            if False:
                while True:
                    i = 10
            torch.utils.rename_privateuse1_backend('foo')
            input_data = torch.randn(3, 4, 5, dtype=torch.float32, device='cpu').to('foo')
            quantized_tensor = torch.quantize_per_tensor(input_data, 0.1, 10, torch.qint8)
            self.assertEqual(quantized_tensor.device, torch.device('foo:0'))
            self.assertEqual(quantized_tensor.dtype, torch.qint8)

        def test_open_device_random():
            if False:
                return 10
            with torch.random.fork_rng(device_type='foo'):
                pass

        def test_open_device_tensor():
            if False:
                print('Hello World!')
            device = self.module.custom_device()
            dtypes = {torch.bool: 'torch.foo.BoolTensor', torch.double: 'torch.foo.DoubleTensor', torch.float32: 'torch.foo.FloatTensor', torch.half: 'torch.foo.HalfTensor', torch.int32: 'torch.foo.IntTensor', torch.int64: 'torch.foo.LongTensor', torch.int8: 'torch.foo.CharTensor', torch.short: 'torch.foo.ShortTensor', torch.uint8: 'torch.foo.ByteTensor'}
            for (tt, dt) in dtypes.items():
                test_tensor = torch.empty(4, 4, dtype=tt, device=device)
                self.assertTrue(test_tensor.type() == dt)
            x = torch.empty(4, 4)
            self.assertFalse(x.is_foo)
            x = x.foo(torch.device('foo'))
            self.assertFalse(self.module.custom_add_called())
            self.assertTrue(x.is_foo)
            y = torch.empty(4, 4)
            self.assertFalse(y.is_foo)
            y = y.foo(torch.device('foo:0'))
            self.assertFalse(self.module.custom_add_called())
            self.assertTrue(y.is_foo)
            z = torch.empty(4, 4)
            self.assertFalse(z.is_foo)
            z = z.foo(0)
            self.assertFalse(self.module.custom_add_called())
            self.assertTrue(z.is_foo)

        def test_open_device_storage():
            if False:
                for i in range(10):
                    print('nop')
            x = torch.empty(4, 4)
            z1 = x.storage()
            self.assertFalse(z1.is_foo)
            z1 = z1.foo()
            self.assertFalse(self.module.custom_add_called())
            self.assertTrue(z1.is_foo)
            with self.assertRaisesRegex(RuntimeError, 'Invalid device'):
                z1.foo(torch.device('cpu'))
            z1 = z1.cpu()
            self.assertFalse(self.module.custom_add_called())
            self.assertFalse(z1.is_foo)
            z1 = z1.foo(device='foo:0', non_blocking=False)
            self.assertFalse(self.module.custom_add_called())
            self.assertTrue(z1.is_foo)
            with self.assertRaisesRegex(RuntimeError, 'Invalid device'):
                z1.foo(device='cuda:0', non_blocking=False)
            y = torch.empty(4, 4)
            z2 = y.untyped_storage()
            self.assertFalse(z2.is_foo)
            z2 = z2.foo()
            self.assertFalse(self.module.custom_add_called())
            self.assertTrue(z2.is_foo)
            self.module.custom_storage_registry()
            z3 = y.untyped_storage()
            self.assertFalse(self.module.custom_storageImpl_called())
            z3 = z3.foo()
            self.assertTrue(self.module.custom_storageImpl_called())

        def test_open_device_storage_pin_memory():
            if False:
                print('Hello World!')
            torch.utils.rename_privateuse1_backend('foo')
            with self.assertRaisesRegex(RuntimeError, 'The custom device module of'):
                torch.utils.generate_methods_for_privateuse1_backend(for_tensor=False, for_module=False, for_storage=True)
            cpu_tensor = torch.empty(3)
            self.assertFalse(cpu_tensor.is_foo)
            self.assertFalse(cpu_tensor.is_pinned('foo'))
            cpu_tensor_pin = cpu_tensor.pin_memory('foo')
            self.assertTrue(cpu_tensor_pin.is_pinned('foo'))
            cpu_storage = cpu_tensor.storage()
            foo_device = torch.device('foo')
            self.assertFalse(cpu_storage.is_pinned('foo'))
            cpu_storage_pin = cpu_storage.pin_memory('foo')
            self.assertFalse(cpu_storage.is_pinned())
            self.assertFalse(cpu_storage.is_pinned('foo'))
            self.assertFalse(cpu_storage.is_pinned(foo_device))
            self.assertFalse(cpu_storage_pin.is_pinned())
            self.assertTrue(cpu_storage_pin.is_pinned('foo'))
            self.assertTrue(cpu_storage_pin.is_pinned(foo_device))
            cpu_storage_pin_already = cpu_storage_pin.pin_memory('foo')
            self.assertTrue(cpu_storage_pin.is_pinned('foo'))
            self.assertTrue(cpu_storage_pin.is_pinned(foo_device))
            self.assertTrue(cpu_storage_pin_already.is_pinned('foo'))
            self.assertTrue(cpu_storage_pin_already.is_pinned(foo_device))
            self.assertFalse(cpu_storage.is_pinned('foo'))
            cpu_storage_pinned = cpu_storage.pin_memory(foo_device)
            self.assertFalse(cpu_storage.is_pinned())
            self.assertFalse(cpu_storage.is_pinned('foo'))
            self.assertFalse(cpu_storage.is_pinned(foo_device))
            self.assertFalse(cpu_storage_pinned.is_pinned())
            self.assertTrue(cpu_storage_pinned.is_pinned('foo'))
            self.assertTrue(cpu_storage_pinned.is_pinned(foo_device))
            cpu_tensor = torch.randn([3, 2, 1, 4])
            cpu_untyped_storage = cpu_tensor.untyped_storage()
            self.assertFalse(cpu_untyped_storage.is_pinned())
            self.assertFalse(cpu_untyped_storage.is_pinned('foo'))
            cpu_untyped_storage_pinned = cpu_untyped_storage.pin_memory('foo')
            self.assertFalse(cpu_untyped_storage_pinned.is_pinned())
            self.assertTrue(cpu_untyped_storage_pinned.is_pinned('foo'))
            self.assertTrue(cpu_untyped_storage_pinned.is_pinned(foo_device))
            cpu_untyped_storage_pinned = cpu_untyped_storage.pin_memory(foo_device)
            self.assertFalse(cpu_untyped_storage_pinned.is_pinned())
            self.assertTrue(cpu_untyped_storage_pinned.is_pinned('foo'))
            self.assertTrue(cpu_untyped_storage_pinned.is_pinned(foo_device))
            with self.assertRaisesRegex(TypeError, 'positional arguments but 3 were given'):
                cpu_untyped_storage_pinned.is_pinned('foo1', 'foo2')
            self.assertFalse(cpu_storage_pinned.is_pinned('hpu'))
            with self.assertRaisesRegex(NotImplementedError, "with arguments from the 'HPU' backend"):
                cpu_storage.pin_memory('hpu')
            self.assertFalse(cpu_untyped_storage_pinned.is_pinned('hpu'))
            with self.assertRaisesRegex(NotImplementedError, "with arguments from the 'HPU' backend"):
                cpu_untyped_storage.pin_memory('hpu')
            invalid_device = torch.device('hpu')
            self.assertFalse(cpu_untyped_storage_pinned.is_pinned(invalid_device))
            with self.assertRaisesRegex(NotImplementedError, "with arguments from the 'HPU' backend"):
                cpu_untyped_storage.pin_memory(invalid_device)

        def test_open_device_serialization():
            if False:
                i = 10
                return i + 15
            self.module.set_custom_device_index(-1)
            storage = torch.UntypedStorage(4, device=torch.device('foo'))
            self.assertEqual(torch.serialization.location_tag(storage), 'foo')
            self.module.set_custom_device_index(0)
            storage = torch.UntypedStorage(4, device=torch.device('foo'))
            self.assertEqual(torch.serialization.location_tag(storage), 'foo:0')
            cpu_storage = torch.empty(4, 4).storage()
            foo_storage = torch.serialization.default_restore_location(cpu_storage, 'foo:0')
            self.assertTrue(foo_storage.is_foo)
            x = torch.empty(4, 4).long()
            y = x.foo()
            self.assertFalse(self.module.check_backend_meta(y))
            self.module.custom_set_backend_meta(y)
            self.assertTrue(self.module.check_backend_meta(y))
            self.module.custom_serialization_registry()
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, 'data.pt')
                torch.save(y, path)
                z1 = torch.load(path)
                self.assertTrue(z1.is_foo)
                self.assertTrue(self.module.check_backend_meta(z1))
                z2 = torch.load(path, map_location='cpu')
                self.assertFalse(z2.is_foo)
                self.assertFalse(self.module.check_backend_meta(z2))

        def test_open_device_storage_resize():
            if False:
                while True:
                    i = 10
            torch.utils.rename_privateuse1_backend('foo')
            cpu_tensor = torch.randn([8])
            foo_tensor = cpu_tensor.foo()
            foo_storage = foo_tensor.storage()
            self.assertTrue(foo_storage.size() == 8)
            foo_storage.resize_(8)
            self.assertTrue(foo_storage.size() == 8)
            with self.assertRaisesRegex(RuntimeError, 'Overflow'):
                foo_storage.resize_(8 ** 29)

        def test_open_device_storage_type():
            if False:
                i = 10
                return i + 15
            torch.utils.rename_privateuse1_backend('foo')
            cpu_tensor = torch.randn([8]).float()
            cpu_storage = cpu_tensor.storage()
            self.assertEqual(cpu_storage.type(), 'torch.FloatStorage')
            foo_tensor = cpu_tensor.foo()
            foo_storage = foo_tensor.storage()
            self.assertEqual(foo_storage.type(), 'torch.storage.TypedStorage')

            class CustomFloatStorage:

                @property
                def __module__(self):
                    if False:
                        i = 10
                        return i + 15
                    return 'torch.' + torch._C._get_privateuse1_backend_name()

                @property
                def __name__(self):
                    if False:
                        return 10
                    return 'FloatStorage'
            try:
                torch.foo.FloatStorage = CustomFloatStorage()
                self.assertEqual(foo_storage.type(), 'torch.foo.FloatStorage')
                foo_tensor2 = torch.randn([8]).int().foo()
                foo_storage2 = foo_tensor2.storage()
                self.assertEqual(foo_storage2.type(), 'torch.storage.TypedStorage')
            finally:
                torch.foo.FloatStorage = None

        def test_open_device_faketensor():
            if False:
                i = 10
                return i + 15
            torch.utils.rename_privateuse1_backend('foo')
            with torch._subclasses.fake_tensor.FakeTensorMode.push():
                a = torch.empty(1, device='foo')
                b = torch.empty(1, device='foo:0')
                result = a + b

        def test_open_device_named_tensor():
            if False:
                while True:
                    i = 10
            torch.utils.rename_privateuse1_backend('foo')
            a = torch.empty([2, 3, 4, 5], device='foo', names=['N', 'C', 'H', 'W'])

        def test_compile_autograd_function_returns_self():
            if False:
                while True:
                    i = 10
            x_ref = torch.randn(4, requires_grad=True)
            out_ref = self.module.custom_autograd_fn_returns_self(x_ref)
            out_ref.sum().backward()
            x_test = x_ref.clone().detach().requires_grad_(True)
            f_compiled = torch.compile(self.module.custom_autograd_fn_returns_self)
            out_test = f_compiled(x_test)
            out_test.sum().backward()
            self.assertEqual(out_ref, out_test)
            self.assertEqual(x_ref.grad, x_test.grad)

        def test_compile_autograd_function_aliasing():
            if False:
                i = 10
                return i + 15
            x_ref = torch.randn(4, requires_grad=True)
            out_ref = torch.ops._test_funcs.custom_autograd_fn_aliasing(x_ref)
            out_ref.sum().backward()
            x_test = x_ref.clone().detach().requires_grad_(True)
            f_compiled = torch.compile(torch.ops._test_funcs.custom_autograd_fn_aliasing)
            out_test = f_compiled(x_test)
            out_test.sum().backward()
            self.assertEqual(out_ref, out_test)
            self.assertEqual(x_ref.grad, x_test.grad)

        def test_open_device_tensor_type_fallback():
            if False:
                i = 10
                return i + 15
            torch.utils.rename_privateuse1_backend('foo')
            x = torch.Tensor([1, 2, 3]).to('foo')
            y = torch.Tensor([1, 0, 2]).to('foo')
            z_cpu = torch.Tensor([0, 2, 1])
            device = self.module.custom_device()
            self.assertTrue(x.device == device)
            self.assertFalse(x.is_cpu)
            z = torch.sub(x, y)
            self.assertEqual(z_cpu, z)

        def test_open_device_tensorlist_type_fallback():
            if False:
                i = 10
                return i + 15
            torch.utils.rename_privateuse1_backend('foo')
            v_foo = torch.Tensor([1, 2, 3]).to('foo')
            z_cpu = torch.Tensor([2, 4, 6])
            x = (v_foo, v_foo)
            y = (v_foo, v_foo)
            device = self.module.custom_device()
            self.assertTrue(v_foo.device == device)
            self.assertFalse(v_foo.is_cpu)
            z = torch._foreach_add(x, y)
            self.assertEqual(z_cpu, z[0])
            self.assertEqual(z_cpu, z[1])
        test_base_device_registration()
        test_before_common_registration()
        test_common_registration()
        test_after_common_registration()
        test_open_device_generator_registration_and_hooks()
        test_open_device_dispatchstub()
        test_open_device_random()
        test_open_device_tensor()
        test_open_device_storage()
        test_open_device_storage_pin_memory()
        test_open_device_serialization()
        test_open_device_storage_resize()
        test_open_device_storage_type()
        test_open_device_faketensor()
        test_open_device_named_tensor()
        test_open_device_quantized()
        test_compile_autograd_function_returns_self()
        test_compile_autograd_function_aliasing()
        test_open_device_tensor_type_fallback()
        test_open_device_tensorlist_type_fallback()
if __name__ == '__main__':
    common.run_tests()