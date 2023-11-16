import os
import tempfile
import unittest
from typing import Optional
import expecttest
from torchgen.gen import _GLOBAL_PARSE_NATIVE_YAML_CACHE
from torchgen.gen_backend_stubs import run
path = os.path.dirname(os.path.realpath(__file__))
gen_backend_stubs_path = os.path.join(path, '../torchgen/gen_backend_stubs.py')

class TestGenBackendStubs(expecttest.TestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        global _GLOBAL_PARSE_NATIVE_YAML_CACHE
        _GLOBAL_PARSE_NATIVE_YAML_CACHE.clear()

    def assert_success_from_gen_backend_stubs(self, yaml_str: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        with tempfile.NamedTemporaryFile(mode='w') as fp:
            fp.write(yaml_str)
            fp.flush()
            run(fp.name, '', True)

    def get_errors_from_gen_backend_stubs(self, yaml_str: str, *, kernels_str: Optional[str]=None) -> str:
        if False:
            while True:
                i = 10
        with tempfile.NamedTemporaryFile(mode='w') as fp:
            fp.write(yaml_str)
            fp.flush()
            try:
                if kernels_str is None:
                    run(fp.name, '', True)
                else:
                    with tempfile.NamedTemporaryFile(mode='w') as kernel_file:
                        kernel_file.write(kernels_str)
                        kernel_file.flush()
                        run(fp.name, '', True, impl_path=kernel_file.name)
            except AssertionError as e:
                return str(e).replace(fp.name, '')
            self.fail('Expected gen_backend_stubs to raise an AssertionError, but it did not.')

    def test_valid_single_op(self) -> None:
        if False:
            i = 10
            return i + 15
        yaml_str = 'backend: XLA\ncpp_namespace: torch_xla\nsupported:\n- abs'
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_multiple_ops(self) -> None:
        if False:
            while True:
                i = 10
        yaml_str = 'backend: XLA\ncpp_namespace: torch_xla\nsupported:\n- add.Tensor\n- abs'
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_zero_ops(self) -> None:
        if False:
            i = 10
            return i + 15
        yaml_str = 'backend: XLA\ncpp_namespace: torch_xla\nsupported:'
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_zero_ops_doesnt_require_backend_dispatch_key(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        yaml_str = 'backend: BAD_XLA\ncpp_namespace: torch_xla\nsupported:'
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_valid_with_autograd_ops(self) -> None:
        if False:
            return 10
        yaml_str = 'backend: XLA\ncpp_namespace: torch_xla\nsupported:\n- abs\nautograd:\n- add.Tensor'
        self.assert_success_from_gen_backend_stubs(yaml_str)

    def test_missing_backend(self) -> None:
        if False:
            while True:
                i = 10
        yaml_str = 'cpp_namespace: torch_xla\nsupported:\n- abs'
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, 'You must provide a value for "backend"')

    def test_empty_backend(self) -> None:
        if False:
            i = 10
            return i + 15
        yaml_str = 'backend:\ncpp_namespace: torch_xla\nsupported:\n- abs'
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, 'You must provide a value for "backend"')

    def test_backend_invalid_dispatch_key(self) -> None:
        if False:
            while True:
                i = 10
        yaml_str = 'backend: NOT_XLA\ncpp_namespace: torch_xla\nsupported:\n- abs'
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, 'unknown dispatch key NOT_XLA\n  The provided value for "backend" must be a valid DispatchKey, but got NOT_XLA.')

    def test_missing_cpp_namespace(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        yaml_str = 'backend: XLA\nsupported:\n- abs'
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, 'You must provide a value for "cpp_namespace"')

    def test_whitespace_cpp_namespace(self) -> None:
        if False:
            while True:
                i = 10
        yaml_str = 'backend: XLA\ncpp_namespace:\t\nsupported:\n- abs'
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, 'You must provide a value for "cpp_namespace"')

    def test_nonlist_supported(self) -> None:
        if False:
            while True:
                i = 10
        yaml_str = 'backend: XLA\ncpp_namespace: torch_xla\nsupported: abs'
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, 'expected "supported" to be a list, but got: abs (of type <class \'str\'>)')

    def test_supported_invalid_op(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        yaml_str = 'backend: XLA\ncpp_namespace: torch_xla\nsupported:\n- abs_BAD'
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, 'Found an invalid operator name: abs_BAD')

    def test_backend_has_no_autograd_key_but_provides_entries(self) -> None:
        if False:
            while True:
                i = 10
        yaml_str = 'backend: Vulkan\ncpp_namespace: torch_vulkan\nsupported:\n- add\nautograd:\n- sub'
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, 'Found an invalid operator name: add')

    def test_backend_autograd_kernel_mismatch_out_functional(self) -> None:
        if False:
            return 10
        yaml_str = 'backend: XLA\ncpp_namespace: torch_xla\nsupported:\n- add.Tensor\nautograd:\n- add.out'
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, 'Currently, all variants of an op must either be registered to a backend key, or to a backend\'s autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! add is listed under "supported", but add_out is listed under "autograd".')

    def test_backend_autograd_kernel_mismatch_functional_inplace(self) -> None:
        if False:
            i = 10
            return i + 15
        yaml_str = 'backend: XLA\ncpp_namespace: torch_xla\nsupported:\n- add.Tensor\nautograd:\n- add_.Tensor'
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, 'Currently, all variants of an op must either be registered to a backend key, or to a backend\'s autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! add is listed under "supported", but add_ is listed under "autograd".')

    def test_op_appears_in_supported_and_autograd_lists(self) -> None:
        if False:
            i = 10
            return i + 15
        yaml_str = 'backend: XLA\ncpp_namespace: torch_xla\nsupported:\n- add.Tensor\nautograd:\n- add.Tensor'
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, 'Currently, all variants of an op must either be registered to a backend key, or to a backend\'s autograd key. They cannot be mix and matched. If this is something you need, feel free to create an issue! add is listed under "supported", but add is listed under "autograd".')

    def test_unrecognized_key(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        yaml_str = 'backend: XLA\ncpp_namespace: torch_xla\nsupported:\n- abs\ninvalid_key: invalid_val'
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, ' contains unexpected keys: invalid_key. Only the following keys are supported: backend, class_name, cpp_namespace, extra_headers, supported, autograd, full_codegen, non_native, ir_gen, symint')

    def test_use_out_as_primary_non_bool(self) -> None:
        if False:
            return 10
        yaml_str = 'backend: XLA\ncpp_namespace: torch_xla\nuse_out_as_primary: frue\nsupported:\n- abs'
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, 'You must provide either True or False for use_out_as_primary. Provided: frue')

    def test_device_guard_non_bool(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        yaml_str = 'backend: XLA\ncpp_namespace: torch_xla\ndevice_guard: frue\nsupported:\n- abs'
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, 'You must provide either True or False for device_guard. Provided: frue')

    def test_incorrect_kernel_name(self) -> None:
        if False:
            i = 10
            return i + 15
        yaml_str = 'backend: XLA\ncpp_namespace: torch_xla\nsupported:\n- abs\nautograd:\n- add.Tensor'
        kernels_str = 'at::Tensor& XLANativeFunctions::absWRONG(at::Tensor& self) {}\nat::Tensor& XLANativeFunctions::add(at::Tensor& self) {}'
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str, kernels_str=kernels_str)
        self.assertExpectedInline(output_error, '\nXLANativeFunctions is missing a kernel definition for abs. We found 0 kernel(s) with that name,\nbut expected 1 kernel(s). The expected function schemas for the missing operator are:\nat::Tensor abs(const at::Tensor & self)\n\n')
if __name__ == '__main__':
    unittest.main()