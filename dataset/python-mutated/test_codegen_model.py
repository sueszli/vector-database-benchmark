import textwrap
import unittest
from typing import cast
import expecttest
import torchgen.dest as dest
import torchgen.gen as gen
import yaml
from torchgen.gen import LineLoader, parse_native_yaml_struct
from torchgen.model import Annotation, CustomClassType, DispatchKey, NativeFunctionsGroup, Type

class TestCodegenModel(expecttest.TestCase):

    def assertParseErrorInline(self, yaml_str: str, expect: str) -> None:
        if False:
            while True:
                i = 10
        es = yaml.load(yaml_str, Loader=LineLoader)
        try:
            parse_native_yaml_struct(es, set())
        except AssertionError as e:
            (msg, _) = str(e).split('  in ', 2)
            self.assertExpectedInline('\n'.join(textwrap.wrap(msg)), expect, skip=1)
            return
        self.fail(msg='Did not raise when expected to')

    def assertUfuncErrorInline(self, yaml_str: str, expect: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        es = yaml.load(yaml_str, Loader=LineLoader)
        parsed_yaml = parse_native_yaml_struct(es, set())
        (native_functions, backend_indices) = (parsed_yaml.native_functions, parsed_yaml.backend_indices)
        grouped_native_functions = gen.get_grouped_native_functions(native_functions)
        assert len(grouped_native_functions) == 1
        g = grouped_native_functions[0]
        assert isinstance(g, NativeFunctionsGroup)
        assert g.out.ufunc_inner_loop
        gen.compute_meta_function_declaration(g)
        dest.compute_native_function_declaration(g, backend_indices[DispatchKey.CPU])
        dest.compute_native_function_declaration(g, backend_indices[DispatchKey.CUDA])
        try:
            dest.compute_ufunc_cpu(g)
            dest.compute_ufunc_cpu_kernel(g)
            dest.compute_ufunc_cuda(g)
        except AssertionError as e:
            (msg, _) = str(e).split('  in ', 2)
            self.assertExpectedInline('\n'.join(textwrap.wrap(msg)), expect, skip=1)
            return
        self.fail(msg='Did not raise when expected to')
    binop_out = 'func: binop.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)'
    ti_binop_out = f'{binop_out}\n  structured: True\n  structured_inherits: TensorIteratorBase'
    ti_binop = 'func: binop(Tensor self, Tensor other) -> Tensor\n  structured_delegate: binop.out\n'
    ti_unop_out = 'func: unop.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)\n  structured: True\n  structured_inherits: TensorIteratorBase'
    ti_unop = 'func: unop(Tensor self) -> Tensor\n  structured_delegate: unop.out\n'

    def test_nonstructured_ufunc(self) -> None:
        if False:
            i = 10
            return i + 15
        yaml_str = f'- {self.binop_out}\n  ufunc_inner_loop:\n    Generic: binop (Bool)\n'
        self.assertParseErrorInline(yaml_str, 'ufunc must be structured')

    def test_overlapping_ufunc_and_dispatch(self) -> None:
        if False:
            i = 10
            return i + 15
        yaml_str = f'- {self.ti_binop_out}\n  ufunc_inner_loop:\n    Generic: binop (Bool)\n  dispatch:\n    CPU: binop_cpu\n'
        self.assertParseErrorInline(yaml_str, 'ufunc should not have explicit dispatch entry for CPU')

    @unittest.expectedFailure
    def test_scalaronly_shadowed(self) -> None:
        if False:
            return 10
        yaml_str = f'- {self.ti_binop_out}\n  ufunc_inner_loop:\n    Generic: binop (Bool)\n    ScalarOnly: binop (Bool)\n'
        self.assertParseErrorInline(yaml_str, '')

    def test_conflicting_ufunc(self) -> None:
        if False:
            return 10
        yaml_str = f'- {self.ti_binop_out}\n  ufunc_inner_loop:\n    Generic: binop (Bool)\n    ScalarOnly: binop_scalar (Bool)\n- {self.ti_binop}\n'
        self.assertUfuncErrorInline(yaml_str, 'ScalarOnly and Generic must have same ufunc name')

    def test_invalid_cudafunctoronself_for_binary_op(self) -> None:
        if False:
            while True:
                i = 10
        yaml_str = f'- {self.ti_unop_out}\n  ufunc_inner_loop:\n    Generic: unop (All)\n    CUDAFunctorOnSelf: unop_self_cuda (All)\n- {self.ti_unop}\n'
        self.assertUfuncErrorInline(yaml_str, 'cannot use CUDAFunctorOnSelf on non-binary function')

    def test_parse_custom_class_type(self) -> None:
        if False:
            return 10
        custom_class_name = 'namespace_foo.class_bar'
        custom_class_name_with_prefix = f'__torch__.torch.classes.{custom_class_name}'
        custom_class_type = cast(CustomClassType, Type.parse(custom_class_name_with_prefix))
        self.assertTrue(isinstance(custom_class_type, CustomClassType))
        self.assertEqual(custom_class_name, custom_class_type.class_name)
        self.assertEqual(custom_class_name_with_prefix, str(custom_class_type))

class TestAnnotation(expecttest.TestCase):

    def test_single_alias_no_write(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        a = Annotation.parse('a')
        self.assertEqual(a.alias_set, tuple('a'))
        self.assertFalse(a.is_write)
        self.assertEqual(a.alias_set_after, tuple())

    def test_single_alias_is_write(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        a = Annotation.parse('a!')
        self.assertEqual(a.alias_set, tuple('a'))
        self.assertTrue(a.is_write)
        self.assertEqual(a.alias_set_after, tuple())

    def test_single_alias_is_write_to_wildcard(self) -> None:
        if False:
            i = 10
            return i + 15
        a = Annotation.parse('a! -> *')
        self.assertEqual(a.alias_set, tuple('a'))
        self.assertTrue(a.is_write)
        self.assertEqual(a.alias_set_after, tuple('*'))

    def test_alias_set(self) -> None:
        if False:
            return 10
        a = Annotation.parse('a|b')
        self.assertEqual(a.alias_set, ('a', 'b'))

    def test_alias_set_is_write_raises_exception(self) -> None:
        if False:
            return 10
        with self.assertRaisesRegex(AssertionError, 'alias set larger than 1 is not mutable'):
            Annotation.parse('a|b!')

    def test_single_alias_is_write_to_alias_set(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        a = Annotation.parse('a! -> a|b')
        self.assertEqual(a.alias_set, tuple('a'))
        self.assertTrue(a.is_write)
        self.assertEqual(a.alias_set_after, ('a', 'b'))

    def test_before_and_after_alias_set_larger_than_1_raises_exception(self) -> None:
        if False:
            return 10
        with self.assertRaisesRegex(AssertionError, 'before alias set and after alias set cannot be larger than 1 at the same time'):
            Annotation.parse('a|b -> c|d')
if __name__ == '__main__':
    unittest.main()