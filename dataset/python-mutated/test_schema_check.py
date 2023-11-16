import os
import sys
import torch
from torch.utils._pytree import tree_map
import unittest
from torch.testing._internal.common_utils import run_tests
from torch.fx.operator_schemas import normalize_function
from torch._subclasses.schema_check_mode import SchemaCheckMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_device_type import ops, OpDTypes, instantiate_device_type_tests
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

def secretly_aliasing(x):
    if False:
        i = 10
        return i + 15
    return x.view(-1)

def secretly_mutating(x):
    if False:
        for i in range(10):
            print('nop')
    x.mul_(2)
    return x * 3

def output_is_input(x):
    if False:
        i = 10
        return i + 15
    return x
custom_lib = torch.library.Library('bad_schemas', 'DEF')
custom_lib.define('secretly_aliasing(Tensor x) -> Tensor')
custom_lib.define('secretly_mutating(Tensor x) -> Tensor')
custom_lib.define('output_is_input(Tensor(a) x) -> Tensor(a)')
custom_lib_cpu = torch.library.Library('bad_schemas', 'IMPL', 'CPU')
custom_lib_cpu.impl('secretly_aliasing', secretly_aliasing)
custom_lib_cpu.impl('secretly_mutating', secretly_mutating)
custom_lib_cpu.impl('output_is_input', output_is_input)
custom_lib_meta = torch.library.Library('bad_schemas', 'IMPL', 'Meta')
custom_lib_meta.impl('secretly_aliasing', secretly_aliasing)
custom_lib_meta.impl('secretly_mutating', secretly_mutating)
custom_lib_meta.impl('output_is_input', output_is_input)

class IncorrectAliasTensor(torch.Tensor):
    ALIAS_ARG_OUT = {'aten::add'}
    ALIAS_OUT_OUT = {'aten::aminmax'}
    MUTATE_ARGS_OUT = {'aten::sub'}
    elem: torch.Tensor
    __slots__ = ['elem']
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        r = torch.Tensor._make_wrapper_subclass(cls, elem.size(), strides=elem.stride(), storage_offset=elem.storage_offset(), dtype=elem.dtype, layout=elem.layout, device=elem.device, requires_grad=kwargs.get('requires_grad', False))
        r.elem = elem.detach() if r.requires_grad else elem
        return r

    def __repr__(self):
        if False:
            print('Hello World!')
        return super().__repr__(tensor_contents=f'{self.elem}')

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if False:
            print('Hello World!')

        def unwrap(e):
            if False:
                i = 10
                return i + 15
            return e.elem if isinstance(e, cls) else e

        def wrap(e):
            if False:
                print('Hello World!')
            return cls(e) if isinstance(e, torch.Tensor) else e
        unwrapped_args = tree_map(unwrap, args)
        out = func(*unwrapped_args, **tree_map(unwrap, kwargs))
        if func._schema.name in IncorrectAliasTensor.ALIAS_ARG_OUT:
            args[0].elem = out
        if func._schema.name in IncorrectAliasTensor.MUTATE_ARGS_OUT:
            args[0].elem = torch.rand(args[0].elem.shape)
        if func._schema.name in IncorrectAliasTensor.ALIAS_OUT_OUT:
            incorrect_out = list(out)
            incorrect_out[0] = incorrect_out[1]
            return tree_map(wrap, tuple(incorrect_out))
        return tree_map(wrap, out)

class TestSchemaCheck(JitTestCase):

    def test_schema_check_mode_operator_order(self):
        if False:
            for i in range(10):
                print('nop')
        with SchemaCheckMode() as schema_check:
            x = torch.rand((3, 3), requires_grad=True)
            x.relu().sin()
        self.assertEqual(['aten::rand', 'aten::relu', 'aten::detach', 'aten::sin'], schema_check.ops)

    def test_schema_check_mode_operator_order_without_grad(self):
        if False:
            return 10
        with SchemaCheckMode() as schema_check:
            x = torch.rand((3, 3), requires_grad=False)
            x.relu().sin()
        self.assertEqual(['aten::rand', 'aten::relu', 'aten::sin'], schema_check.ops)

    def test_schema_check_mode_mutated_aliasing_none(self):
        if False:
            print('Hello World!')
        x = torch.rand((3, 3))
        with SchemaCheckMode() as schema_check:
            actual = x.relu().sin()
        self.assertEqual([], schema_check.mutated)
        self.assertEqual([], schema_check.aliasing)

    def test_schema_check_mode_mutated_aliasing_mutation(self):
        if False:
            for i in range(10):
                print('nop')
        actual = torch.rand((3, 3), requires_grad=False)
        with SchemaCheckMode() as schema_check:
            actual.sinh_()
        self.assertEqual([('aten::sinh_', 'input')], schema_check.mutated)
        self.assertEqual([('aten::sinh_', 'input', 'output_0')], schema_check.aliasing)

    def test_schema_check_mode_mutated_aliasing_resize_(self):
        if False:
            print('Hello World!')
        actual = torch.rand((3, 3), requires_grad=False)
        with SchemaCheckMode() as schema_check:
            actual.resize_(9)
        self.assertEqual([('aten::resize_', 'input')], schema_check.mutated)
        self.assertEqual([('aten::resize_', 'input', 'output_0')], schema_check.aliasing)

    def test_schema_check_mode_mutated_aliasing_aliasing_inputs(self):
        if False:
            while True:
                i = 10
        actual = torch.rand((3, 3))
        y = actual
        with SchemaCheckMode() as schema_check:
            actual.add_(y)
        self.assertEqual([('aten::add_', 'input'), ('aten::add_', 'other')], schema_check.mutated)
        self.assertEqual([('aten::add_', 'input', 'output_0'), ('aten::add_', 'other', 'output_0')], schema_check.aliasing)

    def test_schema_check_mode_mutated_aliasing_as_strided(self):
        if False:
            while True:
                i = 10
        x = torch.rand((3, 6, 4))
        with SchemaCheckMode() as schema_check:
            x.as_strided_([3, 6, 4], [9, 1, 1])
        self.assertEqual([('aten::as_strided_', 'input')], schema_check.mutated)
        self.assertEqual([('aten::as_strided_', 'input', 'output_0')], schema_check.aliasing)

    def test_schema_check_mode_mutated_aliasing_multiple_outputs(self):
        if False:
            print('Hello World!')
        x = torch.arange(9.0)
        m_actual = torch.arange(9.0)
        e_actual = torch.zeros([9], dtype=torch.int32)
        with SchemaCheckMode() as schema_check:
            torch.frexp(x, out=(m_actual, e_actual))
        self.assertEqual([('aten::frexp', 'mantissa'), ('aten::frexp', 'exponent')], schema_check.mutated)
        self.assertEqual([('aten::frexp', 'mantissa', 'output_0'), ('aten::frexp', 'exponent', 'output_1')], schema_check.aliasing)

    def test_schema_check_mode_mutated_aliasing_aliasing_outputs(self):
        if False:
            i = 10
            return i + 15
        x = torch.rand((3, 3))
        actual = torch.zeros(3)
        with SchemaCheckMode() as schema_check:
            torch.aminmax(x, dim=0, out=[actual, actual])
        self.assertEqual([('aten::aminmax', 'min'), ('aten::aminmax', 'max')], schema_check.mutated)
        self.assertEqual([('aten::aminmax', 'min', 'output_0'), ('aten::aminmax', 'min', 'output_1'), ('aten::aminmax', 'max', 'output_0'), ('aten::aminmax', 'max', 'output_1')], schema_check.aliasing)

    def test_schema_check_mode_functionality(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.rand((3, 3), requires_grad=True)
        expected = x.relu().sin()
        with SchemaCheckMode():
            actual = x.relu().sin()
        self.assertEqual(expected, actual)

    def test_schema_check_mode_functionality_default_replaced(self):
        if False:
            while True:
                i = 10
        x = torch.rand((3, 3), requires_grad=True)
        expected = x.add(x, alpha=2)
        with SchemaCheckMode():
            actual = x.add(x, alpha=2)
        self.assertEqual(expected, actual)

    def test_schema_check_mode_functionality_list_input(self):
        if False:
            for i in range(10):
                print('nop')
        a = torch.rand((3, 3))
        b = torch.rand((3, 3))
        c = torch.rand((3, 3))
        expected = torch.linalg.multi_dot([a, b, c])
        with SchemaCheckMode():
            actual = torch.linalg.multi_dot([a, b, c])
        self.assertEqual(expected, actual)

    def test_schema_check_mode_functionality_wildcard_after(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.rand((3, 3))
        expected = x.chunk(6)
        with SchemaCheckMode():
            actual = x.chunk(6)
        self.assertEqual(expected, actual)

    @unittest.skipIf(not torch._C.has_spectral, 'ATen not built with FFT.')
    def test_schema_check_mode_functionality_kwarg_tensor(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.rand((3, 5))
        w = torch.rand(4)
        expected = torch.stft(x, 4, win_length=4, window=w, return_complex=True)
        with SchemaCheckMode():
            actual = torch.stft(x, 4, win_length=4, window=w, return_complex=True)
        self.assertEqual(expected, actual)

    def test_schema_check_mode_functionality_mutable_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        expected = torch.rand((3, 3), requires_grad=False)
        actual = torch.clone(expected)
        expected.sinh_()
        with SchemaCheckMode():
            actual.sinh_()
        self.assertEqual(expected, actual)

    def test_schema_check_mode_functionality_aliasing_inputs(self):
        if False:
            return 10
        expected = torch.rand((3, 3))
        x = expected
        actual = torch.clone(expected)
        y = actual
        expected.add_(x)
        with SchemaCheckMode():
            actual.add_(y)
        self.assertEqual(expected, actual)

    def test_schema_check_mode_functionality_with_multiple_outputs(self):
        if False:
            print('Hello World!')
        x = torch.arange(9.0)
        (m_expected, e_expected) = torch.frexp(x)
        m_actual = torch.arange(9.0)
        e_actual = torch.zeros([9], dtype=torch.int32)
        with SchemaCheckMode():
            torch.frexp(x, out=(m_actual, e_actual))
        self.assertEqual(m_expected, m_actual)
        self.assertEqual(e_expected, e_actual)

    def test_schema_check_mode_functionality_with_multiple_outputs_aliasing(self):
        if False:
            while True:
                i = 10
        x = torch.rand((3, 3))
        actual = torch.zeros(3)
        with SchemaCheckMode():
            torch.aminmax(x, dim=0, out=[actual, actual])
        self.assertEqual(torch.amax(x, dim=0), actual)

    def test_schema_check_mode_functionality_device_input(self):
        if False:
            while True:
                i = 10
        with SchemaCheckMode():
            x = torch.rand((3, 3), device='cpu', dtype=torch.double)
            y = x + x
        self.assertEqual(x + x, y)

    def test_schema_check_mode_functionality_training_op(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.rand((3, 3), requires_grad=True)
        batch = torch.nn.BatchNorm1d(3, track_running_stats=True)
        expected = batch(x)
        with SchemaCheckMode():
            actual = batch(x)
        self.assertEqual(expected, actual)

    def test_schema_check_mode_functionality_nested_training_op(self):
        if False:
            while True:
                i = 10
        actual = torch.rand((3, 3))
        batch = torch.nn.BatchNorm1d(3, track_running_stats=True)
        expected = torch.clone(actual)
        expected.sinh_()
        expected.tanh_()
        expected.relu_()
        expected = batch(expected)
        with SchemaCheckMode():
            actual.sinh_()
            actual.tanh_()
            actual.relu_()
            actual = batch(actual)
        self.assertEqual(expected, actual)

    def test_schema_check_mode_empty_list_input(self):
        if False:
            for i in range(10):
                print('nop')
        expected = torch.atleast_1d([])
        with SchemaCheckMode():
            actual = torch.atleast_1d([])
        self.assertEqual(expected, actual)

    def test_mutation_check_fail(self):
        if False:
            return 10
        with self.assertRaisesRegex(RuntimeError, 'Argument input is not defined as mutable but was mutated'):
            x = torch.rand((3, 3))
            y = torch.rand((3, 3))
            with SchemaCheckMode():
                IncorrectAliasTensor(x).sub(IncorrectAliasTensor(y))

    def test_mutation_check_fail_multiple_operators(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, 'Argument input is not defined as mutable but was mutated'):
            x = torch.rand((3, 3))
            y = torch.rand((3, 3))
            with SchemaCheckMode():
                IncorrectAliasTensor(x).sin().cos().sub(IncorrectAliasTensor(y))

    def test_alias_check_fail_simple(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, 'Argument input is not defined to alias output but was aliasing'):
            x = torch.rand((3, 3), requires_grad=True)
            y = torch.rand((3, 3))
            with SchemaCheckMode():
                IncorrectAliasTensor(x).add(IncorrectAliasTensor(y), alpha=2)

    def test_alias_check_fail_multiple_operators(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, 'Argument input is not defined to alias output but was aliasing'):
            x = torch.rand((3, 3), requires_grad=True)
            y = torch.zeros((3, 3), requires_grad=True)
            with SchemaCheckMode():
                IncorrectAliasTensor(x).sin().relu().add(IncorrectAliasTensor(y), alpha=2)

    def test_alias_check_fail_multiple_operators_centered(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, 'Argument input is not defined to alias output but was aliasing'):
            x = torch.rand((3, 3), requires_grad=True)
            y = torch.zeros((3, 3), requires_grad=True)
            with SchemaCheckMode():
                IncorrectAliasTensor(x).sin().add(IncorrectAliasTensor(y), alpha=2).relu()

    def test_alias_check_fail_outputs_unexpectedly_aliasing(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, 'Outputs 0 and 1 alias unexpectedly'):
            x = torch.rand((3, 3))
            with SchemaCheckMode() as s:
                IncorrectAliasTensor(x).aminmax(dim=0)

    def test_alias_check_fail_custom_ops_secretly_aliasing(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                while True:
                    i = 10
            return torch.ops.bad_schemas.secretly_aliasing(x)
        x = torch.rand((3, 3))
        with self.assertRaisesRegex(RuntimeError, 'not defined to alias output but was aliasing'):
            with SchemaCheckMode() as s:
                out = f(x)

    def test_alias_check_fail_custom_ops_secretly_mutating(self):
        if False:
            return 10

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.ops.bad_schemas.secretly_mutating(x)
        x = torch.rand((3, 3))
        with self.assertRaisesRegex(RuntimeError, 'not defined as mutable but was mutated'):
            with SchemaCheckMode() as s:
                out = f(x)

    def test_alias_check_fail_custom_ops_output_is_input(self):
        if False:
            return 10

        def f(x):
            if False:
                while True:
                    i = 10
            return torch.ops.bad_schemas.output_is_input(x)
        x = torch.rand((3, 3))
        with self.assertRaisesRegex(RuntimeError, 'are not allowed to directly return inputs'):
            with SchemaCheckMode() as s:
                out = f(x)

    def test_is_alias_of_basic(self):
        if False:
            return 10
        x = torch.rand((3, 3), requires_grad=True)
        y = torch.rand((3, 3), requires_grad=True)
        y = x.add(x, alpha=2)
        self.assertTrue(torch._C._is_alias_of(x, x))
        self.assertFalse(torch._C._is_alias_of(x, y))

    def test_is_alias_of_empty_container(self):
        if False:
            for i in range(10):
                print('nop')
        x = []
        y = torch.rand((3, 3), requires_grad=True)
        self.assertFalse(torch._C._is_alias_of(x, x))
        self.assertFalse(torch._C._is_alias_of(x, y))

    def test_overlaps_basic(self):
        if False:
            return 10
        x = torch.rand((3, 3), requires_grad=True)
        y = torch.rand((3, 3), requires_grad=True)
        z = [x, y]
        self.assertTrue(torch._C._overlaps(x, x))
        self.assertFalse(torch._C._overlaps(x, y))
        self.assertTrue(torch._C._overlaps(z, x))
        self.assertTrue(torch._C._overlaps(z, y))

    def test_overlaps_empty_container(self):
        if False:
            print('Hello World!')
        x = []
        y = [torch.rand((3, 3), requires_grad=True)]
        self.assertFalse(torch._C._overlaps(y, x))
        self.assertTrue(torch._C._overlaps(y, y))

    def test_schema_info_bind_basic(self):
        if False:
            while True:
                i = 10

        class SchemaInfoBindTestMode(TorchDispatchMode):

            def __init__(self, test_self):
                if False:
                    print('Hello World!')
                self.test_self = test_self

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    return 10
                named_arg_list = normalize_function(func, args, kwargs, normalize_to_only_use_kwargs=True).kwargs
                schema_info_value_test = torch._C._SchemaInfo(func._schema)
                schema_info_values_test = torch._C._SchemaInfo(func._schema)
                self.test_self.assertFalse(schema_info_value_test.may_alias(torch._C._SchemaArgument(torch._C._SchemaArgType.input, 0), torch._C._SchemaArgument(torch._C._SchemaArgType.input, 1)))
                self.test_self.assertFalse(schema_info_values_test.may_alias(torch._C._SchemaArgument(torch._C._SchemaArgType.input, 0), torch._C._SchemaArgument(torch._C._SchemaArgType.input, 1)))
                for i in named_arg_list:
                    schema_info_value_test.add_argument_value(i, named_arg_list[i])
                schema_info_values_test.add_argument_values(named_arg_list)
                self.test_self.assertTrue(schema_info_value_test.may_alias(torch._C._SchemaArgument(torch._C._SchemaArgType.input, 0), torch._C._SchemaArgument(torch._C._SchemaArgType.input, 1)))
                self.test_self.assertTrue(schema_info_values_test.may_alias(torch._C._SchemaArgument(torch._C._SchemaArgType.input, 0), torch._C._SchemaArgument(torch._C._SchemaArgType.input, 1)))
                return func(*args, **kwargs)
        x = torch.rand((3, 3))
        with SchemaInfoBindTestMode(self) as schemaInfoCheck:
            x.add(x)

class TestSchemaCheckModeOpInfo(JitTestCase):

    @ops(op_db, dtypes=OpDTypes.supported)
    def test_schema_correctness(self, device, dtype, op):
        if False:
            return 10
        if dtype == torch.complex32:
            return
        for sample in op.sample_inputs(device, dtype, requires_grad=False):
            with SchemaCheckMode():
                op(sample.input, *sample.args, **sample.kwargs)
instantiate_device_type_tests(TestSchemaCheckModeOpInfo, globals(), only_for=('cpu', 'cuda'))
if __name__ == '__main__':
    run_tests()