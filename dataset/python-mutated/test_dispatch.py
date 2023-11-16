import torch._C as C
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._python_dispatcher import PythonDispatcher
from collections import namedtuple
import itertools
import os
import re
import torch.utils.cpp_extension
Result = namedtuple('Result', 'state table provenance')
dispatch_keys_to_check = ('Undefined', 'CPU', 'CUDA', 'XLA', 'AutogradOther', 'AutogradCPU', 'AutogradCUDA', 'AutogradXLA')

def extract_dispatch_table_with_keys(table, dispatch_keys):
    if False:
        print('Hello World!')
    extracted = ''
    table_entries = table.split('\n')
    regex = re.compile('registered at .*FallbackKernel\\.cpp.*(\\[)')
    for k in dispatch_keys:
        for t in table_entries:
            if t.startswith(k):
                entry = regex.sub('registered in pytorch framework [', t)
                extracted += entry + '\n'
    return extracted

class TestDispatch(TestCase):
    namespace_index = 0

    def test_all_invariants(self):
        if False:
            while True:
                i = 10
        C._dispatch_check_all_invariants()

    def run_ops(self, name, ops, ctor_order=None, dtor_order=None, results=None, expect_raises=False):
        if False:
            i = 10
            return i + 15
        "\n        Given a list of operator registrations, run the registrations in the\n        order specified by ctor_order, and then run the deregistrations in\n        dtor_order.\n\n        If results is specified, intermediate results are checked for consistency\n        with results stored in results (and stored in results if this is the\n        first time we've seen them).  Results are expected to be equivalent\n        modulo commutativity and inverses (thus, results is keyed on a frozenset\n        of in effect registrations from ops).  Results stores namedtuple\n        Result[state, table, provenance], where state is a string that contains\n        non-derived kernel registered or error message if it doesn't pass;\n        table is a string that contains computed dispatch table entries;\n        provenance is a string that describes how exactly we got this string.\n\n        If expect_raises is True, it is not an error to raise an exception.  Instead,\n        we'll store the exception string (instead of the dispatcher state)\n        in results.  In principle we should flag these differently, but it's\n        very obvious when you get an error in one case but not another.\n        "
        self.__class__.namespace_index += 1
        if results is None:
            results = {}
        if ctor_order is None:
            ctor_order = list(range(len(ops)))
        if dtor_order is None:
            dtor_order = list(reversed(ctor_order))
        refs = [None] * len(ops)
        active_ops = set()
        test_namespace = f'__test{self.namespace_index}__'

        def check_invariants(actual_provenance):
            if False:
                for i in range(10):
                    print('nop')
            C._dispatch_check_invariants(name)
            actual_state = C._dispatch_dump(f'{test_namespace}::{name}').replace(test_namespace, 'test')
            actual_table = C._dispatch_dump_table(f'{test_namespace}::{name}').replace(test_namespace, 'test')
            (expected_state, expected_table, expected_provenance) = results.setdefault(frozenset(active_ops), Result(actual_state, actual_table, actual_provenance))
            self.assertMultiLineEqual(expected_state, actual_state, f'expected from {expected_provenance}; actual from {actual_provenance}')
            self.assertMultiLineEqual(expected_table, actual_table, f'expected from {expected_provenance}; actual from {actual_provenance}')
        results.setdefault(frozenset(), Result('', '', 'hardcoded initial state'))
        check_invariants('initial state')
        set_to_report = frozenset(range(len(ops)))
        for (i, op_ix) in enumerate(ctor_order):
            refs[op_ix] = C._dispatch_library('FRAGMENT', test_namespace, '')
            active_ops.add(op_ix)
            try:
                ops[op_ix](refs[op_ix])
                check_invariants(f'running ctors {ctor_order[:i + 1]}')
            except RuntimeError as e:
                if not expect_raises:
                    raise
                actual = str(e).replace(test_namespace, 'test')
                actual = actual.split('\nException raised from ')[0]
                (expected, _, expected_provenance) = results.setdefault(frozenset(active_ops), Result(actual, '', f'error after running ctors {ctor_order[:i + 1]}'))
                self.assertMultiLineEqual(expected, actual, expected_provenance)
                set_to_report = frozenset(active_ops)
                active_ops.remove(op_ix)
                check_invariants(f"running ctors {ctor_order[:i]} and then failing to run ctor {op_ix} (did this failure leave the dispatcher in a wedged state? it shouldn't!)")
                break
        last_ctor = i
        if expect_raises and len(active_ops) == len(ops):
            refs = None
            self.assertTrue(False, f'expected exception to be raised, but nothing was raised (after running ctors {ctor_order})')
        for (i, op_ix) in enumerate(dtor_order):
            refs[op_ix] = None
            if expect_raises:
                active_ops.discard(op_ix)
            else:
                active_ops.remove(op_ix)
            check_invariants(f'running ctors {ctor_order[:last_ctor + 1]}, then running dtors {dtor_order[:i + 1]}')
        return results[set_to_report][0]

    def commute(self, name, ops, ctor_order=None, expect_raises=False):
        if False:
            i = 10
            return i + 15
        results = {}

        def go(ctor_order):
            if False:
                i = 10
                return i + 15
            for dtor_order in itertools.permutations(range(len(ops))):
                self.run_ops(name, ops, ctor_order, dtor_order, results=results, expect_raises=expect_raises)
        if ctor_order is not None:
            go(ctor_order)
        else:
            for ctor_order in itertools.permutations(range(len(ops))):
                go(ctor_order)
        return results[frozenset(range(len(ops)))]

    def test_def(self):
        if False:
            return 10
        state = self.commute('foo', [lambda m: m.def_('foo(Tensor x) -> Tensor'), lambda m: m.impl_t_t('foo'), lambda m: m.impl_t_t('foo', dispatch='CPU'), lambda m: m.impl_t_t('foo', dispatch='Autograd'), lambda m: m.impl_t_t('foo', dispatch='AutogradCPU')]).state
        self.assertExpectedInline(state, 'name: test::foo\nschema: test::foo(Tensor x) -> Tensor\ndebug: registered at /dev/null:0\nalias analysis kind: FROM_SCHEMA\nCPU: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nAutogradCPU: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nAutograd[alias]: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nCompositeImplicitAutograd[alias]: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\n')

    def test_def_impl_schema_mismatch(self):
        if False:
            print('Hello World!')
        state = self.commute('foo', [lambda m: m.def_('foo(Tensor x, Tensor y) -> Tensor'), lambda m: m.impl_t_t('foo')], expect_raises=True).state
        self.assertExpectedInline(state, "Inferred operator schema for a C++ kernel function doesn't match the expected function schema.\n  operator: test::foo\n  expected schema: test::foo(Tensor x, Tensor y) -> Tensor\n    registered at /dev/null:0\n  inferred schema: (Tensor _0) -> Tensor _0\n    impl_t_t\n  reason: The number of arguments is different. 2 vs 1.")

    def test_def_with_inference(self):
        if False:
            while True:
                i = 10
        state = self.commute('foo', [lambda m: m.def_name_t_t('foo'), lambda m: m.impl_t_t('foo', 'CPU'), lambda m: m.impl_t_t('foo', 'Autograd'), lambda m: m.impl_t_t('foo', 'AutogradCPU')]).state
        self.assertExpectedInline(state, 'name: test::foo\nschema: test::foo(Tensor _0) -> Tensor _0\ndebug: registered at /dev/null:0\nalias analysis kind: CONSERVATIVE\nCPU: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nAutogradCPU: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nAutograd[alias]: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nCompositeImplicitAutograd[alias]: default_def_name_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\n')

    def test_def_only(self):
        if False:
            for i in range(10):
                print('nop')
        state = self.commute('foo', [lambda m: m.def_('foo(Tensor x, Tensor y) -> Tensor')]).state
        self.assertExpectedInline(state, 'name: test::foo\nschema: test::foo(Tensor x, Tensor y) -> Tensor\ndebug: registered at /dev/null:0\nalias analysis kind: FROM_SCHEMA\n')

    def test_impl_only(self):
        if False:
            print('Hello World!')
        state = self.commute('foo', [lambda m: m.impl_t_t('foo'), lambda m: m.impl_t_t('foo', 'CPU'), lambda m: m.impl_t_t('foo', 'Autograd'), lambda m: m.impl_t_t('foo', 'AutogradCPU')]).state
        self.assertExpectedInline(state, 'name: test::foo\nschema: (none)\nCPU: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nAutogradCPU: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nAutograd[alias]: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nCompositeImplicitAutograd[alias]: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\n')

    def test_computed_table(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.commute('foo', [lambda m: m.def_name_t_t('foo'), lambda m: m.impl_t_t('foo', 'CPU', debug='fn_cpu'), lambda m: m.impl_t_t('foo', 'XLA', debug='fn_xla'), lambda m: m.impl_t_t('foo', 'Autograd', debug='fn_autograd'), lambda m: m.impl_t_t('foo', 'AutogradCPU', debug='fn_autogradcpu')])
        (state, table) = (result.state, result.table)
        self.assertExpectedInline(state, 'name: test::foo\nschema: test::foo(Tensor _0) -> Tensor _0\ndebug: registered at /dev/null:0\nalias analysis kind: CONSERVATIVE\nCPU: fn_cpu :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nXLA: fn_xla :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nAutogradCPU: fn_autogradcpu :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nAutograd[alias]: fn_autograd :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nCompositeImplicitAutograd[alias]: default_def_name_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\n')
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)
        self.assertExpectedInline(extracted_table, 'Undefined: default_def_name_t_t [math kernel]\nCPU: fn_cpu [kernel]\nCUDA: default_def_name_t_t [math kernel]\nXLA: fn_xla [kernel]\nAutogradOther: default_def_name_t_t [math kernel]\nAutogradCPU: fn_autogradcpu [kernel]\nAutogradCUDA: default_def_name_t_t [math kernel]\nAutogradXLA: fn_autograd [autograd kernel]\n')

    def test_computed_table_with_cpu_math_autogradcpu_fallthrough(self):
        if False:
            while True:
                i = 10
        global_m = C._dispatch_library('IMPL', '_', 'AutogradCPU')
        result = self.commute('foo', [lambda m: m.def_name_t_t('foo'), lambda m: m.impl_t_t('foo', 'CPU')])
        (state, table) = (result.state, result.table)
        self.assertExpectedInline(state, 'name: test::foo\nschema: test::foo(Tensor _0) -> Tensor _0\ndebug: registered at /dev/null:0\nalias analysis kind: CONSERVATIVE\nCPU: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nCompositeImplicitAutograd[alias]: default_def_name_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\n')
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)
        self.assertExpectedInline(extracted_table, 'Undefined: default_def_name_t_t [math kernel]\nCPU: impl_t_t [kernel]\nCUDA: default_def_name_t_t [math kernel]\nXLA: default_def_name_t_t [math kernel]\nAutogradOther: default_def_name_t_t [math kernel]\nAutogradCPU: registered in pytorch framework [backend fallback]\nAutogradCUDA: default_def_name_t_t [math kernel]\nAutogradXLA: default_def_name_t_t [math kernel]\n')

    def test_computed_table_with_math(self):
        if False:
            for i in range(10):
                print('nop')
        global_m = C._dispatch_library('IMPL', '_', 'AutogradCPU')
        result = self.commute('foo', [lambda m: m.def_('foo(Tensor x) -> Tensor'), lambda m: m.impl_t_t('foo', 'CompositeImplicitAutograd')])
        (state, table) = (result.state, result.table)
        self.assertExpectedInline(state, 'name: test::foo\nschema: test::foo(Tensor x) -> Tensor\ndebug: registered at /dev/null:0\nalias analysis kind: FROM_SCHEMA\nCompositeImplicitAutograd[alias]: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\n')
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)
        self.assertExpectedInline(extracted_table, 'Undefined: impl_t_t [math kernel]\nCPU: impl_t_t [math kernel]\nCUDA: impl_t_t [math kernel]\nXLA: impl_t_t [math kernel]\nAutogradOther: impl_t_t [math kernel]\nAutogradCPU: impl_t_t [math kernel]\nAutogradCUDA: impl_t_t [math kernel]\nAutogradXLA: impl_t_t [math kernel]\n')

    def test_computed_table_with_cpu_math(self):
        if False:
            for i in range(10):
                print('nop')
        global_m = C._dispatch_library('IMPL', '_', 'AutogradCPU')
        result = self.commute('foo', [lambda m: m.def_('foo(Tensor x) -> Tensor'), lambda m: m.impl_t_t('foo', 'CPU', debug='fn_cpu'), lambda m: m.impl_t_t('foo', 'CompositeImplicitAutograd', debug='fn_math')])
        (state, table) = (result.state, result.table)
        self.assertExpectedInline(state, 'name: test::foo\nschema: test::foo(Tensor x) -> Tensor\ndebug: registered at /dev/null:0\nalias analysis kind: FROM_SCHEMA\nCPU: fn_cpu :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nCompositeImplicitAutograd[alias]: fn_math :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\n')
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)
        self.assertExpectedInline(extracted_table, 'Undefined: fn_math [math kernel]\nCPU: fn_cpu [kernel]\nCUDA: fn_math [math kernel]\nXLA: fn_math [math kernel]\nAutogradOther: fn_math [math kernel]\nAutogradCPU: registered in pytorch framework [backend fallback]\nAutogradCUDA: fn_math [math kernel]\nAutogradXLA: fn_math [math kernel]\n')

    def test_computed_table_with_autograd(self):
        if False:
            return 10
        global_m = C._dispatch_library('IMPL', '_', 'AutogradCPU')
        result = self.commute('foo', [lambda m: m.def_('foo(Tensor x) -> Tensor'), lambda m: m.impl_t_t('foo', 'Autograd')])
        (state, table) = (result.state, result.table)
        self.assertExpectedInline(state, 'name: test::foo\nschema: test::foo(Tensor x) -> Tensor\ndebug: registered at /dev/null:0\nalias analysis kind: FROM_SCHEMA\nAutograd[alias]: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\n')
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)
        self.assertExpectedInline(extracted_table, 'AutogradOther: impl_t_t [autograd kernel]\nAutogradCPU: impl_t_t [autograd kernel]\nAutogradCUDA: impl_t_t [autograd kernel]\nAutogradXLA: impl_t_t [autograd kernel]\n')

    def test_computed_table_with_cpu_autograd_math(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.commute('foo', [lambda m: m.def_('foo(Tensor x) -> Tensor'), lambda m: m.impl_t_t('foo', 'CPU', debug='fn_cpu'), lambda m: m.impl_t_t('foo', 'Autograd', debug='fn_autograd'), lambda m: m.impl_t_t('foo', 'CompositeImplicitAutograd', debug='fn_math')])
        (state, table) = (result.state, result.table)
        self.assertExpectedInline(state, 'name: test::foo\nschema: test::foo(Tensor x) -> Tensor\ndebug: registered at /dev/null:0\nalias analysis kind: FROM_SCHEMA\nCPU: fn_cpu :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nAutograd[alias]: fn_autograd :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nCompositeImplicitAutograd[alias]: fn_math :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\n')
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)
        self.assertExpectedInline(extracted_table, 'Undefined: fn_math [math kernel]\nCPU: fn_cpu [kernel]\nCUDA: fn_math [math kernel]\nXLA: fn_math [math kernel]\nAutogradOther: fn_math [math kernel]\nAutogradCPU: fn_autograd [autograd kernel]\nAutogradCUDA: fn_math [math kernel]\nAutogradXLA: fn_math [math kernel]\n')

    def test_computed_table_with_ambiguous_autogradother(self):
        if False:
            print('Hello World!')
        result = self.commute('foo', [lambda m: m.def_('foo(Tensor x) -> Tensor'), lambda m: m.impl_t_t('foo', 'CompositeImplicitAutograd', debug='fn_math'), lambda m: m.impl_t_t('foo', 'FPGA', debug='fn_fpga')])
        (state, table) = (result.state, result.table)
        self.assertExpectedInline(state, 'name: test::foo\nschema: test::foo(Tensor x) -> Tensor\ndebug: registered at /dev/null:0\nalias analysis kind: FROM_SCHEMA\nFPGA: fn_fpga :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nCompositeImplicitAutograd[alias]: fn_math :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\n')
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check + ('FPGA',))
        self.assertExpectedInline(extracted_table, 'Undefined: fn_math [math kernel]\nCPU: fn_math [math kernel]\nCUDA: fn_math [math kernel]\nXLA: fn_math [math kernel]\nAutogradOther: ambiguous_autogradother [ambiguous autogradother]\nAutogradCPU: fn_math [math kernel]\nAutogradCUDA: fn_math [math kernel]\nAutogradXLA: fn_math [math kernel]\nFPGA: fn_fpga [kernel]\n')

    def test_computed_table_with_cpu_defaultbackend(self):
        if False:
            print('Hello World!')
        result = self.commute('foo', [lambda m: m.def_('foo(Tensor x) -> Tensor'), lambda m: m.impl_t_t('foo', 'CPU', debug='fn_cpu'), lambda m: m.impl_t_t('foo', 'CompositeExplicitAutograd', debug='fn_defaultbackend')])
        (state, table) = (result.state, result.table)
        self.assertExpectedInline(state, 'name: test::foo\nschema: test::foo(Tensor x) -> Tensor\ndebug: registered at /dev/null:0\nalias analysis kind: FROM_SCHEMA\nCPU: fn_cpu :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nCompositeExplicitAutograd[alias]: fn_defaultbackend :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\n')
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)
        self.assertExpectedInline(extracted_table, 'Undefined: fn_defaultbackend [default backend kernel]\nCPU: fn_cpu [kernel]\nCUDA: fn_defaultbackend [default backend kernel]\nXLA: fn_defaultbackend [default backend kernel]\nAutogradOther: registered in pytorch framework [backend fallback]\nAutogradCPU: registered in pytorch framework [backend fallback]\nAutogradCUDA: registered in pytorch framework [backend fallback]\nAutogradXLA: registered in pytorch framework [backend fallback]\n')

    def test_computed_table_with_cpu_autograd_defaultbackend(self):
        if False:
            i = 10
            return i + 15
        result = self.commute('foo', [lambda m: m.def_('foo(Tensor x) -> Tensor'), lambda m: m.impl_t_t('foo', 'CPU', debug='fn_cpu'), lambda m: m.impl_t_t('foo', 'Autograd', debug='fn_autograd'), lambda m: m.impl_t_t('foo', 'CompositeExplicitAutograd', debug='fn_defaultbackend')])
        (state, table) = (result.state, result.table)
        self.assertExpectedInline(state, 'name: test::foo\nschema: test::foo(Tensor x) -> Tensor\ndebug: registered at /dev/null:0\nalias analysis kind: FROM_SCHEMA\nCPU: fn_cpu :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nAutograd[alias]: fn_autograd :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nCompositeExplicitAutograd[alias]: fn_defaultbackend :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\n')
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check + ('FPGA',))
        self.assertExpectedInline(extracted_table, 'Undefined: fn_defaultbackend [default backend kernel]\nCPU: fn_cpu [kernel]\nCUDA: fn_defaultbackend [default backend kernel]\nXLA: fn_defaultbackend [default backend kernel]\nAutogradOther: fn_autograd [autograd kernel]\nAutogradCPU: fn_autograd [autograd kernel]\nAutogradCUDA: fn_autograd [autograd kernel]\nAutogradXLA: fn_autograd [autograd kernel]\nFPGA: fn_defaultbackend [default backend kernel]\n')

    def test_computed_table_with_cpu_autograd_math_defaultbackend(self):
        if False:
            while True:
                i = 10
        result = self.commute('foo', [lambda m: m.def_('foo(Tensor x) -> Tensor'), lambda m: m.impl_t_t('foo', 'CPU', debug='fn_cpu'), lambda m: m.impl_t_t('foo', 'Autograd', debug='fn_autograd'), lambda m: m.impl_t_t('foo', 'CompositeImplicitAutograd', debug='fn_math'), lambda m: m.impl_t_t('foo', 'CompositeExplicitAutograd', debug='fn_defaultbackend')])
        (state, table) = (result.state, result.table)
        self.assertExpectedInline(state, 'name: test::foo\nschema: test::foo(Tensor x) -> Tensor\ndebug: registered at /dev/null:0\nalias analysis kind: FROM_SCHEMA\nCPU: fn_cpu :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nAutograd[alias]: fn_autograd :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nCompositeImplicitAutograd[alias]: fn_math :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nCompositeExplicitAutograd[alias]: fn_defaultbackend :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\n')
        extracted_table = extract_dispatch_table_with_keys(table, dispatch_keys_to_check)
        self.assertExpectedInline(extracted_table, 'Undefined: fn_defaultbackend [default backend kernel]\nCPU: fn_cpu [kernel]\nCUDA: fn_defaultbackend [default backend kernel]\nXLA: fn_defaultbackend [default backend kernel]\nAutogradOther: fn_autograd [autograd kernel]\nAutogradCPU: fn_autograd [autograd kernel]\nAutogradCUDA: fn_autograd [autograd kernel]\nAutogradXLA: fn_autograd [autograd kernel]\n')

    def test_multiple_def_error(self):
        if False:
            for i in range(10):
                print('nop')
        ops = [lambda m: m.def_('foo(Tensor x, Tensor y) -> Tensor'), lambda m: m.def_('foo(Tensor x, Tensor y) -> Tensor')]
        self.assertExpectedInline(self.commute('foo', ops, expect_raises=True).state, "Tried to register an operator (test::foo(Tensor x, Tensor y) -> Tensor) with the same name and overload name multiple times. Each overload's schema should only be registered with a single call to def(). Duplicate registration: registered at /dev/null:0. Original registration: registered at /dev/null:0")

    def test_def_with_explicit_alias(self):
        if False:
            while True:
                i = 10
        state = self.commute('foo', [lambda m: m.def_('foo(Tensor x, Tensor y) -> Tensor', alias='PURE_FUNCTION')]).state
        self.assertExpectedInline(state, 'name: test::foo\nschema: test::foo(Tensor x, Tensor y) -> Tensor\ndebug: registered at /dev/null:0\nalias analysis kind: PURE_FUNCTION\n')

    def test_multiple_def_alias_defaulting(self):
        if False:
            print('Hello World!')
        ops = [lambda m: m.def_('foo(Tensor x) -> Tensor', alias='PURE_FUNCTION'), lambda m: m.def_legacy('foo(Tensor x) -> Tensor')]
        self.assertExpectedInline(self.commute('foo', ops, expect_raises=True).state, "Tried to register an operator (test::foo(Tensor x) -> Tensor) with the same name and overload name multiple times. Each overload's schema should only be registered with a single call to def(). Duplicate registration: registered at /dev/null:0. Original registration: registered at /dev/null:0")

    def test_multiple_def_alias_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        ops = [lambda m: m.def_('foo(Tensor x) -> Tensor', alias='PURE_FUNCTION'), lambda m: m.def_('foo(Tensor x) -> Tensor', alias='CONSERVATIVE')]
        self.assertExpectedInline(self.commute('foo', ops, expect_raises=True).state, "Tried to register an operator (test::foo(Tensor x) -> Tensor) with the same name and overload name multiple times. Each overload's schema should only be registered with a single call to def(). Duplicate registration: registered at /dev/null:0. Original registration: registered at /dev/null:0")

    def test_multiple_fallback(self):
        if False:
            return 10
        global_m = C._dispatch_library('IMPL', '_', 'XLA')
        (global_m.fallback_fallthrough(),)
        try:
            (global_m.fallback_fallthrough(),)
        except RuntimeError as e:
            self.assertExpectedInline(str(e), 'Tried to register multiple backend fallbacks for the same dispatch key XLA; previous registration registered at /dev/null:0, new registration registered at /dev/null:0')
        else:
            self.assertTrue(False)

    def test_overwrite_math(self):
        if False:
            for i in range(10):
                print('nop')
        ops = [lambda m: m.impl_t_t('foo', debug='fn1'), lambda m: m.impl_t_t('foo', debug='fn2')]
        self.assertExpectedInline(self.commute('foo', ops, ctor_order=(0, 1)).state, 'name: test::foo\nschema: (none)\nCompositeImplicitAutograd[alias]: fn2 :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\nCompositeImplicitAutograd[alias] (inactive): fn1 :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]\n')

    def test_find_dangling_impls(self):
        if False:
            print('Hello World!')
        dangling_impls = C._dispatch_find_dangling_impls()
        self.assertEqual(0, len(dangling_impls), msg=f'Expect zero dangling impls, but found: {dangling_impls}')

    def test_find_dangling_impls_ext(self):
        if False:
            print('Hello World!')
        extension_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpp_extensions', 'dangling_impl_extension.cpp')
        module = torch.utils.cpp_extension.load(name='dangling_impl_extension', sources=[extension_path], extra_cflags=['-g'], verbose=True)
        impls = C._dispatch_find_dangling_impls()
        self.assertEqual(1, len(impls))
        self.assertEqual(f'name: __test::foo\nschema: (none)\nCPU: registered at {extension_path}:5 :: () -> () [ boxed unboxed ]\n', impls[0])

    def test_dispatch_print_registrations_for_dispatch_key_invalid(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, 'could not parse dispatch key: invalid_key'):
            C._dispatch_print_registrations_for_dispatch_key('invalid_key')

class TestPythonDispatcher(TestCase):

    def test_basic(self):
        if False:
            while True:
                i = 10
        dispatcher = PythonDispatcher()
        dispatcher.register(['CPU', 'XLA', 'Lazy', 'CompositeImplicitAutograd'])
        self.assertExpectedInline(dispatcher.dispatchTable(), '\nComputed Dispatch Table\nkey             kernel\n---------------------------\nCPU             fn_CPU [kernel]\nXLA             fn_XLA [kernel]\nLazy            fn_Lazy [kernel]\nFPGA            fn_CompositeImplicitAutograd [math kernel]\nAutogradOther   fn_CompositeImplicitAutograd [math kernel]\nAutogradCPU     [backend fallback]\nAutogradXLA     [backend fallback]\nAutogradLazy    [backend fallback]\n')

    def test_math_autogradcpu(self):
        if False:
            for i in range(10):
                print('nop')
        dispatcher = PythonDispatcher()
        dispatcher.register(['CPU', 'XLA', 'Lazy', 'CompositeImplicitAutograd', 'AutogradCPU'])
        self.assertExpectedInline(dispatcher.dispatchTable(), '\nComputed Dispatch Table\nkey             kernel\n---------------------------\nCPU             fn_CPU [kernel]\nXLA             fn_XLA [kernel]\nLazy            fn_Lazy [kernel]\nFPGA            fn_CompositeImplicitAutograd [math kernel]\nAutogradOther   fn_CompositeImplicitAutograd [math kernel]\nAutogradCPU     fn_AutogradCPU [kernel]\nAutogradXLA     [backend fallback]\nAutogradLazy    [backend fallback]\n')
        self.assertExpectedInline(dispatcher.registrations(), '\nRegistered Kernels\nkey             kernel\n---------------------------\nCPU             fn_CPU\nXLA             fn_XLA\nLazy            fn_Lazy\nAutogradCPU     fn_AutogradCPU\nCompositeImplicitAutograd[alias] fn_CompositeImplicitAutograd\n')

    def test_defaultbackend_autogradcpu(self):
        if False:
            return 10
        dispatcher = PythonDispatcher()
        dispatcher.register(['CPU', 'XLA', 'Lazy', 'CompositeExplicitAutograd', 'AutogradCPU'])
        self.assertExpectedInline(dispatcher.dispatchTable(), '\nComputed Dispatch Table\nkey             kernel\n---------------------------\nCPU             fn_CPU [kernel]\nXLA             fn_XLA [kernel]\nLazy            fn_Lazy [kernel]\nFPGA            fn_CompositeExplicitAutograd [default backend kernel]\nAutogradOther   [backend fallback]\nAutogradCPU     fn_AutogradCPU [kernel]\nAutogradXLA     [backend fallback]\nAutogradLazy    [backend fallback]\n')
        self.assertExpectedInline(dispatcher.registrations(), '\nRegistered Kernels\nkey             kernel\n---------------------------\nCPU             fn_CPU\nXLA             fn_XLA\nLazy            fn_Lazy\nAutogradCPU     fn_AutogradCPU\nCompositeExplicitAutograd[alias] fn_CompositeExplicitAutograd\n')

    def test_autogradother(self):
        if False:
            return 10
        dispatcher = PythonDispatcher()
        dispatcher.register(['CPU', 'FPGA', 'CompositeImplicitAutograd'])
        self.assertExpectedInline(dispatcher.dispatchTable(), '\nComputed Dispatch Table\nkey             kernel\n---------------------------\nCPU             fn_CPU [kernel]\nXLA             fn_CompositeImplicitAutograd [math kernel]\nLazy            fn_CompositeImplicitAutograd [math kernel]\nFPGA            fn_FPGA [kernel]\nAutogradOther   ambiguous_autogradother [ambiguous autogradother]\nAutogradCPU     [backend fallback]\nAutogradXLA     fn_CompositeImplicitAutograd [math kernel]\nAutogradLazy    fn_CompositeImplicitAutograd [math kernel]\n')
        self.assertExpectedInline(dispatcher.registrations(), '\nRegistered Kernels\nkey             kernel\n---------------------------\nFPGA            fn_FPGA\nCPU             fn_CPU\nCompositeImplicitAutograd[alias] fn_CompositeImplicitAutograd\n')

    def test_duplicate_registrations(self):
        if False:
            return 10
        dispatcher = PythonDispatcher()
        with self.assertRaisesRegex(RuntimeError, 'Overriden is not allowed'):
            dispatcher.register(['CPU', 'CPU'])

    def test_defaultbackend_math(self):
        if False:
            i = 10
            return i + 15
        dispatcher = PythonDispatcher()
        with self.assertRaisesRegex(RuntimeError, 'Registration to both CompositeImplicitAutograd and CompositeExplicitAutograd is not allowed'):
            dispatcher.register(['CompositeExplicitAutograd', 'CompositeImplicitAutograd'])

    def test_quantized_structured_not_implemented(self):
        if False:
            while True:
                i = 10
        x = torch.zeros([1, 1, 1])
        y = torch.zeros([1, 1, 1])
        (scale, zero_point) = (1.0, 0)
        dtype = torch.qint8
        qx = torch.quantize_per_tensor(x, scale, zero_point, dtype)
        qy = torch.quantize_per_tensor(y, scale, zero_point, dtype)
        self.assertRaisesRegex(NotImplementedError, "Could not run 'aten::bmm.out' with arguments from the 'QuantizedCPU' backend.", lambda : torch.bmm(qx, qy))
if __name__ == '__main__':
    run_tests()