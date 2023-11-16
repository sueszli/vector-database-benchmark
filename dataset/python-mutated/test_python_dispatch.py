import tempfile
import torch
from copy import deepcopy
from torch.library import Library, impl, fallthrough_kernel
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch import SymInt
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.cuda.jiterator import _create_jit_fn
import unittest
from torch.testing._internal.common_utils import *
from torch.utils._mode_utils import no_dispatch, all_same_mode
from torch.testing._internal.logging_tensor import LoggingTensor, LoggingTensorReentrant, LoggingTensorMode, log_input, capture_logs, capture_logs_with_logging_tensor_mode
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._pytree import tree_map, tree_map_only
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode, _get_current_dispatch_mode_stack
from torch._custom_op.functional import register_functional_op
from torch._C import DispatchKeySet, DispatchKey
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import ops
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.multiprocessing.reductions import StorageWeakRef
import logging
import sys
import torch._dynamo

class TestDispatcherPythonBindings(TestCase):

    def test_call_boxed(self) -> None:
        if False:
            return 10
        sin = torch._C._dispatch_find_schema_or_throw('aten::sin', '')
        x = torch.randn(3)
        y = torch._C._dispatch_call_boxed(sin, x)
        self.assertEqual(y, x.sin())

class TestPythonRegistration(TestCase):
    test_ns = '_test_python_registration'

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        if hasattr(torch.ops, self.test_ns):
            del torch.ops._test_python_registration

    def test_override_aten_ops_with_multiple_libraries(self) -> None:
        if False:
            while True:
                i = 10
        x = torch.tensor([1, 2])
        my_lib1 = Library('aten', 'IMPL')
        my_lib2 = Library('aten', 'IMPL')

        def my_neg(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return args[0]._neg_view()
        my_lib1.impl('neg', my_neg, 'AutogradCPU')
        self.assertTrue(torch.neg(x).is_neg())
        with self.assertRaisesRegex(RuntimeError, 'operator name does not match namespace'):
            my_lib3 = Library('foo', 'DEF')
            my_lib3.define('neg(Tensor self) -> Tensor')
            my_lib3.impl(torch.ops.aten.neg.default, my_neg, 'AutogradCPU')
            del my_lib3

        def my_mul(*args, **kwargs):
            if False:
                return 10
            return torch.zeros_like(args[0])
        my_lib2.impl('aten::mul.Tensor', my_mul, 'ZeroTensor')
        y = torch._efficientzerotensor(2)
        self.assertFalse(torch.mul(x, y)._is_zerotensor())
        with self.assertRaisesRegex(RuntimeError, 'already a kernel registered from python'):
            my_lib2.impl(torch.ops.aten.mul.Tensor, my_mul, 'ZeroTensor')
        del my_lib1
        self.assertFalse(torch.mul(x, y)._is_zerotensor())
        del my_lib2
        self.assertFalse(torch.neg(x).is_neg())
        self.assertTrue(torch.mul(x, y)._is_zerotensor())

    def test_error_if_fn_not_callable(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(TypeError, 'Input function is required to be a callable'):
            my_lib = Library('aten', 'IMPL')
            my_lib.impl(torch.ops.aten.neg.default, [], 'AutogradCPU')

    def test_finalizer(self):
        if False:
            for i in range(10):
                print('nop')
        impls_refcnt = sys.getrefcount(torch.library._impls)
        lib = Library(self.test_ns, 'FRAGMENT')
        lib.define('foo123(Tensor x) -> Tensor')
        self.assertEqual(sys.getrefcount(lib), 2)
        self.assertEqual(sys.getrefcount(torch.library._impls), impls_refcnt + 1)
        self.assertEqual(sys.getrefcount(lib._op_impls), 3)

        def foo123(x):
            if False:
                for i in range(10):
                    print('nop')
            pass
        lib.impl(f'{self.test_ns}::foo123', foo123, 'CPU')
        key = f'{self.test_ns}/foo123/CPU'
        self.assertTrue(key in torch.library._impls)
        saved_op_impls = lib._op_impls
        self.assertEqual(sys.getrefcount(lib), 2)
        del lib
        self.assertEqual(sys.getrefcount(saved_op_impls), 2)
        self.assertTrue(key not in torch.library._impls)
        self.assertEqual(sys.getrefcount(torch.library._impls), impls_refcnt)

    def test_override_cpu_sum(self) -> None:
        if False:
            i = 10
            return i + 15
        run = [False]

        def my_sum(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            run[0] = True
            return args[0].clone()
        my_lib1 = Library('aten', 'IMPL')
        my_lib1.impl('aten::sum', my_sum, 'CPU')
        x = torch.tensor([1, 2])
        self.assertEqual(torch.sum(x), x)
        self.assertTrue(run[0])
        del my_lib1
        self.assertEqual(torch.sum(x), torch.tensor(3))

    def test_override_cuda_with_jiterator(self) -> None:
        if False:
            while True:
                i = 10

        def override_where_cuda() -> None:
            if False:
                return 10
            not_where_code_string = '\n            template <typename T> T inverted_where(bool cond, T a, T b){\n                return !cond ? a : b;\n            }\n            '
            jitted_where = _create_jit_fn(not_where_code_string)
            CALLED = [False]

            def inverted_where(*args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                CALLED[0] = True
                return jitted_where(*args, **kwargs)
            my_lib = Library('aten', 'IMPL')
            my_lib.impl('aten::where.self', inverted_where, 'CUDA')
            device = 'cuda'
            cond = torch.tensor([True, True, False], device=device, dtype=torch.bool)
            x = torch.tensor([1, 2, 3], device=device)
            y = torch.tensor([-1, -2, -3], device=device)
            self.assertEqual(torch.where(cond, x, y), torch.tensor([-1, -2, 3]))
            self.assertTrue(CALLED[0])
            del my_lib
            self.assertEqual(torch.where(cond, x, y), torch.tensor([1, 2, -3]))

        def override_gelu_cuda() -> None:
            if False:
                print('Hello World!')
            fastest_gelu_code_string = '\n            template <typename T> T fast_gelu(T a){\n                return a > 0 ? a : 0;\n            }\n            '
            jitted_gelu = _create_jit_fn(fastest_gelu_code_string)
            CALLED = [False]

            def fast_gelu(*args, **kwargs):
                if False:
                    return 10
                CALLED[0] = True
                return jitted_gelu(*args, **kwargs)
            my_lib = Library('aten', 'IMPL')
            my_lib.impl('aten::gelu', fast_gelu, 'CUDA')
            x = torch.rand([3, 3], device='cuda', dtype=torch.float)
            self.assertEqual(torch.nn.functional.gelu(x), torch.nn.functional.relu(x))
            self.assertTrue(CALLED[0])
            del my_lib
            self.assertNotEqual(torch.nn.functional.gelu(x), torch.nn.functional.relu(x))

        def override_exp_cuda() -> None:
            if False:
                for i in range(10):
                    print('nop')
            clipped_exp_code_string = '\n            template <typename T> T clipped_exp(T a){\n                return a > T(10.0) ? T(22026.4657948) : exp(a);\n            }\n            '
            jitted_exp = _create_jit_fn(clipped_exp_code_string)
            CALLED = [False]

            def clipped_exp(*args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                CALLED[0] = True
                return jitted_exp(*args, **kwargs)
            my_lib = Library('aten', 'IMPL')
            my_lib.impl('aten::exp', clipped_exp, 'CUDA')
            x = torch.tensor([0.0, 100.0], device='cuda', dtype=torch.float16)
            self.assertEqual(torch.exp(x), torch.tensor([1.0, 22026.4657948], dtype=torch.float16))
            self.assertTrue(CALLED[0])
            del my_lib
            self.assertEqual(torch.exp(x), torch.tensor([1.0, torch.inf], dtype=torch.float16))

        def override_add_cuda() -> None:
            if False:
                while True:
                    i = 10
            buggy_add_code_string = '\n            template <typename T> T buggy_add(T a, T b){\n                return a + b + T(1);\n            }\n            '
            jitted_add = _create_jit_fn(buggy_add_code_string)
            CALLED = [False]

            def buggy_add(*args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                CALLED[0] = True
                return jitted_add(*args, **kwargs)
            my_lib = Library('aten', 'IMPL')
            my_lib.impl('aten::add.Tensor', buggy_add, 'CUDA')
            x_cpu = torch.rand([3, 3], device='cpu')
            y_cpu = torch.rand([3], device='cpu')
            x_cuda = x_cpu.cuda()
            y_cuda = y_cpu.cuda()
            self.assertEqual(x_cuda + y_cuda, x_cpu + y_cpu + 1)
            self.assertTrue(CALLED[0])
            del my_lib
            self.assertEqual(x_cuda + y_cuda, x_cpu + y_cpu)
        if torch.cuda.is_available() and (not TEST_WITH_ROCM):
            override_where_cuda()
            override_gelu_cuda()
            override_exp_cuda()
            override_add_cuda()

    def test_extend_library_with_dispatch_key_arg(self):
        if False:
            while True:
                i = 10

        def my_sum(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            return args[0].clone()
        my_lib1 = Library('aten', 'IMPL', dispatch_key='CPU')
        with self.assertRaisesRegex(RuntimeError, 'inconsistent with the dispatch key'):
            my_lib1.impl('sum', my_sum, 'Conjugate')
        my_lib1.impl('aten::sum', my_sum)
        x = torch.tensor([1, 2])
        self.assertEqual(torch.sum(x), x)
        del my_lib1

    def test_create_new_library(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        my_lib1 = Library(self.test_ns, 'DEF')
        my_lib1.define('sum(Tensor self) -> Tensor')

        @torch.library.impl(my_lib1, 'sum', 'CPU')
        def my_sum(*args, **kwargs):
            if False:
                return 10
            return args[0].clone()
        x = torch.tensor([1, 2])
        op = getattr(torch.ops, self.test_ns).sum
        self.assertEqual(op(x), x)
        my_lib2 = Library(self.test_ns, 'IMPL')

        @torch.library.impl(my_lib2, op.default, 'ZeroTensor')
        def my_sum_zt(*args, **kwargs):
            if False:
                print('Hello World!')
            if args[0]._is_zerotensor():
                return torch._efficientzerotensor(args[0].shape)
            else:
                return args[0].clone()
        y = torch._efficientzerotensor(3)
        self.assertTrue(op(y)._is_zerotensor())
        self.assertEqual(op(x), x)
        del my_lib2
        del my_lib1

    def test_create_new_library_fragment_no_existing(self):
        if False:
            while True:
                i = 10
        my_lib = Library(self.test_ns, 'FRAGMENT')
        my_lib.define('sum2(Tensor self) -> Tensor')

        @torch.library.impl(my_lib, 'sum2', 'CPU')
        def my_sum(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            return args[0]
        x = torch.tensor([1, 2])
        self.assertEqual(getattr(torch.ops, self.test_ns).sum2(x), x)
        del my_lib

    def test_create_new_library_fragment_with_existing(self):
        if False:
            return 10
        my_lib1 = Library(self.test_ns, 'DEF')
        my_lib2 = Library(self.test_ns, 'FRAGMENT')
        my_lib2.define('sum4(Tensor self) -> Tensor')

        @torch.library.impl(my_lib2, 'sum4', 'CPU')
        def my_sum4(*args, **kwargs):
            if False:
                print('Hello World!')
            return args[0]
        x = torch.tensor([1, 2])
        self.assertEqual(getattr(torch.ops, self.test_ns).sum4(x), x)
        my_lib3 = Library(self.test_ns, 'FRAGMENT')
        my_lib3.define('sum3(Tensor self) -> Tensor')

        @torch.library.impl(my_lib3, 'sum3', 'CPU')
        def my_sum3(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            return args[0]
        x = torch.tensor([1, 2])
        self.assertEqual(getattr(torch.ops, self.test_ns).sum3(x), x)
        del my_lib1
        del my_lib2
        del my_lib3

    @unittest.skipIf(IS_WINDOWS, 'Skipped under Windows')
    def test_alias_analysis(self):
        if False:
            for i in range(10):
                print('nop')

        def test_helper(alias_analysis=''):
            if False:
                print('Hello World!')
            my_lib1 = Library(self.test_ns, 'DEF')
            called = [0]

            @torch.library.define(my_lib1, '_op() -> None', alias_analysis=alias_analysis)
            def _op(*args, **kwargs):
                if False:
                    return 10
                called[0] += 1

            @torch.jit.script
            def _test():
                if False:
                    for i in range(10):
                        print('nop')
                torch.ops._test_python_registration._op()
            assert '_test_python_registration::_op' in str(_test.graph)
        with self.assertRaises(AssertionError):
            test_helper('')
        test_helper('CONSERVATIVE')

    def test_error_for_unsupported_ns_or_kind(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'Unsupported kind'):
            my_lib1 = Library('myns', 'BLA')
        for kind in ('DEF', 'FRAGMENT'):
            with self.assertRaisesRegex(ValueError, 'reserved namespace'):
                my_lib1 = Library('prim', kind)

    def test_returning_symint(self) -> None:
        if False:
            print('Hello World!')
        shape_env = ShapeEnv()
        fake_tensor_mode = FakeTensorMode(shape_env=shape_env)
        ft = fake_tensor_mode.from_tensor(torch.rand(2, 3))
        (s0, s1) = ft.shape
        tlib = Library(self.test_ns, 'DEF')
        tlib.define('sqsum(SymInt a, SymInt b) -> SymInt')

        @impl(tlib, 'sqsum', 'CompositeExplicitAutograd')
        def sqsum(a: SymInt, b: SymInt):
            if False:
                print('Hello World!')
            return a * a + b * b
        out = getattr(torch.ops, self.test_ns).sqsum.default(s0, s1)
        out_val = shape_env.evaluate_expr(out.node.expr)
        self.assertEqual(out_val, 13)

    def test_register_functional_op_error_cases(self):
        if False:
            print('Hello World!')
        lib = Library(self.test_ns, 'FRAGMENT')
        with self.assertRaisesRegex(TypeError, 'instance of OpOverload'):
            register_functional_op(lib, 'abs', torch.ops.aten.abs_)
        with self.assertRaisesRegex(RuntimeError, 'Expected op to be mutable'):
            register_functional_op(lib, 'abs', torch.ops.aten.abs_.default)
        with self.assertRaisesRegex(RuntimeError, 'Expected op to be mutable'):
            register_functional_op(lib, 'abs', torch.ops.aten.abs.out)
        schemas = ['foo(Tensor x, Tensor(a!)? y) -> ()', 'foo(Tensor x, Tensor(a!)[] y) -> ()', 'foo(Tensor x, Tensor(a!) y, Tensor(b) z) -> Tensor(b)', 'foo(Tensor x, Tensor(a!) y) -> (Tensor, Tensor(a))']
        del lib
        for schema in schemas:
            lib = Library(self.test_ns, 'FRAGMENT')
            try:
                lib.define(schema)
                with self.assertRaisesRegex(RuntimeError, 'NYI'):
                    register_functional_op(lib, 'foo_functional', getattr(torch.ops, self.test_ns).foo.default)
            finally:
                del lib
                delattr(torch.ops, self.test_ns)

    def _check_is_functional_variant(self, mutable_op, functional_op, args):
        if False:
            return 10
        cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
        functional_result = functional_op(*cloned_args)
        self.assertEqual(cloned_args, args)
        mutable_result = mutable_op(*cloned_args)
        if mutable_result is None:
            flat_mutable_result = []
        else:
            flat_mutable_result = pytree.tree_leaves(mutable_result)
        flat_functional_result = pytree.tree_leaves(functional_result)
        assert len(flat_functional_result) > len(flat_mutable_result)
        self.assertEqual(flat_functional_result[:len(flat_mutable_result)], flat_mutable_result)
        mutated_args = [maybe_mutated_arg for (maybe_mutated_arg, arg) in zip(cloned_args, args) if not torch.allclose(maybe_mutated_arg, arg)]
        self.assertEqual(flat_functional_result[len(flat_mutable_result):], mutated_args)

        def fn(*args):
            if False:
                i = 10
                return i + 15
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            mutable_op(*cloned_args)
            return cloned_args
        gm = make_fx(torch.func.functionalize(fn))(*args)
        has_functional_op = False
        for node in gm.graph.nodes:
            self.assertFalse(node.target is mutable_op)
            if node.target is functional_op:
                has_functional_op = True
        self.assertTrue(has_functional_op)

    def test_register_functional_op_no_returns(self):
        if False:
            i = 10
            return i + 15
        lib = Library(self.test_ns, 'FRAGMENT')
        lib.define('foo(Tensor x, Tensor(a!) y, Tensor z, Tensor(b!) w) -> ()')

        def foo_impl(x, y, z, w):
            if False:
                i = 10
                return i + 15
            y.fill_(3.14)
            w.fill_(2.71)
        lib.impl('foo', foo_impl, 'CPU')
        register_functional_op(lib, 'foo_functional', getattr(torch.ops, self.test_ns).foo.default)
        x = torch.randn([])
        y = torch.randn([])
        z = torch.randn([])
        w = torch.randn([])
        self._check_is_functional_variant(getattr(torch.ops, self.test_ns).foo.default, getattr(torch.ops, self.test_ns).foo_functional.default, (x, y, z, w))

    def test_register_functional_op_one_return(self):
        if False:
            for i in range(10):
                print('nop')
        lib = Library(self.test_ns, 'FRAGMENT')
        lib.define('foo(Tensor x, Tensor(a!) y, Tensor(c!) z, Tensor(b!) w) -> Tensor')

        def foo_impl(x, y, z, w):
            if False:
                print('Hello World!')
            y.fill_(3.14)
            w.fill_(2.71)
            z.fill_(0.99)
            return x.clone()
        lib.impl('foo', foo_impl, 'CPU')
        register_functional_op(lib, 'foo_functional', getattr(torch.ops, self.test_ns).foo.default)
        x = torch.randn([])
        y = torch.randn([])
        z = torch.randn([])
        w = torch.randn([])
        self._check_is_functional_variant(getattr(torch.ops, self.test_ns).foo.default, getattr(torch.ops, self.test_ns).foo_functional.default, (x, y, z, w))

    def test_register_functional_op_multiple_returns(self):
        if False:
            for i in range(10):
                print('nop')
        lib = Library(self.test_ns, 'FRAGMENT')
        lib.define('foo(Tensor x, Tensor(a!) y, Tensor z, Tensor(b!) w) -> (Tensor, Tensor)')

        def foo_impl(x, y, z, w):
            if False:
                while True:
                    i = 10
            y.fill_(3.14)
            w.fill_(2.71)
            return (x.clone(), z.clone())
        lib.impl('foo', foo_impl, 'CPU')
        register_functional_op(lib, 'foo_functional', getattr(torch.ops, self.test_ns).foo.default)
        x = torch.randn([])
        y = torch.randn([])
        z = torch.randn([])
        w = torch.randn([])
        self._check_is_functional_variant(getattr(torch.ops, self.test_ns).foo.default, getattr(torch.ops, self.test_ns).foo_functional.default, (x, y, z, w))

    def test_register_fallthrough(self):
        if False:
            i = 10
            return i + 15
        try:
            my_lib = Library('aten', 'IMPL')
            my_lib.impl('mm', fallthrough_kernel, 'AutocastCPU')
            a = torch.randn(2, 3, device='cpu', dtype=torch.float32)
            b = torch.randn(3, 2, device='cpu', dtype=torch.float32)
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                self.assertEqual(torch.mm(a, b).dtype, torch.float32)
                self.assertEqual(torch.matmul(a, b).dtype, torch.bfloat16)
        finally:
            del my_lib
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            self.assertEqual(torch.mm(a, b).dtype, torch.bfloat16)

class TestPythonDispatch(TestCase):

    def test_basic(self) -> None:
        if False:
            return 10
        with capture_logs() as logs:
            x = LoggingTensor(torch.tensor([3.0]), requires_grad=True)
            log_input('x', x)
            y = x * x
            saved_x = y.grad_fn._saved_self
            grad_y = LoggingTensor(torch.tensor([1.0]))
            log_input('grad_y', grad_y)
            (g,) = torch.autograd.grad((y,), (x,), (grad_y,))
        self.assertEqual(g.elem, torch.tensor([6.0]))
        with torch.no_grad():
            self.assertEqual(saved_x, x)
            self.assertEqual(saved_x._version, x._version)
            x.add_(2)
            self.assertEqual(saved_x, x)
        self.assertExpectedInline('\n'.join(logs), "$0: f32[1] = input('x')\n$1: f32[1] = torch._ops.aten.mul.Tensor($0, $0)\n$2: f32[1] = input('grad_y')\n$3: f32[1] = torch._ops.aten.mul.Tensor($2, $0)\n$4: f32[1] = torch._ops.aten.mul.Tensor($2, $0)\n$5: f32[1] = torch._ops.aten.add.Tensor($4, $3)")

    def test_out(self) -> None:
        if False:
            i = 10
            return i + 15
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1))
            y = LoggingTensor(torch.zeros(1))
            log_input('x', x)
            log_input('y', y)
            torch.abs(x, out=y)
        self.assertEqual(y.elem, torch.ones(1))
        self.assertExpectedInline('\n'.join(logs), "$0: f32[1] = input('x')\n$1: f32[1] = input('y')\n$2: f32[1] = torch._ops.aten.abs.out($0, out=$1)")

    def test_kwarg_only(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1))
            y = LoggingTensor(torch.ones(1, 1))
            z = LoggingTensor(torch.ones(1))
            log_input('x', x)
            log_input('y', y)
            log_input('z', z)
            torch.addmv(x, y, z)
            torch.addmv(x, y, z, beta=1)
            torch.addmv(x, y, z, beta=2)
            torch.addmv(x, y, z, alpha=2)
            torch.addmv(x, y, z, beta=2, alpha=2)
        self.assertExpectedInline('\n'.join(logs), "$0: f32[1] = input('x')\n$1: f32[1, 1] = input('y')\n$2: f32[1] = input('z')\n$3: f32[1] = torch._ops.aten.addmv.default($0, $1, $2)\n$4: f32[1] = torch._ops.aten.addmv.default($0, $1, $2)\n$5: f32[1] = torch._ops.aten.addmv.default($0, $1, $2, beta=2)\n$6: f32[1] = torch._ops.aten.addmv.default($0, $1, $2, alpha=2)\n$7: f32[1] = torch._ops.aten.addmv.default($0, $1, $2, beta=2, alpha=2)")

    def test_kwarg_only_and_positional_default(self) -> None:
        if False:
            i = 10
            return i + 15
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1))
            log_input('x', x)
            torch.ops.aten._foobar(x)
            torch.ops.aten._foobar(x, False)
            torch.ops.aten._foobar(x, arg3=False)
            torch.ops.aten._foobar(x, False, arg3=False)
        self.assertExpectedInline('\n'.join(logs), "$0: f32[1] = input('x')\n$1: f32[1] = torch._ops.aten._foobar.default($0)\n$2: f32[1] = torch._ops.aten._foobar.default($0, False)\n$3: f32[1] = torch._ops.aten._foobar.default($0, arg3=False)\n$4: f32[1] = torch._ops.aten._foobar.default($0, False, arg3=False)")

    def test_produce_real_type(self) -> None:
        if False:
            while True:
                i = 10
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(2, 2))
            log_input('x', x)
            x.to(dtype=torch.double)
            torch.cumprod(x, 0, dtype=torch.double)
            x[:, 1].contiguous(memory_format=torch.contiguous_format)
        self.assertExpectedInline('\n'.join(logs), "$0: f32[2, 2] = input('x')\n$1: f64[2, 2] = torch._ops.aten._to_copy.default($0, dtype=torch.float64)\n$2: f64[2, 2] = torch._ops.aten.cumprod.default($0, 0, dtype=torch.float64)\n$3: f32[2, 2] = torch._ops.aten.slice.Tensor($0, 0, 0, 9223372036854775807)\n$4: f32[2] = torch._ops.aten.select.int($3, 1, 1)\n$5: f32[2] = torch._ops.aten.clone.default($4, memory_format=torch.contiguous_format)")

    def test_optional_tensor_list(self) -> None:
        if False:
            print('Hello World!')

        def weird(xs):
            if False:
                while True:
                    i = 10
            print('woof')
            return torch.empty(())
        my_lib = Library('my_lib', 'DEF')
        my_lib.define('weird(Tensor?[] self) -> Tensor')
        my_lib.impl('weird', weird, 'CPU')
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(2, 2))
            log_input('x', x)
            torch.ops.my_lib.weird.default([None, x])
        self.assertExpectedInline('\n'.join(logs), "$0: f32[2, 2] = input('x')\n$1: f32[] = torch._ops.my_lib.weird.default(['None', '$0'])")

    def test_list_ret(self) -> None:
        if False:
            print('Hello World!')
        for list_type in (list, tuple):

            class A(torch._C.TensorBase):

                @staticmethod
                def __new__(cls, elem):
                    if False:
                        return 10
                    return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

                @classmethod
                def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                    if False:
                        i = 10
                        return i + 15
                    if func.overloadpacket == torch.ops.aten.split:
                        with no_dispatch():
                            return list_type(torch.split(*args))
                    else:
                        raise AssertionError(f'unrecognized func: {func}')
            self.assertEqual(torch.split(A(torch.tensor([0, 1])), 2), torch.split(torch.tensor([0, 1]), 2))

    def test_invalid_ret(self) -> None:
        if False:
            print('Hello World!')

        class A(torch._C.TensorBase):

            @staticmethod
            def __new__(cls, elem):
                if False:
                    i = 10
                    return i + 15
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    return 10
                return 'arf'
        self.assertRaisesRegex(RuntimeError, 'Unable to cast', lambda : A(torch.zeros(1)).neg())
        self.assertRaisesRegex(RuntimeError, 'Unable to cast', lambda : A(torch.zeros(1)).detach())

    def test_detach_appears_twice_when_called_once(self) -> None:
        if False:
            print('Hello World!')
        with capture_logs() as logs:
            x = LoggingTensor(torch.tensor([3.0]), requires_grad=True)
            log_input('x', x)
            x.detach()
        self.assertExpectedInline('\n'.join(logs), "$0: f32[1] = input('x')\n$1: f32[1] = torch._ops.aten.detach.default($0)\n$2: f32[1] = torch._ops.aten.detach.default($1)")

    def test_storage(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        x = LoggingTensor(torch.ones(1))
        storage = x.untyped_storage()
        self.assertRaises(RuntimeError, lambda : storage.data_ptr())

    def test_make_wrapper_subclass_noalloc(self) -> None:
        if False:
            i = 10
            return i + 15
        torch.Tensor._make_wrapper_subclass(LoggingTensor, (1000000000000,))

    def test_version(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        x = LoggingTensor(torch.ones(1))
        prev_vc = x._version
        x.detach().add_(2)
        cur_vc = x._version
        self.assertNotEqual(prev_vc, cur_vc)
        x.data.add_(2)
        self.assertEqual(cur_vc, x._version)

    def test_subclass_priority(self) -> None:
        if False:
            print('Hello World!')

        class ErrorA(RuntimeError):
            pass

        class ErrorB(RuntimeError):
            pass

        class A(torch.Tensor):

            @staticmethod
            def __new__(cls, elem):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    while True:
                        i = 10
                raise ErrorA

        class B(A):

            @staticmethod
            def __new__(cls, elem):
                if False:
                    i = 10
                    return i + 15
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    for i in range(10):
                        print('nop')
                raise ErrorB
        self.assertRaises(ErrorA, lambda : torch.add(A(torch.empty(1)), A(torch.empty(1))))
        self.assertRaises(ErrorB, lambda : torch.add(A(torch.empty(1)), B(torch.empty(1))))
        self.assertRaises(ErrorB, lambda : torch.add(B(torch.empty(1)), A(torch.empty(1))))
        self.assertRaises(ErrorB, lambda : torch.add(B(torch.empty(1)), B(torch.empty(1))))

    def test_format(self) -> None:
        if False:
            return 10
        x = LoggingTensor(torch.ones(1))
        s1 = str(x)
        s2 = repr(x)
        s3 = f'{x}'
        self.assertExpectedInline(s1, 'LoggingTensor(tensor([1.]))')
        self.assertEqual(s1, s2)
        self.assertEqual(s1, s3)

    def test_custom_autograd(self) -> None:
        if False:
            print('Hello World!')
        escape = [None]

        class Square(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                y = x ** 2
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    return 10
                assert isinstance(grad_output, LoggingTensor)
                (x,) = ctx.saved_tensors
                assert isinstance(x, LoggingTensor)
                escape[0] = x
                return grad_output * 2 * x
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1), requires_grad=True)
            log_input('x', x)
            x.grad = LoggingTensor(torch.zeros(1))
            log_input('x.grad', x.grad)
            y = Square.apply(x)
            grad_output = LoggingTensor(torch.ones(1))
            log_input('grad_output', grad_output)
            y.backward(grad_output)
        with torch.no_grad():
            self.assertEqual(escape[0], x)
            self.assertEqual(escape[0]._version, x._version)
            x.add_(2)
            self.assertEqual(escape[0], x)
        self.assertExpectedInline('\n'.join(logs), "$0: f32[1] = input('x')\n$1: f32[1] = input('x.grad')\n$2: f32[1] = torch._ops.aten.pow.Tensor_Scalar($0, 2)\n$3: f32[1] = input('grad_output')\n$4: f32[1] = torch._ops.aten.mul.Tensor($3, 2)\n$5: f32[1] = torch._ops.aten.mul.Tensor($4, $0)\n$6: f32[1] = torch._ops.aten.add_.Tensor($1, $5)")

    def test_subclass_creation(self):
        if False:
            print('Hello World!')

        class Foo(torch.Tensor):
            pass
        err_msg = 'subclass Foo but.*already associated to a python object of type LoggingTensor'
        with self.assertRaisesRegex(RuntimeError, err_msg):
            a = torch.Tensor._make_subclass(Foo, LoggingTensor(torch.rand(2)))
        with self.assertRaisesRegex(RuntimeError, err_msg):
            b = LoggingTensor(torch.rand(2)).as_subclass(Foo)
        with self.assertRaisesRegex(RuntimeError, err_msg):
            Foo(LoggingTensor(torch.rand(2)))
        with self.assertRaisesRegex(TypeError, 'Foo must define __torch_dispatch__'):
            torch.Tensor._make_wrapper_subclass(Foo, (2, 2))

    def test_new_ones(self) -> None:
        if False:
            print('Hello World!')

        class MyTensor(torch.Tensor):
            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    print('Hello World!')
                return MyTensor(3)
        self.assertEqual(type(MyTensor(2).new_ones(3)), MyTensor)

    def test_like(self) -> None:
        if False:
            print('Hello World!')

        class MyTensor(torch.Tensor):
            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    print('Hello World!')
                return MyTensor(3)
        for f in ['empty', 'ones', 'rand', 'randn', 'zeros']:
            f_name = f + '_like'
            self.assertEqual(type(getattr(torch, f_name)(MyTensor(2))), MyTensor)
        self.assertEqual(type(torch.full_like(MyTensor(2), 1.0)), MyTensor)
        self.assertEqual(type(torch.randint_like(MyTensor(2), high=3)), MyTensor)

    def test_make_fx_with_subclass(self) -> None:
        if False:
            print('Hello World!')

        def f(x, y):
            if False:
                print('Hello World!')
            return (x * y, y + y)
        x_a = torch.zeros(4)
        x_b = torch.zeros(4)
        y = torch.ones(4)

        def f_to_trace(x_a, x_b, y):
            if False:
                return 10
            x = TwoTensor(x_a, x_b)
            (out1, out2) = f(x, y)
            (out1_unwrapped_attrs, _) = out1.__tensor_flatten__()
            return (*[getattr(out1, attr) for attr in out1_unwrapped_attrs], out2)
        fx_g = make_fx(f_to_trace, tracing_mode='fake')(x_a, x_b, y)
        self.assertExpectedInline(fx_g.code, '\n\n\ndef forward(self, x_a_1, x_b_1, y_1):\n    mul = torch.ops.aten.mul.Tensor(x_a_1, y_1);  x_a_1 = None\n    mul_1 = torch.ops.aten.mul.Tensor(x_b_1, y_1);  x_b_1 = None\n    add = torch.ops.aten.add.Tensor(y_1, y_1);  y_1 = None\n    return (mul, mul_1, add)\n    ')

    def test_make_wrapper_subclass_propagates_metadata(self) -> None:
        if False:
            while True:
                i = 10

        class WrapperTensor(torch.Tensor):
            elem: torch.Tensor
            __slots__ = ['elem']

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                r = torch.Tensor._make_wrapper_subclass(cls, elem.size(), dtype=elem.dtype, layout=elem.layout, device=elem.device, requires_grad=elem.requires_grad, strides=elem.stride(), storage_offset=elem.storage_offset())
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    i = 10
                    return i + 15
                raise RuntimeError('NYI')
        x = torch.randn(4, 6).t().diagonal(offset=2)
        y = WrapperTensor(x)
        self.assertEqual(y.size(), x.size())
        self.assertEqual(y.stride(), x.stride())
        self.assertEqual(y.storage_offset(), x.storage_offset())

    def test_wrapper_subclass_serializes(self) -> None:
        if False:
            return 10
        with tempfile.TemporaryFile() as f:
            x = LoggingTensor(torch.randn(3))
            torch.save(x, f)
            f.seek(0)
            x_loaded = torch.load(f)
            self.assertTrue(type(x_loaded) is type(x))
            self.assertEqual(x.elem, x_loaded.elem)
            self.assertFalse(x is x_loaded)

    def test_deepcopy_wrapper_subclass(self) -> None:
        if False:
            while True:
                i = 10
        x = LoggingTensor(torch.randn(3))
        x_copy = deepcopy(x)
        self.assertTrue(type(x_copy) is type(x))
        self.assertEqual(x.elem, x_copy.elem)
        self.assertFalse(x is x_copy)

    def test_deepcopy_wrapper_subclass_with_clone_returning_different_type(self) -> None:
        if False:
            return 10

        class MyWrapperTensor(torch.Tensor):
            elem: torch.Tensor
            __slots__ = ['elem']

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                r = torch.Tensor._make_wrapper_subclass(cls, elem.size(), dtype=elem.dtype, layout=elem.layout, device=elem.device, requires_grad=elem.requires_grad, strides=elem.stride(), storage_offset=elem.storage_offset())
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    return 10
                if func.overloadpacket.__name__ == 'clone':
                    return args[0].elem.clone()
                raise RuntimeError('NYI')
        x = MyWrapperTensor(torch.randn(3))
        with self.assertRaisesRegex(RuntimeError, 'for which cloning returns another instance of the same subclass'):
            x_copy = deepcopy(x)

    def test_deepcopy_non_wrapper_subclass(self) -> None:
        if False:
            print('Hello World!')

        class SubTensorError1(torch.Tensor):
            pass

        class SubTensorError2(torch.Tensor):

            def new_empty(self, shape):
                if False:
                    return 10
                return torch.Tensor(shape)
        for error_cls in [SubTensorError1, SubTensorError2]:
            x = error_cls(3)
            with self.assertRaisesRegex(RuntimeError, 'for which that function returns another instance of the same subclass'):
                x_copy = deepcopy(x)

        class SubTensorSuccess(torch.Tensor):

            def new_empty(self, shape):
                if False:
                    while True:
                        i = 10
                return type(self)(shape)
        x = SubTensorSuccess(3)
        x_copy = deepcopy(x)
        self.assertIs(type(x_copy), type(x))

    def test_wrapper_subclass_extra_dispatch_keys(self) -> None:
        if False:
            i = 10
            return i + 15

        class ExtraKeysTensor(torch.Tensor):

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                if False:
                    return 10
                r = torch.Tensor._make_wrapper_subclass(cls, elem.size(), elem.stride(), elem.storage_offset(), torch.contiguous_format, elem.dtype, elem.layout, elem.device, False, False, None, False, False, DispatchKeySet(DispatchKey.NestedTensor))
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    print('Hello World!')
                pass
        x = ExtraKeysTensor(torch.randn(3))
        self.assertTrue(torch._C._dispatch_keys(x).has(DispatchKey.NestedTensor))
        self.assertFalse(torch._C._dispatch_keys(x).has(DispatchKey.AutogradNestedTensor))

    def test_index_put_where_only_index_is_subclass(self) -> None:
        if False:
            while True:
                i = 10
        called_funcs = []

        class MyTensor(torch.Tensor):
            __torch_function__ = torch._C._disabled_torch_function_impl
            elem: torch.Tensor
            __slots__ = ['elem']

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                r = torch.Tensor._make_wrapper_subclass(cls, elem.size(), dtype=elem.dtype, layout=elem.layout, device=elem.device, requires_grad=elem.requires_grad)
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    while True:
                        i = 10
                called_funcs.append(func)
                return MyTensor(torch.tensor(3))
        x = torch.randn(3, 3)
        idxs = (MyTensor(torch.tensor(0)),)
        v = torch.randn(1)
        res = x.index_put_(idxs, v)
        self.assertEqual(called_funcs, [torch.ops.aten.index_put_.default])

    def test_torch_dispatch_mode_basic(self) -> None:
        if False:
            while True:
                i = 10
        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode():
                torch.empty([])
        self.assertExpectedInline('\n'.join(logs), "$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)")

    def test_torch_dispatch_mode_unrelated_tensors(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn([])
        y = torch.randn([])
        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode():
                x + y
        self.assertExpectedInline('\n'.join(logs), '$2: f32[] = torch._ops.aten.add.Tensor($0, $1)')

    def test_nested_push_logging_tensor_mode(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn([])
        y = torch.randn([])
        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode():
                with LoggingTensorMode():
                    torch.empty([])
                    x + y
        self.assertExpectedInline('\n'.join(logs), "$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)\n$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)\n$3: f32[] = torch._ops.aten.add.Tensor($1, $2)\n$3: f32[] = torch._ops.aten.add.Tensor($1, $2)")

    def test_capture_logs_with_torch_dispatch_mode(self):
        if False:
            print('Hello World!')
        x = torch.randn([])
        y = torch.randn([])
        with capture_logs_with_logging_tensor_mode() as logs:
            torch.empty([])
            x + y
        self.assertExpectedInline('\n'.join(logs), "$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)\n$3: f32[] = torch._ops.aten.add.Tensor($1, $2)")
        x = torch.randn([])
        y = torch.randn([])
        with capture_logs_with_logging_tensor_mode() as logs1:
            with capture_logs_with_logging_tensor_mode() as logs2:
                torch.empty([])
                x + y
        self.assertExpectedInline('\n'.join(logs2), "$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)\n$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)\n$3: f32[] = torch._ops.aten.add.Tensor($1, $2)\n$3: f32[] = torch._ops.aten.add.Tensor($1, $2)")
        self.assertEqual(logs1, logs2)

    def test_torch_dispatch_mode_subclass_priority(self) -> None:
        if False:
            i = 10
            return i + 15

        class ErrorA(RuntimeError):
            pass

        class ErrorB(RuntimeError):
            pass

        class A(torch.Tensor):

            @staticmethod
            def __new__(cls, elem):
                if False:
                    i = 10
                    return i + 15
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    for i in range(10):
                        print('nop')
                with AMode():
                    raise ErrorA

        class B(A):

            @staticmethod
            def __new__(cls, elem):
                if False:
                    while True:
                        i = 10
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    return 10
                with BMode():
                    func(*args, **kwargs)

        class AMode(TorchDispatchMode):

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    i = 10
                    return i + 15
                raise ErrorA

        class BMode(TorchDispatchMode):

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    print('Hello World!')
                raise ErrorB
        a = A(torch.empty(1))
        b = B(torch.empty(1))
        with self.assertRaises(ErrorA):
            a + a
        with self.assertRaises(ErrorB):
            a + b
        with self.assertRaises(ErrorA):
            with AMode():
                b + b
        with self.assertRaises(ErrorB):
            with BMode():
                a + a
        with self.assertRaises(ErrorB):
            with BMode():
                a + b

    def test_mode_with_make_subclass(self):
        if False:
            for i in range(10):
                print('nop')

        class SubTensor(torch.Tensor):

            @staticmethod
            def __new__(cls, elem):
                if False:
                    return 10
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

        class BasicMode(TorchDispatchMode):

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    print('Hello World!')
                return func(*args, **kwargs)
        x = torch.randn(3)
        with BasicMode():
            y = SubTensor(x)
        self.assertIsInstance(y, SubTensor)

    def test_torch_dispatch_mode_respects_no_dispatch(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with capture_logs(is_mode=True) as logs1:
            with LoggingTensorMode():
                torch.ones([2, 3])
                with no_dispatch():
                    torch.ones([2, 3])
        with capture_logs(is_mode=True) as logs2:
            with LoggingTensorMode():
                torch.ones([2, 3])
        self.assertEqual(logs1, logs2)

    def test_shallow_copy_and_detach(self) -> None:
        if False:
            print('Hello World!')
        seen = set()
        test_case = self

        class TestMode(TorchDispatchMode):

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    return 10
                tree_map_only(torch.Tensor, lambda t: test_case.assertIn(t, seen), (args, kwargs))
                if kwargs is None:
                    kwargs = {}
                r = func(*args, **kwargs)
                tree_map_only(torch.Tensor, lambda t: seen.add(t), r)
                return r
        with TestMode():
            x = torch.randn(3, requires_grad=True)
            loss = (x * x).sum()
            loss.backward()

    def test_exception_handling(self):
        if False:
            i = 10
            return i + 15

        class A(torch.Tensor):

            @staticmethod
            def __new__(cls, elem):
                if False:
                    while True:
                        i = 10
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

        class AMode(TorchDispatchMode):

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    i = 10
                    return i + 15
                if func.__name__ == 'randn.default':
                    raise RuntimeError()
                return A(torch.zeros(()))
        with AMode():
            try:
                torch.randn(())
            except RuntimeError:
                pass
            self.assertTrue(isinstance(torch.zeros(()), A))

    def test_with_mode_created_separately(self):
        if False:
            print('Hello World!')

        class ErrorA(RuntimeError):
            pass

        class A(TorchDispatchMode):

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    print('Hello World!')
                raise ErrorA()
        x = A()
        with self.assertRaises(ErrorA):
            with x:
                torch.empty([])

    def test_with_nested_modes(self):
        if False:
            for i in range(10):
                print('nop')

        class ErrorA(RuntimeError):

            def __init__(self, msg):
                if False:
                    return 10
                super().__init__(msg)

        class A(TorchDispatchMode):

            def __init__(self, msg):
                if False:
                    i = 10
                    return i + 15
                self.msg = msg

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    while True:
                        i = 10
                raise ErrorA(self.msg)
        with self.assertRaisesRegex(ErrorA, 'layer2'):
            with A('layer1'):
                with A('layer2'):
                    torch.empty([])

    def test_make_subclass_with_modes(self):
        if False:
            print('Hello World!')

        class ModeTensor(torch.Tensor):

            def __new__(cls, elem, mode):
                if False:
                    for i in range(10):
                        print('nop')
                r = torch.Tensor._make_subclass(cls, elem, elem.requires_grad)
                r.elem = elem
                r.mode = mode
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    return 10
                raise NotImplementedError("Shouldn't be here")

        class Mode(TorchDispatchMode):

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    i = 10
                    return i + 15

                def unwrap(e):
                    if False:
                        print('Hello World!')
                    if isinstance(e, ModeTensor):
                        return e.elem
                    else:
                        return e

                def wrap(t):
                    if False:
                        for i in range(10):
                            print('nop')
                    if isinstance(t, torch.Tensor):
                        return ModeTensor(t, self)
                    else:
                        return t
                return wrap(func(*tuple((unwrap(a) for a in args)), **kwargs))

        class BasicMode(TorchDispatchMode):

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    while True:
                        i = 10
                return func(*args, **kwargs)
        x = torch.tensor(4.0)
        with Mode():
            y = x + x
            z = y + y
        self.assertIsInstance(y, ModeTensor)
        self.assertIsInstance(z, ModeTensor)
        with Mode():
            with BasicMode():
                y = x + x
                z = y + y
        self.assertIsInstance(y, ModeTensor)
        self.assertIsInstance(z, ModeTensor)
        assert self.assertRaisesRegex(RuntimeError, 'subclass Mode but.* associated to a python object of type Mode')

    def test_notimplemented_mode(self):
        if False:
            while True:
                i = 10
        sub_count = 0

        class PoliteMode(TorchDispatchMode):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.pre_count = 0
                self.post_count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    print('Hello World!')
                self.pre_count += 1
                if any((t is not torch.Tensor for t in types)):
                    return NotImplemented
                self.post_count += 1
                return func(*args, **kwargs)

        class SubTensor(torch.Tensor):

            def __new__(cls, elem):
                if False:
                    i = 10
                    return i + 15
                r = torch.Tensor._make_wrapper_subclass(cls, elem.shape)
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    while True:
                        i = 10
                nonlocal sub_count
                sub_count += 1

                def unwrap(t):
                    if False:
                        print('Hello World!')
                    if isinstance(t, SubTensor):
                        return t.elem
                    else:
                        return t
                return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            __torch_function__ = torch._C._disabled_torch_function_impl
        a = SubTensor(torch.randn(2))
        with PoliteMode() as mode:
            a.abs()
        self.assertEqual(mode.pre_count, 2)
        self.assertEqual(mode.post_count, 1)
        self.assertEqual(sub_count, 1)
        with PoliteMode():
            with PoliteMode():
                a.abs()

    def test_nesting_same_mode(self):
        if False:
            print('Hello World!')
        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode() as reenabled:
                with reenabled:
                    torch.empty([])
            self.assertExpectedInline('\n'.join(logs), "$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)\n$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)")

    def test_error_using_class_method_on_mode(self):
        if False:
            for i in range(10):
                print('nop')

        class A(TorchDispatchMode):

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    while True:
                        i = 10
                return func(args, kwargs)
        x = torch.tensor(5.0)
        with self.assertRaisesRegex(RuntimeError, 'classmethod is not supported, please make it a plain method'):
            with A():
                x + x

    def test_get_cur_mode(self):
        if False:
            return 10

        class A(TorchDispatchMode):

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        self.assertEqual(_get_current_dispatch_mode(), None)
        with A() as mode1:
            self.assertEqual(_get_current_dispatch_mode(), mode1)
        with mode1:
            with A() as mode2:
                self.assertEqual(_get_current_dispatch_mode(), mode2)

    def test_get_mode_stack(self):
        if False:
            print('Hello World!')

        class A(TorchDispatchMode):

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    print('Hello World!')
                pass
        self.assertEqual(_get_current_dispatch_mode_stack(), [])
        with A() as mode1:
            self.assertEqual(_get_current_dispatch_mode_stack(), [mode1])
        with mode1:
            with A() as mode2:
                self.assertEqual(_get_current_dispatch_mode_stack(), [mode1, mode2])

    def test_all_same_mode(self):
        if False:
            return 10
        x = LoggingTensorMode()
        y = LoggingTensorMode()
        self.assertTrue(all_same_mode([x, x, x]))
        self.assertFalse(all_same_mode([x, None]))
        self.assertFalse(all_same_mode([x, y]))

    def test_tolist_numpy_with_torch_dispatch_mode(self) -> None:
        if False:
            return 10
        x = LoggingTensor(torch.tensor([2.0, 3.0]))
        with self.assertRaisesRegex(RuntimeError, 'is not supported for tensor subclasses.'):
            x.tolist()
        with self.assertRaisesRegex(RuntimeError, 'is not supported for tensor subclasses.'):
            x.numpy()
        with self.assertRaises(AssertionError):
            self.assertEqual(x, None)

    def test_record_stream(self) -> None:
        if False:
            return 10

        class TestMode(TorchDispatchMode):

            def __init__(self, testcase):
                if False:
                    print('Hello World!')
                self.testcase = testcase

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    for i in range(10):
                        print('nop')
                self.testcase.assertEqual(func.name(), 'aten::record_stream')
                self.testcase.assertIsInstance(args[0], torch.Tensor)
                self.testcase.assertIsInstance(args[1], torch.Stream)
                self.testcase.assertEqual(args[1].stream_id, 1)
                self.testcase.assertEqual(args[1].device_index, 2)
                self.testcase.assertEqual(args[1].device_type, 3)
        t = torch.tensor(5.0)
        s = torch.Stream(stream_id=1, device_index=2, device_type=3)
        with TestMode(self):
            t.record_stream(s)

    def test_return_stream(self) -> None:
        if False:
            return 10
        l_def = torch.library.Library('test_return_stream', 'DEF')
        l_def.define('return_stream(Tensor self) -> Stream')
        l_impl = torch.library.Library('test_return_stream', 'IMPL', 'CPU')
        l_impl.impl('return_stream', lambda _: torch.Stream(stream_id=0, device_index=1, device_type=2))

        class TestMode(TorchDispatchMode):

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    return 10
                return torch.Stream(stream_id=1, device_index=2, device_type=3)
        t = torch.tensor(5.0)
        s = torch.ops.test_return_stream.return_stream(t)
        self.assertIsInstance(s, torch.Stream)
        self.assertEqual(s.stream_id, 0)
        self.assertEqual(s.device_index, 1)
        self.assertEqual(s.device_type, 2)
        with TestMode():
            s = torch.ops.test_return_stream.return_stream(t)
        self.assertIsInstance(s, torch.Stream)
        self.assertEqual(s.stream_id, 1)
        self.assertEqual(s.device_index, 2)
        self.assertEqual(s.device_type, 3)

    def test_subclass_autograd_device_check(self) -> None:
        if False:
            print('Hello World!')

        class NonWrapperSubclass(torch.Tensor):
            elem: torch.Tensor
            __slots__ = ['elem']

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                r = torch.Tensor._make_subclass(cls, elem.to('meta'), elem.requires_grad)
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    i = 10
                    return i + 15

                def unwrap(e):
                    if False:
                        i = 10
                        return i + 15
                    return e.elem if isinstance(e, NonWrapperSubclass) else e

                def wrap(e):
                    if False:
                        print('Hello World!')
                    return NonWrapperSubclass(e) if isinstance(e, torch.Tensor) else e
                rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
                logging.getLogger('NonWrapperSubclass').info(f'{func.__module__}.{func.__name__}', args, kwargs, rs)
                return rs
        x = NonWrapperSubclass(torch.tensor([3.0, 4.0], requires_grad=True))
        y = torch.randn(2, requires_grad=True)
        z = x * y
        self.assertIsInstance(z, NonWrapperSubclass)
        z.sum().backward(torch.tensor(1))
        self.assertEqual(x.grad, y)
        self.assertEqual(y.grad, x)

    def test_none_wrapping(self):
        if False:
            return 10

        class SubclassWithNone(torch.Tensor):

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                if False:
                    print('Hello World!')
                r = torch.Tensor._make_wrapper_subclass(cls, elem.size(), dtype=elem.dtype, layout=elem.layout, device=elem.device, requires_grad=elem.requires_grad)
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    while True:
                        i = 10

                def unwrap(e):
                    if False:
                        return 10
                    return e.elem if isinstance(e, SubclassWithNone) else e

                def wrap(e):
                    if False:
                        return 10
                    return SubclassWithNone(e) if isinstance(e, torch.Tensor) else e
                rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
                if func.overloadpacket.__name__ == 'add':
                    return None
                else:
                    return rs
        x = SubclassWithNone(torch.rand(2))
        self.assertIsInstance(x * 2, SubclassWithNone)
        self.assertIsNone(x + 2)
        x.requires_grad_()
        out = x.acos().sum()
        with self.assertRaisesRegex(RuntimeError, 'but got None'):
            out.backward()

    def test_storage_can_be_converted_to_python_object(self):
        if False:
            i = 10
            return i + 15
        s = torch.Storage()
        z = LoggingTensor(torch.empty([]))
        z.set_(s)

    def test_autograd_in_attr(self):
        if False:
            print('Hello World!')
        true_t = torch.rand(2, requires_grad=True)
        t = LoggingTensorReentrant(true_t)
        out = t + 2
        self.assertFalse(out.requires_grad)
        self.assertIsNone(out.grad_fn)
        self.assertTrue(out.elem.requires_grad)
        self.assertIsNotNone(out.elem.grad_fn)
        with self.assertRaisesRegex(RuntimeError, 'does not require grad'):
            out.sum().backward()
        out.elem.sum().backward()
        self.assertIsNone(t.grad)
        self.assertIsNotNone(t.elem.grad)

    def test_dispatch_super_call(self):
        if False:
            i = 10
            return i + 15
        called = []

        class SubTensor(torch.Tensor):

            @staticmethod
            def __new__(cls, elem):
                if False:
                    while True:
                        i = 10
                return torch.Tensor._make_subclass(cls, elem)
            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    i = 10
                    return i + 15
                called.append(func)
                return super().__torch_dispatch__(func, types, args, kwargs)
        x = torch.randn(2)
        y = torch.randn(2)
        self.assertEqual(SubTensor(x) + SubTensor(y), x + y)
        self.assertEqual(called, [torch.ops.aten.add.Tensor])

    def test_dispatch_super_call_list_arg(self):
        if False:
            while True:
                i = 10
        called = []

        class SubTensorWithListArg(torch.Tensor):

            @staticmethod
            def __new__(cls, elem):
                if False:
                    while True:
                        i = 10
                return torch.Tensor._make_subclass(cls, elem)
            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    i = 10
                    return i + 15
                called.append(func)
                return super().__torch_dispatch__(func, types, list(args), kwargs)
        x = torch.randn(2)
        self.assertEqual(SubTensorWithListArg(x).neg(), x.neg())
        self.assertEqual(called, [torch.ops.aten.neg.default])

    def test_dispatch_super_dont_autograd(self):
        if False:
            while True:
                i = 10
        called = []

        class SubTensor(torch.Tensor):

            @staticmethod
            def __new__(cls, elem):
                if False:
                    return 10
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)
            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    while True:
                        i = 10
                called.append(func)
                self.assertTrue(args[0].requires_grad)
                r = super().__torch_dispatch__(func, types, args, kwargs)
                self.assertFalse(r.requires_grad)
                return r
        x = SubTensor(torch.randn(2, requires_grad=True))
        x.neg()
        self.assertEqual(called, [torch.ops.aten.neg.default])

    def test_set_data(self):
        if False:
            i = 10
            return i + 15
        called = 0

        class SubTensor(torch.Tensor):
            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    i = 10
                    return i + 15
                nonlocal called
                called += 1
                return super().__torch_dispatch__(func, types, args, kwargs)
        x = SubTensor(torch.empty(2))
        x.data
        self.assertEqual(called, 1)
        x.data = torch.empty(2)
        self.assertEqual(called, 1)
        x.data
        self.assertEqual(called, 2)
        self.assertIs(type(x), SubTensor)
        x.set_(torch.empty(2))
        self.assertEqual(called, 3)
        x.data
        self.assertEqual(called, 4)
        self.assertIs(type(x), SubTensor)

    def test_construct_int_tensor(self):
        if False:
            while True:
                i = 10

        class SubTensor(torch.Tensor):
            pass
        SubTensor(torch.zeros(2, dtype=torch.int))

    def test_multiple_ops_subclass(self):
        if False:
            for i in range(10):
                print('nop')

        class MySubclass(torch.Tensor):

            @staticmethod
            def __new__(cls, elem):
                if False:
                    while True:
                        i = 10
                r = torch.Tensor._make_subclass(cls, elem)
                return r
            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    return 10
                with no_dispatch():
                    return func(*args, **kwargs)
        x = MySubclass(torch.rand(2, 2, dtype=torch.complex64))
        y = x.conj()
        y.exp()

    @staticmethod
    def subclass_helper(cls, data, use_wrapper_subclass, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if use_wrapper_subclass:
            kwargs['device'] = data.device
            kwargs['dtype'] = data.dtype
            kwargs['layout'] = data.layout
            kwargs['requires_grad'] = True
            return torch.Tensor._make_wrapper_subclass(cls, data.size(), **kwargs)
        else:
            return torch.Tensor._make_subclass(cls, data, True, **kwargs)

    def test_is_contiguous_slow_path(self):
        if False:
            i = 10
            return i + 15
        data = torch.randn(3, 3)
        contiguous_data = data.clone()
        not_contiguous_data = torch.as_strided(data.clone(), (2, 2), (1, 2))
        for use_wrapper_subclass in [True, False]:

            class ExampleTensor1(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        print('Hello World!')
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy='strides')

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        while True:
                            i = 10
                    return NotImplemented

            class ExampleTensor2(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        i = 10
                        return i + 15
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy='strides')

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        for i in range(10):
                            print('nop')
                    if func.overloadpacket == torch.ops.aten.is_contiguous:
                        return contiguous_data.is_contiguous()
                    return NotImplemented

            class ExampleTensor3(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        i = 10
                        return i + 15
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy='strides')

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        return 10
                    if func.overloadpacket == torch.ops.aten.is_contiguous:
                        return not_contiguous_data.is_contiguous()
                    return NotImplemented
            err_msg = "Multiple dispatch failed for 'torch.ops.aten.is_contiguous'"
            e = ExampleTensor1(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.is_contiguous()
            with self.assertRaisesRegex(TypeError, err_msg):
                e.contiguous()
            e = ExampleTensor2(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.is_contiguous(), True)
            e.contiguous()
            err_msg = 'Multiple dispatch failed for'
            e = ExampleTensor3(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.is_contiguous(), False)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.contiguous()

    def test_fancy_strides(self):
        if False:
            return 10
        calls = []

        class ExampleTensor(torch.Tensor):

            @staticmethod
            def __new__(cls, data):
                if False:
                    for i in range(10):
                        print('nop')
                return TestPythonDispatch.subclass_helper(cls, data, False, dispatch_sizes_strides_policy='strides')

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                if func in [torch.ops.aten.is_contiguous.default, torch.ops.aten.is_contiguous.memory_format, torch.ops.aten.is_strides_like_format.default, torch.ops.aten.is_non_overlapping_and_dense.default, torch.ops.aten.stride.default]:
                    calls.append((func, list(args)[1:]))
                    return None
                with no_dispatch():
                    return func(*args, **kwargs)
        e = ExampleTensor(torch.randn(2, 2))
        self.assertFalse(e.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(calls, [(torch.ops.aten.is_contiguous.memory_format, [torch.channels_last])])
        calls.clear()
        self.assertFalse(torch.ops.aten.is_strides_like_format.default(e, torch.channels_last))
        self.assertEqual(calls, [(torch.ops.aten.is_strides_like_format.default, [torch.channels_last])])
        calls.clear()
        self.assertTrue(torch.ops.aten.is_non_overlapping_and_dense.default(e))
        self.assertEqual(calls, [(torch.ops.aten.is_non_overlapping_and_dense.default, [])])

    def test_device_slowpath(self):
        if False:
            return 10
        for use_wrapper_subclass in [True]:

            class ExampleTensor1(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        print('Hello World!')
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_device=True)

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        i = 10
                        return i + 15
                    return NotImplemented

            class ExampleTensor2(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        i = 10
                        return i + 15
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_device=True)

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        for i in range(10):
                            print('nop')
                    if func.overloadpacket == torch.ops.prim.device:
                        return torch.device('meta')
                    return NotImplemented

            class ExampleTensor3(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        while True:
                            i = 10
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_device=True)

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        for i in range(10):
                            print('nop')
                    if func.overloadpacket == torch.ops.prim.device:
                        return torch.device('meta')
                    return NotImplemented
            err_msg = "Multiple dispatch failed for 'torch.ops.prim.device'"
            with self.assertRaisesRegex(TypeError, err_msg):
                e = ExampleTensor1(torch.randn(3, 3), use_wrapper_subclass)
                e.device()
            ten = torch.rand([1])
            e = ExampleTensor2(torch.randn(3, 3, device='cpu'), use_wrapper_subclass)
            self.assertEqual(e.device.type, 'meta')
            self.assertEqual(ten.type_as(e).device.type, 'meta')
            e = ExampleTensor3(torch.randn(3, 3, device='cpu'), use_wrapper_subclass)
            self.assertEqual(e.device.type, 'meta')
            self.assertEqual(ten.type_as(e).device.type, 'meta')

    def test_dim_slowpath(self):
        if False:
            i = 10
            return i + 15
        data = torch.randn(3, 3)
        for use_wrapper_subclass in [True, False]:

            class DimNotImplementedTensor(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        i = 10
                        return i + 15
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy='sizes')

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        return 10
                    return NotImplemented

            class DimImplementedTensor(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        return 10
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy='sizes')

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        return 10
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    return NotImplemented
            err_msg = "Multiple dispatch failed for 'torch.ops.aten.dim'"
            e = DimNotImplementedTensor(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.dim()
            t = DimImplementedTensor(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(t.dim(), 2)

    def test_maybe_tuple_bug(self):
        if False:
            while True:
                i = 10

        class T(torch.Tensor):

            @classmethod
            def __torch_function__(cls, *args, **kwargs):
                if False:
                    return 10
                pass
        a = torch.rand(3)
        a[[T(), T()]]

    def test_standard_is_not_subclass(self):
        if False:
            while True:
                i = 10
        self.assertFalse(torch._C._dispatch_isTensorSubclassLike(torch.empty(0)))

    def test_sym_sizes_strides_slow_path(self):
        if False:
            return 10

        class TestTensor(torch.Tensor):

            @staticmethod
            def __new__(cls, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                r = torch.Tensor._make_wrapper_subclass(cls, (0,), dispatch_sizes_strides_policy='sizes')
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    while True:
                        i = 10
                if func in (torch.ops.aten.sym_size.default, torch.ops.aten.sym_stride.default):
                    from torch._dynamo.source import ConstantSource
                    from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic
                    shape_env = ShapeEnv()
                    si = shape_env.create_symintnode(shape_env.create_symbol(123, source=ConstantSource('abc'), dynamic_dim=DimDynamic.DUCK, constraint_dim=None), hint=123)
                    return (si,)
        t = TestTensor()
        si = t.size()[0]
        self.assertIsInstance(si, torch.SymInt)
        si = t.stride()[0]
        self.assertIsInstance(si, torch.SymInt)

    def test_strides_slow_path(self):
        if False:
            i = 10
            return i + 15
        for use_wrapper_subclass in [True, False]:

            class StridesNotImplemented(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        for i in range(10):
                            print('nop')
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy='strides')

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        for i in range(10):
                            print('nop')
                    return NotImplemented

            class StridesCustomReturn(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        print('Hello World!')
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy='strides')

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        return 10
                    if func == torch.ops.aten.sym_stride.default:
                        return (4, 2)
                    return NotImplemented

            class StridesDefaultReturn(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        for i in range(10):
                            print('nop')
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy='strides')

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        print('Hello World!')
                    if func == torch.ops.aten.sym_stride.default:
                        return None
                    return NotImplemented
            err_msg = "Multiple dispatch failed for 'torch.ops.aten.sym_stride'"
            e = StridesNotImplemented(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.stride()
            e = StridesCustomReturn(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.stride(), (4, 2))
            e = StridesDefaultReturn(torch.randn(6, 2), use_wrapper_subclass)
            self.assertEqual(e.stride(), (2, 1))

    def test_sizes_slow_path(self):
        if False:
            while True:
                i = 10
        for use_wrapper_subclass in [True, False]:
            data = torch.randn(6, 2)

            class SizesNotImplemented(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        for i in range(10):
                            print('nop')
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy='sizes')

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        i = 10
                        return i + 15
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    return NotImplemented

            class SizesCustomReturn(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        print('Hello World!')
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy='sizes')

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        print('Hello World!')
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    if func.overloadpacket == torch.ops.aten.sym_size:
                        return (5, 3)
                    return NotImplemented

            class SizesDefaultReturn(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        i = 10
                        return i + 15
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy='sizes')

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        while True:
                            i = 10
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    if func.overloadpacket == torch.ops.aten.sym_size:
                        return None
                    return NotImplemented
            err_msg = "Multiple dispatch failed for 'torch.ops.aten.sym_size'"
            e = SizesNotImplemented(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.size()
            e = SizesCustomReturn(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.size(), (5, 3))
            e = SizesDefaultReturn(torch.randn(4, 2), use_wrapper_subclass)
            self.assertEqual(e.size(), (4, 2))

    def test_custom_size_policy_dynamic_shapes(self):
        if False:
            for i in range(10):
                print('nop')
        data = torch.randn(6, 2)

        class CustomSizeDynamicShapesTensor(torch.Tensor):

            @staticmethod
            def __new__(cls, inner):
                if False:
                    i = 10
                    return i + 15
                return torch.Tensor._make_wrapper_subclass(cls, inner.size(), inner.stride(), None, None, inner.dtype, inner.layout, inner.device, False, inner.requires_grad, 'sizes')

            def __init__(self, inner):
                if False:
                    i = 10
                    return i + 15
                self.inner = inner

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                if False:
                    return 10
                if func == torch.ops.aten.sym_size.default:
                    return args[0].inner.shape
                if func == torch.ops.aten.sym_stride.default:
                    return args[0].inner.shape
                return NotImplemented
        x = torch.ones(2, 2)

        def trace_fn(x):
            if False:
                return 10
            x_wrapper = CustomSizeDynamicShapesTensor(x)
            return (x_wrapper.size(), x_wrapper.stride())
        fx_g = make_fx(trace_fn, tracing_mode='symbolic')(x)
        self.assertExpectedInline(fx_g.code.strip(), 'def forward(self, x_1):\n    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)\n    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1);  x_1 = None\n    return ((sym_size_int, sym_size_int_1), (sym_size_int, sym_size_int_1))')

    def test_data_ptr_respects_numel_slow_path(self):
        if False:
            for i in range(10):
                print('nop')
        data = torch.randn(6, 2)

        class NumelDefaultReturn(torch.Tensor):

            @staticmethod
            def __new__(cls, data, wrapper):
                if False:
                    i = 10
                    return i + 15
                return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy='sizes')

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                if False:
                    print('Hello World!')
                if func.overloadpacket == torch.ops.aten.dim:
                    return data.dim()
                if func.overloadpacket == torch.ops.aten.numel:
                    numel_called[0] = True
                    return None
                return NotImplemented
        for use_wrapper_subclass in (False, True):
            numel_called = [False]
            e = NumelDefaultReturn(torch.randn(2, 2), use_wrapper_subclass)
            e.data_ptr()
            self.assertTrue(numel_called[0])

    def test_layout_slow_path(self):
        if False:
            while True:
                i = 10
        for use_wrapper_subclass in [True, False]:
            data = torch.randn(6, 2)

            class LayoutNotImplemented(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        return 10
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_layout=True)

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        for i in range(10):
                            print('nop')
                    return NotImplemented

            class LayoutCustomReturn(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        print('Hello World!')
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_layout=True)

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        while True:
                            i = 10
                    if func.overloadpacket == torch.ops.prim.layout:
                        return torch.sparse_csr
                    return NotImplemented

            class LayoutDefaultReturn(torch.Tensor):

                @staticmethod
                def __new__(cls, data, wrapper):
                    if False:
                        while True:
                            i = 10
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_layout=True)

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if False:
                        return 10
                    if func.overloadpacket == torch.ops.prim.layout:
                        return data.layout
                    return NotImplemented
            err_msg = "Multiple dispatch failed for 'torch.ops.prim.layout'"
            e = LayoutNotImplemented(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.layout
            e = LayoutCustomReturn(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.layout, torch.sparse_csr)
            e = LayoutDefaultReturn(torch.randn(4, 2), use_wrapper_subclass)
            self.assertEqual(e.layout, torch.strided)

class TestPythonDispatcher(TestCase):

    def test_basic(self):
        if False:
            print('Hello World!')
        x = torch.randn(2, requires_grad=True)
        r = torch._C._EnablePythonDispatcher()
        torch.add(x, x)

    def test_lstsq(self):
        if False:
            while True:
                i = 10
        a = torch.randn(4, 3)
        b = torch.rand(4, 3)
        expected_shape = torch.linalg.lstsq(a, b).solution.shape
        r = torch._C._EnablePythonDispatcher()
        python_disp_shape = torch.linalg.lstsq(a, b).solution.shape
        self.assertEqual(expected_shape, python_disp_shape)

class TestWrapperSubclassAliasing(TestCase):

    def _test_wrapper_subclass_aliasing(self, op, args, kwargs):
        if False:
            for i in range(10):
                print('nop')

        def to_subclass(t: torch.Tensor):
            if False:
                print('Hello World!')
            return TwoTensor(t, t.clone())
        result_ref = op(*args, **kwargs)
        args_subclass = pytree.tree_map_only(torch.Tensor, to_subclass, args)
        kwargs_subclass = pytree.tree_map_only(torch.Tensor, to_subclass, kwargs)
        result_test = op(*args_subclass, **kwargs_subclass)
        args_ref_flat = pytree.arg_tree_leaves(*args, **kwargs)
        args_ref_flat_tensors = [x for x in args_ref_flat if isinstance(x, torch.Tensor)]
        args_test_flat = pytree.tree_leaves((args_subclass, kwargs_subclass))
        args_test_flat_tensors = [x for x in args_test_flat if isinstance(x, torch.Tensor)]
        result_ref_flat = pytree.tree_leaves(result_ref)
        result_ref_flat_tensors = [x for x in result_ref_flat if isinstance(x, torch.Tensor)]
        result_test_flat = pytree.tree_leaves(result_test)
        result_test_flat_tensors = [x for x in result_test_flat if isinstance(x, torch.Tensor)]
        for (o_ref, o_test) in zip(result_ref_flat_tensors, result_test_flat_tensors):
            for (a_ref, a_test) in zip(args_ref_flat_tensors, args_test_flat_tensors):
                out_is_inpt = o_ref is a_ref
                if out_is_inpt:
                    self.assertTrue(o_test is a_test)
                out_aliases_inpt = StorageWeakRef(o_ref.untyped_storage()) == StorageWeakRef(a_ref.untyped_storage())
                if out_aliases_inpt:
                    self.assertTrue(StorageWeakRef(o_test.untyped_storage()) == StorageWeakRef(a_test.untyped_storage()))
                else:
                    self.assertFalse(StorageWeakRef(o_test.untyped_storage()) == StorageWeakRef(a_test.untyped_storage()))

    @ops([op for op in op_db if op.name in ['mul', 'cat', 'index', 'mul_', 'view', 't_', 'split', 'native_batch_norm']], allowed_dtypes=(torch.float,))
    def test_wrapper_subclass_aliasing(self, device, dtype, op):
        if False:
            i = 10
            return i + 15
        samples = op.sample_inputs(device, dtype)
        sample = first_sample(self, samples)
        args = (sample.input, *sample.args)
        kwargs = sample.kwargs
        self._test_wrapper_subclass_aliasing(op, args, kwargs)

    @ops(custom_op_db, allowed_dtypes=(torch.float,))
    def test_wrapper_subclass_aliasing_custom(self, device, dtype, op):
        if False:
            for i in range(10):
                print('nop')
        samples = op.sample_inputs(device, dtype)
        sample = first_sample(self, samples)
        args = (sample.input, *sample.args)
        kwargs = sample.kwargs
        self._test_wrapper_subclass_aliasing(op, args, kwargs)

    def test_wrapper_subclass_aliasing_conv2d(self, device):
        if False:
            for i in range(10):
                print('nop')
        args = (torch.randn(4, 4, 4, 4), torch.randn(4, 4, 4, 4))
        kwargs = {}
        with torch.inference_mode():
            self._test_wrapper_subclass_aliasing(torch.ops.aten.conv2d.default, args, kwargs)

    def test_wrapper_subclass_aliasing_out_op(self, device):
        if False:
            print('Hello World!')
        args = (torch.ones(4), torch.ones(4))
        kwargs = {'out': torch.empty(4)}
        self._test_wrapper_subclass_aliasing(torch.ops.aten.add.out, args, kwargs)
instantiate_device_type_tests(TestWrapperSubclassAliasing, globals())
if __name__ == '__main__':
    run_tests()