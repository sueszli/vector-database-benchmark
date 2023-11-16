import contextlib
import gc
import io
import math
import os
import random
import sys
import tempfile
import threading
import time
import unittest
import uuid
import warnings
import operator
import subprocess
from copy import deepcopy
from collections import OrderedDict
from itertools import product
from operator import mul
from functools import reduce, partial
import torch
from torch import nn
from torch import inf, nan
from torch.autograd.function import once_differentiable
from torch.autograd.profiler import profile, record_function, emit_nvtx, emit_itt
from torch.autograd.profiler_util import _format_time, EventList, FunctionEvent, FunctionEventAvg
from torch.utils.checkpoint import checkpoint
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfNoLapack, slowTest, IS_WINDOWS, IS_MACOS, disable_gc, gradcheck, gradgradcheck, parametrize, instantiate_parametrized_tests, skipIfMps, set_warn_always_context
from torch.autograd import Variable, Function, detect_anomaly, kineto_available, _calculate_shape
from torch.autograd.function import InplaceFunction
import torch.autograd.forward_ad as fwAD
from torch.autograd.graph import GradientEdge
import torch.autograd._functions
from torch.testing._internal.common_methods_invocations import mask_not_all_zeros
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCPU, onlyCUDA, dtypes, dtypesIfCUDA, deviceCountAtLeast, skipMeta, dtypesIfMPS
from torch.testing._internal.common_dtype import floating_types_and
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
import weakref
import collections
import pickle

def graph_desc(fn):
    if False:
        while True:
            i = 10
    if fn is None:
        return 'None'
    result = type(fn).__name__ + '('
    next_functions = fn.next_functions
    for (next_fn, _) in next_functions:
        result += graph_desc(next_fn)
        result += ', '
    if next_functions:
        result = result[:-2]
    return result + ')'

class TestAutograd(TestCase):

    def test_copy_slices_graph_task_updates(self):
        if False:
            return 10

        def f1(x, y):
            if False:
                while True:
                    i = 10
            out = x.clone().view(-1)
            out += y
            return out

        def f2(x, y):
            if False:
                while True:
                    i = 10
            out = x.clone().view(-1)
            b = out * 2
            out += y
            return out + b
        x = torch.rand(2, requires_grad=True)
        y = torch.rand(2, requires_grad=True)
        y_safe = torch._C._functions.DelayedError('Boom!', 1)(y)
        for f in [f1, f2]:
            out = f(x, y_safe)
            with self.assertRaisesRegex(RuntimeError, 'Boom!'):
                out.sum().backward()
            out = f(x, y_safe)
            with self.assertRaisesRegex(RuntimeError, 'Boom!'):
                torch.autograd.grad(out.sum(), y)
            out = f(x, y_safe)
            torch.autograd.grad(out.sum(), x)
            out = f(x, y_safe)
            torch.autograd.grad(out.sum(), y_safe)
            out = f(x, y_safe)
            torch.autograd.grad(out.sum(), (x, y_safe))

        def f3(x, y):
            if False:
                while True:
                    i = 10
            out = x.clone().view(-1)

            def hook(*args):
                if False:
                    while True:
                        i = 10
                self.assertTrue(False)
            out.register_hook(hook)
            b = out + y
            out += y
            return (out + b, b)
        (out, b) = f3(x, y_safe)
        torch.autograd.grad(out.sum(), (b, y_safe))

    def test_grad_mode_class_decoration(self):
        if False:
            i = 10
            return i + 15
        with self.assertWarnsRegex(UserWarning, 'Decorating classes is deprecated'):

            @torch.no_grad()
            class Foo:

                def __init__(self):
                    if False:
                        while True:
                            i = 10
                    assert not torch.is_grad_enabled()

                def foo(self):
                    if False:
                        print('Hello World!')
                    assert torch.is_grad_enabled()
            foo = Foo()
            foo.foo()
        with warnings.catch_warnings(record=True) as w:

            @torch.no_grad()
            def foo():
                if False:
                    for i in range(10):
                        print('nop')
                assert not torch.is_grad_enabled()
            foo()

            class Foo2:

                @torch.no_grad()
                def __init__(self):
                    if False:
                        i = 10
                        return i + 15
                    assert not torch.is_grad_enabled()

                @torch.no_grad()
                def foo(self):
                    if False:
                        while True:
                            i = 10
                    assert not torch.is_grad_enabled()
            foo2 = Foo2()
            foo2.foo()
        self.assertEqual(len(w), 0)

    def test_tensor_grad_warnings(self):
        if False:
            for i in range(10):
                print('nop')
        dummy = torch.empty(1)
        with warnings.catch_warnings(record=True) as w:
            dummy.requires_grad_()
            foo = dummy.grad
            self.assertEqual(len(w), 0)
            dummy = dummy.clone()
            foo = dummy.grad
            self.assertEqual(len(w), 1)
            dummy.retain_grad()
            foo = dummy.grad
            self.assertEqual(len(w), 1)

    def _function_test(self, cls):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        result = cls.apply(x, 2, y)
        go = torch.ones((), requires_grad=True)
        result.sum().backward(go, create_graph=True)
        self.assertEqual(x.grad, y + torch.ones(5, 5))
        self.assertEqual(y.grad, x + torch.ones(5, 5) * 2)
        self.assertIsNotNone(x.grad.grad_fn)
        self.assertIsNotNone(y.grad.grad_fn)
        return (x, y)

    def test_function(self):
        if False:
            print('Hello World!')

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, tensor1, pyscalar, tensor2):
                if False:
                    for i in range(10):
                        print('nop')
                ctx.pyscalar = pyscalar
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + pyscalar * tensor2 + tensor1 * tensor2

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    print('Hello World!')
                (var1, var2) = ctx.saved_tensors
                self.assertIsInstance(var1, torch.Tensor)
                self.assertIsInstance(var2, torch.Tensor)
                self.assertIsInstance(grad_output, torch.Tensor)
                return (grad_output + grad_output * var2, None, grad_output * ctx.pyscalar + grad_output * var1)
        (x, y) = self._function_test(MyFunction)
        x_grad_desc = graph_desc(x.grad.grad_fn)
        y_grad_desc = graph_desc(y.grad.grad_fn)
        self.assertExpected(x_grad_desc, 'x_grad_desc')
        self.assertExpected(y_grad_desc, 'y_grad_desc')

    def test_once_differentiable(self):
        if False:
            for i in range(10):
                print('nop')

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, tensor1, pyscalar, tensor2):
                if False:
                    while True:
                        i = 10
                ctx.pyscalar = pyscalar
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + pyscalar * tensor2 + tensor1 * tensor2

            @staticmethod
            @once_differentiable
            def backward(ctx, grad_output):
                if False:
                    for i in range(10):
                        print('nop')
                self.assertFalse(torch.is_grad_enabled())
                (t1, t2) = ctx.saved_tensors
                return (grad_output + grad_output * t2, None, grad_output * ctx.pyscalar + grad_output * t1)
        (x, y) = self._function_test(MyFunction)
        self.assertEqual(graph_desc(x.grad.grad_fn), 'CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))')
        self.assertEqual(graph_desc(y.grad.grad_fn), 'CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))')

    def test_function_returns_input(self):
        if False:
            i = 10
            return i + 15

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                return x

            @staticmethod
            def backward(ctx, grad):
                if False:
                    for i in range(10):
                        print('nop')
                return grad * 2
        for shape in [(1,), ()]:
            v = torch.ones(shape, requires_grad=True)
            MyFunction.apply(v).backward()
            self.assertEqual(v.grad, torch.full(shape, 2.0))
            with torch.no_grad():
                v.grad.zero_()
            MyFunction.apply(v.clone()).backward()
            self.assertEqual(v.grad, torch.full(shape, 2.0))

    def test_function_returns_undefined_tensor(self):
        if False:
            print('Hello World!')

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    i = 10
                    return i + 15
                return x * 2

            @staticmethod
            def backward(ctx, grad):
                if False:
                    while True:
                        i = 10
                return None
        x = torch.ones(1, requires_grad=True)
        MyFunction.apply(x).backward()
        self.assertIsNone(x.grad)
        MyFunction.apply(x ** 2).backward()
        self.assertIsNone(x.grad)
        MyFunction.apply(x).sum().backward()
        self.assertIsNone(x.grad)
        self.assertIsNone(torch.autograd.grad(MyFunction.apply(x), x, allow_unused=True)[0])

    def test_materialize_grads(self):
        if False:
            for i in range(10):
                print('nop')

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                return x

            @staticmethod
            def backward(ctx, grad):
                if False:
                    while True:
                        i = 10
                self.assertEqual(grad, torch.zeros(1))
                return grad
        x = torch.ones(1, requires_grad=True)
        torch._C._functions.UndefinedGrad()(MyFunction.apply(x)).backward()

    def test_dont_materialize_grads(self):
        if False:
            print('Hello World!')

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    i = 10
                    return i + 15
                ctx.set_materialize_grads(False)
                return x

            @staticmethod
            def backward(ctx, grad):
                if False:
                    i = 10
                    return i + 15
                self.assertIsNone(grad)
                return grad
        x = torch.ones(1, requires_grad=True)
        torch._C._functions.UndefinedGrad()(MyFunction.apply(x)).backward()

    def test_set_materialize_non_diff_grads(self):
        if False:
            for i in range(10):
                print('nop')

        class Func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    for i in range(10):
                        print('nop')
                out0 = x.clone()
                out1 = x.clone()
                ctx.mark_non_differentiable(out1)
                ctx._materialize_non_diff_grads = False
                return (out0, out1)

            @staticmethod
            def backward(ctx, g0, g1):
                if False:
                    print('Hello World!')
                self.assertIsNone(g1)
                return g0
        a = torch.tensor(1.0, requires_grad=True)
        out = Func.apply(a)[0]
        out.backward()

    def test_legacy_function_deprecation_exception(self):
        if False:
            return 10

        class MyFunction(Function):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x

            def backward(self, grad_output):
                if False:
                    print('Hello World!')
                return grad_output
        with self.assertRaisesRegex(RuntimeError, 'Legacy autograd function with non-static forward method is deprecated'):
            MyFunction()(torch.randn(3, 4))

    class SimulateBackwardError(Function):

        @staticmethod
        def forward(ctx, input):
            if False:
                for i in range(10):
                    print('nop')
            return input.clone()

        @staticmethod
        @once_differentiable
        def backward(ctx, input):
            if False:
                while True:
                    i = 10
            raise Exception('Simulate error on backward pass')

    def test_custom_function_exception(self):
        if False:
            return 10
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        tmp = (t1 + t2) * (t1 + t2)
        t3 = TestAutograd.SimulateBackwardError.apply(tmp)
        with self.assertRaisesRegex(Exception, 'Simulate error on backward pass'):
            t3.sum().backward()

    def test_custom_function_non_tensor_inputs_outputs(self):
        if False:
            return 10

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, t1, t2, scale, t3):
                if False:
                    for i in range(10):
                        print('nop')
                t4 = t1 + t2 * t3
                t5 = t1 * t2 + t3
                t4 *= scale
                t5 *= scale
                ctx.scale = scale
                ctx.save_for_backward(t1, t2, t3)
                return (scale, t4, None, True, t5, 'bar', t1)

            @staticmethod
            @once_differentiable
            def backward(ctx, *grads):
                if False:
                    return 10
                self.assertEqual(7, len(grads))
                self.assertIsNone(grads[0])
                self.assertIsNone(grads[2])
                self.assertIsNone(grads[3])
                self.assertIsNone(grads[5])
                scale = ctx.scale
                (var1, var2, var3) = ctx.saved_tensors
                return (grads[1] * scale + grads[4] * var2 * scale + grads[6], grads[1] * var3 * scale + grads[4] * var1 * scale, None, grads[1] * var2 * scale + grads[4] * scale)
        t1 = torch.rand(10, dtype=torch.double, requires_grad=True)
        t2 = torch.rand(10, dtype=torch.double, requires_grad=True)
        t3 = torch.rand(10, dtype=torch.double)
        scale = random.randint(0, 10)
        res = MyFunction.apply(t1, t2, scale, t3)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual('bar', res[5])
        self.assertEqual(t1, res[6])
        torch.autograd.backward([res[1].sum(), res[4].sum(), res[6].sum()])
        self.assertIsNotNone(t1.grad)
        self.assertIsNotNone(t2.grad)
        self.assertIsNone(t3.grad)

        def foo(t1, t2, t3):
            if False:
                return 10
            res = MyFunction.apply(t1, t2, scale, t3)
            return (res[1], res[4], res[6])
        gradcheck(foo, (t1, t2, t3))

    def test_custom_function_no_tensors(self):
        if False:
            while True:
                i = 10

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, t1, t2, scale, t3):
                if False:
                    print('Hello World!')
                t4 = t1 + t2 * t3
                t5 = t1 * t2 + t3
                t4 *= scale
                t5 *= scale
                return (scale, t4, None, True, t5, 'bar', t1)

            @staticmethod
            @once_differentiable
            def backward(ctx, *args):
                if False:
                    i = 10
                    return i + 15
                return (args[0], args[1], None, args[2])
        t1 = random.random()
        t2 = random.random()
        t3 = random.random()
        scale = random.randint(0, 10)
        res = MyFunction.apply(t1, t2, scale, t3)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual('bar', res[5])
        self.assertEqual(t1, res[6])

    def test_invalid_gradients(self):
        if False:
            return 10

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    i = 10
                    return i + 15
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    print('Hello World!')
                return torch.randn(10, dtype=torch.float)
        with self.assertRaisesRegex(RuntimeError, 'expected shape'):
            input = torch.randn(5, 5, dtype=torch.float, requires_grad=True)
            MyFunction.apply(input).sum().backward()

    def test_unrelated_inputs(self):
        if False:
            for i in range(10):
                print('nop')

        def my_function(x, y):
            if False:
                print('Hello World!')
            return x * x
        x = torch.rand(10, dtype=torch.double, requires_grad=True)
        y = torch.rand(10, dtype=torch.double, requires_grad=True)
        gradcheck(my_function, (x, y))
        gradgradcheck(my_function, (x, y))

    def test_not_implemented_grad(self):
        if False:
            while True:
                i = 10
        a = torch.rand(2, requires_grad=True)
        y = torch.nextafter(a, a).sum()
        with self.assertRaisesRegex(NotImplementedError, 'the derivative for .* is not implemented'):
            y.backward()

    def test_not_implemented_fwad(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(3)
        v = torch.rand(3)
        with fwAD.dual_level():
            dual_x = fwAD.make_dual(x, v)
            err_msg = 'Trying to use forward AD with .* that does not support it'
            hint_msg = 'Running forward AD for an OP that does not implement it should raise a NotImplementedError'
            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                torch.igamma(dual_x, dual_x)

    def test_will_engine_execute_node(self):
        if False:
            i = 10
            return i + 15
        counter = [0]

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                return x * 2

            @staticmethod
            def backward(ctx, gO):
                if False:
                    for i in range(10):
                        print('nop')
                return gO * 2

        def get_grad_fn(t):
            if False:
                for i in range(10):
                    print('nop')
            if t.requires_grad and t.grad_fn is None:
                return t.clone().grad_fn.next_functions[0][0]
            else:
                return t.grad_fn
        a = torch.randn(2, 3, 4, requires_grad=True)
        a2 = torch.randn(2, 3, 4, requires_grad=True)
        b = a * a2
        b2 = b.cos()
        c = MyFunction.apply(b)
        should_execute = list(map(get_grad_fn, (a, b, c)))
        should_not_execute = list(map(get_grad_fn, (a2, b2)))

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            counter[0] += 1
            for g in should_execute:
                self.assertTrue(torch._C._will_engine_execute_node(g))
            for g in should_not_execute:
                self.assertFalse(torch._C._will_engine_execute_node(g))
        b.register_hook(fn)
        c.register_hook(fn)
        out = c.sum()
        torch.autograd.backward(out, inputs=(a, b), retain_graph=True)
        self.assertEqual(counter[0], 2)
        should_execute = list(map(get_grad_fn, (a, a2, b, c)))
        should_not_execute = list(map(get_grad_fn, (b2,)))
        torch.autograd.backward(out, retain_graph=True)
        with self.assertRaisesRegex(RuntimeError, 'are currently running autograd.grad()'):
            torch.autograd.grad(out, (a,))
        a = torch.randn(1, 2, 3, requires_grad=True) * 2
        b = a * 2

        def fn(x):
            if False:
                return 10
            counter[0] += 1
            self.assertTrue(torch._C._will_engine_execute_node(b.grad_fn))
        b.register_hook(fn)
        counter[0] = 0
        torch.autograd.grad(b.sum(), (a,))
        self.assertEqual(counter[0], 1)
        with self.assertRaisesRegex(RuntimeError, 'during the backward pass'):
            torch._C._will_engine_execute_node(out.grad_fn)
        with self.assertRaisesRegex(RuntimeError, 'expects an grad_fn'):
            torch._C._will_engine_execute_node(out)

    def test_custom_function_vmap_defaults(self):
        if False:
            i = 10
            return i + 15

        class MySquare(Function):

            @staticmethod
            def forward(x):
                if False:
                    print('Hello World!')
                return x ** 2

            @staticmethod
            def setup_context(ctx, inputs, output):
                if False:
                    return 10
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            def backward(ctx, gO):
                if False:
                    print('Hello World!')
                (x,) = ctx.saved_tensors
                return gO * 2 * x
        self.assertFalse(MySquare.generate_vmap_rule)
        self.assertTrue(hasattr(MySquare, 'vmap'))

    def test_custom_function_setup_context_simple(self):
        if False:
            for i in range(10):
                print('nop')

        class MySquare(Function):

            @staticmethod
            def forward(x):
                if False:
                    i = 10
                    return i + 15
                return x ** 2

            @staticmethod
            def setup_context(ctx, inputs, output):
                if False:
                    return 10
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            def backward(ctx, gO):
                if False:
                    print('Hello World!')
                (x,) = ctx.saved_tensors
                return gO * 2 * x
        x = torch.randn([], requires_grad=True)
        y = MySquare.apply(x)
        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, 2 * x)

    def test_custom_function_setup_context_multi_output(self):
        if False:
            return 10

        class MySquare(Function):

            @staticmethod
            def forward(x):
                if False:
                    print('Hello World!')
                two_x = x.item() * 2
                return (x ** 2, two_x)

            @staticmethod
            def setup_context(ctx, inputs, output):
                if False:
                    for i in range(10):
                        print('nop')
                (x,) = inputs
                (_, two_x) = output
                ctx.two_x = two_x

            @staticmethod
            @once_differentiable
            def backward(ctx, gO, _):
                if False:
                    i = 10
                    return i + 15
                return gO * ctx.two_x
        x = torch.randn([], requires_grad=True)
        (y, _) = MySquare.apply(x)
        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, 2 * x)

    def test_custom_function_setup_context_multi_input(self):
        if False:
            for i in range(10):
                print('nop')

        class MyReshape(Function):

            @staticmethod
            def forward(x, shape, scale_forward, scale_backward):
                if False:
                    while True:
                        i = 10
                return x.reshape(shape) * scale_forward

            @staticmethod
            def setup_context(ctx, inputs, output):
                if False:
                    while True:
                        i = 10
                (x, shape, scale_forward, scale_backward) = inputs
                ctx.scale_backward = scale_backward
                ctx.x_shape = x.shape

            @staticmethod
            def backward(ctx, gO):
                if False:
                    print('Hello World!')
                return (gO.reshape(ctx.x_shape) * ctx.scale_backward, None, None, None)

        class MyReshapeRef(Function):

            @staticmethod
            def forward(ctx, x, shape, scale_forward, scale_backward):
                if False:
                    print('Hello World!')
                ctx.scale_backward = scale_backward
                ctx.x_shape = x.shape
                return x.reshape(shape) * scale_forward

            @staticmethod
            def backward(ctx, gO):
                if False:
                    print('Hello World!')
                return (gO.reshape(ctx.x_shape) * ctx.scale_backward, None, None, None)

        def test(x, shape, scale_forward, scale_backward):
            if False:
                print('Hello World!')
            y = MyReshape.apply(x, shape, scale_forward, scale_backward).sum()
            (gx,) = torch.autograd.grad(y, x)
            y_expected = MyReshapeRef.apply(x, shape, scale_forward, scale_backward).sum()
            (gx_expected,) = torch.autograd.grad(y_expected, x)
            self.assertEqual(y_expected, y)
            self.assertEqual(gx_expected, gx)
        test(torch.randn(24, requires_grad=True), (3, 8), 7, 11)
        test(torch.randn(2, 3, 4, requires_grad=True), (6, 4), -1, 2)

    def test_accumulate_grad(self):
        if False:
            return 10
        grad_output = torch.ones(5, 5)

        def compute_grad(create_graph):
            if False:
                while True:
                    i = 10
            x = torch.randn(5, 5, requires_grad=True)
            y = x + 2
            y.backward(grad_output, retain_graph=True)
            x_grad = x.grad
            x_grad_clone = x.grad.clone()
            y.backward(grad_output, create_graph=create_graph)
            return (x_grad, x_grad_clone)
        (x_grad, x_grad_clone) = compute_grad(create_graph=False)
        self.assertEqual(x_grad, x_grad_clone * 2)
        (x_grad, x_grad_clone) = compute_grad(create_graph=True)
        self.assertEqual(x_grad, x_grad_clone)

    def test_accumulate_grad_tensor_reference(self):
        if False:
            i = 10
            return i + 15

        def _test_grad_tensor(params_grad_tensor, backward_grad_tensor, should_preserve_reference, create_graph):
            if False:
                for i in range(10):
                    print('nop')
            params = torch.tensor([1.5, 1.5]).requires_grad_()
            params.grad = params_grad_tensor
            grad_saved = params.grad
            params.backward(backward_grad_tensor, create_graph=create_graph)
            self.assertEqual(id(grad_saved) == id(params.grad), should_preserve_reference)
        for create_graph in (False, True):
            _test_grad_tensor(torch.sparse_coo_tensor(torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])), torch.tensor([1.5, 1.5]), False, create_graph)
            _test_grad_tensor(torch.tensor([1.5, 1.5]), torch.tensor([1.5, 1.5]), not create_graph, create_graph)
            _test_grad_tensor(torch.sparse_coo_tensor(torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])), torch.sparse_coo_tensor(torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])), not create_graph, create_graph)

    def test_accumulate_grad_with_zero_numel_grad(self):
        if False:
            return 10
        a = torch.rand(4, 0, requires_grad=True)
        b = torch.rand(4, 1, requires_grad=True)
        c = a + b
        assert c.shape == (4, 0)
        c.sum().backward()
        self.assertEqual(b.grad, torch.zeros(4, 1))
        self.assertEqual(a.grad, torch.zeros(4, 0))

    def test_hessian_vector(self):
        if False:
            return 10
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)
        z = x ** 2 + y * x + y ** 2
        z.backward(torch.ones(2, 2), create_graph=True)
        with torch.no_grad():
            x_grad = 2 * x + y
            y_grad = x + 2 * y
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)
        grad_sum = 2 * x.grad + y.grad
        grad_sum.backward(torch.ones(2, 2))
        x_hv = torch.ones(2, 2) * 5
        y_hv = torch.ones(2, 2) * 4
        self.assertEqual(x.grad, x_grad + x_hv)
        self.assertEqual(y.grad, y_grad + y_hv)

    def test_grad(self):
        if False:
            while True:
                i = 10
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)
        z = x ** 2 + y * x + y ** 2
        z.backward(torch.ones(2, 2), create_graph=True)
        x_grad = 2 * x + y
        y_grad = x + 2 * y
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)
        grad_sum = 2 * x.grad + y.grad
        x_hv = torch.autograd.grad(outputs=[grad_sum], grad_outputs=[torch.ones(2, 2)], inputs=[x], create_graph=True)
        expected_x_hv = torch.ones(2, 2) * 5
        expected_y_hv = torch.ones(2, 2) * 4
        self.assertEqual(x_hv[0], expected_x_hv)
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)
        grad_out = torch.ones(2)
        try:
            torch.autograd.grad(outputs=[grad_sum], grad_outputs=[grad_out], inputs=[x], create_graph=True)
            self.assertFail()
        except RuntimeError as error:
            self.assertEqual(str(error), 'Mismatch in shape: grad_output[0] has a shape of ' + str(grad_out.shape) + ' and output[0] has a shape of ' + str(grad_sum.shape) + '.')

    def test_grad_to_node(self):
        if False:
            print('Hello World!')

        def check_matches(out, inp):
            if False:
                for i in range(10):
                    print('nop')
            ref = torch.autograd.grad(out.sum(), inp)
            edge = torch.autograd.graph.get_gradient_edge(inp)
            new = torch.autograd.grad(out.sum(), edge)
            self.assertEqual(ref, new)
        x = torch.rand(2, requires_grad=True)
        out = x.clone()
        check_matches(out, x)
        x = x.clone()
        out = x.clone()
        check_matches(out, x)
        x = torch.autograd._functions.Resize.apply(x, (2,))
        out = x.clone()
        check_matches(out, x)
        x = torch.var_mean(x)[1]
        out = x.clone()
        check_matches(out, x)

    def test_grad_to_node_set(self):
        if False:
            while True:
                i = 10
        x = torch.rand(2, requires_grad=True)
        x_edge = torch.autograd.graph.get_gradient_edge(x)
        out = x.clone()
        with torch.no_grad():
            x.set_(torch.rand_like(x))
        with self.assertRaisesRegex(RuntimeError, 'to not have been used in the graph'):
            torch.autograd.grad(out.sum(), x)
        torch.autograd.grad(out.sum(), x_edge)

    def test_grad_to_node_inplace(self):
        if False:
            print('Hello World!')
        x = torch.rand(2, requires_grad=True).clone()
        x_edge = torch.autograd.graph.get_gradient_edge(x)
        x *= 2
        (g_old, g_new) = torch.autograd.grad(x.sum(), (x_edge, x))
        self.assertEqual(g_old, 2 * torch.ones_like(x))
        self.assertEqual(g_new, torch.ones_like(x))

    def test_grad_to_node_multi(self):
        if False:
            print('Hello World!')
        x = torch.rand(2, requires_grad=True).clone()
        y = torch.rand(2, requires_grad=True).clone()
        out = x + y
        ref = torch.autograd.grad(out.sum(), (x, y))
        inp_edges = (GradientEdge(x.grad_fn, x.output_nr), GradientEdge(y.grad_fn, y.output_nr))
        new = torch.autograd.grad(out.sum(), inp_edges)
        self.assertEqual(ref, new)

    def test_grad_to_node_materialize(self):
        if False:
            i = 10
            return i + 15
        x = torch.rand(2, requires_grad=True).clone()
        edge_x = GradientEdge(x.grad_fn, x.output_nr)
        y = torch.rand(2, requires_grad=True).clone()
        edge_y = GradientEdge(y.grad_fn, y.output_nr)
        out = x.clone()
        torch.autograd.grad(out.sum(), (edge_x, y), allow_unused=True, materialize_grads=True)
        torch.autograd.grad(out.sum(), (x, y), allow_unused=True, materialize_grads=True)
        torch.autograd.grad(out.sum(), (x, edge_y), allow_unused=True)
        with self.assertRaisesRegex(RuntimeError, 'materialize_grads cannot be used when the given input is a GradientEdge'):
            torch.autograd.grad(out.sum(), (x, edge_y), allow_unused=True, materialize_grads=True)

    def test_backward_to_node(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.rand(2, requires_grad=True).clone()
        edge_x = GradientEdge(x.grad_fn, x.output_nr)
        y = torch.rand(2, requires_grad=True).clone()
        edge_y = GradientEdge(y.grad_fn, y.output_nr)
        out = x.clone()
        torch.autograd.backward(out.sum(), inputs=(edge_x, y))
        torch.autograd.backward(out.sum(), inputs=(x, y))
        torch.autograd.backward(out.sum(), inputs=(x, edge_y))
        torch.autograd.backward(out.sum(), inputs=(edge_x, edge_y))

    def test_grad_nonleaf(self):
        if False:
            i = 10
            return i + 15
        x_init = torch.randn(2, 2, requires_grad=True)
        x = x_init
        y = torch.randn(2, 2, requires_grad=True)
        grad_output = torch.ones(2, 2)

        def fn(x):
            if False:
                while True:
                    i = 10
            return x ** 2 + y * x + y ** 2
        for _ in range(5):
            (grad_x,) = torch.autograd.grad(fn(x), x, grad_outputs=grad_output, create_graph=True)
            grad_x_expected = 2 * x + y
            self.assertIsNone(y.grad)
            self.assertIsNone(x.grad)
            self.assertEqual(grad_x, grad_x_expected)
            x = x + 0.05 * grad_x
        val_init = fn(x_init).sum()
        val_final = fn(x).sum()
        self.assertGreater(val_final, val_init)
        x.backward(grad_output)
        self.assertIsNotNone(y.grad)
        self.assertIsNotNone(x_init.grad)

    def test_grad_nonleaf_many_outputs(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(4, 2, requires_grad=True)
        (a, b) = x.chunk(2)

        def hook(*grads):
            if False:
                for i in range(10):
                    print('nop')
            hook_called[0] = True
        hook_called = [False]
        x.register_hook(hook)
        go = torch.randn(2, 2)
        (grad_a, grad_b) = torch.autograd.grad(a + 2 * b, [a, b], grad_outputs=go, create_graph=True)
        self.assertEqual(grad_a, go)
        self.assertEqual(grad_b, go * 2)
        self.assertFalse(hook_called[0])
        self.assertIsNone(x.grad)

    def test_grad_nonleaf_register_hook(self):
        if False:
            while True:
                i = 10
        x = torch.randn(5, requires_grad=True)
        x_list = x.unbind()
        x0 = x_list[0]
        hook_results = [None]

        def hook(grad):
            if False:
                print('Hello World!')
            hook_results[0] = grad
        x0.register_hook(hook)
        x_list[0].backward()
        self.assertEqual(hook_results[0], torch.tensor(1.0))
        expected_grad = torch.tensor([1.0, 0, 0, 0, 0])
        self.assertEqual(x.grad, expected_grad)
        self.assertIsNone(x_list[0].grad)
        for i in range(1, 5, 1):
            x_list[i].backward()
            self.assertEqual(hook_results[0], None)
            expected_grad[i] = 1.0
            self.assertEqual(x.grad, expected_grad)
            self.assertIsNone(x_list[i].grad)

    def test_grad_materialize_grads(self):
        if False:
            i = 10
            return i + 15
        x = torch.tensor(0.5, requires_grad=True)
        a = torch.tensor(1.0, requires_grad=True)
        y = x * a
        dydx = torch.autograd.grad(y, x, create_graph=True)
        d2ydx2_none = torch.autograd.grad(dydx, x, create_graph=True, allow_unused=True)
        d2ydx2 = torch.autograd.grad(dydx, x, create_graph=True, allow_unused=True, materialize_grads=True)
        d3ydx3 = torch.autograd.grad(d2ydx2, x, materialize_grads=True)
        self.assertIsNone(d2ydx2_none[0])
        self.assertEqual(d2ydx2[0].item(), 0)
        self.assertEqual(d3ydx3[0].item(), 0)
        with self.assertRaisesRegex(ValueError, 'Expected allow_unused to be True or not passed when'):
            torch.autograd.grad(y, x, allow_unused=False, materialize_grads=True)

    def test_post_accumulate_grad_hook_on_non_leaf(self):
        if False:
            for i in range(10):
                print('nop')

        def hook(tensor):
            if False:
                return 10
            tensor.sub_(1.0)
        leaf = torch.rand(3, requires_grad=True)
        non_leaf = 2.0 * leaf
        with self.assertRaisesRegex(RuntimeError, 'post accumulate grad hooks cannot be registered on non-leaf tensors'):
            non_leaf.register_post_accumulate_grad_hook(hook)

    def test_post_accumulate_grad_hook_multiple_hooks(self):
        if False:
            print('Hello World!')

        def hook1(tensor):
            if False:
                print('Hello World!')
            tensor.sub_(tensor.grad)

        def hook2(tensor):
            if False:
                while True:
                    i = 10
            tensor.mul_(4.0)
        tensor = torch.rand(3, requires_grad=True)
        tensor_ref = tensor.clone().detach()
        tensor.register_post_accumulate_grad_hook(hook1)
        tensor.register_post_accumulate_grad_hook(hook2)
        sum = tensor.sum()
        sum.backward()
        self.assertEqual(4.0 * (tensor_ref - 1.0), tensor)

    def test_post_accumulate_grad_hook_multiple_tensors(self):
        if False:
            return 10

        def hook(tensor):
            if False:
                i = 10
                return i + 15
            tensor.sub_(tensor.grad)
        tensor1 = torch.rand(3, requires_grad=True)
        tensor1_ref = tensor1.clone().detach()
        tensor2 = torch.rand(5, requires_grad=True)
        tensor2_ref = tensor2.clone().detach()
        tensor1.register_post_accumulate_grad_hook(hook)
        tensor2.register_post_accumulate_grad_hook(hook)
        tensor1.sum().backward()
        tensor2.sum().backward()
        self.assertEqual(tensor1_ref - 1.0, tensor1)
        self.assertEqual(tensor2_ref - 1.0, tensor2)

    def test_post_accumulate_grad_hook_returns_not_None(self):
        if False:
            print('Hello World!')

        def bad_hook(tensor):
            if False:
                for i in range(10):
                    print('nop')
            return tensor.grad
        tensor = torch.rand(2, 3, requires_grad=True)
        tensor.register_post_accumulate_grad_hook(bad_hook)
        with self.assertRaisesRegex(RuntimeError, 'hooks should return None.'):
            tensor.sum().backward()

    def test_post_accumulate_grad_hook_e2e(self):
        if False:
            while True:
                i = 10

        def setup_optim_in_bwd(model):
            if False:
                return 10
            optims = {}
            handles = []

            def optim_step_hook(param):
                if False:
                    for i in range(10):
                        print('nop')
                optims[param].step()
                optims[param].zero_grad()
            for p in model.parameters():
                optims[p] = torch.optim.Adam([p])
                handles.append(p.register_post_accumulate_grad_hook(optim_step_hook))
            return handles
        model = torch.nn.Linear(3, 2)
        input = torch.rand(2, 3)
        handles = setup_optim_in_bwd(model)
        model_copy = deepcopy(model)
        optim_copy = torch.optim.Adam(model_copy.parameters())
        iters = 5
        for _ in range(iters):
            loss = model(input).sum()
            loss.backward()
            loss_copy = model_copy(input).sum()
            loss_copy.backward()
            optim_copy.step()
            optim_copy.zero_grad()
        params_copy = []
        for (p_reference, p) in zip(model_copy.parameters(), model.parameters()):
            self.assertEqual(p_reference, p)
            params_copy.append(p_reference.clone().detach())
        for h in handles:
            h.remove()
        for _ in range(iters):
            loss = model(input).sum()
            loss.backward()
            loss_copy = model_copy(input).sum()
            loss_copy.backward()
            optim_copy.step()
            optim_copy.zero_grad()
        for (p_static, p_reference, p) in zip(params_copy, model_copy.parameters(), model.parameters()):
            self.assertEqual(p_static, p)
            self.assertNotEqual(p_reference, p)

    def test_post_accumulate_grad_hook_gets_cleaned_up(self):
        if False:
            i = 10
            return i + 15

        def fun_stuff_with_hook():
            if False:
                print('Hello World!')
            thing_to_put_in_hook = torch.rand(3)

            def hook(tensor):
                if False:
                    for i in range(10):
                        print('nop')
                tensor.sub_(tensor.grad)
                tensor.add_(thing_to_put_in_hook)
            tensor = torch.rand(3, requires_grad=True)
            tensor.register_post_accumulate_grad_hook(hook)
            tensor.sum().backward()
            ref = weakref.ref(thing_to_put_in_hook)
            gc.collect()
            return (tensor, ref)
        with disable_gc():
            (tensor, ref) = fun_stuff_with_hook()
            self.assertIsNotNone(ref())
            del tensor
            gc.collect()
            self.assertIsNone(ref())

    def test_post_accumulate_grad_hook_ordering(self):
        if False:
            print('Hello World!')
        tensor = torch.rand(3, requires_grad=True)

        def pre_hook(grad):
            if False:
                for i in range(10):
                    print('nop')
            return grad.sub(2.0)

        def acc_grad_node_pre_hook(grad_out):
            if False:
                i = 10
                return i + 15
            return (grad_out[0].div(5.0),)

        def post_acc_grad_hook(tensor):
            if False:
                for i in range(10):
                    print('nop')
            tensor.grad.add_(0.5)

        def acc_grad_node_post_hook(grad_in, grad_out):
            if False:
                for i in range(10):
                    print('nop')
            tensor.grad = grad_out[0].mul(10)
        acc_grad = tensor.view_as(tensor).grad_fn.next_functions[0][0]
        tensor.register_hook(pre_hook)
        acc_grad.register_prehook(acc_grad_node_pre_hook)
        tensor.register_post_accumulate_grad_hook(post_acc_grad_hook)
        acc_grad.register_hook(acc_grad_node_post_hook)
        tensor.sum().backward()
        self.assertEqual(torch.tensor([3.0, 3.0, 3.0]), tensor.grad)

    def test_hook_with_no_name(self):
        if False:
            i = 10
            return i + 15

        class MyHookClass:

            def __call__(self, grad):
                if False:
                    print('Hello World!')
                return grad.clone()
        x = torch.randn(5, requires_grad=True).clone()
        x.register_hook(MyHookClass())
        x.sum().backward()

    def test_prehook_ordering(self):
        if False:
            print('Hello World!')
        log = []

        def hook1(g):
            if False:
                for i in range(10):
                    print('nop')
            log.append(1)
            return g * 3

        def hook2(gs):
            if False:
                return 10
            log.append(2)
            return tuple((g * 2 for g in gs))
        a = torch.tensor(1.0, requires_grad=True)
        b = a.clone()
        b.grad_fn.register_prehook(hook2)
        b.register_hook(hook1)
        b.grad_fn.register_prehook(hook2)
        acc = b.grad_fn.next_functions[0][0]
        a.register_hook(hook1)
        acc.register_prehook(hook2)
        a.register_hook(hook1)
        b.sum().backward(retain_graph=True)
        self.assertEqual(log, [1, 2, 2, 1, 1, 2])
        log = []
        torch.autograd.grad(b.sum(), inputs=(a,), retain_graph=True)
        self.assertEqual(log, [1, 2, 2, 1, 1])
        log = []
        b.sum().backward(inputs=(b,))
        self.assertEqual(log, [1, 2, 2])
        self.assertEqual(b.grad.item(), 3)

    def test_retains_grad_can_always_observe_tensor_prehook(self):
        if False:
            i = 10
            return i + 15

        def tensor_prehook(g):
            if False:
                print('Hello World!')
            return g * 2
        a = torch.tensor(1.0, requires_grad=True)
        b = a.clone()
        b.register_hook(tensor_prehook)
        b.retain_grad()
        b.register_hook(tensor_prehook)
        b.clone().backward()
        self.assertEqual(b.grad.item(), 4)
        a = torch.tensor(1.0, requires_grad=True)
        b = a.clone()
        b.retain_grad()
        b.register_hook(tensor_prehook)
        b.clone().backward()
        self.assertEqual(b.grad.item(), 2)

    def test_accumulate_grad_posthooks_can_observe_tensor_prehook(self):
        if False:
            return 10
        a = torch.tensor(1.0, requires_grad=True)

        def tensor_prehook(g):
            if False:
                while True:
                    i = 10
            return g * 2

        def posthook(gO, gI):
            if False:
                while True:
                    i = 10
            self.assertTrue(torch.allclose(gI[0], a * 2))
            self.assertEqual(len(gO), 0)

        def prehook(gI):
            if False:
                print('Hello World!')
            self.assertTrue(torch.allclose(gI[0], a * 2))
            self.assertEqual(len(gI), 1)
        b = a.clone()
        acc = b.grad_fn.next_functions[0][0]
        acc.register_hook(posthook)
        acc.register_prehook(prehook)
        a.register_hook(tensor_prehook)
        b.backward()

    def test_hook_edge_case_when_called_with_grad(self):
        if False:
            for i in range(10):
                print('nop')
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        c = b * 2
        tensor_hook_count = [0]
        prehook_count = [0]
        posthook_count = [0]

        def reset_counts():
            if False:
                for i in range(10):
                    print('nop')
            nonlocal tensor_hook_count, prehook_count, posthook_count
            tensor_hook_count = [0]
            prehook_count = [0]
            posthook_count = [0]

        def tensor_prehook(g):
            if False:
                for i in range(10):
                    print('nop')
            tensor_hook_count[0] += 1

        def prehook(g):
            if False:
                for i in range(10):
                    print('nop')
            prehook_count[0] += 1

        def posthook(gI, gO):
            if False:
                return 10
            posthook_count[0] += 1
        a.register_hook(tensor_prehook)
        b.register_hook(tensor_prehook)
        acc = b.grad_fn.next_functions[0][0]
        acc.register_hook(posthook)
        acc.register_prehook(prehook)
        b.grad_fn.register_hook(posthook)
        b.grad_fn.register_prehook(prehook)
        torch.autograd.grad(c, inputs=b, retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 1)
        self.assertEqual(posthook_count[0], 0)
        self.assertEqual(prehook_count[0], 0)
        reset_counts()
        torch.autograd.grad(c, inputs=(a, b), retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 2)
        self.assertEqual(posthook_count[0], 1)
        self.assertEqual(prehook_count[0], 1)
        reset_counts()
        c.backward(retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 2)
        self.assertEqual(posthook_count[0], 2)
        self.assertEqual(prehook_count[0], 2)
        reset_counts()
        c.backward(inputs=(a, b), retain_graph=True)
        self.assertEqual(tensor_hook_count[0], 2)
        self.assertEqual(posthook_count[0], 2)
        self.assertEqual(prehook_count[0], 2)

    def test_sharded_grad(self):
        if False:
            while True:
                i = 10
        leaves = [torch.zeros(5, 5, requires_grad=True) for _ in range(10)]
        intermediates = [l * i + l * l for (i, l) in enumerate(leaves)]
        loss = sum((v * i for (i, v) in enumerate(intermediates))).sum()

        def group(l, group_size):
            if False:
                for i in range(10):
                    print('nop')
            return (l[i:i + group_size] for i in range(0, len(l), group_size))
        shard_size = 2
        d_intermediates = [d_i for intermediates_batch in group(intermediates, shard_size) for d_i in torch.autograd.grad(loss, intermediates_batch)]
        torch.autograd.backward(intermediates, d_intermediates)
        for (i, l) in enumerate(leaves):
            self.assertEqual(l.grad, i * i * (1 + l))

    def test_backward_badcalls(self):
        if False:
            i = 10
            return i + 15
        x = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, 'does not require grad'):
            x.backward()

    def test_grad_badcalls(self):
        if False:
            while True:
                i = 10
        x = torch.ones(1)
        y = x ** 2
        with self.assertRaisesRegex(RuntimeError, 'does not require grad'):
            torch.autograd.grad(x, y)
        with self.assertRaisesRegex(RuntimeError, 'does not require grad'):
            torch.autograd.grad(y, x)
        x = torch.ones(1, requires_grad=True)
        y = x ** 2
        torch.autograd.grad(y, x)

    def test_grad_empty_inputs(self):
        if False:
            print('Hello World!')
        x = torch.tensor([1.0], requires_grad=True)
        with self.assertRaisesRegex(ValueError, 'grad requires non-empty inputs.'):
            torch.autograd.grad(2 * x, [], grad_outputs=torch.tensor([1.0]))

    def test_grad_fn_badcalls(self):
        if False:
            return 10
        error_regex = 'expected .* arguments, got .* instead'
        x = torch.ones(1, requires_grad=True)
        y = x ** 2
        with self.assertRaisesRegex(TypeError, error_regex):
            y.grad_fn(x.detach(), x.detach())
        with self.assertRaisesRegex(TypeError, error_regex):
            y.grad_fn()
        y.grad_fn(x.detach())

    def test_grad_unreachable(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        z = x * 2
        w = y * 2
        (grad_x, grad_y) = torch.autograd.grad(x * 2, [x, y], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_y)
        z = torch.ones(1, requires_grad=True)
        (grad_x, grad_z) = torch.autograd.grad(x * 2, [x, z], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_z)
        with self.assertRaisesRegex(RuntimeError, 'Set allow_unused=True'):
            (grad_x, grad_y) = torch.autograd.grad(x * 2, [x, y], allow_unused=False)

    def test_grad_unreachable_discovery(self):
        if False:
            while True:
                i = 10

        class MyFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                return x

            @staticmethod
            def backward(ctx, x):
                if False:
                    return 10
                self.fail('This node should not be executed!')
        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        (gY,) = torch.autograd.grad(x, (y,), allow_unused=True)
        self.assertIsNone(gY)
        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        z = torch.randn(1, requires_grad=True)
        (gY, gZ) = torch.autograd.grad(x + z, (y, z), allow_unused=True)
        self.assertIsNone(gY)
        self.assertIsNotNone(gZ)
        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        torch.autograd.backward(x, inputs=(y,))
        self.assertIsNone(y.grad)

    def test_grad_batched_grad(self):
        if False:
            print('Hello World!')
        x = torch.randn(2, 2, requires_grad=True)
        out = x.clone()
        batched_grad = torch.arange(3).expand(2, 2, 3).transpose(0, 2)
        (grad,) = torch.autograd.grad(out, (x,), (batched_grad,), is_grads_batched=True)
        self.assertEqual(grad, torch.arange(3).expand(2, 2, 3).transpose(0, 2).to(dtype=grad.dtype))
        grad_out = torch.ones(2, 2)
        with self.assertRaisesRegex(RuntimeError, 'If `is_grads_batched=True`, we interpret the first'):
            torch.autograd.grad(outputs=out, grad_outputs=(grad_out,), inputs=(x,), is_grads_batched=True)
        out = x.sum()
        batched_grad = torch.arange(3)
        (grad,) = torch.autograd.grad(out, (x,), (batched_grad,), is_grads_batched=True)
        self.assertEqual(grad, torch.arange(3).expand(2, 2, 3).transpose(0, 2).to(dtype=grad.dtype))
        grad_out = torch.ones(2).unsqueeze(1)
        with self.assertRaisesRegex(RuntimeError, 'If `is_grads_batched=True`, we interpret the first'):
            torch.autograd.grad(outputs=out, grad_outputs=(grad_out,), inputs=(x,), is_grads_batched=True)

    def test_hooks(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
        y.requires_grad_(True)
        counter = [0]

        def bw_hook(inc, grad):
            if False:
                i = 10
                return i + 15
            self.assertIsInstance(grad, torch.Tensor)
            counter[0] += inc
        z = x ** 2 + x * 2 + x * y + y
        x.register_hook(lambda *args: bw_hook(0, *args))
        test = z.register_hook(lambda *args: bw_hook(1, *args))
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 1)
        test2 = z.register_hook(lambda *args: bw_hook(2, *args))
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 4)
        test2.remove()
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 5)

        def bw_hook_modify(grad):
            if False:
                return 10
            return grad.mul(2)
        test.remove()
        z.register_hook(bw_hook_modify)
        with torch.no_grad():
            y.grad.zero_()
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(y.grad, (x + 1) * 2)
        y.register_hook(bw_hook_modify)
        with torch.no_grad():
            y.grad.zero_()
        z.backward(torch.ones(5, 5))
        self.assertEqual(y.grad, (x + 1) * 4)

    def _get_mul2(self, use_custom_function):
        if False:
            for i in range(10):
                print('nop')
        if use_custom_function:

            class Mul2(Function):

                @staticmethod
                def forward(ctx, x):
                    if False:
                        print('Hello World!')
                    return x * 2

                @staticmethod
                def backward(ctx, gO):
                    if False:
                        i = 10
                        return i + 15
                    return gO * 2
            return Mul2.apply
        else:
            return lambda x: x * 2

    def test_grad_fn_prehooks(self):
        if False:
            for i in range(10):
                print('nop')
        for use_custom_function in (True, False):
            mul2 = self._get_mul2(use_custom_function)
            a = torch.tensor([1.0], requires_grad=True)
            b = mul2(a)
            post_counter = [0]
            pre_counter = [0]

            def posthook(grad_input, grad_output):
                if False:
                    while True:
                        i = 10
                self.assertEqual(pre_counter[0], 3)
                self.assertTrue(torch.allclose(grad_output[0], torch.ones(1) * 8))
                self.assertTrue(torch.allclose(grad_input[0], torch.ones(1) * 16))
                post_counter[0] += 1
                return grad_input

            def prehook(grad_output):
                if False:
                    print('Hello World!')
                pre_counter[0] += 1
                return (grad_output[0] * 2,)
            b.grad_fn.register_hook(posthook)
            b.grad_fn.register_hook(posthook)
            b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(lambda x: None)
            b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(lambda x: x)
            b.grad_fn.register_prehook(lambda x: None)
            b.sum().backward()
            self.assertEqual(post_counter[0], 2)
            self.assertEqual(pre_counter[0], 3)
            a = torch.rand(3, 3, requires_grad=True)
            b = mul2(a)

            def prehook(grad_output):
                if False:
                    return 10
                pre_counter[0] += 1
                return None
            b.grad_fn.register_prehook(prehook)
            b.sum().backward()
            self.assertEqual(pre_counter[0], 4)
            self.assertTrue(torch.allclose(a.grad, torch.ones(3, 3) * 2))

    def test_grad_fn_prehooks_multiple_outputs(self):
        if False:
            while True:
                i = 10
        b = torch.rand(3, 3, requires_grad=True)
        (var, mean) = torch.var_mean(b, dim=0)
        (var + mean).sum().backward()
        a = b.detach().requires_grad_()
        counter = [0]

        def prehook(grad_output):
            if False:
                for i in range(10):
                    print('nop')
            (gvar, gmean) = grad_output
            counter[0] += 1
            return (gvar * 2, gmean * 2)
        (var, mean) = torch.var_mean(a, dim=0)
        mean.grad_fn.register_prehook(prehook)
        (var + mean).sum().backward()
        self.assertEqual(counter[0], 1)
        self.assertTrue(torch.allclose(a.grad, b.grad * 2))

        class DoubleMul2(Function):

            @staticmethod
            def forward(ctx, x, a, y):
                if False:
                    while True:
                        i = 10
                ctx.a = a
                return (a * x * 2, a, a * y * 2)

            @staticmethod
            def backward(ctx, g1, _a, g2):
                if False:
                    return 10
                return (ctx.a * g1 * 2, None, ctx.a * g2 * 2)
        counter = [0]

        def prehook(grad_output):
            if False:
                for i in range(10):
                    print('nop')
            (g1, ga, g2) = grad_output
            self.assertIsNone(ga)
            counter[0] += 1
            return (g1 * 2, None, g2 * 2)
        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        k = 3
        (c, _, d) = DoubleMul2.apply(a, k, b)
        c.grad_fn.register_prehook(prehook)
        (c + d).sum().backward()
        self.assertEqual(counter[0], 1)
        self.assertTrue(torch.allclose(a.grad, torch.ones(1) * 4 * k))
        self.assertTrue(torch.allclose(b.grad, torch.ones(1) * 4 * k))

    def test_grad_fn_prehooks_remove_hooks(self):
        if False:
            print('Hello World!')
        for use_custom_function in (True, False):
            mul2 = self._get_mul2(use_custom_function)
            a = torch.rand(3, 3, requires_grad=True)
            b = mul2(a)
            counter = [0]

            def prehook(grad_output):
                if False:
                    return 10
                counter[0] += 1
                return None
            handle = b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(prehook)
            handle.remove()
            b.sum().backward()
            self.assertTrue(torch.allclose(a.grad, torch.ones(3, 3) * 2))
            self.assertEqual(counter[0], 1)
            a = torch.rand(3, 3, requires_grad=True)
            b = mul2(a)
            counter = [0]

            def prehook1(grad_output):
                if False:
                    print('Hello World!')
                handle2.remove()
                handle3.remove()
                return None

            def prehook2(grad_output):
                if False:
                    return 10
                counter[0] += 1
                return None
            b.grad_fn.register_prehook(prehook1)
            handle2 = b.grad_fn.register_prehook(prehook2)
            handle3 = b.grad_fn.register_prehook(prehook2)
            handle3.remove()
            b.sum().backward()
            self.assertTrue(torch.allclose(a.grad, torch.ones(3, 3) * 2))
            self.assertEqual(counter[0], 1)

    def test_hooks_cpp(self):
        if False:
            while True:
                i = 10
        bn = torch.nn.BatchNorm1d(5, affine=False)
        bn.double()
        bn.eval()
        counter = [0]

        def bw_hook(grad):
            if False:
                for i in range(10):
                    print('nop')
            counter[0] += 1
            return grad * 2
        x = torch.ones(5, 5, dtype=torch.double, requires_grad=True)
        z = bn(x)
        z.register_hook(bw_hook)
        z.sum().backward()
        self.assertEqual(counter[0], 1, msg='bw_hook not called')
        self.assertEqual(x.grad, torch.ones(5, 5, dtype=torch.double) * 2, atol=1e-05, rtol=0)

    def test_hook_none(self):
        if False:
            while True:
                i = 10

        class NoneGradientFunction(Function):

            @staticmethod
            def forward(ctx, x, y):
                if False:
                    while True:
                        i = 10
                assert ctx.needs_input_grad[0]
                assert not ctx.needs_input_grad[1]
                return (x, y)

            @staticmethod
            def backward(ctx, grad_x, grad_y):
                if False:
                    i = 10
                    return i + 15
                return (grad_x, None)
        was_called = [False]

        def hook(grad):
            if False:
                while True:
                    i = 10
            self.assertIsNotNone(grad)
            was_called[0] = True
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5)
        (rx, ry) = NoneGradientFunction.apply(x, y)
        rx.register_hook(hook)
        ry.register_hook(hook)
        sum(rx, ry).sum().backward()
        self.assertTrue(was_called[0])

    def test_retain_grad(self):
        if False:
            while True:
                i = 10
        input = torch.rand(1, 3, requires_grad=True)
        h1 = input * 3
        out = (h1 * h1).sum()
        h1.retain_grad()
        h1.retain_grad()
        out.backward(retain_graph=True)
        self.assertEqual(h1 * 2, h1.grad)
        out.backward(retain_graph=True)
        self.assertEqual(h1 * 4, h1.grad)
        with torch.no_grad():
            input.grad.zero_()
        input.retain_grad()
        input.retain_grad()
        out.backward()
        self.assertEqual(input * 18, input.grad)

    def test_retain_grad_inplace(self):
        if False:
            return 10
        a = torch.tensor([1.0], requires_grad=True).clone()
        a.retain_grad()
        a.mul_(2)
        a.sum().backward()
        self.assertEqual(a.grad, torch.tensor([1.0]))
        a = torch.tensor([1.0], requires_grad=True).clone()
        a.retain_grad()
        a.mul_(2)
        a.mul_(2)
        a.sum().backward()
        self.assertEqual(a.grad, torch.tensor([1.0]))

    def test_retains_grad_inplace_multiple_outputs(self):
        if False:
            i = 10
            return i + 15

        class DoubleMul(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    print('Hello World!')
                return (x * 2, x * 3)

            @staticmethod
            def backward(ctx, g1, g2):
                if False:
                    return 10
                return g1 * 2 + g2 * 3
        var_mean = partial(torch.var_mean, dim=0)
        for fn in (DoubleMul.apply, var_mean):
            b = torch.rand(3, 3, requires_grad=True)
            (var, mean) = fn(b)
            var.retain_grad()
            mean.retain_grad()
            var.mul_(2)
            (var + mean).sum().backward()
            gvar = var.grad
            gmean = mean.grad
            a = b.detach().requires_grad_(True)
            (var, mean) = fn(a)
            var.mul_(2)
            out = (var + mean).sum()
            (gvar_expected, gmean_expected) = torch.autograd.grad(out, inputs=(var, mean))
            self.assertTrue(torch.allclose(gvar, gvar_expected))
            self.assertTrue(torch.allclose(gmean, gmean_expected))

    def test_retain_grad_inplace_over_view(self):
        if False:
            return 10
        base = torch.tensor([1.0], requires_grad=True).clone()
        view = base[:]
        view2 = base[:]
        view.retain_grad()
        view2.retain_grad()
        view.mul_(2)
        (view + view2).sum().backward()
        self.assertEqual(view.grad, view2.grad)
        self.assertEqual(view.grad, torch.tensor([1.0]))

    def test_tensor_hooks_inplace(self):
        if False:
            print('Hello World!')
        count1 = [0]
        count2 = [0]

        def fn1(grad):
            if False:
                return 10
            count1[0] += 1
            self.assertEqual(grad, torch.tensor([4.0]))
            return grad * 2

        def fn2(grad):
            if False:
                print('Hello World!')
            count2[0] += 1
            self.assertEqual(grad, torch.tensor([1.0]))
            return grad * 2
        a = torch.tensor([1.0], requires_grad=True)
        b = a.clone()
        b.register_hook(fn1)
        b.mul_(2)
        b.register_hook(fn2)
        b.sum().backward()
        self.assertEqual(count1[0], 1)
        self.assertEqual(count2[0], 1)
        self.assertEqual(a.grad, torch.tensor([8.0]))
        count3 = [0]

        def fn3(grad):
            if False:
                i = 10
                return i + 15
            count3[0] += 1
            self.assertEqual(grad, torch.tensor([4.0]))
            return grad * 2
        a = torch.tensor([1.0], requires_grad=True)
        b = a.clone()
        b.register_hook(fn3)
        b.mul_(2)
        b.mul_(2)
        b.sum().backward()
        self.assertEqual(count1[0], 1)
        self.assertEqual(a.grad, torch.tensor([8.0]))

    def test_tensor_hooks_inplace_multiple_outputs(self):
        if False:
            print('Hello World!')

        class DoubleMul(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                return (x * 2, x * 3)

            @staticmethod
            def backward(ctx, g1, g2):
                if False:
                    for i in range(10):
                        print('nop')
                return g1 * 2 + g2 * 3
        var_mean = partial(torch.var_mean, dim=0)
        for fn in (DoubleMul.apply, var_mean):
            counts = [0, 0, 0]

            def fn0(grad):
                if False:
                    i = 10
                    return i + 15
                counts[0] += 1
                self.assertEqual(grad, torch.ones_like(out1) * 2)

            def fn1(grad):
                if False:
                    while True:
                        i = 10
                counts[1] += 1
                self.assertEqual(grad, torch.ones_like(out1) * 3)

            def fn2(grad):
                if False:
                    for i in range(10):
                        print('nop')
                counts[2] += 1
                self.assertEqual(grad, torch.ones_like(out1))
            b = torch.rand(3, 3, requires_grad=True)
            (out1, out2) = fn(b)
            out1.register_hook(fn0)
            out2.register_hook(fn1)
            out1.mul_(2)
            out1.register_hook(fn2)
            (out1 + out2 * 3).sum().backward()
            self.assertEqual(counts, [1, 1, 1])

    def test_tensor_hooks_inplace_over_view(self):
        if False:
            return 10
        count = [0]

        def fn0(grad):
            if False:
                print('Hello World!')
            self.fail()

        def fn1(grad):
            if False:
                while True:
                    i = 10
            self.fail()

        def fn2(grad):
            if False:
                i = 10
                return i + 15
            count[0] += 1
            self.assertEqual(grad, torch.tensor([1.0]))
        base = torch.tensor([1.0], requires_grad=True).clone()
        view = base[:]
        view2 = base[:]
        view.register_hook(fn0)
        view2.register_hook(fn1)
        view.mul_(2)
        view2.grad_fn
        view2.register_hook(fn2)
        (view + view2).sum().backward()
        self.assertEqual(count[0], 1)

    def test_retain_grad_cycle(self):
        if False:
            return 10
        x = torch.ones(5, 5, requires_grad=True)

        def run_test():
            if False:
                print('Hello World!')
            y = x * 2
            y.retain_grad()
            return (y / 2, torch._C._WeakTensorRef(y))
        (z, ref) = run_test()
        self.assertTrue(ref.expired())
        z.sum().backward()

    def test_backward(self):
        if False:
            print('Hello World!')
        v = torch.randn(5, 5, requires_grad=True)
        x = torch.randn(5, 5, requires_grad=True)
        y = (torch.rand(5, 5) + 0.1).requires_grad_(True)
        z = torch.randn(5, 5, requires_grad=True)
        grad_output = torch.randn(5, 5)
        v.backward(grad_output)
        self.assertEqual(v.grad, grad_output)
        a = x + y * z + 4 * z ** 2 * x / y
        a.backward(grad_output)
        x_grad = 4 * z.pow(2) / y + 1
        y_grad = z - 4 * x * z.pow(2) / y.pow(2)
        z_grad = 8 * x * z / y + y
        self.assertEqual(x.grad, x_grad * grad_output)
        self.assertEqual(y.grad, y_grad * grad_output)
        self.assertEqual(z.grad, z_grad * grad_output)

    def test_to_sparse_backward(self):
        if False:
            for i in range(10):
                print('nop')
        to_attr_names = ('to_dense', 'to_sparse', 'to_sparse_csr', 'to_sparse_csc', 'to_sparse_bsr', 'to_sparse_bsc')
        to_params = ((), (), (), (), (2,), (2,))
        to_attr_names_params = dict(zip(to_attr_names, to_params))

        def check_inversion_possible(t, layout1, layout1_params, layout2, layout2_params):
            if False:
                i = 10
                return i + 15
            l = (layout1, layout2)
            p = (layout1_params, layout2_params)
            for (l1, l2, p1, p2) in ((*l, *p), (*l[::-1], *p[::-1])):
                try:
                    to_l1 = getattr(t, l1)(*p1)
                    to_l2 = getattr(to_l1, l2)(*p2)
                except RuntimeError:
                    return False
            return True
        self_strided = torch.rand(4, 4, dtype=torch.double) + 1
        grad_strided = torch.rand(4, 4, dtype=torch.double) + 1
        for from_to_attr in to_attr_names:
            from_params = to_attr_names_params[from_to_attr]
            self_from = getattr(self_strided, from_to_attr)(*from_params).requires_grad_(True)
            for to_to_attr in to_attr_names[1:]:
                to_params = to_attr_names_params[to_to_attr]
                if check_inversion_possible(self_strided, from_to_attr, from_params, to_to_attr, to_params):
                    self_to = getattr(self_from, to_to_attr)(*to_params)
                    grad_to = getattr(grad_strided, to_to_attr)(*to_params)
                    grad_res = torch.autograd.grad(self_to, self_from, grad_to)[0]
                    self.assertEqual(grad_res.layout, self_from.layout)
                    self.assertEqual(grad_res.to_dense(), grad_strided)

    def test_sparse_mm_backward(self):
        if False:
            print('Hello World!')
        size = (3, 3)
        mm_test_cases = product(*([False, True],) * 4)
        for (a_req_grad, a_is_sparse, b_req_grad, b_is_sparse) in mm_test_cases:
            if not ((a_is_sparse or b_is_sparse) and (a_req_grad or b_req_grad)):
                continue
            a = torch.randn(size)
            if a_is_sparse:
                a = a.to_sparse().detach()
            b = torch.randn(size)
            if b_is_sparse:
                b = b.to_sparse().detach()
            a = a.requires_grad_(a_req_grad)
            b = b.requires_grad_(b_req_grad)
            r = a.mm(b)
            s = r.sum().backward()
            a_grad = None if a.grad is None else a.grad.clone().detach()
            b_grad = None if b.grad is None else b.grad.clone().detach()
            a = (a.to_dense() if a.is_sparse else a).clone().detach().requires_grad_(a_req_grad)
            b = (b.to_dense() if b.is_sparse else b).clone().detach().requires_grad_(b_req_grad)
            r = a.mm(b)
            r.sum().backward()
            self.assertEqual(a_grad, a.grad)
            self.assertEqual(b_grad, b.grad)

    def test_multi_backward(self):
        if False:
            print('Hello World!')
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        q = torch.randn(5, 5, requires_grad=True)
        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)
        q2 = q * 2
        z = x + y + q2
        c = a * b + q2
        grad_z = torch.randn(5, 5)
        grad_c = torch.randn(5, 5)
        torch.autograd.backward([z, c], [grad_z, grad_c])
        self.assertEqual(x.grad, grad_z)
        self.assertEqual(y.grad, grad_z)
        self.assertEqual(a.grad, grad_c * b)
        self.assertEqual(b.grad, grad_c * a)
        self.assertEqual(q.grad, (grad_c + grad_z) * 2)

    def test_multi_backward_no_grad(self):
        if False:
            while True:
                i = 10
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=False)
        z = x + y
        q = y * 2

        def call_backwards():
            if False:
                print('Hello World!')
            torch.autograd.backward([z, q], [torch.ones(5, 5), torch.ones(5, 5)])
        self.assertRaises(RuntimeError, call_backwards)

    def test_backward_with_inputs(self):
        if False:
            print('Hello World!')
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

        def fn():
            if False:
                while True:
                    i = 10
            return x ** 2 + y * x + y ** 2
        gradient = torch.ones(2, 2)
        x_grad_expected = 2 * x + y
        y_grad_expected = x + 2 * y

        @torch.no_grad()
        def reset_grad():
            if False:
                i = 10
                return i + 15
            x.grad.zero_()
            y.grad.zero_()
        torch.autograd.backward(fn(), gradient, inputs=[x, y])
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(y.grad, y_grad_expected)
        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=[x])
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(y.grad, torch.zeros(2, 2), exact_dtype=False)
        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=[y])
        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, torch.zeros(2, 2), exact_dtype=False)
        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=y)
        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, torch.zeros(2, 2), exact_dtype=False)
        reset_grad()
        self.assertRaisesRegex(RuntimeError, 'cannot be empty', lambda : torch.autograd.backward(fn(), gradient, inputs=[]))

    def test_backward_with_nonleaf_inputs(self):
        if False:
            return 10
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        x_nonleaf = x * 1
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        out = x_nonleaf ** 2 + y * x_nonleaf + y ** 2
        out.backward(torch.ones(2, 2, dtype=torch.double), create_graph=True, inputs=[x, y, x_nonleaf])
        x_grad_expected = 2 * x + y
        y_grad_expected = x + 2 * y
        x_non_leaf_expected = 2 * x_nonleaf + y
        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(x_nonleaf.grad, x_non_leaf_expected)
        out.backward(torch.ones(2, 2, dtype=torch.double), create_graph=True, inputs=[z])
        self.assertIsNone(z.grad)

    def test_dependent_backward(self):
        if False:
            return 10
        x = torch.randn(10, requires_grad=True)
        y = x ** 2
        z = y ** 3
        go_y = torch.randn(10)
        go_z = torch.randn(10)
        torch.autograd.backward([y, z], [go_y, go_z])
        xd = x
        self.assertEqual(x.grad, 2 * xd * go_y + 6 * xd.pow(5) * go_z)

    def test_save_output_nr(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(10, requires_grad=True)

        class MultiOutputFn(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                return (x[:5], x[5:])

            @staticmethod
            def backward(ctx, *grad):
                if False:
                    while True:
                        i = 10
                return torch.cat(grad)
        (a, b) = MultiOutputFn.apply(x)
        self.assertEqual(b.output_nr, 1)

        class TestFn(Function):

            @staticmethod
            def forward(ctx, b):
                if False:
                    for i in range(10):
                        print('nop')
                ctx.save_for_backward(b)
                return b * 2

            @staticmethod
            def backward(ctx, grad_b):
                if False:
                    return 10
                (b,) = ctx.saved_tensors
                self.assertEqual(b.output_nr, 1)
        TestFn.apply(b).sum().backward()

    def test_first_grad_fn_access_in_no_grad_mode(self):
        if False:
            i = 10
            return i + 15
        a = torch.tensor([1 + 1j], requires_grad=True).clone()
        v = a.real
        a.add_(1)
        with torch.autograd.grad_mode.no_grad():
            v.grad_fn

    def test_free_deep_graph(self):
        if False:
            i = 10
            return i + 15

        def scope():
            if False:
                i = 10
                return i + 15
            depth = 150000
            x = torch.randn(1, requires_grad=True)
            y = x.clone()
            for _ in range(depth):
                y = y + y * 1e-06
        scope()

    def test_free_deep_graph_complicated(self):
        if False:
            for i in range(10):
                print('nop')

        def scope():
            if False:
                print('Hello World!')
            depth = 100000
            randchoice = torch.randint(2, [depth, 2])
            x = torch.randn(1, requires_grad=True)
            y = x.clone()
            prev_values = [None, None]
            for _ in range(depth):
                prev_tensors = [tensor for tensor in prev_values[:-1] if tensor is not None]
                prev_values.append(y)
                prev_values.pop(0)
                y += y * 1e-06
                nprev = len(prev_tensors)
                if nprev == 2:
                    y += randchoice[depth].mul(torch.cat(prev_tensors)).sum()
        scope()

    def test_free_deep_graph_pyfunction(self):
        if False:
            while True:
                i = 10

        class MyOp(Function):

            @staticmethod
            def forward(ctx, tensor1, tensor2):
                if False:
                    print('Hello World!')
                return tensor1 + tensor2

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    for i in range(10):
                        print('nop')
                return (grad_output, grad_output)

        def scope():
            if False:
                return 10
            depth = 150000
            x = torch.randn(1, requires_grad=True)
            y = x.clone()
            for _ in range(depth):
                y = MyOp.apply(y, y)
        scope()

    def test_no_unnecessary_save(self):
        if False:
            return 10
        mu = torch.ones(1, requires_grad=True)
        x = torch.empty(1)
        loss = 0
        for i in range(3):
            x.detach_()
            x.copy_(mu + i)
            ft = torch.tensor([float(i)])
            multiplied = x * ft
            s = multiplied.sum()
            loss += s
        loss.backward()

    def test_no_grad(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
        with torch.no_grad():
            w = x + y

        def adder(x, y):
            if False:
                i = 10
                return i + 15
            return x + y
        adders = [torch.no_grad()(adder), torch.no_grad(adder)]
        for adder in adders:
            z = adder(x, y)
            self.assertFalse(w.requires_grad)
            self.assertRaises(RuntimeError, lambda : w.backward(torch.ones(5, 5)))
            self.assertIsNone(w.grad_fn)
            self.assertFalse(z.requires_grad)
            self.assertRaises(RuntimeError, lambda : z.backward(torch.ones(5, 5)))
            self.assertIsNone(z.grad_fn)
        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            w = adder(x, y)
            self.assertFalse(torch.is_grad_enabled())

    def test_enable_grad_decorator_no_paren(self):
        if False:
            while True:
                i = 10
        x = torch.ones(1, requires_grad=True)

        @torch.enable_grad
        def doubler(x):
            if False:
                while True:
                    i = 10
            return x * 2
        with torch.no_grad():
            z = doubler(x)
        self.assertTrue(z.requires_grad)

    def test_set_grad_generator_functions(self):
        if False:
            print('Hello World!')

        @torch.no_grad()
        def gen_no_grad():
            if False:
                i = 10
                return i + 15
            for i in range(10):
                self.assertEqual(torch.is_grad_enabled(), False)
                yield i
        with torch.enable_grad():
            for _ in gen_no_grad():
                self.assertEqual(torch.is_grad_enabled(), True)

        @torch.enable_grad()
        def gen_enable_grad():
            if False:
                print('Hello World!')
            for i in range(10):
                self.assertEqual(torch.is_grad_enabled(), True)
                yield i
        with torch.no_grad():
            for _ in gen_enable_grad():
                self.assertEqual(torch.is_grad_enabled(), False)

    def test_set_grad_generator_functions_recursive(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.enable_grad()
        def enable_grad_decorator_recursive(depth):
            if False:
                print('Hello World!')
            self.assertTrue(torch.is_grad_enabled())
            if depth > 0:
                no_grad_decorator_recursive(depth - 1)
                self.assertTrue(torch.is_grad_enabled())

        @torch.no_grad()
        def no_grad_decorator_recursive(depth):
            if False:
                i = 10
                return i + 15
            self.assertFalse(torch.is_grad_enabled())
            if depth > 0:
                enable_grad_decorator_recursive(depth - 1)
                self.assertFalse(torch.is_grad_enabled())

        def enable_grad_context_manager_recursive(depth):
            if False:
                return 10
            with torch.enable_grad():
                self.assertTrue(torch.is_grad_enabled())
                if depth > 0:
                    no_grad_context_manager_recursive(depth - 1)
                    self.assertTrue(torch.is_grad_enabled())

        def no_grad_context_manager_recursive(depth):
            if False:
                for i in range(10):
                    print('nop')
            with torch.no_grad():
                self.assertFalse(torch.is_grad_enabled())
                if depth > 0:
                    enable_grad_context_manager_recursive(depth - 1)
                    self.assertFalse(torch.is_grad_enabled())
        with torch.enable_grad():
            self.assertTrue(torch.is_grad_enabled())
            enable_grad_decorator_recursive(10)
            self.assertTrue(torch.is_grad_enabled())
            enable_grad_context_manager_recursive(10)
            self.assertTrue(torch.is_grad_enabled())
        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            enable_grad_decorator_recursive(10)
            self.assertFalse(torch.is_grad_enabled())
            enable_grad_context_manager_recursive(10)
            self.assertFalse(torch.is_grad_enabled())

    def test_set_grad_coroutines(self):
        if False:
            i = 10
            return i + 15

        @torch.no_grad()
        def coro_no_grad(n=10):
            if False:
                print('Hello World!')
            self.assertFalse(torch.is_grad_enabled())
            for i in range(n):
                self.assertFalse(torch.is_grad_enabled())
                r = (yield i)
                self.assertFalse(torch.is_grad_enabled())
                self.assertEqual(i, r)
            self.assertFalse(torch.is_grad_enabled())

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            if False:
                print('Hello World!')
            self.assertTrue(torch.is_grad_enabled())
            for i in range(n):
                self.assertTrue(torch.is_grad_enabled())
                r = (yield i)
                self.assertTrue(torch.is_grad_enabled())
                self.assertEqual(i, r)
            self.assertTrue(torch.is_grad_enabled())
        with torch.enable_grad():
            self.assertTrue(torch.is_grad_enabled())
            (coro, r) = (coro_no_grad(), None)
            try:
                while True:
                    self.assertTrue(torch.is_grad_enabled())
                    r = coro.send(r)
                    self.assertTrue(torch.is_grad_enabled())
            except StopIteration:
                pass
        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            (coro, r) = (coro_enable_grad(), None)
            try:
                while True:
                    self.assertFalse(torch.is_grad_enabled())
                    r = coro.send(r)
                    self.assertFalse(torch.is_grad_enabled())
            except StopIteration:
                pass

    def test_set_grad_coroutines_benign_exceptions(self):
        if False:
            i = 10
            return i + 15

        class RecoverableException(Exception):
            pass

        @torch.no_grad()
        def coro_no_grad(n=10):
            if False:
                i = 10
                return i + 15
            has_raised = False
            for i in range(n):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield (-i if has_raised else i)
                except RecoverableException:
                    self.assertFalse(torch.is_grad_enabled())
                    has_raised = True

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            if False:
                while True:
                    i = 10
            has_raised = False
            for i in range(n):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield (-i if has_raised else i)
                except RecoverableException:
                    self.assertTrue(torch.is_grad_enabled())
                    has_raised = True
        with torch.enable_grad():
            coro = coro_no_grad()
            assert 0 == next(coro)
            try:
                while True:
                    r = coro.throw(RecoverableException)
                    self.assertLess(r, 0)
            except StopIteration:
                pass
        with torch.no_grad():
            coro = coro_enable_grad()
            assert 0 == next(coro)
            try:
                while True:
                    r = coro.throw(RecoverableException)
                    self.assertLess(r, 0)
            except StopIteration:
                pass

    def test_set_grad_coroutines_critical_exceptions(self):
        if False:
            while True:
                i = 10

        class UnrecoverableException(Exception):
            pass

        class SecondaryException(Exception):
            pass

        @torch.no_grad()
        def coro_no_grad(n=10):
            if False:
                i = 10
                return i + 15
            has_raised = False
            for i in range(n):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield (-i if has_raised else i)
                except UnrecoverableException:
                    self.assertFalse(torch.is_grad_enabled())
                    raise SecondaryException

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            if False:
                while True:
                    i = 10
            has_raised = False
            for i in range(n):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield (-i if has_raised else i)
                except UnrecoverableException:
                    self.assertTrue(torch.is_grad_enabled())
                    raise SecondaryException
        with torch.enable_grad():
            coro = coro_no_grad()
            assert 0 == next(coro)
            with self.assertRaises(SecondaryException):
                coro.throw(UnrecoverableException)
        with torch.no_grad():
            coro = coro_enable_grad()
            assert 0 == next(coro)
            with self.assertRaises(SecondaryException):
                coro.throw(UnrecoverableException)

    def test_set_grad_coroutines_exit(self):
        if False:
            while True:
                i = 10

        @torch.no_grad()
        def coro_no_grad(state):
            if False:
                while True:
                    i = 10
            for i in range(10):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield i
                except GeneratorExit:
                    self.assertFalse(torch.is_grad_enabled())
                    state.add('GeneratorExit')
                    raise

        @torch.enable_grad()
        def coro_enable_grad(state):
            if False:
                i = 10
                return i + 15
            for i in range(10):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield i
                except GeneratorExit:
                    self.assertTrue(torch.is_grad_enabled())
                    state.add('GeneratorExit')
                    raise
        state = set()
        with torch.enable_grad():
            coro = coro_no_grad(state)
            for i in range(5):
                next(coro)
            coro.close()
        self.assertTrue('GeneratorExit' in state)
        state = set()
        with torch.no_grad():
            coro = coro_enable_grad(state)
            for i in range(5):
                next(coro)
            coro.close()
        self.assertTrue('GeneratorExit' in state)

    def test_no_grad_python_function(self):
        if False:
            return 10
        'Python Functions should respect grad mode.'
        x = torch.ones(5, 5, requires_grad=True)

        class MyOp(Function):

            @staticmethod
            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x + 1

            @staticmethod
            def backward(self, dy):
                if False:
                    for i in range(10):
                        print('nop')
                return dy
        with torch.no_grad():
            y = MyOp.apply(x)
        self.assertFalse(y.requires_grad)

    def test_indexing(self):
        if False:
            print('Hello World!')
        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        def compare(x, y, idx, indexed_tensor, indexed_var):
            if False:
                while True:
                    i = 10
            indexed_var_t = indexed_var.data
            if not isinstance(indexed_tensor, torch.Tensor):
                indexed_var_t = indexed_var_t[0]
            self.assertEqual(indexed_tensor, indexed_var_t)
            indexed_var.sum().backward()
            expected_grad = torch.empty(x.size()).fill_(0)
            expected_grad[idx] = 1
            self.assertEqual(y.grad, expected_grad)

        def check_index(x, y, idx):
            if False:
                i = 10
                return i + 15
            if y.grad is not None:
                with torch.no_grad():
                    y.grad.zero_()
            indexed_tensor = x[idx]
            indexed_var = y[idx]
            compare(x, y, idx, indexed_tensor, indexed_var)
        check_index(x, y, 1)
        check_index(x, y, (1, 1))
        check_index(x, y, slice(1, None))
        check_index(x, y, slice(None, 2))
        check_index(x, y, (slice(None, 2), 2))
        check_index(x, y, (slice(1, 2), 2))
        check_index(x, y, (1, slice(2, None)))
        check_index(x, y, (slice(None, None), slice(2, None)))
        check_index(x, y, torch.LongTensor([0, 2]))
        check_index(x, y, torch.rand(4, 4).bernoulli().bool())
        check_index(x, y, (Ellipsis, slice(2, None)))
        check_index(x, y, ([0], [0]))
        check_index(x, y, ([1, 2, 3], [0]))
        check_index(x, y, ([1, 2], [2, 1]))
        check_index(x, y, ([[1, 2], [3, 0]], [[0, 1], [2, 3]]))
        check_index(x, y, [slice(None), [2, 3]])
        check_index(x, y, [[2, 3], slice(None)])
        check_index(x, y, [0])
        check_index(x, y, ([0],))
        x = torch.arange(1.0, 49).view(4, 3, 4)
        y = Variable(x, requires_grad=True)
        check_index(x, y, (slice(None), [0], [0]))
        check_index(x, y, ([0], [0], slice(None)))
        check_index(x, y, (slice(None), [0, 1, 2], [0]))
        check_index(x, y, ([0, 1, 2], [0], slice(None)))
        check_index(x, y, (slice(None), [1, 2], [2, 1]))
        check_index(x, y, ([1, 2], [2, 1], slice(None)))
        check_index(x, y, (slice(None), [[1, 2], [2, 0]], [[0, 1], [2, 3]]))
        check_index(x, y, ([[1, 2], [3, 0]], [[0, 1], [2, 2]], slice(None)))
        check_index(x, y, (slice(None), slice(None), [2, 1]))
        check_index(x, y, (slice(None), [2, 1], slice(None)))
        check_index(x, y, ([2, 1], slice(None), slice(None)))
        check_index(x, y, ([0],))
        check_index(x, y, ([0], slice(None)))
        check_index(x, y, ([0], Ellipsis))
        check_index(x, y, ([1, 2], [0, 1]))
        check_index(x, y, ([1, 2], [0, 1], Ellipsis))
        check_index(x, y, (Ellipsis, [1, 2], [0, 1]))
        z = torch.LongTensor([0, 1])
        zv = Variable(z, requires_grad=False)
        seq = [z, Ellipsis]
        seqv = [zv, Ellipsis]
        if y.grad is not None:
            with torch.no_grad():
                y.grad.zero_()
        indexed_tensor = x[seq]
        indexed_var = y[seqv]
        compare(x, y, seq, indexed_tensor, indexed_var)

    def test_indexing_duplicates(self):
        if False:
            print('Hello World!')
        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)
        idx = torch.LongTensor([1, 1, 3, 2, 1, 2])
        y[idx].sum().backward()
        expected_grad = torch.zeros(4, 4)
        for i in idx:
            expected_grad[i] += 1
        self.assertEqual(y.grad, expected_grad)
        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)
        idx = [[1, 1, 3, 2, 1, 2], [0]]
        y[idx].sum().backward()
        expected_grad = torch.zeros(4, 4)
        for i in idx[0]:
            for j in idx[1]:
                expected_grad[i][j] += 1
        self.assertEqual(y.grad, expected_grad)
        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)
        idx = [[[1, 2], [0, 0]], [[0, 1], [1, 1]]]
        y[idx].sum().backward()
        expected_grad = torch.tensor([[0.0, 2.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(y.grad, expected_grad)
        x = torch.arange(1.0, 65).view(4, 4, 4)
        y = Variable(x, requires_grad=True)
        idx = [[1, 1, 1], slice(None), slice(None)]
        y[idx].sum().backward()
        expected_grad = torch.empty(4, 4, 4).zero_()
        expected_grad[1].fill_(3)
        self.assertEqual(y.grad, expected_grad)

    def test_index_backward_does_not_save_tensor(self):
        if False:
            return 10
        a = torch.tensor([1.0, 0, 0])
        b = torch.zeros(3, requires_grad=True)
        tensor = b + 0
        tensor[a != 0] = tensor[a != 0]
        tensor.backward(torch.zeros_like(tensor))

    def test_volatile_deprecated(self):
        if False:
            print('Hello World!')
        v = torch.autograd.torch.randn(3, 3)
        with warnings.catch_warnings(record=True) as w:
            self.assertFalse(v.volatile)
        self.assertIn('volatile', str(w[0].message))

    def test_saved_variables_deprecated(self):
        if False:
            i = 10
            return i + 15

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, tensor1, tensor2):
                if False:
                    for i in range(10):
                        print('nop')
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + tensor2

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    return 10
                (var1, var2) = ctx.saved_variables
                return (grad_output, grad_output)
        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter('always')
            x = torch.randn((3, 3), requires_grad=True)
            y = torch.randn((3, 3), requires_grad=True)
            MyFunction.apply(x, y).sum().backward()
            has_deprecated = ('deprecated' in str(warn) and 'saved_variables' in str(warn) for warn in warns)
            has_deprecated = reduce(lambda x, y: x or y, has_deprecated)
            self.assertTrue(has_deprecated)

    def test_requires_grad(self):
        if False:
            return 10
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        z = torch.randn(5, 5, requires_grad=True)
        a = x + y
        self.assertFalse(a.requires_grad)
        b = a + z
        self.assertTrue(b.requires_grad)

        def error():
            if False:
                i = 10
                return i + 15
            raise RuntimeError
        a._backward_hooks = OrderedDict()
        x._backward_hooks = OrderedDict()
        y._backward_hooks = OrderedDict()
        a._backward_hooks['test'] = error
        x._backward_hooks['test'] = error
        y._backward_hooks['test'] = error
        b.backward(torch.ones(5, 5))

    def test_requires_grad_(self):
        if False:
            return 10
        x = torch.randn(5, 5)
        y = torch.randn(5, 5, requires_grad=True)
        self.assertIs(x, x.requires_grad_())
        self.assertTrue(x.requires_grad)
        self.assertIs(y, y.requires_grad_())
        self.assertTrue(y.requires_grad)
        self.assertIs(x, x.requires_grad_(True))
        self.assertTrue(x.requires_grad)
        self.assertIs(y, y.requires_grad_(True))
        self.assertTrue(y.requires_grad)
        z = x * y
        self.assertRaises(RuntimeError, lambda : z.requires_grad_(False))
        self.assertIs(z, z.requires_grad_())
        self.assertTrue(z.requires_grad)
        self.assertIs(z, z.requires_grad_(True))
        self.assertTrue(z.requires_grad)
        self.assertIs(x, x.requires_grad_(False))
        self.assertFalse(x.requires_grad)
        self.assertIs(y, y.requires_grad_(False))
        self.assertFalse(y.requires_grad)

    def test_requires_grad_inplace(self):
        if False:
            i = 10
            return i + 15
        a = torch.randn(5, 5)
        b = torch.randn(5, 5, requires_grad=True)
        a += b
        self.assertTrue(a.requires_grad)
        a = torch.randn(5, 5) + 0
        b = torch.randn(5, 5, requires_grad=True)
        a += b
        self.assertTrue(a.requires_grad)

    def test_no_requires_grad_inplace(self):
        if False:
            print('Hello World!')
        a = torch.randn(2, 3)
        a.add_(5)
        a.requires_grad = True
        a.sum().backward()
        self.assertEqual(a.grad, torch.ones(2, 3))
        a = torch.randn(2, 3)
        b = a[:]
        b.add_(5)
        a.requires_grad = True
        a.sum().backward()
        self.assertEqual(a.grad, torch.ones(2, 3))
        a = torch.randn(2, 3)
        b = a[:]
        a.requires_grad = True
        with self.assertRaises(RuntimeError):
            a.add_(5)
        with self.assertRaises(RuntimeError):
            b.add_(5)

    def test_attribute_deletion(self):
        if False:
            return 10
        x = torch.randn((5, 5), requires_grad=True)
        del x.grad
        self.assertIsNone(x.grad)
        with self.assertRaises(RuntimeError):
            del x.data
        with self.assertRaises(TypeError):
            x.data = None
        with self.assertRaises(RuntimeError):
            del x.requires_grad
        with self.assertRaises(RuntimeError):
            del x._grad_fn
        with self.assertRaises(RuntimeError):
            del x._backward_hooks

    def test_duplicate_backward_root(self):
        if False:
            return 10
        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)
        x = a * b
        grad_output = torch.randn_like(x)
        torch.autograd.backward([x, x], [grad_output, grad_output])
        self.assertEqual(a.grad, b * grad_output * 2)
        self.assertEqual(b.grad, a * grad_output * 2)

    def test_backward_no_grad(self):
        if False:
            for i in range(10):
                print('nop')
        a = torch.randn(5, 5, requires_grad=True)
        b = a + 2
        with self.assertRaises(RuntimeError):
            torch.autograd.backward([b], [None])

    def test_backward_twice_with_saved_values(self):
        if False:
            return 10
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))
        self.assertRaisesRegex(RuntimeError, 'Specify retain_graph=True', lambda : c.backward(torch.tensor([1, 1, 1], dtype=torch.double)))

    def test_backward_twice_retained_graph_with_saved_values(self):
        if False:
            while True:
                i = 10
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double), retain_graph=True)
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_twice_without_saved_values(self):
        if False:
            while True:
                i = 10
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = b + 1
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_twice_retained_graph_without_saved_values(self):
        if False:
            i = 10
            return i + 15
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double), retain_graph=True)
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_create_graph_warns(self):
        if False:
            print('Hello World!')
        with set_warn_always_context(True):
            b = torch.randn(3, requires_grad=True, dtype=torch.double)
            c = b * b
            with warnings.catch_warnings(record=True) as ws:
                c.backward(torch.ones_like(c), create_graph=True)
            b.grad = None
            self.assertTrue(any(('Using backward() with create_graph=True' in str(w.message) for w in ws)))
            with warnings.catch_warnings(record=True) as ws:
                torch.autograd.grad(c, b, torch.ones_like(c), create_graph=True)
            self.assertFalse(any(('Using backward() with create_graph=True' in str(w.message) for w in ws)))

    def test_next_functions(self):
        if False:
            while True:
                i = 10
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        a = x + y
        self.assertIsNotNone(a.grad_fn)
        next_functions = a.grad_fn.next_functions
        self.assertEqual(len(next_functions), 2)
        self.assertIsInstance(next_functions[0][0], torch._C._functions.AccumulateGrad)
        self.assertEqual(next_functions[0][1], 0)
        self.assertIsInstance(next_functions[1][0], torch._C._functions.AccumulateGrad)
        self.assertEqual(next_functions[1][1], 0)
        b = a + 5
        next_functions = b.grad_fn.next_functions
        self.assertEqual(len(next_functions), 2)
        self.assertIs(next_functions[0][0], a.grad_fn)
        self.assertIs(next_functions[1][0], None)

    def test_inplace(self):
        if False:
            while True:
                i = 10
        x = torch.ones(5, 5, requires_grad=True)
        y = Variable(torch.ones(5, 5) * 4, requires_grad=True)
        z = x * y
        q = z + y
        w = z * y
        z.add_(2)
        q.backward(torch.ones(5, 5), retain_graph=True)
        self.assertRaises(RuntimeError, lambda : w.backward(torch.ones(5, 5)))
        z = x * y
        q = z * y
        r = z + y
        w = z.add_(y)
        w.backward(torch.ones(5, 5), retain_graph=True)
        r.backward(torch.ones(5, 5), retain_graph=True)
        self.assertRaises(RuntimeError, lambda : q.backward(torch.ones(5, 5)))
        with torch.no_grad():
            x.grad.zero_()
        m = x / 2
        z = m + y / 8
        q = z * y
        r = z + y
        prev_version = z._version
        w = z.exp_()
        self.assertNotEqual(z._version, prev_version)
        r.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(x.grad, torch.ones(5, 5) / 2)
        w.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(x.grad, torch.empty(5, 5).fill_((1 + math.e) / 2))
        self.assertRaises(RuntimeError, lambda : q.backward(torch.ones(5, 5)))
        leaf = torch.ones(5, 5, requires_grad=True)
        x = leaf.clone()
        x.add_(10)
        self.assertEqual(x, torch.ones(5, 5) * 11)
        y = x + 2
        y.backward(torch.ones(5, 5))
        self.assertEqual(leaf.grad, torch.ones(5, 5))
        z = x * y
        x.add_(2)
        self.assertRaises(RuntimeError, lambda : z.backward(torch.ones(5, 5)))

    def test_mark_non_differentiable(self):
        if False:
            for i in range(10):
                print('nop')

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, input):
                if False:
                    for i in range(10):
                        print('nop')
                output = input > 0
                ctx.mark_non_differentiable(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    for i in range(10):
                        print('nop')
                return (grad_output * 0).to(torch.double)
        x = torch.randn(5, 5, requires_grad=True)
        mask = MyFunction.apply(x)
        self.assertFalse(mask.requires_grad)
        y = x.masked_fill(mask, 0)
        y.sum().backward()

    def test_mark_non_differentiable_mixed(self):
        if False:
            while True:
                i = 10

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, input):
                if False:
                    while True:
                        i = 10
                a = input + 1
                b = input + 2
                ctx.mark_non_differentiable(a)
                return (a, b)

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                if False:
                    while True:
                        i = 10
                self.assertTrue((grad_a == 0).all())
                self.assertTrue((grad_b == 1).all())
                return grad_b
        x = torch.randn(5, 5, requires_grad=True)
        (a, b) = MyFunction.apply(x)
        self.assertFalse(a.requires_grad)
        self.assertTrue(b.requires_grad)
        b.sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5))

    def test_mark_non_differentiable_none(self):
        if False:
            i = 10
            return i + 15

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, input):
                if False:
                    i = 10
                    return i + 15
                output = input.clone()
                ctx.mark_non_differentiable(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    for i in range(10):
                        print('nop')
                return None
        x = torch.randn(5, 5, requires_grad=True)
        r = MyFunction.apply(x * x)
        (r * x).sum().backward()

    def test_return_duplicate(self):
        if False:
            for i in range(10):
                print('nop')

        class DoubleDuplicate(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                output = x * 2
                return (output, output)

            @staticmethod
            def backward(ctx, grad1, grad2):
                if False:
                    while True:
                        i = 10
                return grad1 * 2 + grad2 * 2

        def fn(x):
            if False:
                print('Hello World!')
            (a, b) = DoubleDuplicate.apply(x)
            self.assertIs(a, b)
            return a + b
        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        gradcheck(fn, [x])
        gradgradcheck(fn, [x])

    def test_return_duplicate_inplace(self):
        if False:
            return 10

        class DoubleInplace(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    print('Hello World!')
                x.mul_(2)
                ctx.mark_dirty(x)
                return (x, x)

            @staticmethod
            def backward(ctx, grad1, grad2):
                if False:
                    return 10
                return grad1 * 2 + grad2 * 2

        def inplace_fn(x):
            if False:
                i = 10
                return i + 15
            (a, b) = DoubleInplace.apply(x.clone())
            self.assertIs(a, b)
            return a + b
        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        gradcheck(inplace_fn, [x])
        gradgradcheck(inplace_fn, [x])
        self.assertRaises(RuntimeError, lambda : InplaceFunction.apply(x))
        self.assertRaises(RuntimeError, lambda : InplaceFunction.apply(x.clone()[0]))

    def _test_setitem(self, size, index):
        if False:
            for i in range(10):
                print('nop')
        x = torch.ones(*size, requires_grad=True)
        y = x + 2
        y_version = y._version
        y[index] = 2
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad = torch.ones(*size)
        expected_grad[index] = 0
        self.assertEqual(x.grad, expected_grad)

    def _test_setitem_tensor(self, size, index):
        if False:
            print('Hello World!')
        x = torch.ones(*size, requires_grad=True)
        y = x + 2
        y_version = y._version
        value = x.new(x[index].size()).fill_(7)
        value.requires_grad = True
        y[index] = value
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad_input = torch.ones(*size)
        expected_grad_input[index] = 0
        self.assertEqual(x.grad, expected_grad_input)
        self.assertEqual(value.grad, torch.ones_like(value))
        x = torch.randn(4, requires_grad=True)
        y = torch.zeros(2, 3, 4)
        y[1] = x
        y.backward(torch.randn(2, 3, 4))
        self.assertEqual(x.size(), x.grad.size())

    def test_setitem(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_setitem((5, 5), 1)
        self._test_setitem((5,), 1)
        self._test_setitem((1,), 0)
        self._test_setitem((10,), [[0, 4, 2]])
        self._test_setitem((5, 5), [[0, 4], [2, 2]])
        self._test_setitem((5, 5, 5), [slice(None), slice(None), [1, 3]])
        self._test_setitem((5, 5, 5), [slice(None), [1, 3], slice(None)])
        self._test_setitem((5, 5, 5), [[1, 3], slice(None), slice(None)])
        self._test_setitem((5, 5, 5), [slice(None), [2, 4], [1, 3]])
        self._test_setitem((5, 5, 5), [[1, 3], [2, 4], slice(None)])
        self._test_setitem_tensor((5, 5), 3)
        self._test_setitem_tensor((5, 5), [[0, 1], [1, 0]])
        self._test_setitem_tensor((5,), 3)
        self._test_setitem_tensor((5,), Variable(torch.LongTensor([3]), requires_grad=False).sum())
        self._test_setitem_tensor((5,), [[0, 1, 2, 3]])
        self._test_setitem_tensor((5, 5, 5), [slice(None), slice(None), [1, 3]])
        self._test_setitem_tensor((5, 5, 5), [slice(None), [1, 3], slice(None)])
        self._test_setitem_tensor((5, 5, 5), [[1, 3], slice(None), slice(None)])
        self._test_setitem_tensor((5, 5, 5), [slice(None), [2, 4], [1, 3]])
        self._test_setitem_tensor((5, 5, 5), [[1, 3], [2, 4], slice(None)])
        self._test_setitem_tensor((5, 5, 5), [Variable(torch.LongTensor([1, 3]), requires_grad=False), [2, 4], slice(None)])

    def test_setitem_mask(self):
        if False:
            return 10
        mask = torch.BoolTensor(5, 5).bernoulli_()
        self._test_setitem((5, 5), Variable(mask))
        self._test_setitem((5,), Variable(mask[0]))
        self._test_setitem((1,), Variable(mask[0, 0:1]))
        self._test_setitem_tensor((5, 5), Variable(mask))
        self._test_setitem_tensor((5,), Variable(mask[0]))

    def test_select_sum(self):
        if False:
            while True:
                i = 10
        x = torch.randn(10, dtype=torch.double, requires_grad=True)

        def func(x):
            if False:
                while True:
                    i = 10
            return x.select(0, 1).sum()
        gradcheck(func, [x])
        gradgradcheck(func, [x])

    def test_diagonal_expanded_v(self):
        if False:
            print('Hello World!')
        value = torch.rand([])
        v_expanded = torch.tensor(value).expand(10)
        a = torch.rand(10, 10, dtype=torch.double, requires_grad=True)
        (result,) = torch.autograd.grad(a.diagonal(), a, v_expanded)
        self.assertEqual(result, torch.eye(10, dtype=torch.double) * value)

    def test_select_expanded_v(self):
        if False:
            while True:
                i = 10
        v_expanded = torch.rand(10).expand(10, 10)
        a = torch.rand(10, 10, 10, requires_grad=True)
        (result,) = torch.autograd.grad(a[0], a, v_expanded)
        expected = torch.zeros(10, 10, 10)
        expected[0] = v_expanded
        self.assertEqual(result, expected)

    def test_slice_expanded_v(self):
        if False:
            print('Hello World!')
        v_expanded = torch.rand(10, 1).expand(2, 10, 10)
        a = torch.rand(10, 10, 10, requires_grad=True)
        (result,) = torch.autograd.grad(a[3:5], a, v_expanded)
        expected = torch.zeros(10, 10, 10)
        expected[3:5] = v_expanded
        self.assertEqual(result, expected)

    def test_unused_output(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(10, 10, requires_grad=True)
        outputs = x.chunk(5)
        o = outputs[2]
        o = o * 4 + 2
        o.sum().backward()
        expected_grad = torch.zeros(10, 10)
        expected_grad[4:6] = 4
        self.assertEqual(x.grad, expected_grad)
        with torch.no_grad():
            x.grad.zero_()
        grad_output = torch.randn(2, 10)
        outputs = x.chunk(5)
        outputs[0].backward(grad_output)
        expected_grad = torch.zeros(10, 10)
        expected_grad[:2] = grad_output
        self.assertEqual(x.grad, expected_grad)

    def _test_sparse_gather(self, size_x, size_ind, dim):
        if False:
            while True:
                i = 10
        x = torch.randn(size_x, requires_grad=True)
        if len(size_ind) > 0 and len(size_x) > 0:
            ind = torch.randint(x.size(dim), size_ind)
        else:
            ind = torch.zeros(size_ind, dtype=torch.int64)
        out = torch.gather(x, dim, ind, sparse_grad=False)
        grad = torch.rand_like(out)
        out.backward(grad)
        grad_dense = x.grad.clone()
        x.grad = None
        out = torch.gather(x, dim, ind, sparse_grad=True)
        out.backward(grad)
        self.assertEqual(grad_dense, x.grad.to_dense())

    def test_sparse_gather_dim0(self):
        if False:
            print('Hello World!')
        self._test_sparse_gather((10, 10), (5, 10), 0)

    def test_sparse_gather_dim1(self):
        if False:
            print('Hello World!')
        self._test_sparse_gather((10, 10, 5), (10, 5, 5), 1)

    def test_sparse_gather_dim_neg(self):
        if False:
            while True:
                i = 10
        self._test_sparse_gather((10, 10, 5), (10, 10, 2), -1)

    def test_sparse_gather_ind_scalar(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_sparse_gather((10,), (), 0)

    def test_sparse_gather_x_scalar(self):
        if False:
            return 10
        self._test_sparse_gather((), (2,), 0)

    def test_sparse_gather_both_scalar(self):
        if False:
            i = 10
            return i + 15
        self._test_sparse_gather((), (), 0)

    def test_gc_in_destructor(self):
        if False:
            i = 10
            return i + 15
        "\n        Previously, if a Function destructor triggered a garbage collection,\n        the Variable's tp_dealloc handler would get called twice leading to a\n        segfault.\n        "

        class CollectOnDelete(Function):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x

            def backward(self, grad_output):
                if False:
                    while True:
                        i = 10
                return grad_output

            def __del__(self):
                if False:
                    return 10
                gc.collect()
        for _ in range(10):
            CollectOnDelete().forward(torch.randn(1, requires_grad=True)).backward()

    def test_naughty_autograd_function_attribute_access(self):
        if False:
            print('Hello World!')

        class Id(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    i = 10
                    return i + 15
                return x

            @staticmethod
            def backward(ctx, grad_x):
                if False:
                    print('Hello World!')
                return grad_x
        with self.assertWarnsRegex(DeprecationWarning, 'should not be instantiated'):
            f = Id()
        self.assertIsInstance(f, Id)
        x = torch.zeros(1, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, 'non-static forward method is deprecated'):
            f(x)
        t = Id.apply(x)
        self.assertEqual(t.grad_fn.name(), 'IdBackward')
        t = torch.ones(1, requires_grad=True)
        t._backward_hooks = {}
        with self.assertRaisesRegex(RuntimeError, "Attribute '_register_hook_dict' is invalid"):
            f._register_hook_dict(t)
        with self.assertRaisesRegex(RuntimeError, "Attribute 'register_hook' is invalid"):
            f.register_hook(lambda x, y: None)
        with self.assertRaisesRegex(RuntimeError, "Attribute 'next_functions' is invalid"):
            f.next_functions
        with self.assertRaisesRegex(RuntimeError, "Attribute 'name' is invalid"):
            f.name()
        with self.assertRaisesRegex(RuntimeError, 'underlying PyNode has already been deallocated'):
            f.metadata

    @unittest.expectedFailure
    def test_naughty_anomaly_access(self):
        if False:
            print('Hello World!')

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                return x

            @staticmethod
            def backward(ctx, g):
                if False:
                    while True:
                        i = 10
                return g
        x = torch.zeros(1, requires_grad=True)
        y = MyFunction.apply(x)
        y.backward()
        y.grad_fn.metadata
        g = y.grad_fn
        del y
        g.metadata

    def test_naughty_autograd_function_stashing_ctx(self):
        if False:
            print('Hello World!')
        saved_ctx = []

        class Id(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, grad_x):
                if False:
                    while True:
                        i = 10
                saved_ctx.append(ctx)
                return ctx.saved_tensors
        p = torch.zeros(1, requires_grad=True)
        loss = Id.apply(p)
        loss.backward(retain_graph=True)
        del loss
        self.assertRaises(RuntimeError, lambda : saved_ctx[0].saved_tensors)

    def test_custom_autograd_repeated_grad_grad(self):
        if False:
            print('Hello World!')

        def mult1(x):
            if False:
                while True:
                    i = 10
            return x.prod(dim=-1).prod(dim=-1)

        class Mult(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    print('Hello World!')
                y = mult1(x)
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    return 10
                (x, y) = ctx.saved_tensors
                return (grad_output * y)[:, None, None] / x
        mult2 = Mult.apply

        def check_gradgrad_repeated(x, y):
            if False:
                return 10
            (gy,) = torch.autograd.grad(y[0], x, create_graph=True)
            (ggy_1,) = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            (gy,) = torch.autograd.grad(y[0], x, create_graph=True)
            (ggy_2,) = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            self.assertEqual(ggy_1[0, 0, 1], ggy_2[0, 0, 1])
        x = torch.ones(2, 4, 4).requires_grad_()
        check_gradgrad_repeated(x, mult1(x))
        check_gradgrad_repeated(x, mult2(x))

    def test_custom_autograd_no_early_free(self):
        if False:
            for i in range(10):
                print('nop')

        class Double(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    print('Hello World!')
                y = x ** 2
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    while True:
                        i = 10
                (x, _) = ctx.saved_tensors
                return grad_output * 2 * x

        class Double2(Double):

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    while True:
                        i = 10
                (x, y) = ctx.saved_tensors
                return grad_output * 2 * y / x
        double = Double.apply
        double2 = Double2.apply
        x = torch.tensor(2).double().requires_grad_()
        self.assertTrue(gradcheck(double, x))
        self.assertTrue(gradgradcheck(double, x))
        self.assertTrue(gradcheck(double2, x))
        self.assertTrue(gradgradcheck(double2, x))
        y = double(x)
        torch.autograd.grad(y, x, create_graph=True)
        torch.autograd.grad(y, x)
        y = double2(x)
        torch.autograd.grad(y, x, create_graph=True)
        torch.autograd.grad(y, x)

    def test_detach(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(10, 10, requires_grad=True)
        y = x + 2
        y = y.detach()
        z = y * 4 + 2
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)
        x = torch.randn(10, 10, requires_grad=True)
        y = x * 2
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)
        z = x + y
        z.sum().backward()
        self.assertEqual(x.grad, torch.ones(10, 10))
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        a = x * 2
        (y + a).sum().backward(retain_graph=True)
        a.detach_()
        self.assertFalse(a.requires_grad)
        (y + a).sum().backward()
        self.assertEqual(x.grad, torch.ones(10, 10) * 2)
        self.assertEqual(y.grad, torch.ones(10, 10) * 2)
        view = x.narrow(0, 1, 4)
        self.assertRaisesRegex(RuntimeError, 'view', lambda : view.detach_())

    def test_detach_base(self):
        if False:
            return 10
        'detaching base does not detach view'
        x = torch.randn(10, 10, requires_grad=True)
        view = x.narrow(0, 1, 4)
        x.detach_()
        self.assertFalse(x.requires_grad)
        self.assertTrue(view.requires_grad)
        self.assertIsNotNone(view.grad_fn)
        self.assertIs(view._base, x)

    def test_detach_then_inplace_raises_in_autograd(self):
        if False:
            print('Hello World!')
        x = torch.randn([], requires_grad=True)
        orig_x = x.detach().clone()
        y = x ** 2
        z = x.detach()
        z.zero_()
        with self.assertRaisesRegex(RuntimeError, 'has been modified by an inplace'):
            y.backward()

    def _test_type_conversion_backward(self, t):
        if False:
            return 10
        fvar = Variable(t(torch.randn(5, 5).float()), requires_grad=True)
        fvar.double().sum().backward()
        self.assertEqual(fvar.grad, torch.ones_like(fvar))
        self.assertEqual(type(fvar.grad), type(fvar))
        dvar = Variable(t(torch.randn(5, 5).double()), requires_grad=True)
        dvar.float().sum().backward()
        self.assertEqual(dvar.grad, torch.ones_like(dvar))
        self.assertEqual(type(dvar.grad), type(dvar))

    def test_type_conversions(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(5, 5)
        self.assertIsInstance(x.float(), torch.FloatTensor)
        self.assertIsInstance(x.int(), torch.IntTensor)
        if torch.cuda.is_available():
            self.assertIsInstance(x.float().cuda(), torch.cuda.FloatTensor)
            self.assertIsInstance(x.int().cuda(), torch.cuda.IntTensor)
            self.assertIsInstance(x.int().cuda().cpu(), torch.IntTensor)
            if torch.cuda.device_count() >= 2:
                x2 = x.float().cuda(1)
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                x2 = x.float().cuda()
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 0)
                x2 = x2.cuda(1)
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                y = Variable(torch.randn(5).cuda(1), requires_grad=True)
                y.cpu().sum().backward()
                self.assertIs(y.grad.get_device(), 1)
                self.assertIs(y.long().get_device(), 1)
        for t in [torch.DoubleTensor, torch.FloatTensor, torch.IntTensor, torch.ByteTensor]:
            for y_var in (True, False):
                y = torch.randint(5, (5, 5), dtype=t.dtype)
                y = Variable(y) if y_var else y
                self.assertIsInstance(x.type(t), t)
                self.assertIsInstance(x.type_as(y), t)
                t_dtype = t().dtype
                self.assertIsInstance(x.type(t_dtype), t)
                self.assertIs(t_dtype, x.type(t_dtype).dtype)
                self.assertEqual(y.data_ptr(), y.type(t).data_ptr())
                if torch.cuda.is_available():
                    for x_cuda in (True, False):
                        for y_cuda in (True, False):
                            x_c = x.cuda() if x_cuda else x
                            y_c = y.cuda() if y_cuda else y
                            (_, y_type) = y_c.type().rsplit('.', 1)
                            y_typestr = ('torch.cuda.' if y_cuda else 'torch.') + y_type
                            self.assertEqual(y_c.type(), x_c.type(y_typestr).type())
                            self.assertIs(y_c.dtype, x_c.type(y_c.dtype).dtype)
                            self.assertEqual(y_c.data_ptr(), y_c.cuda().data_ptr() if y_cuda else y_c.data_ptr())
        self._test_type_conversion_backward(lambda x: x)
        if torch.cuda.is_available():
            self._test_type_conversion_backward(lambda x: x.cuda())
            if torch.cuda.device_count() >= 2:
                self._test_type_conversion_backward(lambda x: x.cuda(0))
                self._test_type_conversion_backward(lambda x: x.cuda(1))

    def test_isolated_node(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        a = x + y
        b = torch.max(a, 1, True)[1].repeat(1, 5).double()
        o = (b + a).sum()
        o.backward()

    def test_shape(self):
        if False:
            while True:
                i = 10
        x = torch.randn(3, 4)
        self.assertEqual(2, len(x.shape))
        self.assertEqual(x.shape[0], 3)
        self.assertEqual(x.shape[1], 4)

    def test_numpy_requires_grad(self):
        if False:
            print('Hello World!')
        x = torch.randn(2, 2, requires_grad=True)
        err_msg_outputs = "Can't call numpy\\(\\) on Tensor that requires grad. Use tensor.detach\\(\\).numpy\\(\\) instead."
        with self.assertRaisesRegex(RuntimeError, err_msg_outputs):
            x.numpy()
        with torch.no_grad():
            x.numpy()
        x = torch.randn(2, 2)
        x.numpy()
        with torch.no_grad():
            x.numpy()

    def test_return_leaf(self):
        if False:
            i = 10
            return i + 15

        class Identity(Function):

            @staticmethod
            def forward(ctx, a, b):
                if False:
                    while True:
                        i = 10
                return (a, a + b)

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                if False:
                    return 10
                return (grad_a + grad_b, grad_b)
        hook_called = [False]
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        (q, p) = Identity.apply(x, y)

        def hook(grad):
            if False:
                for i in range(10):
                    print('nop')
            hook_called[0] = True
            self.assertEqual(grad, torch.ones(5, 5))
        q.register_hook(hook)
        (q + p + x).sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5) * 3)
        self.assertEqual(y.grad, torch.ones(5, 5))
        self.assertTrue(hook_called[0])

    def test_return_leaf_inplace(self):
        if False:
            while True:
                i = 10

        class Inplace(InplaceFunction):

            @staticmethod
            def forward(ctx, a, b):
                if False:
                    return 10
                ctx.mark_dirty(a)
                return (a.add_(b), b + 2)

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                if False:
                    while True:
                        i = 10
                return (grad_a, grad_a + grad_b)
        x = torch.randn(5, 5)
        y = torch.randn(5, 5, requires_grad=True)
        (q, p) = Inplace.apply(x, y)
        self.assertIs(q, x)
        self.assertIs(q.grad_fn.__class__, Inplace._backward_cls)
        self.assertTrue(q.requires_grad)
        q.sum().backward()
        self.assertEqual(y.grad, torch.ones(5, 5))

    def test_leaf_assignment(self):
        if False:
            while True:
                i = 10
        x = torch.randn(5, 5)
        y = torch.randn(5, requires_grad=True)
        z = torch.randn(5, requires_grad=True)
        x[0] = y
        x[1] = 2 * z
        self.assertTrue(x.requires_grad)
        self.assertIsNot(x.grad_fn, None)
        x.sum().backward()
        self.assertEqual(y.grad, torch.ones(5))
        self.assertEqual(z.grad, torch.ones(5) * 2)

    def test_no_grad_assignment(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5)
        with torch.no_grad():
            x[0] = y
        self.assertTrue(x.requires_grad)
        self.assertIsNone(x.grad_fn)

    def test_no_grad_modifies_version(self):
        if False:
            return 10
        x = torch.randn(5, requires_grad=True)
        y = torch.randn(5, requires_grad=True)
        z = (x * y).sum()
        with torch.no_grad():
            x *= 2
        self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation', lambda : z.backward())

    def test_increment_version(self):
        if False:
            i = 10
            return i + 15
        a = torch.rand(5, requires_grad=True)
        v = a._version
        torch.autograd.graph.increment_version(a)
        self.assertEqual(a._version, v + 1)
        a = torch.zeros(5, dtype=torch.int)
        v = a._version
        torch.autograd.graph.increment_version(a)
        self.assertEqual(a._version, v + 1)
        with torch.inference_mode():
            a = torch.rand(5, requires_grad=True)
        msg = 'update to inference tensor outside InferenceMode'
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.autograd.graph.increment_version(a)

    def test_no_grad_input(self):
        if False:
            while True:
                i = 10

        class MyFunction(Function):

            @staticmethod
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x

            @staticmethod
            def backward(self, grad_output):
                if False:
                    return 10
                return grad_output
        x = torch.randn(5, requires_grad=True)
        with torch.no_grad():
            y = MyFunction.apply(x)
        self.assertTrue(x.requires_grad)
        self.assertIsNone(y.grad_fn)

    def test_backward_copy(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5, requires_grad=True)
        a = x + 2
        b = y + 2
        c = x + 2
        add1 = a + b
        add2 = add1 + c
        for _ in range(4):
            a = a * 2
            b = b * 2
            c = c * 2
        branch = a + b + c
        out = add2 + branch
        grad_output = torch.ones(5, 5)
        out.backward(grad_output)
        self.assertEqual(x.grad, torch.ones(5, 5) * 34)
        self.assertEqual(y.grad, torch.ones(5, 5) * 17)

    def test_save_none_for_backward(self):
        if False:
            return 10
        test_case = self

        class MyFn(Function):

            @staticmethod
            def forward(ctx, input):
                if False:
                    i = 10
                    return i + 15
                ctx.save_for_backward(None, input, None)
                return input * input

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    return 10
                (n1, input, n2) = ctx.saved_tensors
                test_case.assertIsNone(n1)
                test_case.assertIsNone(n2)
                return 2 * input * grad_output
        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn.apply(x)
        y.sum().backward()
        self.assertEqual(x.grad, 2 * x)

    def test_too_many_grads(self):
        if False:
            return 10

        class MyFn(Function):

            @staticmethod
            def forward(ctx, input):
                if False:
                    return 10
                return input

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    print('Hello World!')
                return (grad_output, None, None)
        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn.apply(x)
        y.sum().backward()
        self.assertEqual(x.grad, torch.ones_like(x))

    def test_pickle(self):
        if False:
            print('Hello World!')
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=False)

        def assert_strict_equal(var1, var2):
            if False:
                while True:
                    i = 10
            self.assertEqual(var1, var2)
            self.assertEqual(var1.requires_grad, var2.requires_grad)
        serialized = [pickle.dumps([x, y], protocol=p) for p in range(3)]
        for dump in serialized:
            (xc, yc) = pickle.loads(dump)
            assert_strict_equal(xc, x)
            assert_strict_equal(yc, y)

    def test_dep_nograd(self):
        if False:
            i = 10
            return i + 15

        class F1(Function):

            @staticmethod
            def forward(ctx, input):
                if False:
                    for i in range(10):
                        print('nop')
                out = torch.randn(input.size())
                ctx.mark_non_differentiable(out)
                return (input, out)

            @staticmethod
            def backward(ctx, grad_output, ignored):
                if False:
                    for i in range(10):
                        print('nop')
                return grad_output

        class F2(Function):

            @staticmethod
            def forward(ctx, input, ignored):
                if False:
                    print('Hello World!')
                return input

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    while True:
                        i = 10
                return (grad_output, None)
        x = torch.randn(5, requires_grad=True)
        (a, b) = F1.apply(x)
        b = b + 1
        self.assertTrue(a.requires_grad)
        self.assertFalse(b.requires_grad)
        c = F2.apply(a, b)
        c.backward(torch.ones(c.size()))
        self.assertEqual(x.grad, torch.ones(x.size()))

    def test_set_grad_enabled(self):
        if False:
            print('Hello World!')
        x = torch.tensor([1.0], requires_grad=True)
        with torch.set_grad_enabled(False):
            y = x * 2
        self.assertFalse(y.requires_grad)
        with torch.set_grad_enabled(True):
            y = x * 2
        self.assertTrue(y.requires_grad)
        with torch.set_grad_enabled(False):
            torch.set_grad_enabled(True)
            y = x * 2
        self.assertTrue(y.requires_grad)

    def test_set_grad_enabled_wraps(self):
        if False:
            while True:
                i = 10
        for decorator in [True, False]:
            with torch.enable_grad():
                self.assertTrue(torch.is_grad_enabled())
                if decorator:

                    @torch.set_grad_enabled(False)
                    def inner_func(x):
                        if False:
                            return 10
                        return x.sin()
                else:

                    def inner_func(x):
                        if False:
                            for i in range(10):
                                print('nop')
                        return x.sin()
                    obj = torch.set_grad_enabled(False)
                    self.assertTrue(not torch.is_grad_enabled())
                    inner_func = obj(inner_func)
                    self.assertTrue(torch.is_grad_enabled())
                self.assertTrue(torch.is_grad_enabled())
                x = torch.zeros(1, requires_grad=True)
                self.assertTrue(not inner_func(x).requires_grad)

    def test_simple_reentrant(self):
        if False:
            return 10
        y_data = torch.randn(2, 2)

        class Reenter(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                with torch.enable_grad():
                    ctx.x = Variable(x, requires_grad=True)
                    ctx.y = Variable(y_data, requires_grad=True)
                    ctx.output_var = ctx.x * ctx.y
                return ctx.output_var.detach()

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    i = 10
                    return i + 15
                with torch.enable_grad():
                    ctx.output_var.sum().backward()
                return ctx.x.grad * grad_output
        x = torch.randn(2, 2, requires_grad=True)
        out = Reenter.apply(x)
        out.sum().backward()
        self.assertEqual(x.grad, y_data)

    def test_reentrant_child_error(self):
        if False:
            for i in range(10):
                print('nop')
        a = torch.rand(3, 3, requires_grad=True)
        c = a * a
        b = torch.rand(3, 3, requires_grad=True)
        e = b * b
        f = TestAutograd.SimulateBackwardError.apply(e)
        reentrant_root = f.sum()

        class ReentrantFunc(Function):

            @staticmethod
            def forward(ctx, inp):
                if False:
                    return 10
                return inp.clone()

            @staticmethod
            def backward(ctx, grad):
                if False:
                    return 10
                reentrant_root.backward()
                return grad
        d = ReentrantFunc.apply(c)
        with self.assertRaisesRegex(Exception, 'Simulate error'):
            d.sum().backward()

    def test_var_mean_differentiable(self):
        if False:
            while True:
                i = 10
        dim = [2, 4]
        keepdim = False
        input1 = torch.randn(3, 4, 5, 6, 2, 3, requires_grad=True)
        input2 = deepcopy(input1)
        (var1, mean1) = torch.var_mean(input1, dim=dim, keepdim=keepdim)
        var2 = input2.var(dim=dim, keepdim=keepdim)
        mean2 = input2.mean(dim=dim, keepdim=keepdim)
        grad = torch.randn(3, 4, 6, 3, requires_grad=True)
        r1 = var1 * var1 * mean1 * mean1
        r2 = var2 * var2 * mean2 * mean2
        self.assertEqual(r1, r2, rtol=0.01, atol=0.0)
        torch.autograd.backward(r1, grad)
        torch.autograd.backward(r2, grad)
        self.assertEqual(input1.grad, input2.grad, rtol=0.01, atol=0.0)

    @skipIfNoLapack
    def test_lobpcg(self):
        if False:
            print('Hello World!')

        def func(k, A, largest=True, B=None):
            if False:
                print('Hello World!')
            X_shape = list(A.shape)
            X_shape[-1] = k
            X = torch.eye(A.size(-2), k, dtype=A.dtype, device=A.device)
            if A.dim() > 2:
                X = X.expand(X_shape)
            (D, U) = torch.lobpcg(A=A, k=k, B=B, X=X, largest=largest)
            (_, idx) = U.abs().max(-2, keepdim=True)
            sign = U.gather(-2, idx).sign()
            U = U * sign
            return (D, U)

        def run_symeig_test(k, sizes, largest=True):
            if False:
                print('Hello World!')
            A = torch.rand(*sizes).double()
            A = A @ A.mT / 10
            A.requires_grad_(True)
            gradcheck(lambda A: func(k, A, largest), A, check_batched_grad=False)
            D_grad = torch.rand(*A.shape[:-2], k) / 100
            U_grad = torch.rand(*A.shape[:-1], k) / 100
            gradgradcheck(lambda A: func(k, A, largest), A, [D_grad, U_grad], atol=0.0001, check_batched_grad=False)
            A = A.detach().requires_grad_(True)
            (D, U) = func(k, A, largest)
            (D.sum() + U.sum()).backward()
            self.assertEqual(A.grad, A.grad.mT)
        for largest in [True, False]:
            run_symeig_test(1, (6, 6), largest=largest)
            run_symeig_test(1, (2, 6, 6), largest=largest)
            run_symeig_test(1, (2, 2, 6, 6), largest=largest)
            run_symeig_test(2, (6, 6), largest=largest)
            run_symeig_test(2, (2, 6, 6), largest=largest)
            run_symeig_test(2, (2, 2, 6, 6), largest=largest)
            run_symeig_test(3, (9, 9), largest=largest)
            run_symeig_test(3, (2, 9, 9), largest=largest)
            run_symeig_test(3, (2, 2, 9, 9), largest=largest)

    def test_variable_traverse(self):
        if False:
            for i in range(10):
                print('nop')

        def get_out_and_unrefed_cycle():
            if False:
                i = 10
                return i + 15
            inp = torch.randn(10, requires_grad=True)
            tmp = inp.view(10, 1)
            out = tmp.view(10)
            my_list = []
            my_list.append(tmp)
            my_list.append(my_list)
            return out
        out = get_out_and_unrefed_cycle()
        gc.collect()
        out.backward(torch.randn(out.size()))

    def test_pow_zero_tensor_gradient(self):
        if False:
            return 10

        def run_test(input_size, exponent):
            if False:
                while True:
                    i = 10
            input = torch.zeros(*input_size, requires_grad=True)
            input.pow(exponent).sum().backward()
            self.assertEqual(input.grad.abs().sum(), 0)
        run_test((10,), torch.zeros(10))
        run_test((10, 10), torch.zeros(10, 10))
        run_test((10,), 0)

    def test_current_graph_task_id(self):
        if False:
            print('Hello World!')
        id = [-1]

        def hook(_):
            if False:
                while True:
                    i = 10
            id[0] = torch._C._current_graph_task_id()
        t = torch.tensor(1.0, requires_grad=True).clone()
        t.register_hook(hook)
        t.backward(retain_graph=True)
        base = id[0]
        t.backward(retain_graph=True)
        self.assertEqual(id[0] - base, 1)
        t.backward(retain_graph=True)
        self.assertEqual(id[0] - base, 2)
        self.assertEqual(torch._C._current_graph_task_id(), -1)

    def test_current_graph_task_execution_order(self):
        if False:
            i = 10
            return i + 15
        predicted = [None]

        def hook(_):
            if False:
                while True:
                    i = 10
            predicted[0] = torch._C._current_graph_task_execution_order()

        def names(nodes):
            if False:
                for i in range(10):
                    print('nop')
            return ', '.join([node.name().split(' ')[-1] for node in nodes]) + '\n'

        def grad_fns(*tensors):
            if False:
                for i in range(10):
                    print('nop')
            out = []
            for t in tensors:
                if t.requires_grad and t.grad_fn is None:
                    out.append(t.clone().grad_fn.next_functions[0][0])
                else:
                    out.append(t.grad_fn)
            return out
        actual = []

        def register_logging_hooks(*tensors):
            if False:
                while True:
                    i = 10

            def get_hook(i):
                if False:
                    for i in range(10):
                        print('nop')

                def hook(t_):
                    if False:
                        print('Hello World!')
                    actual.append(tensors[i])
                return hook
            for (i, t) in enumerate(tensors):
                t.register_hook(get_hook(i))
        t = torch.tensor(1.0, requires_grad=True).clone().sin().exp()
        t.register_hook(hook)
        with torch.autograd.set_multithreading_enabled(False):
            t.backward()
        self.assertExpectedInline(names(predicted[0]), 'ExpBackward0, SinBackward0, CloneBackward0, torch::autograd::AccumulateGrad\n')
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(2.0, requires_grad=True)
        c = b.sin()
        d = a.cos()
        out = c * d
        register_logging_hooks(a, b, c, d, out)
        out.register_hook(hook)
        with torch.autograd.set_multithreading_enabled(False):
            out.backward()
        self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []
        a = torch.tensor(1.0, requires_grad=True)
        b = a.sin()
        c = a.cos()
        out = b * c
        register_logging_hooks(a, b, c, out)
        out.register_hook(hook)
        with torch.autograd.set_multithreading_enabled(False):
            out.backward()
        self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        out = b.sin()
        out2 = b.cos()
        out3 = b.cos()
        register_logging_hooks(a, b, out, out2, out3)
        out3.register_hook(hook)
        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.grad((out, out3, out2), inputs=(a,))
        self.assertExpectedInline(names(predicted[0]), 'CosBackward0, CosBackward0, SinBackward0, MulBackward0, torch::autograd::AccumulateGrad\n')
        actual = []
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        out = b.sin()
        register_logging_hooks(a, b, out)
        out.register_hook(hook)
        with torch.autograd.set_multithreading_enabled(False):
            out.backward()
        self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []
        a = torch.tensor(1.0, requires_grad=True)
        b = a * 2
        out = b.sin()
        register_logging_hooks(a, b, out)
        out.register_hook(hook)
        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.grad((out,), inputs=(a, b))
        self.assertEqual(names(predicted[0]), 'SinBackward0, MulBackward0, torch::autograd::AccumulateGrad\n')
        actual = []
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)
        c = a * b
        out = c.sin()
        register_logging_hooks(a, b, c, out)
        out.register_hook(hook)
        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.grad((out,), inputs=(a,))
        self.assertEqual(names(predicted[0]), 'SinBackward0, MulBackward0, torch::autograd::AccumulateGrad\n')
        actual = []
        with self.assertRaisesRegex(RuntimeError, 'should only be called during the backward pass'):
            torch._C._current_graph_task_execution_order()
        t = torch.tensor(1.0, requires_grad=True).clone().sin().exp()
        t.register_hook(hook)
        with self.assertRaisesRegex(RuntimeError, 'expects the current backward to be executed with multithreading disabled'):
            t.backward()

    def test_view_replay_enabled(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                print('Hello World!')
            out = x.clone().view(-1)
            out.add_(1)
            return out
        x = torch.ones(2, 2, requires_grad=True)
        with torch.autograd._force_original_view_tracking(False):
            out = f(x)
            self.assertTrue('AsStridedBackward' in str(out.grad_fn))
            self.assertFalse(torch.autograd.is_view_replay_enabled())
        self.assertFalse(torch.autograd.is_view_replay_enabled())
        with torch.autograd._force_original_view_tracking(True):
            out = f(x)
            self.assertTrue('ViewBackward' in str(out.grad_fn))
            self.assertTrue(torch.autograd.is_view_replay_enabled())
        out = f(x)
        self.assertTrue('AsStridedBackward' in str(out.grad_fn))
        self.assertFalse(torch.autograd.is_view_replay_enabled())
        with torch.autograd._force_original_view_tracking(False):
            torch.autograd._force_original_view_tracking(True)
            out = f(x)
            self.assertTrue('ViewBackward' in str(out.grad_fn))
            self.assertTrue(torch.autograd.is_view_replay_enabled())
        self.assertFalse(torch.autograd.is_view_replay_enabled())
        torch.autograd._force_original_view_tracking(False)
        out = f(x)
        self.assertTrue('AsStridedBackward' in str(out.grad_fn))
        self.assertFalse(torch.autograd.is_view_replay_enabled())
        torch.autograd._force_original_view_tracking(True)
        out = f(x)
        self.assertTrue('ViewBackward' in str(out.grad_fn))
        self.assertTrue(torch.autograd.is_view_replay_enabled())

    def test_unsafe_set_version_counter(self):
        if False:
            print('Hello World!')
        x = torch.ones(2, requires_grad=True).clone()
        x.add_(1)
        x.add_(2)
        self.assertEqual(2, x._version)
        with torch.autograd._unsafe_preserve_version_counter(x):
            x.mul_(2)
            x.mul_(3)
        self.assertEqual(2, x._version)
        torch._C._autograd._unsafe_set_version_counter(x, 0)
        self.assertEqual(0, x._version)
        with self.assertRaisesRegex(RuntimeError, 'Cannot set'):
            torch._C._autograd._unsafe_set_version_counter(x, -1)

    def test_current_node(self):
        if False:
            i = 10
            return i + 15
        pr = []

        class MyMode(TorchDispatchMode):

            def __torch_dispatch__(self, func, types, args, kwargs=None):
                if False:
                    print('Hello World!')
                node = torch._C._current_autograd_node()
                node_name = node.__class__.__name__ if node else 'None'
                pr.append(f'Running {func} from within {node_name}')
                return func(*args, **kwargs or {})
        with MyMode():
            pr.append('FW')
            a = torch.rand(10, requires_grad=True)
            b = a.mul(2).div(3).sum()
            pr.append('BW')
            b.backward()
            pr.append('Done')
        self.assertExpectedInline('\n'.join(pr), 'FW\nRunning aten.rand.default from within None\nRunning aten.mul.Tensor from within None\nRunning aten.div.Tensor from within None\nRunning aten.sum.default from within None\nBW\nRunning aten.ones_like.default from within None\nRunning aten.expand.default from within SumBackward0\nRunning aten.div.Tensor from within DivBackward0\nRunning aten.mul.Tensor from within MulBackward0\nRunning aten.detach.default from within AccumulateGrad\nRunning aten.detach.default from within AccumulateGrad\nDone')

    def test_profiler(self):
        if False:
            print('Hello World!')
        x = torch.randn(10, 10)
        with profile(use_kineto=kineto_available()) as p:
            self.assertTrue(torch.autograd._profiler_enabled())
            y = x * 2 + 4
        self.assertFalse(torch.autograd._profiler_enabled())
        names = ['aten::mul', 'aten::add']
        found_indices = set()
        for evt in p.function_events:
            if evt.name in names:
                found_indices.add(names.index(evt.name))
        self.assertEqual(len(found_indices), len(names))

    def test_profiler_seq_nr(self):
        if False:
            print('Hello World!')
        with profile(use_kineto=kineto_available()) as p:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            s = z.sum(dim=None)
            s.backward()
        print(p.key_averages().table(sort_by='self_cpu_time_total', row_limit=-1))
        autograd_ops = {('aten::add', 'Add'): [], ('aten::sum', 'Sum'): []}
        accumulate_ops = []
        found_empty = False
        for e in p.function_events:
            for ((fwd_name, bwd_name), ops) in autograd_ops.items():
                if e.name == fwd_name or (bwd_name in e.name and 'Backward' in e.name):
                    ops.append(e)
            if 'AccumulateGrad' in e.name:
                accumulate_ops.append(e)
            if e.name == 'aten::empty':
                self.assertEqual(e.sequence_nr, -1)
                found_empty = True
        for (idx, ((fwd_name, bwd_name), ops)) in enumerate(autograd_ops.items()):
            self.assertEqual(len(ops), 3)
            self.assertEqual(ops[0].name, fwd_name)
            self.assertEqual(ops[1].name, f'autograd::engine::evaluate_function: {bwd_name}Backward{idx}')
            self.assertEqual(ops[2].name, f'{bwd_name}Backward{idx}')
            self.assertGreaterEqual(ops[0].sequence_nr, 0)
            self.assertEqual(ops[1].sequence_nr, ops[0].sequence_nr)
            self.assertEqual(ops[2].sequence_nr, ops[0].sequence_nr)
            self.assertEqual(ops[0].fwd_thread, 0)
            self.assertEqual(ops[1].fwd_thread, ops[0].thread)
            self.assertEqual(ops[2].fwd_thread, ops[0].thread)
        self.assertTrue(found_empty)

    def test_profiler_unboxed_only(self):
        if False:
            i = 10
            return i + 15
        x = torch.rand(3, 4)
        with torch.autograd.profiler.profile(use_kineto=kineto_available()) as prof:
            x.resize_([3, 2])

    def test_profiler_propagation(self):
        if False:
            while True:
                i = 10

        def foo(x):
            if False:
                print('Hello World!')
            with record_function('in_foo') as rf:
                return x * 2
        x = torch.rand(3, 4)
        traced_foo = torch.jit.trace(foo, x)

        def bar(x):
            if False:
                return 10
            with record_function('in_bar') as rf:
                fut = torch.jit._fork(traced_foo, x)
                y = torch.jit._wait(fut)
                with record_function('in_bar_after_wait') as rf2:
                    y = y * 2
                return y
        traced_bar = torch.jit.trace(bar, x)
        with profile(use_kineto=kineto_available()) as p:
            traced_bar(x)
        found_foo = False
        found_bar = False
        found_bar_after_wait = False
        for info in p.function_events:
            if info.name == 'in_foo':
                self.assertFalse(found_foo)
                found_foo = True
            elif info.name == 'in_bar':
                self.assertFalse(found_bar)
                found_bar = True
            elif info.name == 'in_bar_after_wait':
                self.assertFalse(found_bar_after_wait)
                found_bar_after_wait = True
        self.assertTrue(found_foo)
        self.assertTrue(found_bar)
        self.assertTrue(found_bar_after_wait)

    def test_record_function_callbacks(self):
        if False:
            while True:
                i = 10
        x = torch.randn(10, 10)
        with profile(use_kineto=kineto_available()) as p:
            with record_function('foo'):
                y = x * 2 + 4
        function_events = p.function_events
        foo_event = [event for event in function_events if 'foo' in event.name][0]
        self.assertEqual(foo_event.count, 1)

    def test_record_function_legacy(self):
        if False:
            return 10
        x = torch.randn(10, 10)
        with profile(use_kineto=kineto_available()) as p:
            handle = torch.ops.profiler._record_function_enter('bar', None)
            try:
                y = x * 2 + 4
            finally:
                torch.ops.profiler._record_function_exit(handle)
        function_events = p.function_events
        foo_event = [event for event in function_events if 'bar' in event.name][0]
        self.assertEqual(foo_event.count, 1)

    def test_profiler_aggregation_fake(self):
        if False:
            return 10
        events = EventList()
        id = [0]

        def get_id():
            if False:
                print('Hello World!')
            id[0] = id[0] + 1
            return id[0]
        threads = [[1, [(0, 1, get_id()), (1, 2, get_id())]], [0, [(0, 2, get_id()), (1, 2, get_id()), (1, 3, get_id())]]]
        for (thread, ranges) in threads:
            for range in ranges:
                assert len(range) == 3
                events.append(FunctionEvent(id=range[2], node_id=0, name='', thread=thread, start_us=range[0], end_us=range[1]))
        events._populate_cpu_children()
        res = [[], [], [], [], [4]]

        def get_children_ids(event):
            if False:
                print('Hello World!')
            return [child.id for child in event.cpu_children]
        assert [get_children_ids(event) for event in events] == res

    def test_profiler_aggregation_table(self):
        if False:
            return 10
        '\n        Test if the profiling result is aggregated for `str(prof)`\n\n        See: https://github.com/pytorch/pytorch/issues/37500\n        '
        x = torch.randn(1024)
        with torch.autograd.profiler.profile(use_kineto=kineto_available()) as prof:
            torch.einsum('i->', x)
        prof_str = str(prof)
        prof_table = prof.table()
        self.assertEqual(prof_table, prof_str)

    def test_profiler_function_event_avg(self):
        if False:
            print('Hello World!')
        avg = FunctionEventAvg()
        avg.add(FunctionEvent(id=0, node_id=0, name='foo', thread=0, start_us=10, end_us=15))
        avg.add(FunctionEvent(id=1, node_id=0, name='foo', thread=0, start_us=20, end_us=30))
        avg.add(avg)
        self.assertEqual(avg.key, 'foo')
        self.assertEqual(avg.count, 4)
        self.assertEqual(avg.cpu_time_total, 30)
        self.assertEqual(avg.self_cpu_time_total, 30)
        self.assertEqual(avg.cuda_time_total, 0)
        self.assertEqual(avg.cpu_time, 7.5)
        self.assertEqual(avg.cuda_time_total, 0)

    def test_profiler_shapes(self):
        if False:
            return 10
        print('')
        layer1 = torch.nn.Linear(20, 30)
        layer2 = torch.nn.Linear(30, 40)
        input = torch.randn(128, 20)
        with profile(record_shapes=True, use_kineto=kineto_available()) as prof:
            layer2(layer1(input))
        print(prof.function_events)
        linear_expected_shapes = [[[128, 20], [30, 20], [30]], [[128, 30], [40, 30], [40]]]
        found_indices = set()
        for event in prof.function_events:
            if event.name == 'aten::linear':
                self.assertTrue(event.input_shapes in linear_expected_shapes)
                found_indices.add(linear_expected_shapes.index(event.input_shapes))
        self.assertEqual(len(found_indices), len(linear_expected_shapes))

    def test_profiler_aggregation_lstm(self):
        if False:
            return 10
        print('')
        rnn = torch.nn.LSTM(10, 20, 2)
        total_time_s = 0
        with profile(record_shapes=True, use_kineto=kineto_available()) as prof:
            for i in range(20):
                input = torch.randn(5, 3, 10)
                h = torch.randn(2, 3, 20)
                c = torch.randn(2, 3, 20)
                start = time.time()
                rnn(input, (h, c))
                end = time.time()
                total_time_s += end - start
        print(prof.table(sort_by='self_cpu_time_total', row_limit=10, header='TEST'))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by='self_cpu_time_total', row_limit=10))
        print(prof.table(sort_by='self_cpu_time_total', row_limit=10, max_src_column_width=300, header='TEST', top_level_events_only=True))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by='self_cpu_time_total', row_limit=10, top_level_events_only=True))
        total_time_us = total_time_s * 1000.0 * 1000.0
        print('Total time based on python measurements: ', _format_time(total_time_us))
        print('CPU time measurement python side overhead: {:.2f}%'.format((total_time_us / prof.self_cpu_time_total - 1.0) * 100.0))
        if sys.platform != 'win32':
            with tempfile.NamedTemporaryFile() as trace_file:
                prof.export_chrome_trace(trace_file.name)

    def test_record_function(self):
        if False:
            print('Hello World!')
        x = torch.randn(10, 10)

        def forward(x):
            if False:
                while True:
                    i = 10
            with record_function('outer'):
                y = x * 2 + 4
                with record_function('inner'):
                    y = y - 1
            y = y / 1
        forward(x)
        with profile(use_kineto=kineto_available()) as p:
            forward(x)
        events = p.function_events
        important_events = ['outer', 'aten::mul', 'aten::add', 'inner', 'aten::sub', 'aten::div']
        idx = 0
        for info in events:
            if info.name == important_events[idx]:
                idx = idx + 1
            if idx == len(important_events):
                break
        self.assertEqual(idx, len(important_events))

        @record_function('my_func')
        def f(x, y):
            if False:
                print('Hello World!')
            return x + y
        with profile(use_kineto=kineto_available()) as p:
            f(1, 2)
        self.assertTrue('my_func' in str(p))

    def test_record_function_multithreaded(self):
        if False:
            while True:
                i = 10
        rf = record_function('outer')
        rf.__enter__()
        with record_function('inner'):
            rf.__exit__(None, None, None)
        with record_function('inner'):
            rf.__enter__()
        rf.__exit__(None, None, None)

    def test_dir(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(10, 10)
        keys = dir(x)
        self.assertIn('shape', keys)
        y = torch.randn(10, 10, dtype=torch.cfloat)
        imag_key = 'imag'
        self.assertRaises(RuntimeError, lambda : hasattr(x, imag_key))
        self.assertTrue(hasattr(y, imag_key))
        keys.remove(imag_key)
        for key in keys:
            self.assertTrue(hasattr(x, key))

    def test_inplace_on_view_saved_output(self):
        if False:
            while True:
                i = 10
        dealloc = [0]

        class IncrementOnDelete:

            def __del__(self):
                if False:
                    i = 10
                    return i + 15
                dealloc[0] += 1

        def test():
            if False:
                for i in range(10):
                    print('nop')
            root = torch.randn(3, 3, requires_grad=True)
            copy = root.clone()
            copy.grad_fn.register_hook(IncrementOnDelete())
            view = copy.view(9)
            torch.nn.functional.relu(view, inplace=True)
        test()
        self.assertEqual(dealloc[0], 1)

    def test_inplace_on_view_leaf_errors(self):
        if False:
            while True:
                i = 10
        x = torch.zeros(1, requires_grad=True)
        y = x.view_as(x)
        with self.assertRaisesRegex(RuntimeError, 'a view of a leaf Variable that requires grad is being used in an in-place operation.'):
            y.add_(1)

    def test_inplace_on_view_backward(self):
        if False:
            print('Hello World!')
        net = nn.Sequential(nn.InstanceNorm2d(2), nn.ReLU(True))
        x = torch.tensor([[[[1.0, 1.0]]]], requires_grad=True)
        (g,) = torch.autograd.grad(net(x).pow(2), [x], grad_outputs=x.new_ones(x.shape), create_graph=True)
        torch.autograd.grad(g.sum(), [x])
        self.assertEqual(x, torch.tensor([[[[1.0, 1.0]]]]))
        inputs = torch.ones((1, 3, 256, 256), requires_grad=True)
        tmp1 = (inputs + 1).view_as(inputs)
        tmp2 = torch.nn.functional.threshold(tmp1, 0.0, 0.0, True)
        prob_interpolated = torch.sigmoid(tmp2)
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=inputs, grad_outputs=torch.ones(prob_interpolated.size()), create_graph=True, retain_graph=True)[0]
        gradient_penalty = gradients.sum()
        gradient_penalty.backward()
        fn = gradient_penalty.grad_fn.next_functions[0][0].next_functions[1][0]
        self.assertEqual(fn.name(), 'ThresholdBackwardBackward0')

    def test_inplace_on_view_weak_grad_fn(self):
        if False:
            while True:
                i = 10
        a = torch.arange(10.0, requires_grad=True)
        b = a.narrow(0, 0, 2).clone().view(-1)
        b.relu_()
        c = b.clone()
        del b
        gc.collect()
        s = c.sum()
        s.backward()
        self.assertEqual(s, torch.tensor(1.0))
        a = torch.rand(10, requires_grad=True).narrow(0, 0, 10)
        with self.assertRaises(RuntimeError):
            b = a.relu_()

    def test_out_variant_raises_when_inputs_require_grad(self):
        if False:
            print('Hello World!')
        a = torch.randn(2, 2, requires_grad=True)
        b = torch.randn(2, 2, requires_grad=True)
        x = torch.zeros_like(a)
        self.assertRaisesRegex(RuntimeError, 'out=', lambda : torch.mul(a, b, out=x))
        with torch.no_grad():
            torch.mul(a, b, out=x)
            self.assertEqual(x, a * b)
        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        x = torch.zeros(2, 2, requires_grad=True)
        self.assertRaisesRegex(RuntimeError, 'out=', lambda : torch.mul(a, b, out=x))

    def test_anomaly_detect_nan(self):
        if False:
            for i in range(10):
                print('nop')
        size = 10

        class MyFunc(Function):

            @staticmethod
            def forward(ctx, inp1, inp2, fail_0th):
                if False:
                    return 10
                ctx.fail_0th = fail_0th
                return inp1.sum(0, keepdim=True)

            @staticmethod
            def backward(ctx, gO):
                if False:
                    print('Hello World!')
                gI = gO.clone().expand(size)
                gI[0] = 0
                gI[0] /= 0
                if ctx.fail_0th:
                    return (gI, None, None)
                else:
                    return (None, gI, None)
        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, inp, True)
        out.backward()
        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, inp, True)
        with self.assertRaisesRegex(RuntimeError, "Function 'MyFuncBackward' returned nan values in its 0th output."):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    out.backward()
            self.assertIn('No forward pass information', str(w[0].message))
        inp = torch.rand(size, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "Function 'MyFuncBackward' returned nan values in its 1th output."):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    out = MyFunc.apply(inp, inp, False)
                    out.backward()
            self.assertIn('MyFunc.apply', str(w[0].message))

    def test_calculate_shape_util(self):
        if False:
            i = 10
            return i + 15
        out = torch.randn(10, 5, requires_grad=True)
        grad = torch.randn(5, 10, requires_grad=True)
        (out_shape, grad_shape) = _calculate_shape(out, grad, False)
        assert out_shape == torch.Size([10, 5])
        assert grad_shape == torch.Size([5, 10])
        out = torch.nested.as_nested_tensor([torch.randn(10, 5, requires_grad=True), torch.randn(10, 5, requires_grad=True), torch.randn(10, 5, requires_grad=True)])
        grad = torch.nested.as_nested_tensor([torch.randn(5, 10, requires_grad=True), torch.randn(5, 10, requires_grad=True)])
        (out_shape, grad_shape) = _calculate_shape(out, grad, False)
        assert torch.equal(out_shape, torch.tensor([[10, 5], [10, 5], [10, 5]]))
        assert torch.equal(grad_shape, torch.tensor([[5, 10], [5, 10]]))

    def test_nested_anomaly_detect_nan(self):
        if False:
            while True:
                i = 10
        size = 10

        class MyFunc(Function):

            @staticmethod
            def forward(ctx, inp1, fail_0th):
                if False:
                    print('Hello World!')
                ctx.fail_0th = fail_0th
                ctx.save_for_backward(inp1)
                return inp1.sum(0, keepdim=True)

            @staticmethod
            def backward(ctx, gO):
                if False:
                    for i in range(10):
                        print('nop')
                (inp,) = ctx.saved_tensors
                fail_0th = ctx.fail_0th
                g = gO.clone().expand(size)
                gI = MyFunc2.apply(g * inp, g + inp, fail_0th)
                return (gI, None)

        class MyFunc2(Function):

            @staticmethod
            def forward(ctx, inp1, inp2, fail_0th):
                if False:
                    while True:
                        i = 10
                ctx.fail_0th = fail_0th
                return inp1 * 2.0 + inp2

            @staticmethod
            def backward(ctx, gO):
                if False:
                    return 10
                fail_0th = ctx.fail_0th
                g1 = gO.clone()
                g2 = gO.clone()
                g1[0] = 0
                g2[0] = 0
                if fail_0th:
                    g1[0] /= 0
                else:
                    g2[0] /= 0
                return (g1, g2, None)
        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, True)
        (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
        gsum = ginp.sum()
        gsum.backward()
        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, True)
        (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
        gsum = ginp.sum()
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaisesRegex(RuntimeError, "Function 'MyFunc2Backward' returned nan values in its 0th output."):
                with detect_anomaly():
                    gsum.backward()
        self.assertIn('No forward pass information', str(w[1].message))
        inp = torch.rand(size, requires_grad=True)
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaisesRegex(RuntimeError, "Function 'MyFunc2Backward' returned nan values in its 1th output."):
                with detect_anomaly():
                    out = MyFunc.apply(inp, False)
                    (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
                    gsum = ginp.sum()
                    gsum.backward()
        self.assertIn('MyFunc2.apply', str(w[1].message))
        self.assertIn('MyFunc.apply', str(w[2].message))

    def test_anomaly_grad_warnings(self):
        if False:
            i = 10
            return i + 15

        class StdErrDiverter:

            def __enter__(self):
                if False:
                    print('Hello World!')
                self.stderr_orig = sys.stderr
                self.stderr_new = io.StringIO()
                sys.stderr = self.stderr_new
                return self

            def __exit__(self, *args):
                if False:
                    print('Hello World!')
                self.captured = self.stderr_new.getvalue()
                sys.stderr = self.stderr_orig
        with self.assertRaisesRegex(RuntimeError, 'one of the variables needed for gradient computation has been modified by an inplace operation'):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    a = torch.randn(5, requires_grad=True)
                    d1 = a + 1
                    d2 = d1 ** 2
                    d1 += 1
                    torch.autograd.grad(d2.sum(), a)
        self.assertEqual(len(w), 2)
        self.assertIn('Anomaly Detection has been enabled', str(w[0].message))
        self.assertIn('Error detected in PowBackward0', str(w[1].message))
        with self.assertRaisesRegex(RuntimeError, 'one of the variables needed for gradient computation has been modified by an inplace operation'):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    warnings.simplefilter('error')
                    with StdErrDiverter() as s:
                        a = torch.randn(5, requires_grad=True)
                        d1 = a + 1
                        d2 = d1 ** 2
                        d1 += 1
                        torch.autograd.grad(d2.sum(), a)
        self.assertEqual(len(w), 1)
        self.assertIn('Anomaly Detection has been enabled', str(w[0].message))
        self.assertIn('Error detected in PowBackward0', s.captured)

    def test_anomaly_assign_parent_cleanup(self):
        if False:
            for i in range(10):
                print('nop')

        def get_ref():
            if False:
                for i in range(10):
                    print('nop')
            x = torch.randn(2, 2, requires_grad=True)
            t = x.exp()
            with detect_anomaly():
                grad = torch.autograd.grad(t, x, torch.ones_like(t), create_graph=True)

            class Foo:
                pass
            my_obj = Foo()
            meta_dict = t.grad_fn.metadata
            meta_dict[0] = my_obj
            ref = weakref.ref(my_obj)
            return (t, ref)
        (t, ref) = get_ref()
        self.assertIsNotNone(ref())
        del t
        self.assertIsNone(ref())

    def test_nested_anomaly_printstack_cleanup(self):
        if False:
            while True:
                i = 10

        def get_ref():
            if False:
                while True:
                    i = 10

            class MyFunc(Function):

                @staticmethod
                def forward(ctx, x):
                    if False:
                        i = 10
                        return i + 15
                    ctx.save_for_backward(x)
                    return x

                @staticmethod
                def backward(ctx, gO):
                    if False:
                        i = 10
                        return i + 15
                    (x,) = ctx.saved_tensors
                    return MyFunc2.apply(x)

            class MyFunc2(Function):

                @staticmethod
                def forward(ctx, x):
                    if False:
                        for i in range(10):
                            print('nop')
                    return x

                @staticmethod
                def backward(ctx, gO):
                    if False:
                        return 10
                    return gO + float('NaN')
            inp = torch.rand(1, requires_grad=True)
            out = MyFunc.apply(inp)
            (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
            with warnings.catch_warnings(record=True) as w:
                with self.assertRaisesRegex(RuntimeError, "Function 'MyFunc2Backward' returned nan values in its 0th output."):
                    with detect_anomaly():
                        ginp.backward()

            class Foo:
                pass
            my_obj = Foo()
            meta_dict = out.grad_fn.metadata
            meta_dict[0] = my_obj
            ref = weakref.ref(my_obj)
            return (out, ref)
        (t, ref) = get_ref()
        self.assertIsNotNone(ref())
        del t
        self.assertIsNone(ref())

    def test_anomaly_mode_no_check_nan(self):
        if False:
            return 10

        class MyFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, inp):
                if False:
                    return 10
                return inp.clone()

            @staticmethod
            def backward(ctx, gO):
                if False:
                    i = 10
                    return i + 15
                return torch.tensor(float('nan')).expand(10, 10)

        def run_fn(a):
            if False:
                print('Hello World!')
            out = MyFunc.apply(a)
            return out.sum()
        with warnings.catch_warnings(record=True) as w:
            with torch.autograd.detect_anomaly(check_nan=False):
                inp = torch.rand(10, 10, requires_grad=True)
                out = run_fn(inp)
                out.backward(retain_graph=True)
                with torch.autograd.detect_anomaly(check_nan=True):
                    with self.assertRaisesRegex(RuntimeError, "Function 'MyFuncBackward' returned nan values in its 0th output."):
                        out.backward(retain_graph=True)
                out.backward()

    def test_no_grad_copy(self):
        if False:
            i = 10
            return i + 15

        class MyFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                if False:
                    return 10
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                if False:
                    print('Hello World!')
                MyFunc.static_grad_ptr = grad.data_ptr()
                return (grad, grad)

        class NonContGradFunc(Function):

            @staticmethod
            def forward(ctx, inp1):
                if False:
                    print('Hello World!')
                ctx.size = inp1.size()
                return torch.tensor([1.0])

            @staticmethod
            def backward(ctx, grad):
                if False:
                    print('Hello World!')
                return torch.ones(1).expand(ctx.size)
        a = torch.randn(5, 6, requires_grad=True)
        b = torch.randn(5, 6, requires_grad=True)
        NonContGradFunc.apply(MyFunc.apply(a, b)).backward()
        self.assertFalse(a.grad.data_ptr() == MyFunc.static_grad_ptr)
        self.assertFalse(b.grad.data_ptr() == MyFunc.static_grad_ptr)
        a.grad = b.grad = None
        MyFunc.apply(a, b)[1][0].backward()
        p_g = MyFunc.static_grad_ptr
        p_a = a.grad.data_ptr()
        p_b = b.grad.data_ptr()
        self.assertFalse(p_a == p_b)
        self.assertTrue(p_a == p_g or p_b == p_g)

    def test_no_grad_copy_sparse(self):
        if False:
            for i in range(10):
                print('nop')

        class MyFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                if False:
                    for i in range(10):
                        print('nop')
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                if False:
                    return 10
                MyFunc.static_grad_ptr = grad._values().data_ptr()
                return (grad, grad)

        class NonContGradFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                if False:
                    print('Hello World!')
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                if False:
                    print('Hello World!')
                v = torch.rand(1, 3)
                i = torch.ones(1, 1, dtype=torch.long)
                nv = v.expand(8, 3)
                ni = i.expand(1, 8)
                ngrad = torch.sparse_coo_tensor(ni, nv, (10, 3), dtype=torch.float32)
                NonContGradFunc.static_grad_ptr = ngrad._values().data_ptr()
                return (ngrad, ngrad)
        a = torch.randn(10, 3, requires_grad=True)
        b = torch.randn(10, 3, requires_grad=True)
        input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        offsets = torch.tensor([0, 4])
        import torch.nn.functional as F
        emb_matrix = MyFunc.apply(a, b)
        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        loss.backward(retain_graph=True)
        p_g = MyFunc.static_grad_ptr
        p_a = a.grad._values().data_ptr()
        p_b = b.grad._values().data_ptr()
        self.assertFalse(p_a == p_b)
        self.assertTrue(p_a == p_g or p_b == p_g)
        for i in range(10):
            loss.backward(retain_graph=True)
        a.grad = b.grad = None
        emb_matrix = NonContGradFunc.apply(a, b)
        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        loss.backward(retain_graph=True)
        p_g = NonContGradFunc.static_grad_ptr
        p_a = a.grad._values().data_ptr()
        p_b = b.grad._values().data_ptr()
        self.assertFalse(p_a == p_b)
        self.assertFalse(p_a == p_g)
        self.assertFalse(p_b == p_g)
        for i in range(10):
            loss.backward(retain_graph=True)

    def test_gradcheck_single_input(self):
        if False:
            i = 10
            return i + 15

        def check(fast_mode):
            if False:
                i = 10
                return i + 15

            def f(inp):
                if False:
                    return 10
                return inp.mul(5)
            gradcheck(f, torch.rand(10, dtype=torch.float64, requires_grad=True), fast_mode=fast_mode)
            gradgradcheck(f, torch.rand(10, dtype=torch.float64, requires_grad=True), fast_mode=fast_mode)
        check(fast_mode=True)
        check(fast_mode=False)

    @parametrize('layout', (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc))
    def test_gradcheck_input(self, layout):
        if False:
            for i in range(10):
                print('nop')
        if layout in {torch.sparse_bsr, torch.sparse_bsc}:
            blocksize = (2, 2)
            size = (4, 8)
        else:
            blocksize = None
            size = (2, 2)

        def check(fast_mode, masked):
            if False:
                for i in range(10):
                    print('nop')

            def fn(sparse):
                if False:
                    print('Hello World!')
                return torch.sum(sparse)
            gradcheck(fn, torch.rand(size, dtype=torch.double).to_sparse(layout=layout, blocksize=blocksize).requires_grad_(), masked=masked, check_batched_grad=False, fast_mode=fast_mode)
        for (fast_mode, masked) in product(*[(True, False)] * 2):
            check(fast_mode=fast_mode, masked=masked)

    def test_gradcheck_nondeterministic(self):
        if False:
            i = 10
            return i + 15

        class NonDetFunc(Function):

            @staticmethod
            def forward(ctx, x, jitter=0.0):
                if False:
                    print('Hello World!')
                ctx._jitter = jitter
                return x

            @staticmethod
            def backward(ctx, grad_out):
                if False:
                    for i in range(10):
                        print('nop')
                return (NonDetFunc.apply(grad_out, ctx._jitter) * (1 + torch.rand_like(grad_out) * ctx._jitter), None)

        def check(fast_mode):
            if False:
                print('Hello World!')
            inp = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
            gradcheck(lambda x: NonDetFunc.apply(x, 0.0), inp, check_batched_grad=False, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, 'Backward is not reentrant'):
                gradcheck(lambda x: NonDetFunc.apply(x, 1e-06), inp, check_batched_grad=False, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, 'Backward is not reentrant'):
                gradgradcheck(lambda x: NonDetFunc.apply(x, 1e-12), inp, check_batched_grad=False, fast_mode=fast_mode)
            gradcheck(lambda x: NonDetFunc.apply(x, 0.0), inp, nondet_tol=1e-05, check_batched_grad=False, fast_mode=fast_mode)
            gradcheck(lambda x: NonDetFunc.apply(x, 1e-06), inp, nondet_tol=1e-05, check_batched_grad=False, fast_mode=fast_mode)
            gradgradcheck(lambda x: NonDetFunc.apply(x, 1e-12), inp, nondet_tol=1e-05, check_batched_grad=False, fast_mode=fast_mode)
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_validates_inputs(self):
        if False:
            while True:
                i = 10

        def check(fast_mode):
            if False:
                while True:
                    i = 10
            x = torch.rand(10, requires_grad=True).to_sparse()
            self.assertTrue(gradcheck(lambda x: x.to_dense(), (x,), check_batched_grad=False, atol=0.1, fast_mode=fast_mode, masked=True))
            self.assertFalse(gradcheck(lambda x: x.to_dense(), (x,), masked=False, check_batched_grad=False, raise_exception=False, fast_mode=fast_mode))
            self.assertTrue(gradcheck(lambda x: x.to_dense(masked_grad=False), (x,), masked=False, atol=0.1, check_batched_grad=False, raise_exception=False, fast_mode=fast_mode))
            x = torch.rand(10, requires_grad=False)
            with self.assertRaisesRegex(ValueError, 'at least one input tensor to require gradient'):
                gradcheck(lambda x: x, (x,), raise_exception=False, fast_mode=fast_mode)
            x = torch.ones(1, dtype=torch.float32, requires_grad=True)
            with self.assertWarnsRegex(UserWarning, 'Input #0 requires gradient and is not a double precision'):
                self.assertTrue(gradcheck(lambda x: x, (x,), atol=0.1, fast_mode=fast_mode))
            x = torch.ones(1, dtype=torch.float64, requires_grad=True)
            x = x.expand((2, 2))
            with self.assertRaisesRegex(RuntimeError, 'The 0th input has a dimension with stride 0'):
                gradcheck(lambda x: x, (x,), raise_exception=False, fast_mode=fast_mode)
        check(fast_mode=True)
        check(fast_mode=False)

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), 'MKL-DNN build is disabled')
    def test_gradcheck_validates_input_mkldnn(self):
        if False:
            print('Hello World!')
        x = torch.ones(1).to_mkldnn().requires_grad_()
        with self.assertWarnsRegex(UserWarning, 'Input #0 requires gradient and is not a double precision'):
            with self.assertRaisesRegex(ValueError, 'MKLDNN inputs are not support for forward AD gradcheck.'):
                gradcheck(lambda x: x.to_dense(), (x,), raise_exception=False, fast_mode=False, check_forward_ad=True, atol=0.1, rtol=0.1)
        with self.assertWarnsRegex(UserWarning, 'Input #0 requires gradient and is not a double precision'):
            with self.assertRaisesRegex(ValueError, 'MKLDNN inputs are not support for forward AD gradcheck.'):
                gradcheck(lambda x: x.to_dense(), (x,), raise_exception=False, fast_mode=True, check_forward_ad=True, atol=0.1, rtol=0.1)

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), 'MKL-DNN build is disabled')
    def test_gradcheck_test_outputs(self):
        if False:
            print('Hello World!')

        def check(fast_mode):
            if False:
                print('Hello World!')
            x = torch.rand(10, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(ValueError, 'Sparse output is not supported at gradcheck yet'):
                gradcheck(lambda x: x, (x,), masked=True, check_batched_grad=False, raise_exception=False, fast_mode=fast_mode)
            root = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
            with self.assertRaisesRegex(ValueError, 'MKLDNN output is not supported at gradcheck yet'):
                gradcheck(lambda x: x.to_mkldnn(), (root,), check_batched_grad=False, raise_exception=False, fast_mode=fast_mode)
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_check_no_differentiable_outputs(self):
        if False:
            while True:
                i = 10

        def check(fast_mode):
            if False:
                return 10
            x = torch.ones((1,), requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, 'Numerical gradient for function expected to be zero'):
                gradcheck(lambda x: torch.tensor([x]), x)
            self.assertFalse(gradcheck(lambda x: torch.tensor([x]), x, raise_exception=False, fast_mode=fast_mode))
            self.assertTrue(gradcheck(lambda x: (), (x,), fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_check_batched_grad(self):
        if False:
            return 10

        def check(fast_mode):
            if False:
                i = 10
                return i + 15
            x = torch.rand(10, dtype=torch.double, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(RuntimeError, 'gradcheck or gradgradcheck failed while testing batched gradient'):
                gradcheck(lambda x: x.to_dense(), (x,), masked=True, check_batched_grad=True, fast_mode=fast_mode)
            self.assertFalse(gradcheck(lambda x: x.to_dense(), (x,), masked=True, check_batched_grad=True, raise_exception=False, fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_backward_mul_by_grad_output(self):
        if False:
            while True:
                i = 10

        def check(fast_mode):
            if False:
                return 10

            def fn(x):
                if False:
                    print('Hello World!')

                def hook(grad):
                    if False:
                        i = 10
                        return i + 15
                    if grad is not None:
                        return grad.to_dense().to_sparse(1)
                    return grad
                y = x.clone()
                y.register_hook(hook)
                return y.to_dense()
            x = torch.ones((2, 2), dtype=torch.double, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(RuntimeError, 'grad is sparse tensor, but has incorrect sparse_dim'):
                gradcheck(fn, (x,), atol=0.1, masked=True, check_batched_grad=False, fast_mode=fast_mode)
            self.assertFalse(gradcheck(fn, (x,), atol=0.1, masked=True, check_batched_grad=False, raise_exception=False, fast_mode=fast_mode))

            def fn2(x):
                if False:
                    for i in range(10):
                        print('nop')
                y = x.clone()
                y.register_hook(lambda x: x + 0.01)
                return y
            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, 'backward not multiplied by grad_output'):
                gradcheck(fn2, (x,), atol=0.1, fast_mode=fast_mode)
            self.assertFalse(gradcheck(fn2, (x,), atol=0.1, raise_exception=False, fast_mode=fast_mode))

            def fn3(x):
                if False:
                    print('Hello World!')
                y = x.clone().to_dense()
                y.register_hook(lambda x: x + 0.01)
                return y
            x = torch.ones(1, dtype=torch.double, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(RuntimeError, 'backward not multiplied by grad_output'):
                gradcheck(fn3, (x,), atol=0.1, masked=True, check_batched_grad=False, fast_mode=fast_mode)
            self.assertFalse(gradcheck(fn3, (x,), atol=0.1, masked=True, check_batched_grad=False, raise_exception=False, fast_mode=fast_mode))

            class Test(Function):

                @staticmethod
                def forward(ctx, x):
                    if False:
                        i = 10
                        return i + 15
                    return x

                @staticmethod
                def backward(ctx, x):
                    if False:
                        for i in range(10):
                            print('nop')
                    return x.to_sparse()
            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, 'grad is incorrect layout'):
                gradcheck(Test.apply, (x,), check_batched_grad=False, fast_mode=fast_mode)
            self.assertFalse(gradcheck(Test.apply, (x,), check_batched_grad=False, raise_exception=False, fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_undefined_grad(self):
        if False:
            for i in range(10):
                print('nop')

        def check(fast_mode):
            if False:
                i = 10
                return i + 15

            def fn(x):
                if False:
                    i = 10
                    return i + 15

                def hook(x):
                    if False:
                        print('Hello World!')
                    if x is None:
                        raise RuntimeError('x is undefined')
                y = x.clone()
                y.register_hook(hook)
                return y
            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertWarnsRegex(UserWarning, 'Backwards compatibility: New undefined gradient support checking feature'):
                with self.assertRaisesRegex(RuntimeError, 'Expected backward function to handle undefined output grads'):
                    gradcheck(fn, (x,), fast_mode=fast_mode)
                self.assertFalse(gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_jacobian_mismatch(self):
        if False:
            print('Hello World!')

        def check(fast_mode):
            if False:
                print('Hello World!')

            def fn(x):
                if False:
                    while True:
                        i = 10
                y = x.clone()
                y.register_hook(lambda x: x + 0.01)
                return y
            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, 'Jacobian mismatch for output 0 with respect to input 0'):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            self.assertFalse(gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode))
            x_c = torch.ones(2, 2, requires_grad=True, dtype=torch.complex128)
            with self.assertRaisesRegex(RuntimeError, 'While considering the imaginary part of complex outputs only'):
                gradcheck(fn, (x_c,), fast_mode=False)
            self.assertFalse(gradcheck(fn, (x_c,), raise_exception=False, fast_mode=False))

            def fn2(x):
                if False:
                    return 10
                y = torch.complex(x, x)
                y.register_hook(lambda x: x + 0.01)
                return y
            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, 'While considering the imaginary part of complex outputs only'):
                gradcheck(fn2, (x,), fast_mode=False)
            self.assertFalse(gradcheck(fn2, (x,), raise_exception=False, fast_mode=False))

            def fn3(x):
                if False:
                    i = 10
                    return i + 15
                y = torch.real(x)
                y.register_hook(lambda x: x + 0.01)
                return y
            with self.assertRaisesRegex(RuntimeError, 'Jacobian mismatch for output 0 with respect to input 0'):
                gradcheck(fn3, (x_c,), fast_mode=False)
            self.assertFalse(gradcheck(fn3, (x_c,), raise_exception=False, fast_mode=False))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_dense_and_sparse_inputs(self):
        if False:
            while True:
                i = 10

        def check(fast_mode):
            if False:
                return 10

            def fn(x, y):
                if False:
                    i = 10
                    return i + 15
                return x * y.coalesce().to_dense()
            a = torch.rand(2, 2, dtype=torch.double, requires_grad=True)
            b = torch.rand(2, 2, dtype=torch.double).to_sparse().requires_grad_(True)
            self.assertTrue(gradcheck(fn, (a, b), masked=True, check_batched_grad=False, fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), 'MKL-DNN build is disabled')
    def test_gradcheck_multiple_mkldnn_inputs(self):
        if False:
            i = 10
            return i + 15

        def check(fast_mode):
            if False:
                while True:
                    i = 10

            def fn(x, y):
                if False:
                    return 10
                return x + y.to_dense()
            a = torch.rand(10, requires_grad=True)
            b = torch.rand(10, dtype=torch.float32).to_mkldnn().requires_grad_(True)
            self.assertTrue(gradcheck(fn, (a, b), atol=0.1, check_batched_grad=False, fast_mode=fast_mode))

            def fn2(x, y):
                if False:
                    i = 10
                    return i + 15
                return x.to_dense() + y.to_dense()
            c = torch.rand(10, dtype=torch.float32).to_mkldnn().requires_grad_(True)
            self.assertTrue(gradcheck(fn, (a, c), atol=0.1, check_batched_grad=False, fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_output_shape_or_dtype_depend_on_values(self):
        if False:
            return 10

        def check(fast_mode):
            if False:
                while True:
                    i = 10

            def fn(x):
                if False:
                    for i in range(10):
                        print('nop')
                if torch.all(x >= 1):
                    return torch.cat([x, x])
                else:
                    return x
            a = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(AssertionError, 'return outputs with the same shape when inputs are perturbed'):
                self.assertTrue(gradcheck(fn, (a,), fast_mode=fast_mode))

            def fn2(x):
                if False:
                    i = 10
                    return i + 15
                if torch.all(x >= 1):
                    return x.to(torch.float32)
                else:
                    return x
            with self.assertRaisesRegex(AssertionError, 'return outputs with the same dtype when inputs are perturbed'):
                self.assertTrue(gradcheck(fn2, (a,), fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_complex_non_complex_outputs(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(x, y):
            if False:
                i = 10
                return i + 15
            z = torch.complex(x, y)
            return (z, x + 1)
        a = torch.ones(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.ones(2, 2, requires_grad=True, dtype=torch.float64)
        self.assertTrue(gradcheck(fn, (a, b)))

        def fn2(z):
            if False:
                i = 10
                return i + 15
            return (z, torch.real(z))
        c = torch.ones(2, 2, requires_grad=True, dtype=torch.complex128)
        self.assertTrue(gradcheck(fn2, c))

    def test_gradcheck_get_numerical_jacobian(self):
        if False:
            i = 10
            return i + 15
        from torch.autograd.gradcheck import get_numerical_jacobian

        def fn(inputs):
            if False:
                while True:
                    i = 10
            x = inputs[0]
            y = inputs[1]
            return (2 * x + y, x + 2 * y)
        a = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        with self.assertWarnsRegex(UserWarning, "get_numerical_jacobian was part of PyTorch's private API"):
            jacobian = get_numerical_jacobian(fn, (a, b), target=a, eps=1e-06)
        self.assertEqual(jacobian[0], 2 * torch.eye(4, dtype=torch.double))
        with self.assertWarnsRegex(UserWarning, "get_numerical_jacobian was part of PyTorch's private API"):
            jacobian = get_numerical_jacobian(fn, (a, b), eps=1e-06)
        self.assertEqual(jacobian[0], 2 * torch.eye(4, dtype=torch.double))
        self.assertEqual(jacobian[1], 1 * torch.eye(4, dtype=torch.double))
        with self.assertRaisesRegex(ValueError, 'Expected grad_out to be 1.0'):
            jacobian = get_numerical_jacobian(fn, (a, b), eps=1e-06, grad_out=2.0)

    def test_gradcheck_get_analytical_jacobian(self):
        if False:
            while True:
                i = 10
        from torch.autograd.gradcheck import get_analytical_jacobian

        def fn(x, y):
            if False:
                return 10
            return (2 * x + y, x + 2 * y)
        a = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        outputs = fn(a, b)
        with self.assertWarnsRegex(UserWarning, "get_analytical_jacobian was part of PyTorch's private API"):
            (jacobians, reentrant, correct_grad_sizes, correct_grad_types) = get_analytical_jacobian((a, b), outputs[0])
        self.assertEqual(jacobians[0], 2 * torch.eye(4, dtype=torch.double))
        self.assertEqual(jacobians[1], 1 * torch.eye(4, dtype=torch.double))
        self.assertTrue(reentrant)

        class NonDetFunc(Function):

            @staticmethod
            def forward(ctx, x, jitter=0.0):
                if False:
                    print('Hello World!')
                ctx._jitter = jitter
                return x

            @staticmethod
            def backward(ctx, grad_out):
                if False:
                    for i in range(10):
                        print('nop')
                return (NonDetFunc.apply(grad_out, ctx._jitter) * (1 + torch.rand_like(grad_out) * ctx._jitter), None)
        outputs = NonDetFunc.apply(a, 1e-06)
        with self.assertWarnsRegex(UserWarning, "get_analytical_jacobian was part of PyTorch's private API"):
            (jacobians, reentrant, correct_grad_sizes, correct_grad_types) = get_analytical_jacobian((a,), outputs)
        self.assertFalse(reentrant)
        with self.assertRaisesRegex(ValueError, 'Expected grad_out to be 1.0'):
            (jacobians, _, _, _) = get_analytical_jacobian((a,), outputs, grad_out=2.0)

    def test_gradcheck_custom_error(self):
        if False:
            i = 10
            return i + 15
        from torch.autograd.gradcheck import GradcheckError

        def check(fast_mode):
            if False:
                print('Hello World!')

            def fn(x):
                if False:
                    i = 10
                    return i + 15
                y = x.clone()
                y.register_hook(lambda x: x + 0.01)
                return y
            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(GradcheckError, 'Jacobian mismatch for output 0 with respect to input 0'):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, 'Jacobian mismatch for output 0 with respect to input 0'):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            self.assertFalse(gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode))

            def fn2(x):
                if False:
                    return 10
                raise RuntimeError('Not a GradcheckError!')
            with self.assertRaisesRegex(RuntimeError, 'Not a GradcheckError!'):
                gradcheck(fn2, (x,), fast_mode=fast_mode, raise_exception=False)
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_forward_ad(self):
        if False:
            i = 10
            return i + 15

        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return (x + y, y)

        def bad_fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            is_running_forward_ad = fwAD._current_level >= 0
            if is_running_forward_ad:
                (y_p, y_d) = fwAD.unpack_dual(y)
                y = fwAD.make_dual(y_p, y_d * 1.1)
            return (x + y, y)
        err_msg = 'Jacobian computed with forward mode mismatch for output 0 with respect to input 1'
        for fast_mode in [True, False]:
            x = torch.rand(2, dtype=torch.double, requires_grad=True)
            y = torch.rand(2, dtype=torch.double, requires_grad=True)
            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

            def basic_mul(x):
                if False:
                    print('Hello World!')
                return torch.view_as_real(torch.resolve_conj(x * 1j))
            gradcheck(basic_mul, x, check_forward_ad=True, fast_mode=fast_mode)
            x = torch.rand(2, dtype=torch.cdouble, requires_grad=True)
            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            y = torch.rand(2, dtype=torch.cdouble, requires_grad=True)
            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

    def test_gradcheck_forward_ad_runs_with_no_requires_grad(self):
        if False:
            return 10

        class UserFn(Function):

            @staticmethod
            def forward(ctx, x, y):
                if False:
                    i = 10
                    return i + 15
                if fwAD._current_level >= 0:
                    self.assertFalse(x.requires_grad)
                    self.assertFalse(y.requires_grad)
                return (x.clone(), y.clone())

            @staticmethod
            def jvp(ctx, x_t, y_t):
                if False:
                    print('Hello World!')
                return (x_t, y_t)
        x = torch.rand(2, dtype=torch.double, requires_grad=True)
        y = torch.rand(2, dtype=torch.double, requires_grad=True)
        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=False, check_backward_ad=False, check_batched_grad=False, check_batched_forward_grad=False)
        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=True, check_backward_ad=False, check_batched_grad=False, check_batched_forward_grad=False)
        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=True, check_backward_ad=False, check_batched_grad=False, check_batched_forward_grad=True)
        x = torch.rand(2, dtype=torch.double, requires_grad=True)
        y = torch.rand(2, dtype=torch.double, requires_grad=False)
        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=True, check_backward_ad=False, check_batched_grad=False, check_batched_forward_grad=True)

    def test_gradcheck_forward_ad_respects_requires_grad(self):
        if False:
            return 10
        jvp_count = [0]

        class UserFn(Function):

            @staticmethod
            def forward(ctx, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return (x.clone(), y.clone())

            @staticmethod
            def jvp(ctx, x_t, y_t):
                if False:
                    while True:
                        i = 10
                jvp_count[0] += 1
                return (x_t, y_t)
        x = torch.rand(1, dtype=torch.double, requires_grad=True)
        y = torch.rand(1, dtype=torch.double, requires_grad=True)
        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=False, check_backward_ad=False, check_batched_grad=False, check_batched_forward_grad=False)
        self.assertEqual(jvp_count[0], 2)
        jvp_count = [0]
        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=True, check_backward_ad=False, check_batched_grad=False, check_batched_forward_grad=False)
        self.assertEqual(jvp_count[0], 6)
        jvp_count = [0]
        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=True, check_backward_ad=False, check_batched_grad=False, check_batched_forward_grad=True)
        self.assertEqual(jvp_count[0], 12)
        jvp_count = [0]
        x = torch.rand(1, dtype=torch.double, requires_grad=True)
        y = torch.rand(1, dtype=torch.double, requires_grad=False)
        gradcheck(UserFn.apply, (x, y), check_forward_ad=True, check_undefined_grad=True, check_backward_ad=False, check_batched_grad=False, check_batched_forward_grad=True)
        self.assertEqual(jvp_count[0], 5)

    def test_gradcheck_check_forward_or_backward_only(self):
        if False:
            return 10
        'Depending on settings for check_forward_ad and check_backward_ad, the\n        correct codepaths should be reached (or not reached)\n        '
        fwd_fail_err_msg = 'FAIL FWD'
        bwd_fail_err_msg = 'FAIL BWD'

        class UserFn(Function):

            @staticmethod
            def forward(ctx, foo, fwd_bad, bwd_bad):
                if False:
                    while True:
                        i = 10
                ctx.fwd_bad = fwd_bad
                ctx.bwd_bad = bwd_bad
                return foo * 2

            @staticmethod
            def vjp(ctx, gO):
                if False:
                    i = 10
                    return i + 15
                if ctx.bwd_bad:
                    raise RuntimeError(bwd_fail_err_msg)
                else:
                    return (2 * gO, None, None)

            @staticmethod
            def jvp(ctx, gI, _1, _2):
                if False:
                    return 10
                if ctx.fwd_bad:
                    raise RuntimeError(fwd_fail_err_msg)
                else:
                    return 2 * gI
        for fast_mode in (True, False):
            for check_forward_ad in (True, False):
                for check_backward_ad in (True, False):
                    for fwd_bad in (True, False):
                        for bwd_bad in (True, False):
                            fwd_should_fail = fwd_bad and check_forward_ad
                            bwd_should_fail = bwd_bad and check_backward_ad

                            def run():
                                if False:
                                    return 10
                                gradcheck(UserFn.apply, (x, fwd_bad, bwd_bad), check_forward_ad=check_forward_ad, check_backward_ad=check_backward_ad, check_undefined_grad=check_backward_ad, check_batched_grad=check_backward_ad, fast_mode=fast_mode)
                            x = torch.rand(2, dtype=torch.double, requires_grad=True)
                            if not check_forward_ad and (not check_backward_ad):
                                with self.assertRaisesRegex(AssertionError, 'Expected at least one of'):
                                    run()
                                continue
                            if not fwd_should_fail and (not bwd_should_fail):
                                run()
                            else:
                                if fwd_should_fail:
                                    fail_msg = fwd_fail_err_msg
                                if bwd_should_fail:
                                    fail_msg = bwd_fail_err_msg
                                with self.assertRaisesRegex(RuntimeError, fail_msg):
                                    run()

    def test_gradcheck_forward_ad_batched_grad(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.rand(2, dtype=torch.double, requires_grad=True)

        def fn1(a: torch.Tensor, b: int):
            if False:
                print('Hello World!')
            return (a.clone(), a + 1)
        gradcheck(fn1, (x, 1), check_forward_ad=True, check_backward_ad=False, check_batched_grad=False, check_undefined_grad=False, check_batched_forward_grad=True)

        def fn2(a: torch.Tensor, c: torch.Tensor):
            if False:
                for i in range(10):
                    print('nop')
            return a.clone()
        gradcheck(fn2, (x, x.clone()), check_forward_ad=True, check_backward_ad=False, check_batched_grad=False, check_undefined_grad=False, check_batched_forward_grad=True)

        class Fn(Function):

            @staticmethod
            def forward(ctx, foo):
                if False:
                    return 10
                return foo * 2

            @staticmethod
            def vjp(ctx, gO):
                if False:
                    print('Hello World!')
                return gO * 2

            @staticmethod
            def jvp(ctx, gI):
                if False:
                    return 10
                torch.randn_like(gI)
                return gI * 2
        msg = 'vmap: We do not yet support calling random operations inside of vmap'
        with self.assertRaisesRegex(RuntimeError, msg):
            gradcheck(Fn.apply, (x,), check_forward_ad=True, check_batched_forward_grad=True)

    def test_version_counter(self):
        if False:
            print('Hello World!')
        x = torch.randn(1, 2)
        x_saved_version = x._version
        x.add_(1).add_(1)
        self.assertTrue(x._version > x_saved_version)
        xz = x[:]
        self.assertTrue(x._version == xz._version)
        xz.add_(1)
        self.assertTrue(x._version == xz._version)
        x_saved_version = x._version
        x.data = torch.randn(2, 3)
        self.assertTrue(x._version == x_saved_version)
        x.add_(1)
        self.assertTrue(x._version > x_saved_version)
        self.assertTrue(x._version == xz._version)
        xz.add_(1)
        self.assertTrue(x._version == xz._version)

    def test_set_data_tensorimpl_type(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(1, 2)
        x_s = torch.sparse_coo_tensor(torch.zeros([1, 1]), torch.ones([1]))
        with self.assertRaisesRegex(RuntimeError, 'incompatible tensor type'):
            x.data = x_s

    def test_set_data_preserve_pyobj(self):
        if False:
            i = 10
            return i + 15
        a = torch.randn(1, 2)
        b = torch.randn(1, 2)
        b_id_saved = id(b)
        b.data = a
        self.assertTrue(b_id_saved == id(b))

    def test_set_data_self_requires_grad(self):
        if False:
            while True:
                i = 10
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(2.0)
        c = torch.tensor(3, dtype=torch.int64)
        a.data = b
        with self.assertRaisesRegex(RuntimeError, 'must be floating point or complex dtype'):
            a.data = c

    @unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
    def test_thread_shutdown(self):
        if False:
            return 10
        code = 'import torch\nfrom torch.autograd import Function\nclass MyFunction(Function):\n    @staticmethod\n    def forward(ctx, x):\n        return x\n\n    @staticmethod\n    def backward(ctx, grad):\n        return grad\n\n# Run on cuda if it is available to ensure that the worker thread\n# is properly initialized by the time we exit.\ndevice = "cuda" if torch.cuda.is_available() else "cpu"\n\nfor shape in [(1,), ()]:\n    v = torch.ones(shape, requires_grad=True, device=device)\n    MyFunction.apply(v).backward()\n'
        s = TestCase.runWithPytorchAPIUsageStderr(code)
        if TEST_CUDA or torch.backends.mps.is_available():
            self.assertRegex(s, 'PYTORCH_API_USAGE torch.autograd.thread_shutdown')
        else:
            self.assertNotRegex(s, 'PYTORCH_API_USAGE torch.autograd.thread_shutdown')

    @unittest.skipIf(IS_MACOS, 'Fails with SIGBUS on macOS; https://github.com/pytorch/pytorch/issues/25941')
    def test_deep_reentrant(self):
        if False:
            while True:
                i = 10

        class DeepReentrant(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                with torch.enable_grad():
                    ctx.x = Variable(x.detach(), requires_grad=True)
                    ctx.x = ctx.x - 1
                return ctx.x.detach()

            @staticmethod
            def backward(ctx, x):
                if False:
                    return 10
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    DeepReentrant.apply(ctx.x).sum().backward()
                return x
        v = torch.tensor(2000.0, requires_grad=True)
        DeepReentrant.apply(v).sum().backward()
        v2 = torch.tensor(200.0, requires_grad=True)
        DeepReentrant.apply(v2).sum().backward()

    def test_reentrant_priority(self):
        if False:
            for i in range(10):
                print('nop')
        order = []

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                return x

            @staticmethod
            def backward(ctx, x):
                if False:
                    for i in range(10):
                        print('nop')
                order.append('MyFunction')
                return x

        class Reentrant(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                with torch.enable_grad():
                    ctx.x = Variable(x.detach(), requires_grad=True)
                    ctx.x = ctx.x - 1
                return ctx.x.detach()

            @staticmethod
            def backward(ctx, x):
                if False:
                    for i in range(10):
                        print('nop')
                order.append('Reentrant')
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    Reentrant.apply(ctx.x).backward()
                return x
        a = MyFunction.apply(torch.tensor(6.0, requires_grad=True))
        b = Reentrant.apply(torch.tensor(9.0, requires_grad=True))
        v = a * b
        v.backward()
        self.assertEqual(len(order), 11)
        self.assertEqual(order.count('Reentrant'), 10)
        self.assertEqual(order[-1], 'MyFunction')

    @slowTest
    def test_checkpointing(self):
        if False:
            i = 10
            return i + 15
        num_inp = 2000
        nz_inp = 10
        nz_out = 10
        nz_bottleneck = 1000
        module = nn.Sequential(nn.Linear(nz_inp, nz_bottleneck), nn.ReLU(), nn.Linear(nz_bottleneck, nz_inp))
        feat_combined = []
        for r in range(num_inp):
            data_r = torch.empty(1, nz_inp)
            data_r.uniform_()
            data_r.requires_grad = True
            feat_r = checkpoint(module, data_r, use_reentrant=True)
            feat_combined.append(feat_r)
        mean_combined = torch.stack(feat_combined).mean()
        mean_combined.backward()

    def _test_checkpointing_non_reentrant_autocast(self, device_type):
        if False:
            i = 10
            return i + 15
        for enabled in [True, False]:

            def foo(x, y, z):
                if False:
                    return 10
                x = torch.mm(x, y)
                y = torch.mm(x, z)
                z = torch.mm(z, z)
                expected_dtype = torch.float32 if not enabled else torch.bfloat16
                self.assertEqual(expected_dtype, z.dtype)
                return z
            x = torch.randn(3, 3, requires_grad=True)
            y = torch.randn(3, 3, requires_grad=True)
            z = torch.randn(3, 3, requires_grad=True)
            if device_type == 'cuda':
                x = x.cuda()
                y = y.cuda()
                z = z.cuda()
            with torch.autocast(enabled=enabled, device_type=device_type, dtype=torch.bfloat16):
                loss = checkpoint(foo, x, y, z, use_reentrant=False)
                loss = loss.sum()
            loss.backward()

    def test_checkpointing_non_reentrant_autocast_cpu(self):
        if False:
            return 10
        '\n        Test that autocast args such as the dtype are preserved during non-reentrant\n        checkpoint recomputation on CPU.\n        '
        self._test_checkpointing_non_reentrant_autocast(device_type='cpu')

    @unittest.skipIf(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), 'Test requires CUDA bf16 support')
    def test_checkpointing_non_reentrant_autocast_gpu(self):
        if False:
            return 10
        '\n        Test that autocast args/kwargs such as the dtype are preserved during\n        non-reentrant checkpoint recomputation on GPU.\n        '
        self._test_checkpointing_non_reentrant_autocast(device_type='cuda')

    @unittest.skipIf(not torch.cuda.is_available(), 'Test requires CUDA')
    @slowTest
    def test_checkpointing_without_reentrant_memory_savings(self):
        if False:
            print('Hello World!')

        class MyModel(nn.Module):

            def __init__(self, n, use_checkpoint, use_reentrant):
                if False:
                    print('Hello World!')
                super().__init__()
                self.n = n
                self.use_checkpoint = use_checkpoint
                self.use_reentrant = use_reentrant
                self.layers = nn.ModuleList()
                for i in range(self.n):
                    layer = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 256))
                    self.layers.append(layer)
                for layer in self.layers:
                    for lin in layer:
                        lin.weight.grad = torch.ones_like(lin.weight)
                        lin.bias.grad = torch.ones_like(lin.bias)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                for i in range(self.n):
                    if not self.use_checkpoint:
                        x = self.layers[i](x)
                    else:
                        x = checkpoint(self.layers[i], x, use_reentrant=self.use_reentrant)
                return x
        model_no_checkpoint = MyModel(8, use_checkpoint=False, use_reentrant=False).cuda()
        model_reentrant_checkpoint = MyModel(8, use_checkpoint=True, use_reentrant=True).cuda()
        model_no_reentrant_checkpoint = MyModel(8, use_checkpoint=True, use_reentrant=False).cuda()
        x = torch.randn(100, 256, requires_grad=True, device='cuda')
        torch.cuda.reset_peak_memory_stats()
        loss = model_no_checkpoint(x.clone()).sum()
        loss.backward()
        mem_no_checkpoint = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        loss = model_reentrant_checkpoint(x.clone()).sum()
        loss.backward()
        mem_reentrant_checkpoint = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        loss = model_no_reentrant_checkpoint(x.clone()).sum()
        loss.backward()
        mem_no_reentrant_checkpoint = torch.cuda.max_memory_allocated()
        self.assertTrue(mem_reentrant_checkpoint < mem_no_checkpoint)
        self.assertTrue(mem_no_reentrant_checkpoint < mem_no_checkpoint)

    def test_checkpointing_without_reentrant_custom_function_works(self):
        if False:
            while True:
                i = 10
        msg = 'Unpack is being triggered for a tensor that was already unpacked once'

        class MyFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x, y, z):
                if False:
                    for i in range(10):
                        print('nop')
                w = x * y * z
                out = w + w
                ctx.save_for_backward(x, y, z, w, out)
                return out

            @staticmethod
            def backward(ctx, grad_out):
                if False:
                    for i in range(10):
                        print('nop')
                (x, y, z, w, out) = ctx.saved_tensors
                with self.assertRaisesRegex(RuntimeError, msg):
                    (x_2, y_2, z_2, w_2, out_2) = ctx.saved_tensors
                return (x, y, z)
        x = torch.tensor(1.0, requires_grad=True)
        y = torch.tensor(2.0, requires_grad=True)
        z = torch.tensor(3.0, requires_grad=True)

        def foo(x, y, z):
            if False:
                i = 10
                return i + 15
            x = x * y * z
            y = y * y * z
            z = z * z
            out = MyFunc.apply(x, y, z)
            return out
        out = checkpoint(foo, x, y, z, use_reentrant=False)
        out.sum().backward()

    def test_checkpointing_without_reentrant_with_context_fn(self):
        if False:
            while True:
                i = 10

        class VerboseTorchDispatchMode(TorchDispatchMode):

            def __init__(self):
                if False:
                    return 10
                self.operators = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    return 10
                if kwargs is None:
                    kwargs = {}
                self.operators.append(func.__name__)
                return func(*args, **kwargs)
        x = torch.tensor(1.0, requires_grad=True)
        verbose_mode = VerboseTorchDispatchMode()

        def context_fn():
            if False:
                for i in range(10):
                    print('nop')
            return (verbose_mode, contextlib.nullcontext())
        out = checkpoint(lambda x: x.exp(), x, use_reentrant=False, context_fn=context_fn)
        self.assertEqual(verbose_mode.operators, ['exp.default'])
        verbose_mode.operators = []

        def context_fn():
            if False:
                i = 10
                return i + 15
            return (contextlib.nullcontext(), verbose_mode)
        out = checkpoint(lambda x: x.exp(), x, use_reentrant=False, context_fn=context_fn)
        out.backward()
        self.assertEqual(verbose_mode.operators, ['exp.default', 'detach.default', 'detach.default'])
        with self.assertRaisesRegex(Exception, 'only supported when use_reentrant=False'):
            out = checkpoint(lambda x: x.sin(), x, use_reentrant=True, context_fn=context_fn)

    def test_checkpoint_warns_if_use_reentrant_not_passed_explcitly(self):
        if False:
            i = 10
            return i + 15
        a = torch.randn(1, requires_grad=True)
        with warnings.catch_warnings(record=True) as w:
            checkpoint(lambda x: x, a, use_reentrant=False)
        self.assertEqual(len(w), 0)
        with warnings.catch_warnings(record=True) as w:
            checkpoint(lambda x: x, a)
        self.assertEqual(len(w), 1)
        self.assertIn('please pass in use_reentrant=True or use_reentrant=False explicitly', str(w[0].message))

    def test_checkpoint_detects_non_determinism(self):
        if False:
            return 10

        def save_3_tensors(x):
            if False:
                for i in range(10):
                    print('nop')
            out = x.sin().exp()
            out = out.sin()
            return out

        def save_2_tensors(x):
            if False:
                while True:
                    i = 10
            return x.sin().exp()

        def save_2_tensors_alt(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.sin() * torch.tensor([1.0, 2.0])

        def get_non_det_fn(orig_fn, recompute_fn):
            if False:
                i = 10
                return i + 15
            counter = [0]

            def fn(x):
                if False:
                    while True:
                        i = 10
                if counter[0] == 0:
                    counter[0] += 1
                    return orig_fn(x)
                else:
                    return recompute_fn(x)
            return fn
        a = torch.randn(1, requires_grad=True)
        fn = get_non_det_fn(orig_fn=save_3_tensors, recompute_fn=save_2_tensors)
        with self.assertRaisesRegex(RuntimeError, 'A different number of tensors was saved'):
            out = checkpoint(fn, a, use_reentrant=False)
            out.backward()
        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_3_tensors)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            with self.assertRaisesRegex(RuntimeError, 'trying to save more tensors during recomputation'):
                out = checkpoint(fn, a, use_reentrant=False)
                out.backward()
        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_3_tensors)
        out = checkpoint(fn, a, use_reentrant=False)
        out.backward()
        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)
        with self.assertRaisesRegex(RuntimeError, 'tensors have different metadata'):
            out = checkpoint(fn, a, use_reentrant=False)
            out.backward()
        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)
        with self.assertRaisesRegex(RuntimeError, 'You are seeing this error because you passed `debug=True` to checkpoint'):
            out = checkpoint(fn, a, use_reentrant=False, debug=True)
            out.backward()
        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)
        with self.assertRaisesRegex(RuntimeError, 'You are seeing this error because you passed `debug=True` to checkpoint'):
            with torch.utils.checkpoint.set_checkpoint_debug_enabled(True):
                out = checkpoint(fn, a, use_reentrant=False, debug=False)
                out.backward()
        fn = get_non_det_fn(orig_fn=save_2_tensors, recompute_fn=save_2_tensors_alt)
        with self.assertRaisesRegex(RuntimeError, 'Recomputed values for the following tensors have different'):
            with torch.utils.checkpoint.set_checkpoint_debug_enabled(False):
                out = checkpoint(fn, a, use_reentrant=False, debug=True)
                out.backward()

    def test_access_saved_tensor_twice_without_recomputation_works(self):
        if False:
            i = 10
            return i + 15
        count = [0]

        def foo(a):
            if False:
                for i in range(10):
                    print('nop')
            count[0] += 1
            b = a * a
            c = a * b
            d = torch.exp(a)
            return d
        a = torch.randn(5, requires_grad=True)
        d = checkpoint(foo, a, use_reentrant=False)
        self.assertEqual(count[0], 1)
        d.grad_fn._saved_result
        self.assertEqual(count[0], 2)
        d.grad_fn._saved_result
        self.assertEqual(count[0], 3)
        d.sum().backward()
        self.assertEqual(count[0], 4)
        with self.assertRaisesRegex(RuntimeError, 'or directly access saved tensors after they have already been freed'):
            d.grad_fn._saved_result

    @slowTest
    @parametrize('input_requires_grad', [True, False])
    def test_checkpointing_without_reentrant(self, input_requires_grad):
        if False:
            i = 10
            return i + 15
        '\n        Basic test for checkpoint without reentrant autograd.\n        '
        num_inp = 2000
        nz_inp = 10
        nz_out = 10
        nz_bottleneck = 1000
        module = nn.Sequential(nn.Linear(nz_inp, nz_bottleneck), nn.ReLU(), nn.Linear(nz_bottleneck, nz_inp))

        class MyModule(nn.Module):

            def __init__(self, mod):
                if False:
                    print('Hello World!')
                super().__init__()
                self.module = mod

            def forward(self, data):
                if False:
                    i = 10
                    return i + 15
                return self.module(data)
        module = MyModule(mod=module)
        module_copy = deepcopy(module)
        feat_combined = []
        feat_combined_no_checkpoint = []
        for r in range(num_inp):
            data_r = torch.empty(1, nz_inp)
            data_r.uniform_()
            data_r.requires_grad = input_requires_grad
            data_r_copy = data_r.clone()
            feat_r = checkpoint(module, data=data_r, use_reentrant=False)
            feat_combined.append(feat_r)
            feat_r_no_checkpoint = module_copy(data_r)
            feat_combined_no_checkpoint.append(feat_r_no_checkpoint)
        mean_combined = torch.stack(feat_combined).mean()
        mean_combined.backward()
        mean_combined_no_checkpoint = torch.stack(feat_combined_no_checkpoint).mean()
        mean_combined_no_checkpoint.backward()
        for (checkpoint_param, param) in zip(module.parameters(), module_copy.parameters()):
            self.assertEqual(checkpoint_param.grad, param.grad)

    def test_checkpoint_valid_reset_on_error(self):
        if False:
            while True:
                i = 10
        a = torch.randn(2, 2, requires_grad=True)
        with self.assertRaisesRegex(Exception, 'Checkpointing is not compatible with .grad()'):
            b = checkpoint(torch.exp, a, use_reentrant=True).sum()
            torch.autograd.grad(b, (a,))
        c = checkpoint(torch.exp, a, use_reentrant=True).sum()
        c.backward()

    @parametrize('use_reentrant', [True, False])
    def test_checkpointing_without_reentrant_detached_tensor(self, use_reentrant):
        if False:
            return 10

        class NoGradModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)
                self.lin2 = nn.Linear(2, 2, bias=False)

            def forward(self, x):
                if False:
                    return 10
                with torch.no_grad():
                    return self.lin2(self.linear(x))
        module = NoGradModule()
        err_ctx = self.assertRaisesRegex(RuntimeError, 'none of output has requires_grad=True') if use_reentrant else contextlib.nullcontext()
        a = torch.randn(2, 2, requires_grad=True)
        for _ in range(3):
            with err_ctx:
                out = checkpoint(module, a, use_reentrant=use_reentrant)
                out += a
                out.sum().backward()

    def test_checkpointing_without_reentrant_correct_grad(self):
        if False:
            i = 10
            return i + 15
        '\n        Verifies that correct gradients are calculated for checkpoint\n        without reentrant autograd, for both backward() and autograd.grad().\n        '
        a = torch.randn(2, 2, requires_grad=True)
        b = torch.exp(a).sum()
        b.backward()
        b_grad = a.grad
        a.grad = None
        c = checkpoint(torch.exp, a, use_reentrant=False).sum()
        c.backward()
        c_grad = a.grad
        a.grad = None
        d = checkpoint(torch.exp, a, use_reentrant=False).sum()
        (d_grad,) = torch.autograd.grad(d, (a,))
        self.assertEqual(b_grad, c_grad)
        self.assertEqual(b_grad, d_grad)

    def test_checkpointing_without_reentrant_dataparallel(self):
        if False:
            i = 10
            return i + 15
        '\n        Verifies gradient correctness when checkpoint without reentrant autograd\n        is used in conjunction with DataParallel.\n        '

        class LinearModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)

            def forward(self, inp):
                if False:
                    return 10
                return self.linear(inp)
        a = torch.randn(2, 2, requires_grad=True)
        if torch.cuda.is_available():
            a = a.cuda()
        model = LinearModule()
        if torch.cuda.is_available():
            model = model.cuda()
        b = deepcopy(model)(a).sum()
        b.backward()
        b_grad = a.grad
        a.grad = None
        module = torch.nn.DataParallel(deepcopy(model))
        c = checkpoint(module, a, use_reentrant=False).sum()
        c.backward()
        c_grad = a.grad
        self.assertEqual(b_grad, c_grad)

    def test_checkpointing_without_reentrant_parameter_used_in_an_out(self):
        if False:
            print('Hello World!')
        '\n        Ensures that gradient hooks are only called once per tensor.\n        '
        w = torch.randn(10, 10, requires_grad=True)
        count = 0

        def hook(grad):
            if False:
                while True:
                    i = 10
            nonlocal count
            count += 1
        w.register_hook(hook)
        x = torch.rand(10, 10, requires_grad=True)
        h = w * x
        out = checkpoint(lambda x: w * x, h, use_reentrant=False)
        out.sum().backward()
        self.assertEqual(count, 1)

    def test_checkpointing_without_reentrant_arbitrary_input_output(self):
        if False:
            print('Hello World!')
        '\n        Ensures checkpointing without reentrant autograd works with functions\n        with arbitrary input/output structures.\n        '

        class MyModel(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.layer = torch.nn.Linear(5, 5, bias=False)

            def forward(self, dict_input):
                if False:
                    return 10
                tensor = dict_input['tensor']
                return {'result': self.layer(tensor)}
        model_no_checkpoint = MyModel()
        model_checkpoint_without_reentrant = deepcopy(model_no_checkpoint)
        inp = {'tensor': torch.randn(5, 5)}
        out_no_checkpoint = model_no_checkpoint(inp)['result'].sum()
        out_checkpoint = checkpoint(model_checkpoint_without_reentrant, inp, use_reentrant=False)['result'].sum()
        self.assertEqual(out_checkpoint, out_no_checkpoint)
        out_no_checkpoint.backward()
        out_checkpoint.backward()
        for (param, checkpoint_param) in zip(model_no_checkpoint.parameters(), model_checkpoint_without_reentrant.parameters()):
            self.assertEqual(param.grad, checkpoint_param.grad)

    def test_callback_adds_callback(self):
        if False:
            i = 10
            return i + 15
        called = [0]

        def callback_final():
            if False:
                while True:
                    i = 10
            called[0] += 1

        def callback_adds_callback():
            if False:
                i = 10
                return i + 15
            called[0] += 1
            Variable._execution_engine.queue_callback(callback_final)

        class MyFunc(Function):

            @staticmethod
            def forward(ctx, input):
                if False:
                    while True:
                        i = 10
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, grad):
                if False:
                    while True:
                        i = 10
                Variable._execution_engine.queue_callback(callback_adds_callback)
                return grad
        a = torch.rand((3, 3), requires_grad=True)
        b = MyFunc.apply(a)
        b.sum().backward()
        self.assertEqual(called[0], 2)

    def _test_reentrant_with_callbacks(self, install_callbacks_in_depths):
        if False:
            print('Hello World!')
        counter = {}
        counter['inner'] = 0
        counter['outer'] = 0

        def inc_inner_counter():
            if False:
                while True:
                    i = 10
            counter['inner'] += 1

        def inc_outer_counter():
            if False:
                for i in range(10):
                    print('nop')
            counter['outer'] += 1

        class MyFunc(Function):

            @staticmethod
            def forward(ctx, input):
                if False:
                    return 10
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, input):
                if False:
                    while True:
                        i = 10
                if 1 in install_callbacks_in_depths:
                    Variable._execution_engine.queue_callback(inc_inner_counter)
                return input

        class MyReentrantFunc(Function):

            @staticmethod
            def forward(ctx, input):
                if False:
                    return 10
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, input):
                if False:
                    while True:
                        i = 10
                if 0 in install_callbacks_in_depths:
                    Variable._execution_engine.queue_callback(inc_outer_counter)
                tmp_inp = input.detach().requires_grad_()
                with torch.enable_grad():
                    tmp_out = MyFunc.apply(tmp_inp).sum()
                tmp_out.backward()
                return input
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = MyReentrantFunc.apply(t1)
        t3 = t2.sum()
        torch.autograd.backward([t3])
        return counter

    def test_reentrant_with_callbacks_depth_0(self):
        if False:
            i = 10
            return i + 15
        ret = self._test_reentrant_with_callbacks([0])
        self.assertEqual(1, ret['outer'])
        self.assertEqual(0, ret['inner'])

    def test_reentrant_with_callbacks_depth_1(self):
        if False:
            while True:
                i = 10
        ret = self._test_reentrant_with_callbacks([1])
        self.assertEqual(0, ret['outer'])
        self.assertEqual(1, ret['inner'])

    def test_reentrant_with_callbacks_both_depths(self):
        if False:
            print('Hello World!')
        ret = self._test_reentrant_with_callbacks([0, 1])
        self.assertEqual(1, ret['outer'])
        self.assertEqual(1, ret['inner'])

    def test_reentrant_with_leaf_variable_hook(self):
        if False:
            for i in range(10):
                print('nop')
        handle = None
        param = torch.rand(10, requires_grad=True)

        def add_gradient_penalty_to_grad(grad):
            if False:
                i = 10
                return i + 15
            handle.remove()
            old_param_grad = grad
            param.grad = None
            with torch.enable_grad():
                g = grad.detach().requires_grad_()
                new_param = param.detach().requires_grad_()
                out = (g * 2 + new_param).sum()
                out.backward()
            res = g.grad + grad
            param.grad = old_param_grad
            return res
        handle = param.register_hook(add_gradient_penalty_to_grad)
        tmp = param * param
        loss = tmp.sum()
        loss.backward()

    def test_reentrant_with_non_leaf_variable_hook(self):
        if False:
            print('Hello World!')
        handle = None
        param = torch.rand(10, requires_grad=True)

        def manual_increase_gradient(grad):
            if False:
                print('Hello World!')
            handle.remove()
            with torch.enable_grad():
                g = grad.detach().requires_grad_()
                out = (g * 2 + 5).sum()
                out.backward()
            res = g.grad + grad
            return res
        tmp = param * param
        handle = tmp.register_hook(manual_increase_gradient)
        loss = tmp.sum()
        loss.backward()
        self.assertEqual(param.grad, 6 * param)

    def test_grad_fn_attr_bindings(self):
        if False:
            for i in range(10):
                print('nop')
        a = torch.ones(1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        out1 = torch.stack([a, b], dim=0)
        out2 = a * 2 * b
        self.assertEqual(out2.grad_fn._saved_self, a * 2)
        self.assertIsInstance(out2.grad_fn._saved_self, torch.Tensor)
        self.assertIsInstance(out2.grad_fn._raw_saved_self, torch._C._autograd.SavedTensor)
        self.assertEqual(out1.grad_fn._saved_dim, 0)
        self.assertIsInstance(out1.grad_fn._saved_dim, int)
        out2.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)
        out2.sum().backward()
        with self.assertRaisesRegex(RuntimeError, 'after they have already been freed'):
            out2.grad_fn._saved_self
        self.assertEqual(out1.grad_fn._saved_dim, 0)
        a = torch.ones(2, 2, requires_grad=True)
        indices = torch.tensor([0, 1])
        out = a[:, indices]
        self.assertEqual(out.grad_fn._saved_indices, (None, indices))
        self.assertIsInstance(out.grad_fn._saved_indices[1], torch.Tensor)
        self.assertIsInstance(out.grad_fn._raw_saved_indices[1], torch._C._autograd.SavedTensor)
        self.assertEqual(out.grad_fn._saved_self_sym_sizes, a.shape)
        self.assertIsInstance(out.grad_fn._saved_self_sym_sizes[0], int)
        out.grad_fn._raw_saved_indices[1].register_hooks(lambda x: x, lambda x: x)
        with self.assertRaisesRegex(RuntimeError, 'None is forbidden'):
            out.grad_fn._raw_saved_indices[0].register_hooks(lambda x: x, lambda x: x)
        out = a.mean()
        self.assertEqual(out.grad_fn._saved_self_sym_sizes, a.shape)
        a = torch.ones(2, 2, requires_grad=True)
        out = a * a
        out.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)
        out.sum().backward()
        with self.assertRaisesRegex(RuntimeError, 'after it has been freed'):
            out.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)
        a = torch.ones(1, 1, 2, requires_grad=True)
        out = torch.nn.functional.interpolate(a, 4, mode='linear')
        self.assertEqual(out.grad_fn._saved_output_size, (4,))
        self.assertIsInstance(out.grad_fn._saved_output_size[0], int)
        self.assertEqual(out.grad_fn._saved_align_corners, False)
        self.assertIsInstance(out.grad_fn._saved_align_corners, bool)
        if hasattr(out.grad_fn, '_saved_scale_factors'):
            self.assertIsNone(out.grad_fn._saved_scale_factors)
        else:
            self.assertIsNone(out.grad_fn._saved_scales)
        a = torch.ones(1, 1, 3, 3, requires_grad=True)
        out = nn.Conv2d(1, 1, 3)(a)
        self.assertEqual(out.grad_fn._saved_bias_sym_sizes_opt, (1,))
        out = nn.Conv2d(1, 1, 3, bias=False)(a)
        self.assertEqual(out.grad_fn._saved_bias_sym_sizes_opt, (0,))
        a = torch.ones(1, 3, 3, requires_grad=True)
        out = torch.addbmm(a.squeeze(0), a, a)
        self.assertEqual(out.grad_fn._saved_batch1_sym_argsize_0, 1)
        self.assertEqual(out.grad_fn._saved_batch1_sym_argsize_1, 3)
        a = torch.ones(1, 1, 3, 3, requires_grad=True)
        out = torch.nn.functional.unfold(a, 3)
        self.assertEqual(out.grad_fn._saved_self_sym_argsize_minus_2, 3)
        self.assertEqual(out.grad_fn._saved_self_sym_argsize_minus_1, 3)
        a = torch.ones(1, 1, 2, requires_grad=True)
        out = torch.nn.functional.interpolate(a, scale_factor=0.5, mode='linear')
        self.assertEqual(out.grad_fn._saved_scales, 0.5)
        a = torch.ones(2, 2, requires_grad=True)
        out = torch.pdist(a, p=1)
        self.assertEqual(out.grad_fn._saved_p, 1.0)
        self.assertIsInstance(out.grad_fn._saved_p, float)
        a = torch.ones(1, 1, 2, requires_grad=True)
        out = torch.logit(a, 1.0)
        self.assertEqual(out.grad_fn._saved_eps, 1.0)
        self.assertIsInstance(out.grad_fn._saved_eps, float)
        out = torch.logit(a)
        self.assertIsNone(out.grad_fn._saved_eps)
        if torch._C.has_lapack:
            a = torch.ones(1, 1, requires_grad=True)
            (q, r) = torch.linalg.qr(a, mode='reduced')
            self.assertEqual(q.grad_fn._saved_mode, 'reduced')
        a = torch.tensor([1.0], requires_grad=True)
        out = torch.div(a, 2.0, rounding_mode='trunc')
        self.assertEqual(out.grad_fn._saved_rounding_mode, 'trunc')
        out = torch.div(a, 2.0, rounding_mode=None)
        self.assertIsNone(out.grad_fn._saved_rounding_mode)
        x = torch.zeros(5, requires_grad=True)
        out = torch.threshold(x, threshold=1 + 0j, value=1 + 0j)
        self.assertIsInstance(out.grad_fn._saved_threshold, complex)
        cfloat = torch.tensor(1 + 0j, dtype=torch.complex64)
        out = torch.threshold(x, threshold=cfloat, value=1 + 0j)
        self.assertIsInstance(out.grad_fn._saved_threshold, complex)
        out = torch.threshold(x, threshold=1.0, value=1.0)
        self.assertIsInstance(out.grad_fn._saved_threshold, float)
        out = torch.threshold(x, threshold=1, value=1)
        self.assertIsInstance(out.grad_fn._saved_threshold, int)
        out = torch.threshold(x, threshold=False, value=False)
        self.assertIsInstance(out.grad_fn._saved_threshold, bool)
        a = torch.ones(2, 2, requires_grad=True)
        out = a.as_strided((3,), (1,), 1)
        self.assertEqual(out.grad_fn._saved_storage_offset, 1)
        self.assertIsInstance(out.grad_fn._saved_storage_offset, int)
        out = a.as_strided((3,), (1,))
        self.assertIsNone(out.grad_fn._saved_storage_offset)
        a = torch.ones(2, requires_grad=True)
        out = torch.tanh(a)
        self.assertEqual(out, out.grad_fn._saved_result)
        a = torch.randn(3, 5, requires_grad=True)
        b = torch.tensor([1, 0, 4])
        loss = nn.NLLLoss()
        out = loss(a, b)
        self.assertIsNone(out.grad_fn._saved_weight)
        loss = nn.NLLLoss(weight=torch.ones((5,)))
        out = loss(a, b)
        self.assertEqual(out.grad_fn._saved_weight, torch.ones((5,)))
        out.sum().backward()
        with self.assertRaisesRegex(RuntimeError, 'after they have already been freed'):
            out.grad_fn._saved_weight
        num_tensors = 3
        input_tensors = [torch.ones(2, 2, requires_grad=True) for _ in range(num_tensors)]
        scalars = [0.0 for _ in range(num_tensors)]
        results = torch._foreach_maximum(input_tensors, scalars)
        for t in results:
            self.assertEqual(t.grad_fn._saved_scalars, scalars)

    def test_cant_create_saved_tensors(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'Trying to create a SavedTensor object from Python is forbidden'):
            torch.autograd.SavedTensor()

    def test_custom_function_saved_tensors(self):
        if False:
            print('Hello World!')

        def getFn(save=True):
            if False:
                print('Hello World!')

            class MyFn(Function):

                @staticmethod
                def forward(ctx, x):
                    if False:
                        for i in range(10):
                            print('nop')
                    if save:
                        ctx.save_for_backward(x, None)
                    return x

                @staticmethod
                def backward(ctx, g):
                    if False:
                        return 10
                    return g
            return MyFn
        a = torch.randn(5, requires_grad=True)
        y = getFn(True).apply(a)
        self.assertEqual((a, None), y.grad_fn.saved_tensors)
        saved = y.grad_fn._raw_saved_tensors
        self.assertIsInstance(saved[0], torch._C._autograd.SavedTensor)
        self.assertIsInstance(saved[1], torch._C._autograd.SavedTensor)
        with self.assertRaisesRegex(RuntimeError, 'None is forbidden'):
            saved[1].register_hooks(lambda x: x, lambda x: x)
        with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
            saved[0].register_hooks(lambda x: x)
        with self.assertRaisesRegex(TypeError, 'incompatible function arguments'):
            saved[0].register_hooks(1, 1)
        saved[0].register_hooks(lambda x: x, lambda x: x)
        with self.assertRaisesRegex(RuntimeError, 'already been set'):
            saved[0].register_hooks(lambda x: x, lambda x: x)
        y.sum().backward()
        del saved
        with self.assertRaisesRegex(RuntimeError, 'after they have already been freed'):
            y.grad_fn._raw_saved_tensors
        with self.assertRaisesRegex(RuntimeError, 'after they have already been freed'):
            y.grad_fn.saved_tensors
        y = getFn(False).apply(a)
        self.assertEqual(y.grad_fn.saved_tensors, ())
        self.assertEqual(y.grad_fn._raw_saved_tensors, ())

    def test_autograd_node_isinstance(self):
        if False:
            return 10
        Node = torch.autograd.graph.Node
        a = torch.rand(3, 3, requires_grad=True)
        b = a.exp()
        self.assertIsInstance(b.grad_fn, Node)
        self.assertTrue(issubclass(type(b.grad_fn), Node))
        self.assertTrue(Node not in type(b.grad_fn).mro())
        self.assertNotIsInstance(torch._C._functions.AccumulateGrad, Node)
        self.assertTrue(issubclass(torch._C._functions.AccumulateGrad, Node))
        self.assertIsInstance(b.grad_fn.next_functions[0][0], Node)
        self.assertTrue(issubclass(torch._C._functions.DelayedError, Node))
        self.assertNotIsInstance(None, Node)
        self.assertNotIsInstance(1, Node)
        self.assertNotIsInstance(Node, Node)
        self.assertTrue(issubclass(Node, Node))
        self.assertTrue(issubclass(torch.autograd.function.BackwardCFunction, Node))

        class Func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                self.assertIsInstance(ctx, Node)
                return x

            @staticmethod
            def backward(ctx, x):
                if False:
                    i = 10
                    return i + 15
                self.assertIsInstance(ctx, Node)
                return x
        out = Func.apply(a)
        self.assertIsInstance(out.grad_fn, Node)
        self.assertTrue(issubclass(type(out.grad_fn), Node))
        self.assertTrue(Node not in type(out.grad_fn).mro())
        out.sum().backward()

    def test_autograd_views_codegen(self):
        if False:
            print('Hello World!')

        def run_test(grad_mode, requires_grad, is_view, should_raise_tuple):
            if False:
                print('Hello World!')

            def maybe_check_raise(fn, should_raise):
                if False:
                    i = 10
                    return i + 15
                self.assertTrue(should_raise is None or isinstance(should_raise, str))
                if should_raise is not None:
                    with self.assertRaisesRegex(RuntimeError, should_raise):
                        fn()
                else:
                    fn()
            inp = torch.rand(2, requires_grad=requires_grad).clone()
            with torch.set_grad_enabled(grad_mode):
                out = inp.view_as(inp)
            self.assertTrue(out._is_view() == is_view)
            maybe_check_raise(lambda : out.add_(1), should_raise_tuple[0])
            inp = torch.rand(2, requires_grad=requires_grad).clone()
            with torch.set_grad_enabled(grad_mode):
                out = inp.unbind()
            self.assertTrue(out[0]._is_view() == is_view)
            self.assertTrue(out[1]._is_view() == is_view)
            maybe_check_raise(lambda : out[0].add_(1), should_raise_tuple[1])
            maybe_check_raise(lambda : out[1].add_(1), should_raise_tuple[2])
        run_test(grad_mode=True, requires_grad=False, is_view=True, should_raise_tuple=(None, None, None))
        inp_change_err = 'Output {} of UnbindBackward0 is a view and is being modified inplace.'
        run_test(grad_mode=True, requires_grad=True, is_view=True, should_raise_tuple=(None, inp_change_err.format('0'), inp_change_err.format('1')))
        leaf_grad_err = 'A view was created in no_grad mode and is being modified inplace'
        run_test(grad_mode=False, requires_grad=True, is_view=True, should_raise_tuple=(leaf_grad_err, leaf_grad_err, leaf_grad_err))
        run_test(grad_mode=False, requires_grad=False, is_view=True, should_raise_tuple=(None, None, None))

    def test_inplace_not_requires_grad(self):
        if False:
            print('Hello World!')

        class MyFn(torch.autograd.Function):

            @staticmethod
            def forward(ctx, inp):
                if False:
                    for i in range(10):
                        print('nop')
                return inp.view_as(inp)

            @staticmethod
            def backward(ctx, grad):
                if False:
                    i = 10
                    return i + 15
                return grad
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        view_a = MyFn.apply(a)
        with self.assertRaisesRegex(RuntimeError, 'This view was created inside a custom Function'):
            view_a += b
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        view_a = MyFn.apply(a)
        with self.assertRaisesRegex(RuntimeError, 'This view was created inside a custom Function'):
            view_a.copy_(b)
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        view_a = a.unbind()[0]
        with self.assertRaisesRegex(RuntimeError, 'This view is the output of a function that returns multiple views.'):
            view_a.copy_(b)
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        a.select(1, 0).copy_(b)

    def _do_test_autograd_simple_views_python(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        bw_called = [0]
        ga_nz = [False]

        class IdOneOutput(Function):

            @staticmethod
            def forward(ctx, a, b, make_view):
                if False:
                    return 10
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                return a

            @staticmethod
            def backward(ctx, ga):
                if False:
                    for i in range(10):
                        print('nop')
                bw_called[0] += 1
                return (ga, None, None)

        class IdTwoOutput(Function):

            @staticmethod
            def forward(ctx, a, b, make_view):
                if False:
                    return 10
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                return (a, a + b)

            @staticmethod
            def backward(ctx, ga, gab):
                if False:
                    return 10
                bw_called[0] += 1
                if ga.eq(0).all():
                    ga_nz[0] = False
                else:
                    ga_nz[0] = True
                return (ga + gab, gab, None)

        class ViewOfTemp(Function):

            @staticmethod
            def forward(ctx, a, make_view):
                if False:
                    print('Hello World!')
                ctx.save_for_backward(a)
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                b = a.clone()
                return b.select(0, 0)

            @staticmethod
            def backward(ctx, grad):
                if False:
                    for i in range(10):
                        print('nop')
                bw_called[0] += 1
                (a,) = ctx.saved_tensors
                res = torch.zeros_like(a)
                res.select(0, 0).copy_(grad)
                return (res, None)
        fn_id_to_inplace_on_view_err_msg = {'one_output': 'Output 0 of IdOneOutputBackward is a view and is being modified inplace. This view was created inside a custom Function', 'two_output': 'Output 0 of IdTwoOutputBackward is a view and is being modified inplace. This view is the output of a function that returns multiple views.', 'view_of_temp': 'Output 0 of ViewOfTempBackward is a view and is being modified inplace. This view was created inside a custom Function'}
        for fn_id in ['one_output', 'two_output', 'view_of_temp']:
            for inplace in [True, False]:
                for make_view in [True, False]:
                    output_is_a_view = make_view or fn_id == 'view_of_temp'

                    def fn(a, b):
                        if False:
                            while True:
                                i = 10
                        a = a.clone()
                        b = b.clone()
                        if fn_id == 'two_output':
                            (tmp1, tmp2) = IdTwoOutput.apply(a, b, make_view)
                            if inplace:
                                tmp1 += 3
                                tmp2 += 3
                            else:
                                tmp1 = tmp1 + 3
                                tmp2 = tmp2 + 3
                            tmp = tmp1 * tmp2
                        else:
                            if fn_id == 'one_output':
                                tmp = IdOneOutput.apply(a, b, make_view)
                            else:
                                tmp = ViewOfTemp.apply(a + b, make_view)
                            if inplace:
                                tmp += 3
                            else:
                                tmp = tmp + 3
                        return tmp.sum()
                    a = torch.ones(2, dtype=dtype, requires_grad=True)
                    b = torch.ones(2, dtype=dtype, requires_grad=True)
                    err_msg = fn_id_to_inplace_on_view_err_msg[fn_id]
                    if not inplace or not output_is_a_view:
                        gradcheck(fn, (a, b), check_batched_grad=False)
                    bw_called[0] = 0
                    ga_nz[0] = True
                    if inplace and output_is_a_view:
                        with self.assertRaisesRegex(RuntimeError, err_msg):
                            fn(a, b)
                    else:
                        fn(a, b).abs().backward()
                    expected_called = 1
                    expected_ga_nz = True
                    if output_is_a_view and inplace:
                        expected_called = 0
                    self.assertTrue(bw_called[0] == expected_called)
                    self.assertTrue(ga_nz[0] == expected_ga_nz)

    def test_autograd_simple_views_python(self):
        if False:
            print('Hello World!')
        self._do_test_autograd_simple_views_python(torch.double)
        self._do_test_autograd_simple_views_python(torch.cdouble)

    def test_autograd_inplace_views_creation_meta(self):
        if False:
            i = 10
            return i + 15

        class Func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    print('Hello World!')
                return x.view_as(x)

            @staticmethod
            def backward(ctx, x):
                if False:
                    print('Hello World!')
                return x
        view_custom = Func.apply

        def run_test(fn, fn_type, grad_mode_view, grad_mode_iview, requires_grad, error1, error2):
            if False:
                for i in range(10):
                    print('nop')
            base = torch.rand(2, 3, requires_grad=requires_grad).clone()
            with torch.set_grad_enabled(grad_mode_view):
                if fn_type == 'multi_view':
                    inp = base.unbind()[0]
                elif fn_type == 'custom':
                    inp = view_custom(base)
                else:
                    inp = base.view_as(base)
            with torch.set_grad_enabled(grad_mode_iview):
                if error1 is not None:
                    with self.assertRaisesRegex(RuntimeError, error1):
                        fn(inp)
                    return
                else:
                    fn(inp)
            if error2 is not None:
                with self.assertRaisesRegex(RuntimeError, error2):
                    inp.add_(1)
            else:
                inp.add_(1)
        no_grad_err = 'A view was created in no_grad mode'
        multi_view_err = 'function that returns multiple views'
        custom_err = 'view was created inside a custom Function'

        def run_tests(fn):
            if False:
                return 10
            for fn_type in ('normal', 'multi_view', 'custom'):
                for grad_mode_view in (True, False):
                    for grad_mode_iview in (True, False):
                        for requires_grad in (True, False):
                            error1 = None
                            error2 = None
                            if requires_grad:
                                if not grad_mode_view and grad_mode_iview:
                                    error1 = no_grad_err
                                if not grad_mode_view and (not grad_mode_iview):
                                    error2 = no_grad_err
                                if fn_type == 'multi_view':
                                    if grad_mode_view and grad_mode_iview:
                                        error1 = multi_view_err
                                    if grad_mode_view and (not grad_mode_iview):
                                        error2 = multi_view_err
                                if fn_type == 'custom':
                                    if grad_mode_view and grad_mode_iview:
                                        error1 = custom_err
                                    if grad_mode_view and (not grad_mode_iview):
                                        error2 = custom_err
                            run_test(fn, fn_type, grad_mode_view, grad_mode_iview, requires_grad, error1, error2)
        run_tests(lambda v: v.as_strided_((1, 0), (2, 2)))
        run_tests(lambda v: v.transpose_(0, 0))
        run_tests(lambda v: v.t_())
        run_tests(lambda v: v.squeeze_(0))
        run_tests(lambda v: v.unsqueeze_(0))
        run_tests(lambda v: v.swapdims_(0, 0))
        run_tests(lambda v: v.swapaxes_(0, 0))

    def test_autograd_inplace_view_of_view(self):
        if False:
            return 10
        x = torch.zeros(2)
        with torch.no_grad():
            y = x.view(2)
        y.requires_grad_(True)
        z = y.view(2)
        with self.assertRaisesRegex(RuntimeError, 'a view of a view .* is being .* inside the no_grad block'):
            z /= 2
        x = torch.zeros(2)
        with torch.inference_mode():
            y = x.view(2)
        y.requires_grad_(True)
        z = y.view(2)
        with self.assertRaisesRegex(RuntimeError, 'a view of a view .* is being .* inside the inference_mode'):
            z /= 2

    def test_autograd_inplace_views_cross_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        a_orig = torch.rand(3, 3, requires_grad=True, dtype=torch.complex64)
        a = a_orig.clone()
        b = torch.view_as_real(a)
        b = b.transpose(0, 1)
        b += 1
        b.backward(torch.arange(0, 18, dtype=torch.float).view(3, 3, 2))
        non_inplace_grad = a_orig.grad
        a_orig = torch.rand(3, 3, requires_grad=True, dtype=torch.complex64)
        a = a_orig.clone()
        b = torch.view_as_real(a)
        b.transpose_(0, 1)
        b += 1
        b.backward(torch.arange(0, 18, dtype=torch.float).view(3, 3, 2))
        inplace_grad = a_orig.grad
        self.assertEqual(non_inplace_grad.T, inplace_grad)

    def test_autograd_multiple_views_python(self):
        if False:
            i = 10
            return i + 15
        bw_called = [0]

        class ComplexView(Function):

            @staticmethod
            def forward(ctx, a, idx):
                if False:
                    for i in range(10):
                        print('nop')
                res = a.narrow(0, idx, 1)
                res = a.select(0, idx)
                ctx.save_for_backward(a)
                ctx.idx = idx
                return res

            @staticmethod
            def backward(ctx, grad):
                if False:
                    i = 10
                    return i + 15
                bw_called[0] += 1
                (a,) = ctx.saved_tensors
                res = torch.zeros_like(a)
                res.select(0, ctx.idx).copy_(grad)
                return (res, None)
        a = torch.ones(2, requires_grad=True)
        idx = 1
        bw_called[0] = 0
        out = ComplexView.apply(a.clone(), idx)
        out.sum().backward()
        self.assertTrue(bw_called[0] == 1)
        out = ComplexView.apply(a.clone(), idx)
        with self.assertRaisesRegex(RuntimeError, 'Output 0 of ComplexViewBackward is a view and is being modified inplace'):
            out += 1

    def test_autograd_python_custom_function_inplace(self):
        if False:
            for i in range(10):
                print('nop')
        bw_called = [0]

        class MyAdder(Function):

            @staticmethod
            def forward(ctx, a, b):
                if False:
                    print('Hello World!')
                a.add_(b)
                ctx.mark_dirty(a)
                return a

            @staticmethod
            def backward(ctx, grad):
                if False:
                    return 10
                bw_called[0] += 1
                return (grad, grad)
        a = torch.ones(2, requires_grad=True)
        b = torch.ones(2, requires_grad=True)
        c = MyAdder.apply(a.clone(), b)
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)
        bw_called[0] = 0
        c = MyAdder.apply(a.clone(), b)
        c += 2
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)
        bw_called[0] = 0
        c = MyAdder.apply(a.clone().view_as(a), b)
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        class MyAdderBad(Function):

            @staticmethod
            def forward(ctx, a, b):
                if False:
                    for i in range(10):
                        print('nop')
                c = 3 * a
                c.add_(b)
                ctx.mark_dirty(c)
                return c

            @staticmethod
            def backward(ctx, grad):
                if False:
                    print('Hello World!')
                bw_called[0] += 1
                grad = 3 * grad
                return (grad, grad)
        a = torch.ones(2, requires_grad=True)
        b = torch.ones(2, requires_grad=True)
        with warnings.catch_warnings(record=True) as w:
            MyAdderBad.apply(a.clone(), b)
        self.assertEqual(len(w), 1)

        class MyBadAdder(Function):

            @staticmethod
            def forward(ctx, a, b):
                if False:
                    print('Hello World!')
                a.add_(b)
                ctx.mark_dirty(a)
                return (a, a + b)

            @staticmethod
            def backward(ctx, ga, gab):
                if False:
                    return 10
                bw_called[0] += 1
                return (ga + gab, ga + gab)
        bw_called[0] = 0
        (c, d) = MyBadAdder.apply(a.clone(), b)
        (c * d).sum().backward()
        self.assertTrue(bw_called[0] == 1)
        bw_called[0] = 0
        (c, d) = MyBadAdder.apply(a.clone(), b)
        c += 2
        (c * d).sum().backward()
        self.assertTrue(bw_called[0] == 1)
        inplace_on_view_err = 'your Function modifies inplace an input that is a view of another Tensor'
        with self.assertRaisesRegex(RuntimeError, inplace_on_view_err):
            (c, d) = MyBadAdder.apply(a.clone().view_as(a), b)

        class MyOutPlaceAdder(Function):

            @staticmethod
            def forward(ctx, a, b):
                if False:
                    return 10
                a.add_(b)
                ctx.mark_dirty(a)
                return (a.clone(), a + b)

            @staticmethod
            def backward(ctx, ga, gab):
                if False:
                    return 10
                bw_called[0] += 1
                return (ga + gab, ga + 2 * gab)

        def fn(a, b):
            if False:
                while True:
                    i = 10
            orig_a = a.clone().view_as(a)
            (c, d) = MyOutPlaceAdder.apply(orig_a, b)
            return (c * d).sum()
        bad_mark_dirty_err = 'Some elements marked as dirty during the forward method were not returned as output.'
        with self.assertRaisesRegex(RuntimeError, bad_mark_dirty_err):
            fn(a, b)

    def test_custom_function_mark_dirty_not_differentiable(self):
        if False:
            i = 10
            return i + 15

        def get_custom_fn(jvp_err):
            if False:
                return 10

            class InplaceMul(torch.autograd.Function):

                @staticmethod
                def forward(ctx, x):
                    if False:
                        for i in range(10):
                            print('nop')
                    result = x.mul_(2)
                    ctx.mark_dirty(result)
                    return result

                @staticmethod
                def backward(ctx, grad_output):
                    if False:
                        i = 10
                        return i + 15
                    pass

                @staticmethod
                def jvp(ctx, x_t):
                    if False:
                        while True:
                            i = 10
                    if jvp_err:
                        return x_t
                    else:
                        return x_t.mul_(2)
            return InplaceMul
        for (requires_grad, jvp_err) in product([True, False], repeat=2):
            InplaceMul = get_custom_fn(jvp_err)
            z = torch.tensor(1.0, requires_grad=requires_grad)
            x = z.clone()
            y = InplaceMul.apply(x)
            self.assertTrue(x is y)
            self.assertEqual(x, z * 2)
            with fwAD.dual_level():
                x_tangent = torch.ones_like(x)
                x_dual = fwAD.make_dual(x, x_tangent)
                if jvp_err:
                    bad_mark_dirty_err = 'jvp function must modify the corresponding gradient inplace'
                    with self.assertRaisesRegex(RuntimeError, bad_mark_dirty_err):
                        InplaceMul.apply(x_dual)
                else:
                    out_dual = InplaceMul.apply(x_dual)
                    (_, out_tangent) = fwAD.unpack_dual(out_dual)
                    self.assertTrue(out_dual is x_dual)
                    self.assertTrue(out_tangent is x_tangent)

    def test_named_tensor_for_complex_views(self):
        if False:
            i = 10
            return i + 15
        names = ['batch', 'height', 'width', 'complex']
        z = torch.ones((2, 1, 2, 2), requires_grad=True)
        z_named = z.refine_names(*names)
        z_complex = torch.view_as_complex(z_named.rename(None)).refine_names(*names[:-1])
        z_complex.sum().abs().backward()
        expected = torch.ones_like(z_complex).rename(None)
        abs_1_1j = abs(1 + 1j)
        expected.fill_(complex(abs_1_1j / 2, abs_1_1j / 2))
        self.assertEqual(z.grad, torch.view_as_real(expected))

    def test_custom_function_return_view_in_nograd(self):
        if False:
            while True:
                i = 10

        class Alias(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    print('Hello World!')
                return x[:]

            @staticmethod
            def backward(ctx, gx):
                if False:
                    for i in range(10):
                        print('nop')
                return gx
        inp = torch.rand(2, requires_grad=True)
        with torch.no_grad():
            output = Alias.apply(inp)
        with torch.no_grad():
            expected_output = inp[:]
        self.assertEqual(output.requires_grad, expected_output.requires_grad)
        leaf_grad_err = 'A view was created in no_grad mode and is being modified inplace'
        with self.assertRaisesRegex(RuntimeError, leaf_grad_err):
            output.zero_()

    def test_custom_function_preserve_torch_function_when_return_as_is(self):
        if False:
            print('Hello World!')

        class Custom(torch.Tensor):

            def __init__(self, data):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self._data = data

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if False:
                    for i in range(10):
                        print('nop')
                kwargs = {} if kwargs is None else kwargs
                args = tuple((a._data if isinstance(a, cls) else a for a in args))
                out = func(*args, **kwargs)
                if isinstance(out, torch.Tensor):
                    out = cls(out)
                return out

        class Fn(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input):
                if False:
                    for i in range(10):
                        print('nop')
                return input

            @staticmethod
            def backward(ctx):
                if False:
                    i = 10
                    return i + 15
                pass
        x = Custom(torch.randn(2, 3))
        y = Fn.apply(x)
        self.assertTrue(isinstance(y, Custom))

    def test_grad_mode_restored_reentrant(self):
        if False:
            i = 10
            return i + 15

        class MyFunction(Function):

            @staticmethod
            def forward(ctx, inp):
                if False:
                    while True:
                        i = 10
                return inp.clone()

            @staticmethod
            def backward(ctx, go):
                if False:
                    for i in range(10):
                        print('nop')
                original = torch._C.is_grad_enabled()
                with torch.enable_grad():
                    self.assertTrue(torch._C.is_grad_enabled())
                    foo = torch.rand(go.size(), requires_grad=True)
                    (grad,) = torch.autograd.grad(foo ** 3, foo, grad_outputs=go)
                    self.assertTrue(torch._C.is_grad_enabled())
                self.assertTrue(torch._C.is_grad_enabled() == original)
                return grad
        inp = torch.rand(3, requires_grad=True)
        MyFunction.apply(inp).sum().backward()
        MyFunction.apply(inp).sum().backward(create_graph=True)

    def test_power_function(self):
        if False:
            return 10
        a = torch.tensor([0.0, 0.0, 0.0])
        b = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        c = torch.sum(a ** b)
        c.backward()
        self.assertEqual(b.grad, torch.tensor([-inf, 0.0, 0.0]))
        s = 0
        b = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        c = torch.sum(s ** b)
        c.backward()
        self.assertEqual(b.grad, torch.tensor([-inf, 0.0, 0.0]))

    def test_custom_function_error(self):
        if False:
            return 10

        class BadFw(Function):

            @staticmethod
            def backward(ctx, foo):
                if False:
                    i = 10
                    return i + 15
                return foo

        class BadBw(Function):

            @staticmethod
            def forward(ctx, foo):
                if False:
                    i = 10
                    return i + 15
                return foo.clone()

        class BadBw2(Function):

            @staticmethod
            def forward(ctx, foo):
                if False:
                    i = 10
                    return i + 15
                return foo.clone()

            @staticmethod
            def backward(ctx, foo):
                if False:
                    return 10
                return foo

            @staticmethod
            def vjp(ctx, foo):
                if False:
                    i = 10
                    return i + 15
                return foo

        class BadJvp(Function):

            @staticmethod
            def forward(ctx, foo):
                if False:
                    for i in range(10):
                        print('nop')
                return foo.clone()
        inp = torch.rand(1, requires_grad=True)
        with self.assertRaisesRegex(NotImplementedError, 'must implement the forward'):
            BadFw.apply(inp)
        with self.assertRaisesRegex(RuntimeError, 'must implement either the backward'):
            BadBw.apply(inp).sum().backward()
        with self.assertRaisesRegex(RuntimeError, "Implementing both 'backward' and 'vjp'"):
            BadBw2.apply(inp).sum().backward()
        with self.assertRaisesRegex(RuntimeError, 'must implement the jvp function'):
            with fwAD.dual_level():
                d = fwAD.make_dual(inp, torch.rand_like(inp))
                res = BadJvp.apply(d)

    def test_custom_function_forward_mode_view_checks(self):
        if False:
            i = 10
            return i + 15
        flag_to_error = {'ok': None, 'not_a_view': 'jvp is not returning a view', 'not_a_view_of_inp': 'jvp is not returning a view of the given', 'not_a_view_of_inp_base': 'jvp is not returning a view of the same base'}

        class ViewFn(Function):

            @staticmethod
            def forward(ctx, foo, flag):
                if False:
                    for i in range(10):
                        print('nop')
                ctx.flag = flag
                ctx.size = foo.size()
                return foo.narrow(0, 0, 2)

            @staticmethod
            def vjp(ctx, gO):
                if False:
                    while True:
                        i = 10
                gI = gO.new_zeros(ctx.size)
                gI.narrow(0, 0, 2).copy_(gO)
                return (gI, None)

            @staticmethod
            def jvp(ctx, gI, _):
                if False:
                    for i in range(10):
                        print('nop')
                res = gI.narrow(0, 0, 2)
                if ctx.flag != 'ok':
                    res = res.clone()
                if ctx.flag in ['not_a_view_of_inp', 'not_a_view_of_inp_base']:
                    res = res.view_as(res)
                return res
        inp = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        for (flag, msg) in flag_to_error.items():

            def test_fn(inp):
                if False:
                    return 10
                if flag == 'not_a_view_of_inp_base':
                    inp = inp.view_as(inp)
                return ViewFn.apply(inp, flag)
            if msg is None:
                gradcheck(test_fn, inp, check_forward_ad=True)
            else:
                with self.assertRaisesRegex(RuntimeError, msg):
                    gradcheck(test_fn, inp, check_forward_ad=True)

    def test_custom_function_forward_mode_inplace_checks(self):
        if False:
            print('Hello World!')

        class InplaceFn(Function):

            @staticmethod
            def forward(ctx, foo, flag):
                if False:
                    for i in range(10):
                        print('nop')
                ctx.mark_dirty(foo)
                ctx.flag = flag
                foo.mul_(2)
                return foo

            @staticmethod
            def vjp(ctx, gO):
                if False:
                    return 10
                return (2 * gO, None)

            @staticmethod
            def jvp(ctx, gI, _):
                if False:
                    while True:
                        i = 10
                if ctx.flag:
                    return 2 * gI
                else:
                    gI.mul_(2)
                    return gI
        inp = torch.rand(4, 4, dtype=torch.double, requires_grad=True)

        def test_fn(inp, flag):
            if False:
                for i in range(10):
                    print('nop')
            inp = inp.clone()
            return InplaceFn.apply(inp, flag)
        gradcheck(test_fn, (inp, False), check_forward_ad=True)
        with self.assertRaisesRegex(RuntimeError, 'inplace custom Function is not modifying the forward mode gradients inplace'):
            gradcheck(test_fn, (inp, True), check_forward_ad=True)

    def test_custom_function_forward_mode_wrong_formula(self):
        if False:
            i = 10
            return i + 15

        class UserFn(Function):

            @staticmethod
            def forward(ctx, foo, should_fail):
                if False:
                    while True:
                        i = 10
                ctx.should_fail = should_fail
                return foo * 2

            @staticmethod
            def vjp(ctx, gO):
                if False:
                    while True:
                        i = 10
                return (2 * gO, None)

            @staticmethod
            def jvp(ctx, gI, _):
                if False:
                    print('Hello World!')
                if ctx.should_fail:
                    return 3 * gI
                else:
                    return 2 * gI
        inp = torch.rand(10, dtype=torch.double, requires_grad=True)
        gradcheck(UserFn.apply, (inp, False), check_forward_ad=True)
        with self.assertRaisesRegex(RuntimeError, 'Jacobian computed with forward mode mismatch for output 0'):
            gradcheck(UserFn.apply, (inp, True), check_forward_ad=True)

    def test_custom_function_forward_mode_non_tensor_before_tensor_args(self):
        if False:
            print('Hello World!')

        class MyFn(torch.autograd.Function):

            @staticmethod
            def forward(ctx, nt, x, nt2, y):
                if False:
                    while True:
                        i = 10
                return x * 2 + y * 3

            @staticmethod
            def jvp(ctx, nt, x_t, nt2, y_t):
                if False:
                    print('Hello World!')
                self.assertIsNone(nt)
                self.assertIsNone(nt2)
                return x_t * 2 + y_t * 3
        x = torch.tensor(1.0, dtype=torch.double)
        t = torch.tensor(1.0, dtype=torch.double)
        y = torch.tensor(1.0, dtype=torch.double)
        with fwAD.dual_level():
            dual_x = fwAD.make_dual(x, t)
            MyFn.apply(1, dual_x, 1, y)
        gradcheck(MyFn.apply, (1, x.requires_grad_(True), 1, y.requires_grad_(True)), check_forward_ad=True, check_backward_ad=False, check_batched_grad=False)

    def test_custom_function_forward_mode_forward_is_no_op(self):
        if False:
            while True:
                i = 10
        error_regex = "A custom Function's forward is returning a view \\(or an input as-is\\)"
        return_lambdas = {'view_as': lambda x: x.view_as(x), 'self': lambda x: x, 'mul_by_2': lambda x: x * 2}
        for (k, fn) in return_lambdas.items():

            class MyFn(torch.autograd.Function):

                @staticmethod
                def forward(ctx, x, y):
                    if False:
                        print('Hello World!')
                    return (x + y, x)

                @staticmethod
                def vjp(ctx, gO1, gO2):
                    if False:
                        for i in range(10):
                            print('nop')
                    return (gO1 + gO2, gO1)

                @staticmethod
                def jvp(ctx, x_t, y_t):
                    if False:
                        i = 10
                        return i + 15
                    return (x_t + y_t, fn(x_t))
            a = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
            t = torch.tensor(1.0, dtype=torch.double)
            b = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
            c = torch.tensor(1.0, dtype=torch.double)
            t2 = torch.tensor(1.0, dtype=torch.double)
            d = torch.tensor(1.0, dtype=torch.double)
            with fwAD.dual_level():
                a_dual = fwAD.make_dual(a, t)
                c_dual = fwAD.make_dual(c, t2)
                if k == 'view_as':
                    (_, out2) = MyFn.apply(a_dual, b)
                    self.assertTrue(fwAD.unpack_dual(out2).tangent._base is t)
                    (_, out2) = MyFn.apply(c_dual, d)
                    self.assertTrue(fwAD.unpack_dual(out2).tangent._base is t2)
                else:
                    with self.assertRaisesRegex(RuntimeError, error_regex):
                        MyFn.apply(a_dual, b)
                    with self.assertRaisesRegex(RuntimeError, error_regex):
                        MyFn.apply(c_dual, d)
            if k == 'view_as':
                gradcheck(MyFn.apply, (a, c), check_forward_ad=True)
            else:
                with self.assertRaisesRegex(RuntimeError, error_regex):
                    gradcheck(MyFn.apply, (a, c), check_forward_ad=True)

    def test_custom_function_save_for_forward(self):
        if False:
            i = 10
            return i + 15

        class Func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x: torch.Tensor, y: torch.Tensor, z: int):
                if False:
                    i = 10
                    return i + 15
                ctx.save_for_backward(x, y)
                ctx.save_for_forward(x, y)
                ctx.z = z
                ctx.prod = x * y
                return z * ctx.prod

            @staticmethod
            def jvp(ctx, x_t, y_t, _):
                if False:
                    return 10
                (x_p, y_p) = ctx.saved_tensors
                z = ctx.z
                return z * (y_p * x_t + x_p * y_t)

            @staticmethod
            def vjp(ctx, grad_out):
                if False:
                    print('Hello World!')
                (x, y) = ctx.saved_tensors
                z = ctx.z
                return (z * grad_out * y, z * grad_out * x, None)
        a = torch.tensor(1.0, requires_grad=True, dtype=torch.double)
        t = torch.tensor(1.0, dtype=torch.double)
        b = torch.tensor(2.0, requires_grad=True, dtype=torch.double)
        c = 4
        with fwAD.dual_level():
            a_dual = fwAD.make_dual(a, t)
            out = Func.apply(a_dual, b, c)
            out.backward()
        gradcheck(Func.apply, (a, b, c), check_forward_ad=True)

        class Func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x: torch.Tensor):
                if False:
                    for i in range(10):
                        print('nop')
                ctx.save_for_backward(x)
                return x.clone()

            @staticmethod
            def jvp(ctx, x_t):
                if False:
                    while True:
                        i = 10
                self.assertEqual(len(ctx.saved_tensors), 0)
                return x_t

            @staticmethod
            def vjp(ctx, grad_out):
                if False:
                    for i in range(10):
                        print('nop')
                (x,) = ctx.saved_tensors
                self.assertEqual(len(ctx.saved_tensors), 1)
                return grad_out
        with fwAD.dual_level():
            a_dual = fwAD.make_dual(a, t)
            out = Func.apply(a_dual)
            out.backward()
        gradcheck(Func.apply, (a,), check_forward_ad=True)

    def test_custom_function_forward_mode_non_differentiable(self):
        if False:
            return 10

        class Func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x, y):
                if False:
                    return 10
                out = y.clone()
                ctx.mark_non_differentiable(out)
                return (x.clone(), out)

            @staticmethod
            def jvp(ctx, x_tangent, y_tangent):
                if False:
                    while True:
                        i = 10
                return (x_tangent, None)
        x = torch.tensor(2.0)
        x_tangent = torch.tensor(1.0)
        y = torch.tensor(3.0)
        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            (_, out2_dual) = Func.apply(x_dual, y)
            self.assertEqual(fwAD.unpack_dual(out2_dual).tangent, None)
        y = torch.tensor(3)

        class Func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x, y):
                if False:
                    print('Hello World!')
                return (x.clone(), y.clone())

            @staticmethod
            def jvp(ctx, x_tangent, y_tangent):
                if False:
                    return 10
                self.assertIsNone(y_tangent)
                return (x_tangent, None)
        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            (_, out2_dual) = Func.apply(x_dual, y)
            self.assertEqual(fwAD.unpack_dual(out2_dual).tangent, None)

        class FuncWrong(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x, y):
                if False:
                    return 10
                out = y.clone()
                ctx.mark_non_differentiable(out)
                return (x.clone(), out)

            @staticmethod
            def jvp(ctx, x_tangent, y_tangent):
                if False:
                    i = 10
                    return i + 15
                return (x_tangent, x_tangent.clone())
        with fwAD.dual_level():
            x_dual = fwAD.make_dual(x, x_tangent)
            with self.assertRaisesRegex(RuntimeError, 'You should return None at that position instead'):
                FuncWrong.apply(x_dual, y)

    def test_custom_function_local_inplace(self):
        if False:
            return 10

        class MyFn(torch.autograd.Function):

            @staticmethod
            def forward(ctx, inp, inplace):
                if False:
                    i = 10
                    return i + 15
                view = inp.clone()[:3]
                if inplace:
                    view += 2
                return view

            @staticmethod
            def backward(ctx, grad):
                if False:
                    print('Hello World!')
                return (grad, None)
        base = torch.rand(10, requires_grad=True)
        foo = MyFn.apply(base, False)
        self.assertEqual(foo.grad_fn.__class__.__name__, 'MyFnBackward')
        foo = MyFn.apply(base, True)
        self.assertEqual(foo.grad_fn.__class__.__name__, 'MyFnBackward')

    def test_integer_outputs(self):
        if False:
            i = 10
            return i + 15
        inp = torch.rand(4, requires_grad=True)
        out = inp.argmax()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)
        out = inp.argmin()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)
        out = inp.argsort()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)
        val = torch.rand((), requires_grad=True)
        out = torch.searchsorted(inp, val)
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)
        bins = torch.linspace(0, 1.0, steps=100, requires_grad=True)
        vals = torch.rand(5, 5, requires_grad=True)
        out = torch.bucketize(vals, bins)
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)
        val = torch.empty(5).requires_grad_()
        out = val.count_nonzero()
        self.assertFalse(out.requires_grad)

        def assert_only_first_requires_grad(res):
            if False:
                return 10
            if not isinstance(res, tuple):
                res = (res,)
            self.assertTrue(res[0].requires_grad)
            for out in res[1:]:
                if out is not None:
                    self.assertFalse(out.requires_grad)
        for sort in [True, False]:
            for return_inverse in [True, False]:
                for return_counts in [True, False]:
                    res = torch.unique(inp, sorted=sort, return_inverse=return_inverse, return_counts=return_counts)
                    assert_only_first_requires_grad(res)
                    res = torch.unique(inp, sorted=sort, return_inverse=return_inverse, return_counts=return_counts, dim=0)
                    assert_only_first_requires_grad(res)
                    res = torch.unique_consecutive(inp, return_inverse=return_inverse, return_counts=return_counts)
                    assert_only_first_requires_grad(res)
                    res = torch.unique_consecutive(inp, return_inverse=return_inverse, return_counts=return_counts, dim=0)
                    assert_only_first_requires_grad(res)
                    res = torch._unique(inp, sorted=sort, return_inverse=return_inverse)
                    assert_only_first_requires_grad(res)
                    res = torch._VF.unique_dim(inp, dim=0, sorted=sort, return_inverse=return_inverse, return_counts=return_counts)
                    assert_only_first_requires_grad(res)
                    res = torch._unique2(inp, sorted=sort, return_inverse=return_inverse, return_counts=return_counts)
                    assert_only_first_requires_grad(res)

    def test_custom_function_cycle(self):
        if False:
            while True:
                i = 10

        class MyFn(Function):

            @staticmethod
            def forward(ctx, x, metadata):
                if False:
                    for i in range(10):
                        print('nop')
                x = x.clone()
                ctx.meta = metadata
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                if False:
                    print('Hello World!')
                (x,) = ctx.saved_tensors
                self.assertEqual(x, 3.14)
                self.assertEqual(ctx.meta['foo'], 3.14)
                return (gO * x, None)

        def get_refs(with_backward):
            if False:
                for i in range(10):
                    print('nop')
            a = torch.tensor(3.14, requires_grad=True)
            metadata = {}
            out = MyFn.apply(a, metadata)
            metadata['foo'] = out
            if with_backward:
                out.sum().backward()
                self.assertEqual(a.grad, a)
            return torch._C._WeakTensorRef(out)
        with disable_gc():
            ref = get_refs(False)
            self.assertFalse(ref.expired())
        gc.collect()
        self.assertTrue(ref.expired())
        with disable_gc():
            ref = get_refs(True)
            self.assertFalse(ref.expired())
        gc.collect()
        self.assertTrue(ref.expired())

    def test_create_graph_and_full_backward_hook_cycle(self):
        if False:
            return 10

        class TestCls:
            pass

        def get_ref(input_requires_grad, nb_hooks):
            if False:
                print('Hello World!')
            t = torch.randn(10, requires_grad=input_requires_grad)
            a = torch.tensor(1.0, requires_grad=True)

            class Test(nn.Module):

                def forward(self, x):
                    if False:
                        return 10
                    return x ** 2 * a ** 2
            mod = Test()
            for _ in range(nb_hooks):
                mod.register_full_backward_hook(lambda a, b, c: None)
            tmp = mod(t)
            test = TestCls()
            ref = weakref.ref(test)
            tmp.grad_fn.metadata['a'] = test
            with set_warn_always_context(True):
                with warnings.catch_warnings(record=True) as w:
                    tmp.exp().sum().backward(create_graph=True)
                    self.assertTrue(len(w) == 1)
                    self.assertTrue('Using backward() with create_graph=True' in str(w[0].message))
            a.grad = None
            t.grad = None
            return ref
        for nb_hooks in (1, 2, 3):
            for input_requires_grad in (True, False):
                ref_ = get_ref(input_requires_grad=input_requires_grad, nb_hooks=nb_hooks)
                gc.collect()
                self.assertIsNone(ref_())

    @parametrize('use_custom_function', [True, False])
    @parametrize('use_tensor_hook', [True, False])
    def test_hook_closure_cycle(self, use_custom_function, use_tensor_hook):
        if False:
            print('Hello World!')

        class Function(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                return x

            @staticmethod
            def backward(ctx, grad):
                if False:
                    i = 10
                    return i + 15
                return grad

        class Test:
            pass
        count = [0]

        def scope():
            if False:
                while True:
                    i = 10
            a = torch.tensor(1.0, requires_grad=True)
            if use_custom_function:
                b = Function.apply(a)
            else:
                b = a.clone()
            grad_fn_b = b.grad_fn
            obj = Test()

            def hook(*args):
                if False:
                    return 10
                grad_fn_b
                obj
                count[0] += 1
            if use_tensor_hook:
                b.register_hook(hook)
            else:
                b.grad_fn.register_hook(hook)
            c = b.clone()
            ref = weakref.ref(obj)
            return (c, ref)
        with disable_gc():
            (out, ref) = scope()
            out.backward(retain_graph=True)
            gc.collect()
            out.backward(retain_graph=True)
            self.assertEqual(count[0], 2)
            self.assertIsNotNone(ref())
            del out
            gc.collect()
            self.assertIsNone(ref())

    def test_full_backward_hook_double_backward(self):
        if False:
            return 10
        x = torch.rand(1, requires_grad=True)
        y = torch.rand_like(x)
        func = torch.nn.MSELoss()
        counter = [0]

        def hook(module, grad_input, grad_output):
            if False:
                for i in range(10):
                    print('nop')
            counter[0] += 1
        func.register_full_backward_hook(hook)
        f = func(x, y)
        (gradx_f,) = torch.autograd.grad(f, x, create_graph=True)
        self.assertEqual(counter[0], 1)
        _ = torch.autograd.grad(gradx_f, x)
        self.assertEqual(counter[0], 1)

    def test_input_buffer_accum(self):
        if False:
            while True:
                i = 10
        leaf = torch.rand(2, 2, requires_grad=True)
        ind = torch.tensor([[0, 0]], dtype=torch.long)
        out2 = leaf.gather(0, ind, sparse_grad=True)
        out1 = leaf.clone()
        grad_out1_original = torch.rand_like(out1)
        grad_out1 = grad_out1_original.clone()
        grad_out2 = torch.rand_like(out2)
        torch.autograd.backward((out1, out2), (grad_out1, grad_out2))
        self.assertEqual(grad_out1, grad_out1_original)

    def test_no_unnecessary_unwrapping(self):
        if False:
            i = 10
            return i + 15
        a = torch.randn(5, requires_grad=True)
        a_orig = a.detach().clone()
        b = a * a
        c = a * b
        d = torch.exp(a)
        self.assertIs(b.grad_fn._saved_self, a)
        self.assertIs(b.grad_fn._saved_other, a)
        self.assertIs(c.grad_fn._saved_self, a)
        self.assertIs(c.grad_fn._saved_other, b)
        self.assertEqual(d.grad_fn._saved_result, d)
        self.assertIsNot(d.grad_fn._saved_result, d)
        c.sum().backward()
        with self.assertRaisesRegex(RuntimeError, 'after they have already been freed'):
            c.grad_fn._saved_self
        self.assertEqual(a, a_orig)

    def test_saved_variable_version_counter(self):
        if False:
            for i in range(10):
                print('nop')
        a = torch.rand(2, requires_grad=True)
        b = torch.exp(a)
        b_unpacked = b.grad_fn._saved_result
        self.assertEqual(b, b_unpacked)
        self.assertEqual(b._version, b_unpacked._version)
        with torch.no_grad():
            b += 1
        self.assertEqual(b, b_unpacked)
        self.assertEqual(b._version, b_unpacked._version)

    def test_saved_variable_packing_unpacking_saved_original_with_hooks(self):
        if False:
            while True:
                i = 10

        def test(get_input, is_leaf):
            if False:
                for i in range(10):
                    print('nop')
            a = get_input()
            grad_fn = a.grad_fn
            y = a * a
            y.grad_fn._raw_saved_self.register_hooks(lambda x: 2 * x, lambda x: x / 2)
            self.assertEqual(a, y.grad_fn._saved_self)
            if not is_leaf:
                self.assertIs(grad_fn, y.grad_fn._saved_self.grad_fn)
                y.sum().backward()
            else:
                y.sum().backward()
                self.assertEqual(2 * a, a.grad)
            a = get_input()
            grad_fn = a.grad_fn
            y = a * a
            y.grad_fn._raw_saved_self.register_hooks(lambda x: 2 * x, lambda x: x)
            self.assertEqual(2 * a, y.grad_fn._saved_self)
            if not is_leaf:
                self.assertIs(grad_fn, y.grad_fn._saved_self.grad_fn)
                y.sum().backward()
            else:
                y.sum().backward()
                self.assertEqual(3 * a, a.grad)
            a = get_input()
            grad_fn = a.grad_fn
            y = a ** 3
            y.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: x)
            s = torch.sum(y)
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            if not is_leaf:
                self.assertIs(grad_fn, y.grad_fn._saved_self.grad_fn)
                g.sum().backward()
            else:
                g.sum().backward()
                self.assertEqual(6 * a, a.grad)
            a = get_input()
            y = a * a
            y.grad_fn._raw_saved_self.register_hooks(lambda x: x, lambda x: 1)
            with self.assertRaisesRegex(TypeError, 'Output of saved tensor unpack_hook expected to be a Tensor'):
                print(y.grad_fn._saved_self)
            a = get_input()
            y = a * a
            with self.assertRaisesRegex(TypeError, 'missing 1 required positional argument'):
                y.grad_fn._raw_saved_self.register_hooks(lambda x, b: x, lambda x: x)
            a = get_input()
            y = a * a
            with self.assertRaisesRegex(TypeError, 'missing 1 required positional argument'):
                y.grad_fn._raw_saved_self.register_hooks(lambda x, b: (x, b), lambda x: x)

            def inplace_double(x):
                if False:
                    return 10
                x *= 2
                return x
            a = get_input()
            t = a * a
            with self.assertRaisesRegex(RuntimeError, 'A saved tensor pack hook is modifying its input in place.'):
                t.grad_fn._raw_saved_self.register_hooks(inplace_double, lambda x: x / 2)
        test(lambda : torch.randn(5, requires_grad=True), True)
        test(lambda : 1 + torch.randn(5, requires_grad=True), False)

    def test_saved_variable_saved_original_inplace_detach(self):
        if False:
            return 10
        a = torch.tensor(1.0, requires_grad=True).clone()
        b = a.sin()
        a.detach_()
        with self.assertRaisesRegex(RuntimeError, 'Trying to use a saved tensor that has been detached'):
            b.backward()
        a = torch.tensor(1.0, requires_grad=True).clone()
        b = a.exp()
        a.detach_()
        b.backward()

    def test_saved_variable_packing_unpacking_did_not_save_original_with_hooks(self):
        if False:
            for i in range(10):
                print('nop')
        a = torch.randn(5, requires_grad=True)
        y = torch.exp(a)
        y.grad_fn._raw_saved_result.register_hooks(lambda x: x, lambda x: x)
        self.assertEqual(y, y.grad_fn._saved_result)
        self.assertIs(y.grad_fn, y.grad_fn._saved_result.grad_fn)
        y.sum().backward()
        self.assertEqual(a.grad, y)

    def test_saved_variable_packing_unpacking_saved_original_with_default_hooks(self):
        if False:
            print('Hello World!')

        def pack(x):
            if False:
                for i in range(10):
                    print('nop')
            warnings.warn('pack')
            return x
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            a = torch.ones(5, requires_grad=True)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                y = a * a
                self.assertEqual(len(w), 2)
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = a * a
            self.assertEqual(a, y.grad_fn._saved_self)
            self.assertEqual(a, y.grad_fn._saved_other)
            y.sum().backward()
            self.assertEqual(2 * a, a.grad)
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x / 2):
            a = torch.randn(5, requires_grad=True)
            y = a * a
            self.assertEqual(a, y.grad_fn._saved_self)
            self.assertEqual(a, y.grad_fn._saved_other)
            y.sum().backward()
            self.assertEqual(2 * a, a.grad)
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = a * a
            self.assertEqual(2 * a, y.grad_fn._saved_self)
            self.assertEqual(2 * a, y.grad_fn._saved_other)
            y.sum().backward()
            self.assertEqual(4 * a, a.grad)
        a = torch.randn(5, requires_grad=True)
        y = a * a
        self.assertEqual(a, y.grad_fn._saved_self)
        self.assertEqual(a, y.grad_fn._saved_other)
        y.sum().backward()
        self.assertEqual(2 * a, a.grad)

    def test_saved_variable_packing_unpacking_did_not_save_original_with_default_hooks(self):
        if False:
            return 10
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = torch.exp(a)
            self.assertEqual(y, y.grad_fn._saved_result)
            y.sum().backward()
            self.assertEqual(a.grad, y)

    def test_setting_default_saved_variable_hooks_twice_should_not_fail(self):
        if False:
            return 10
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
                pass

    def test_setting_default_saved_variable_hooks_twice_should_use_inner(self):
        if False:
            return 10
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 3 * x, lambda x: 3 * x):
            b = torch.randn(5, requires_grad=True)
            with torch.autograd.graph.saved_tensors_hooks(lambda x: 5 * x, lambda x: 5 * x):
                a = torch.randn(5, requires_grad=True)
                y = a * a
            z = b * b
        y.sum().backward()
        z.sum().backward()
        self.assertEqual(2 * 5 * 5 * a, a.grad)
        self.assertEqual(2 * 3 * 3 * b, b.grad)

    def test_disabling_saved_tensor_hooks(self):
        if False:
            print('Hello World!')
        with torch.autograd.graph.disable_saved_tensors_hooks('error message'):
            with self.assertRaisesRegex(RuntimeError, 'error message'):
                with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
                    pass
        self.assertTrue(torch._C._autograd._saved_tensors_hooks_is_enabled())
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            with self.assertRaisesRegex(RuntimeError, 'error message'):
                with torch.autograd.graph.disable_saved_tensors_hooks('error message'):
                    pass
        self.assertTrue(torch._C._autograd._saved_tensors_hooks_is_enabled())

    def test_disabling_saved_tensor_hooks_nested(self):
        if False:
            while True:
                i = 10
        with torch.autograd.graph.disable_saved_tensors_hooks('outer'):
            with torch.autograd.graph.disable_saved_tensors_hooks('inner'):
                with self.assertRaisesRegex(RuntimeError, 'inner'):
                    with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
                        pass
            self.assertFalse(torch._C._autograd._saved_tensors_hooks_is_enabled())
        self.assertTrue(torch._C._autograd._saved_tensors_hooks_is_enabled())

    def test_saved_tensor_hooks_custom_error_propagaation(self):
        if False:
            while True:
                i = 10

        class CustomError(Exception):
            pass

        class error_on_pack_hook(torch.autograd.graph.saved_tensors_hooks):

            def __init__(self):
                if False:
                    while True:
                        i = 10

                def pack_hook(x):
                    if False:
                        i = 10
                        return i + 15
                    raise CustomError('pack')
                super().__init__(pack_hook, lambda x: x)

        class error_on_unpack_hook(torch.autograd.graph.saved_tensors_hooks):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')

                def unpack_hook(x):
                    if False:
                        i = 10
                        return i + 15
                    raise CustomError('unpack')
                super().__init__(lambda x: x, unpack_hook)
        a = torch.tensor(1.0, requires_grad=True)
        with error_on_pack_hook():
            with self.assertRaisesRegex(CustomError, 'pack'):
                out = torch.sin(a)
        with error_on_unpack_hook():
            out = torch.sin(a)
            with self.assertRaisesRegex(CustomError, 'unpack'):
                out.backward()

    def test_saved_tensor_hooks_custom_function_intermediates(self):
        if False:
            i = 10
            return i + 15

        class Func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                intermediate = x.exp()
                ctx.save_for_backward(intermediate.clone().detach_().requires_grad_(True))
                return x.exp()

            @staticmethod
            def backward(ctx, grad_out):
                if False:
                    while True:
                        i = 10
                (intermediate,) = ctx.saved_tensors
                return grad_out * intermediate
        a = torch.tensor(1.0, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            out = Func.apply(a)
        out.backward()

    def test_save_on_cpu_and_checkpoint(self):
        if False:
            print('Hello World!')
        a = torch.randn(2, 2, requires_grad=True)
        b = a.pow(2).pow(2).pow(2).pow(2)
        b.sum().backward()
        b_grad = a.grad.clone()
        a.grad.zero_()
        with torch.autograd.graph.save_on_cpu():
            h = a.pow(2)
            h = checkpoint(lambda x: x.pow(2).pow(2), h, use_reentrant=False)
            c = h.pow(2)
        c.sum().backward()
        c_grad = a.grad.clone()
        a.grad.zero_()

        def f(a):
            if False:
                return 10
            h = a.pow(2)
            with torch.autograd.graph.save_on_cpu():
                h = h.pow(2).pow(2)
            return h.pow(2)
        d = checkpoint(f, a, use_reentrant=False)
        d.sum().backward()
        d_grad = a.grad.clone()
        self.assertEqual(b_grad, c_grad)
        self.assertEqual(b_grad, d_grad)

    def test_pack_hook_with_inplace_modification_should_fail(self):
        if False:
            print('Hello World!')
        a = torch.randn(5, requires_grad=True)

        def inc(x):
            if False:
                return 10
            x += 1
            return x
        with torch.autograd.graph.saved_tensors_hooks(inc, lambda x: x):
            with self.assertRaisesRegex(RuntimeError, 'A saved tensor pack hook is modifying its input in place.'):
                y = torch.exp(a)
        y = torch.exp(a)
        with self.assertRaisesRegex(RuntimeError, 'A saved tensor pack hook is modifying its input in place.'):
            y.grad_fn._raw_saved_result.register_hooks(inc, lambda x: x)

    def test_saving_variable_to_disk(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as tmp_dir:

            def pack(x):
                if False:
                    return 10
                name = os.path.join(tmp_dir, str(uuid.uuid4()))
                torch.save(x, name)
                return name

            def unpack(name):
                if False:
                    print('Hello World!')
                return torch.load(name)
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                a = torch.ones(5, requires_grad=True)
                y = a * a
                self.assertEqual(a, y.grad_fn._saved_self)
                y.sum().backward()
                self.assertEqual(2 * a, a.grad)

    def test_default_saved_variable_hooks_double_backward(self):
        if False:
            return 10
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = a ** 3
            s = torch.sum(y)
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            g.sum().backward()
            self.assertEqual(6 * a, a.grad)
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = a ** 3
            s = torch.sum(y)
        (g,) = torch.autograd.grad(s, (a,), create_graph=True)
        g.sum().backward()
        self.assertEqual(6 * 2 * a, a.grad)
        a = torch.randn(5, requires_grad=True)
        y = a ** 3
        s = torch.sum(y)
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            g.sum().backward()
            self.assertEqual(6 * 4 * a, a.grad)
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            a = torch.randn(5, requires_grad=True)
            y = a ** 3
            s = torch.sum(y)
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            g.sum().backward()
            self.assertEqual(6 * 8 * a, a.grad)

    def test_wrapped_number_saved_variable_hooks(self):
        if False:
            return 10

        def err_hook(x):
            if False:
                print('Hello World!')
            raise RuntimeError('this hook should not be called')
        with torch.autograd.graph.saved_tensors_hooks(err_hook, err_hook):
            a = torch.randn(5, requires_grad=True)
            out = (a * 3).sum()
            torch.autograd.grad(out, (a,))

    def test_graph_save_on_cpu(self):
        if False:
            print('Hello World!')

        def test(get_input, cuda, pin_memory):
            if False:
                for i in range(10):
                    print('nop')
            with torch.autograd.graph.save_on_cpu(pin_memory):
                a = get_input()
                if cuda:
                    a.cuda()
                y = a * a
                self.assertEqual(a, y.grad_fn._saved_self)
                self.assertEqual(a, y.grad_fn._saved_other)
                self.assertEqual(a.dtype, y.grad_fn._saved_self.dtype)
                self.assertEqual(a.layout, y.grad_fn._saved_self.layout)
                if y.is_sparse:
                    y = y.to_dense()
                y.sum().backward()
                actual = 2 * a
                expected = a.grad
                if a.is_sparse:
                    actual = actual.coalesce()
                    expected = expected.coalesce()
                self.assertEqual(actual, expected)
        for cuda in [False] + ([True] if torch.cuda.is_available() else []):
            for pin_memory in [True, False]:
                test(lambda : torch.randn(5, requires_grad=True), cuda, pin_memory)
                test(lambda : torch.randn(5, requires_grad=True, dtype=torch.double), cuda, pin_memory)
                x = torch.sparse_coo_tensor(torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0]), requires_grad=True)
                test(lambda : x, cuda, pin_memory)

    @unittest.skipIf(not TEST_CUDA, 'test requires CUDA')
    def test_graph_save_on_cpu_cuda(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                print('Hello World!')
            a = x + 1
            return a * a
        a = torch.ones(1, requires_grad=True, device='cuda')
        y = f(a)
        memory_with_grad = torch.cuda.memory_allocated()
        del a
        del y
        a = torch.ones(1, requires_grad=True, device='cuda')
        with torch.no_grad():
            y = f(a)
        memory_without_grad = torch.cuda.memory_allocated()
        self.assertGreater(memory_with_grad, memory_without_grad)
        del a
        del y
        with torch.autograd.graph.save_on_cpu():
            a = torch.ones(1, requires_grad=True, device='cuda')
            y = f(a)
            memory_with_hooks = torch.cuda.memory_allocated()
            self.assertEqual(memory_with_hooks, memory_without_grad)

    def test_multi_grad_hooks(self):
        if False:
            print('Hello World!')
        t1 = torch.rand(2, requires_grad=True)
        t2 = torch.rand(2, requires_grad=True)
        t3 = torch.rand(2, requires_grad=True)
        t4 = torch.rand(2, requires_grad=True)
        res = [None] * 4
        count = [0]

        def hook(grads):
            if False:
                return 10
            nonlocal res
            count[0] += 1
            res = [g is not None for g in grads]
        handle = torch.autograd.graph.register_multi_grad_hook((t1, t2, t3, t4), hook)
        out = t2 * t3
        out.sum().backward(inputs=(t2, t3), retain_graph=True)
        self.assertEqual(count[0], 1)
        self.assertEqual(res, [False, True, True, False])
        out.sum().backward(inputs=(t1, t4), retain_graph=True)
        self.assertEqual(count[0], 1)
        out.sum().backward(inputs=(t1, t3), retain_graph=True)
        self.assertEqual(count[0], 2)
        self.assertEqual(res, [False, False, True, False])

        class Func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x

            @staticmethod
            def backward(ctx, gO):
                if False:
                    print('Hello World!')
                raise RuntimeError('error message')
        out = Func.apply(t2) * t3
        with self.assertRaisesRegex(RuntimeError, 'error message'):
            out.sum().backward(inputs=(t2, t3), retain_graph=True)
        self.assertEqual(count[0], 2)
        handle.remove()
        out.sum().backward(inputs=(t1, t3), retain_graph=True)
        self.assertEqual(count[0], 2)

    def test_pynode_destruction_deadlock(self):
        if False:
            return 10
        script = "\nimport torch\n\nclass Foo(torch.autograd.Function):\n    @staticmethod\n    def forward(ctx, x):\n        return x.clone()\n\n    @staticmethod\n    def forward(ctx, gO):\n        return gO.clone()\n\ndef get_out():\n    inp = torch.rand(2, requires_grad=True)\n\n    # The python function is first so that it runs\n    # last in the backward pass\n    right = Foo.apply(inp)\n\n    # An op that creates new memory\n    left1 = inp.clone()\n    # An op that saves its input\n    left2 = left1 ** 2\n\n    # Inplace modify so that the backward for\n    # left2 always raises an error\n    left1 += 1\n\n    # An op that takes both side as input.\n    # After running, both side's last op will be in\n    # the ready queue\n    # And the op for left will run first as it was\n    # executed last during the forward\n    out = left2 + right\n\n    return out\n\n# Nothing should be global variables here as, from what\n# I can see, python leaks all the global objects\nget_out().sum().backward()\n\n# This used to deadlock when the PyNode is being destroyed after\n# the error is raised.\n"
        try:
            subprocess.check_output([sys.executable, '-c', script], stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.realpath(__file__)), timeout=20)
        except subprocess.TimeoutExpired as e:
            self.fail(msg='Example code timed out! See the code sample in the test for details.')
        except subprocess.CalledProcessError as e:
            err_msg = 'RuntimeError: one of the variables needed for gradient computation'
            self.assertTrue(err_msg in e.output.decode('utf-8'))

    def test_view_func_replay(self):
        if False:
            return 10

        def _assert_match_metadata(a, b):
            if False:
                while True:
                    i = 10
            self.assertEqual(a.size(), b.size())
            self.assertEqual(a.stride(), b.stride())
            self.assertEqual(a.storage_offset(), b.storage_offset())

        def _test_op(fn, inp, args):
            if False:
                return 10
            out = fn(inp, *args)
            self.assertTrue(out._is_view)
            self.assertTrue(out._base is inp)
            new_inp = inp.clone()
            _assert_match_metadata(new_inp, inp)
            new_out = out._view_func(new_inp)
            _assert_match_metadata(new_out, out)
        _test_op(torch.select, torch.rand(2, 2), (0, 0))
        _test_op(torch.as_strided, torch.rand(2, 2), ((4,), (1,)))
        _test_op(torch.view_as_complex, torch.rand(2, 2), ())
        _test_op(torch.view_as_real, torch.rand(2, 2, dtype=torch.cfloat), ())

    def test_setup_context_when_forward_has_default_args(self):
        if False:
            for i in range(10):
                print('nop')

        class PowFunction(Function):

            @staticmethod
            def forward(x, y=3):
                if False:
                    i = 10
                    return i + 15
                return torch.pow(x, y)

            @staticmethod
            def setup_context(ctx, inputs, output):
                if False:
                    while True:
                        i = 10
                (x, y) = inputs
                ctx.save_for_backward(x)
                ctx.y = y

            @staticmethod
            def backward(ctx, gO):
                if False:
                    while True:
                        i = 10
                (x,) = ctx.saved_tensors
                y = ctx.y
                return (gO * y * torch.pow(x, y - 1), None)

        class PowFunctionWithClassmethod(Function):

            @classmethod
            def forward(cls, x, y=3):
                if False:
                    return 10
                return torch.pow(x, y)

            @classmethod
            def setup_context(cls, ctx, inputs, output):
                if False:
                    for i in range(10):
                        print('nop')
                (x, y) = inputs
                ctx.save_for_backward(x)
                ctx.y = y

            @classmethod
            def backward(cls, ctx, gO):
                if False:
                    while True:
                        i = 10
                (x,) = ctx.saved_tensors
                y = ctx.y
                return (gO * y * torch.pow(x, y - 1), None)
        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(8.0)
        y_expected = torch.tensor(12.0)
        y1 = PowFunction.apply(x)
        (y1_expected,) = torch.autograd.grad(y1, x)
        y2 = PowFunctionWithClassmethod.apply(x)
        (y2_expected,) = torch.autograd.grad(y2, x)
        self.assertEqual(y, y1)
        self.assertEqual(y_expected, y1_expected)
        self.assertEqual(y, y2)
        self.assertEqual(y_expected, y2_expected)

def index_perm_variable(shape, max_indices):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(shape, tuple):
        shape = (shape,)
    index = torch.randperm(max_indices).narrow(0, 0, reduce(mul, shape)).view(shape)
    return index

def bernoulli_scalar():
    if False:
        print('Hello World!')
    return torch.tensor(0, dtype=torch.uint8).bernoulli_()

class TestAutogradForwardModeBatchedGrad(TestCase):

    def test_out_of_place_basic(self):
        if False:
            i = 10
            return i + 15
        a = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        b = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        self.assertTrue(gradcheck(torch.sin, a, check_forward_ad=True, check_batched_grad=True, check_batched_forward_grad=True))
        self.assertTrue(gradcheck(torch.add, (a, b), check_forward_ad=True, check_batched_grad=True, check_batched_forward_grad=True))

    def test_out_of_place_not_same_layout(self):
        if False:
            i = 10
            return i + 15
        input = torch.zeros([2, 2]).transpose(0, 1)
        tangent = torch.zeros([2, 2, 2])

        def jvp(tangent):
            if False:
                print('Hello World!')
            with fwAD.dual_level():
                x = fwAD.make_dual(input, tangent)
                return fwAD.unpack_dual(x)[1]
        x_tangent = torch._vmap_internals._vmap(jvp, 0, 0)(tangent)
        self.assertIsNot(x_tangent, tangent)

    def test_inplace_on_view_same_layout(self):
        if False:
            print('Hello World!')
        input = torch.zeros([2, 2])
        tangent = torch.zeros([2, 2, 2])
        base = torch.zeros([2, 2])
        view = base.view_as(base)

        def jvp(tangent):
            if False:
                while True:
                    i = 10
            with fwAD.dual_level():
                x = fwAD.make_dual(input, tangent)
                view.copy_(x)
                return (fwAD.unpack_dual(x)[1], fwAD.unpack_dual(view)[1], fwAD.unpack_dual(view._base)[1])
        (x_tangent, view_tangent, base_tangent) = torch._vmap_internals._vmap(jvp, 0, 0)(tangent)
        self.assertFalse(view_tangent._is_view())
        self.assertIs(view_tangent, base_tangent)
        self.assertIs(x_tangent, tangent)

    def test_inplace_on_view_not_same_layout(self):
        if False:
            print('Hello World!')
        input = torch.zeros([2, 2])
        tangent = torch.zeros([2, 2, 2])
        view = torch.zeros([2, 2]).transpose(0, 1)

        def jvp(tangent):
            if False:
                return 10
            with fwAD.dual_level():
                x = fwAD.make_dual(input, tangent)
                view.copy_(x)
                return (fwAD.unpack_dual(x)[1], fwAD.unpack_dual(view)[1], fwAD.unpack_dual(view._base)[1])
        (x_tangent, view_tangent, base_tangent) = torch._vmap_internals._vmap(jvp, 0, 0)(tangent)
        self.assertIs(view_tangent._base, base_tangent)
        self.assertIs(x_tangent, tangent)
        self.assertIsNot(view_tangent, tangent)

    def test_metadata_check_for_storage_numel_skipped(self):
        if False:
            while True:
                i = 10
        primal = torch.randn(5)[:4].detach()
        self.assertEqual(len(primal.storage()), 5)
        tangent = torch.randn(10, 4)

        def jvp(tangent):
            if False:
                while True:
                    i = 10
            with fwAD.dual_level():
                dual = fwAD.make_dual(primal, tangent)
                (_, unpacked_tangent) = fwAD.unpack_dual(dual)
                self.assertIs(tangent, unpacked_tangent)
                with self.assertRaisesRegex(RuntimeError, 'can access memory outside of `tensor`'):
                    dual.as_strided((5,), (1,), 0)
            return unpacked_tangent
        torch._vmap_internals._vmap(jvp, 0, 0)(tangent)

class TestAutogradForwardMode(TestCase):

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        while fwAD._current_level >= 0:
            fwAD.exit_dual_level()
        super().tearDown()

    def test_forward_level_cleanup(self):
        if False:
            for i in range(10):
                print('nop')

        def get_tensor_and_weak_ref():
            if False:
                while True:
                    i = 10
            t = torch.rand(2, requires_grad=True)
            return (t, torch._C._WeakTensorRef(t))
        (t, t_ref) = get_tensor_and_weak_ref()
        self.assertFalse(t_ref.expired())
        del t
        self.assertTrue(t_ref.expired())
        foo = torch.rand(2)
        with fwAD.dual_level():
            (tangent, tangent_ref) = get_tensor_and_weak_ref()
            self.assertFalse(tangent_ref.expired())
            dual = fwAD.make_dual(foo, tangent)
            self.assertFalse(tangent_ref.expired())
            self.assertTrue(fwAD.unpack_dual(dual)[1] is tangent)
            del tangent
            self.assertFalse(tangent_ref.expired())
            del dual
            self.assertTrue(tangent_ref.expired())

    def test_size_check(self):
        if False:
            while True:
                i = 10
        foo = torch.rand(2)
        tangent = torch.rand(3)
        with fwAD.dual_level():
            with self.assertRaisesRegex(RuntimeError, 'Trying to set a forward gradient that has a different size'):
                dual = fwAD.make_dual(foo, tangent)
            dual = fwAD.make_dual(foo, tangent[1:])

    def test_metadata_check_checks_storage_numel(self):
        if False:
            return 10
        primal = torch.randn(5)[:4].detach()
        self.assertEqual(len(primal.storage()), 5)
        tangent = torch.randn(4)
        with fwAD.dual_level():
            dual = fwAD.make_dual(primal, tangent)
            (_, unpacked_tangent) = fwAD.unpack_dual(dual)
            tangent_clone = tangent.clone()
            unpacked_tangent *= 2
            self.assertTrue(torch.allclose(tangent_clone, tangent))
            dual.as_strided((5,), (1,), 0)

    def test_metadata_check_checks_ignores_size_zero(self):
        if False:
            print('Hello World!')
        a = torch.ones(0).as_strided((0, 1), (1, 1), 0)
        b = torch.ones(0).as_strided((0, 1), (1, 0), 0)
        with fwAD.dual_level():
            dual = fwAD.make_dual(a, b)
            torch.diagonal(dual, offset=0)
        input = torch.rand([0, 1], dtype=torch.complex128, requires_grad=True)
        func = partial(torch.diagonal, offset=0)
        torch.autograd.gradcheck(func, (input,), check_forward_ad=True)

    def test_metadata_check_when_primal_has_conj_bit(self):
        if False:
            while True:
                i = 10
        a = torch.randn(2, 2, dtype=torch.cdouble).conj()
        b = torch.rand_like(a)
        self.assertTrue(torch.is_conj(a))
        self.assertEqual(len(a.storage()), len(b.storage()))
        with fwAD.dual_level():
            dual = fwAD.make_dual(a, b)
            dual[1:]

    def test_metadata_check_when_primal_has_neg_bit(self):
        if False:
            return 10
        a = torch.randn(2, 2, dtype=torch.cdouble).conj().imag
        b = torch.randn(2, 2, dtype=torch.cdouble).imag
        self.assertTrue(torch.is_neg(a))
        self.assertEqual(len(a.storage()), len(b.storage()))
        with fwAD.dual_level():
            dual = fwAD.make_dual(a, b)
            dual[1:]

    def test_metadata_check_check_conj(self):
        if False:
            for i in range(10):
                print('nop')
        keys = {'NEITHER': lambda x: x, 'CONJ': lambda x: x.conj(), 'NEG': lambda x: x._neg_view()}
        for (primal_key, tangent_key) in product(keys, keys):
            x = keys[primal_key](torch.randn(2, 3, 4, dtype=torch.cdouble))
            t = keys[tangent_key](torch.randn(2, 3, 4, dtype=torch.cdouble))
            if primal_key == tangent_key:
                with fwAD.dual_level():
                    dual = fwAD.make_dual(x, t)
                    self.assertTrue(fwAD.unpack_dual(dual).tangent is t)
                    torch.real(dual)
                    torch.imag(dual)
            else:
                with fwAD.dual_level():
                    dual = fwAD.make_dual(x, t)
                    self.assertTrue(fwAD.unpack_dual(dual).tangent is not t)
                    torch.real(dual)
                    torch.imag(dual)

    def test_metadata_check_ignore_storage_offset_for_zero_numel_tensor(self):
        if False:
            return 10
        a = torch.tensor([1.0]).as_strided((0,), (1,), 1)
        b = torch.tensor([1.0]).as_strided((0,), (1,), 2)
        with fwAD.dual_level():
            dual_input = fwAD.make_dual(a, b)
            self.assertIs(fwAD.unpack_dual(dual_input).tangent, b)
        a = torch.tensor([1.0]).as_strided((1,), (2,), 0)
        b = torch.tensor([1.0]).as_strided((1,), (1,), 0)
        with fwAD.dual_level():
            dual_input = fwAD.make_dual(a, b)
            dual_input[1:]

    def test_default_level(self):
        if False:
            return 10
        foo = torch.rand(2)
        bar = torch.rand(2)
        with fwAD.dual_level():
            baz = fwAD.make_dual(foo, bar)
            (baz_primal, baz_tangent) = fwAD.unpack_dual(baz)
        self.assertEqual(baz_primal, foo)
        self.assertIs(baz_tangent, bar)
        (baz_primal, baz_tangent) = fwAD.unpack_dual(baz)
        self.assertEqual(baz_primal, foo)
        self.assertEqual(baz_tangent, None)

    def test_fwd_grad_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        enabled = fwAD._is_fwd_grad_enabled()
        self.assertTrue(enabled)
        try:
            torch._C._set_fwd_grad_enabled(False)
            enabled = fwAD._is_fwd_grad_enabled()
            self.assertFalse(enabled)
        finally:
            torch._C._set_fwd_grad_enabled(True)
        enabled = fwAD._is_fwd_grad_enabled()
        self.assertTrue(enabled)

    def test_set_fwd_grad_enabled(self):
        if False:
            print('Hello World!')
        try:
            torch._C._set_fwd_grad_enabled(False)
            enabled = fwAD._is_fwd_grad_enabled()
            self.assertFalse(enabled)
            with fwAD._set_fwd_grad_enabled(True):
                enabled = fwAD._is_fwd_grad_enabled()
                self.assertTrue(enabled)
            enabled = fwAD._is_fwd_grad_enabled()
            self.assertFalse(enabled)
        finally:
            torch._C._set_fwd_grad_enabled(True)

    def test_nested_level(self):
        if False:
            while True:
                i = 10
        with fwAD.dual_level() as level:
            self.assertEqual(level, 0)
        with fwAD.dual_level():
            with self.assertRaisesRegex(RuntimeError, 'Nested forward mode AD is not supported at the moment'):
                nest_level = fwAD.enter_dual_level()

    def test_set_fw_grad_having_own_fw_grad_at_same_level(self):
        if False:
            while True:
                i = 10
        foo = torch.rand(2)
        bar = torch.rand(2)
        baz = torch.rand(2)
        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)
            with self.assertRaisesRegex(RuntimeError, 'has a forward gradient at the same level'):
                fwAD.make_dual(baz, dual)

    def test_codegen_ignores_undefined_outputs(self):
        if False:
            for i in range(10):
                print('nop')
        weight = torch.randn(6, 1, 30, 30)
        inp = torch.rand((1, 1, 32, 32))
        out = torch.nn.functional.conv2d(inp, weight)
        grad_out = torch.ones_like(out)
        with fwAD.dual_level():
            dual_weight = fwAD.make_dual(weight, torch.ones_like(weight))
            (grad_input, _, _) = torch.ops.aten.convolution_backward(grad_out, inp, dual_weight, (0,), (1, 1), (0, 0), (1, 1), False, (0, 0), 1, (False, True, False))
        self.assertIsNone(grad_input)

    def test_make_dual_inference_tensor_in_inference_mode(self):
        if False:
            print('Hello World!')
        with torch.inference_mode():
            foo = torch.rand(2)
            bar = torch.rand(2)
            foo_copy = foo.clone()
            with fwAD.dual_level():
                dual = fwAD.make_dual(foo, bar)
                self.assertFalse(dual._is_view())
                dual += 1
                self.assertFalse(torch.allclose(foo, foo_copy))

    def test_make_dual_torch_dispatch(self):
        if False:
            print('Hello World!')
        counter = [0]

        class MySubclass(torch.Tensor):

            def __new__(cls, data=None):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.Tensor._make_subclass(cls, data)
            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if False:
                    for i in range(10):
                        print('nop')
                if func.overloadpacket == torch.ops.aten.alias:
                    counter[0] += 1
                    with torch.overrides.enable_reentrant_dispatch():
                        foo = torch.rand(1, requires_grad=True)
                        self.assertIsNotNone(foo.exp().grad_fn)
                with no_dispatch():
                    return func(*args, **kwargs)
        a = torch.tensor(1.0)
        s = MySubclass(a)
        with fwAD.dual_level():
            fwAD.make_dual(s, torch.rand_like(s))
            self.assertEqual(counter[0], 1)
            fwAD.make_dual(torch.rand_like(s), s)
            self.assertEqual(counter[0], 1)

    def test_make_dual_forbid_integral_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        primal_f = torch.ones(2, 2, dtype=torch.float)
        primal_l = torch.ones(2, 2, dtype=torch.long)
        tangent_f = torch.ones(2, 2, dtype=torch.float)
        tangent_l = torch.ones(2, 2, dtype=torch.long)
        with fwAD.dual_level():
            with self.assertRaisesRegex(ValueError, 'Expected tangent to be floating point or complex'):
                fwAD.make_dual(primal_f, tangent_l)
            with self.assertRaisesRegex(ValueError, 'Expected primal to be floating point or complex'):
                fwAD.make_dual(primal_l, tangent_l)
            with self.assertRaisesRegex(ValueError, 'Expected primal to be floating point or complex'):
                fwAD.make_dual(primal_l, tangent_f)

    def test_print(self):
        if False:
            return 10
        with fwAD.dual_level() as level:
            a = torch.rand(3)
            self.assertFalse('tangent=' in str(a))
            b = fwAD.make_dual(a, torch.rand(3))
            self.assertFalse('tangent=' in str(a))
            self.assertTrue('tangent=' in str(b))
            (b_primal, b_tangent) = fwAD.unpack_dual(b)
            self.assertFalse('tangent=' in str(b_primal))
            self.assertFalse('tangent=' in str(b_tangent))

    def test_basic_packing_unpacking(self):
        if False:
            return 10
        foo = torch.rand(2)
        bar = torch.rand(2)
        with fwAD.dual_level():
            baz = fwAD.make_dual(foo, bar)
            (baz_primal, baz_tangent) = fwAD.unpack_dual(baz)
            self.assertEqual(baz_primal, foo)
            self.assertIs(baz_tangent, bar)
            self.assertIsNot(baz_primal, fwAD.unpack_dual(baz).primal)
            self.assertEqual(baz_primal, fwAD.unpack_dual(baz).primal)
            self.assertIs(baz_tangent, fwAD.unpack_dual(baz).tangent)
            (foo_primal, foo_tangent) = fwAD.unpack_dual(foo)
            self.assertEqual(foo_primal, foo)
            self.assertIsNone(foo_tangent)

    def test_advanced_packing_unpacking(self):
        if False:
            i = 10
            return i + 15
        foo = torch.rand(2)
        bar = torch.ones(2)
        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)
            self.assertEqual(dual.storage().data_ptr(), foo.storage().data_ptr())
            self.assertEqual(foo._version, dual._version)
            foo.add_(1)
            self.assertEqual(foo._version, dual._version)
            (dual_primal, dual_tangent) = fwAD.unpack_dual(dual)
            self.assertEqual(dual_primal.storage().data_ptr(), foo.storage().data_ptr())
            self.assertEqual(dual_tangent.storage().data_ptr(), bar.storage().data_ptr())
            self.assertIs(dual_tangent, bar)
            self.assertEqual(foo._version, dual_primal._version)
            foo.add_(1)
            self.assertEqual(foo._version, dual_primal._version)
            self.assertEqual(bar._version, dual_tangent._version)
            bar.add_(1)
            self.assertEqual(bar._version, dual_tangent._version)
        with fwAD.dual_level():
            foo.requires_grad_()
            bar.requires_grad_()
            dual = fwAD.make_dual(foo, bar)
            (p, t) = fwAD.unpack_dual(dual)
            (gfoo, gbar) = torch.autograd.grad(p.sum(), (foo, bar), retain_graph=True, allow_unused=True)
            self.assertEqual(gfoo, torch.ones_like(foo))
            self.assertIsNone(gbar)
            (gfoo, gbar) = torch.autograd.grad(t.sum(), (foo, bar), retain_graph=True, allow_unused=True)
            self.assertIsNone(gfoo)
            self.assertEqual(gbar, torch.ones_like(bar))
            detached_dual = dual.detach()
            out = detached_dual * 2
            (p, t) = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertEqual(p, foo * 2)
            self.assertIsNone(t)
            with torch.no_grad():
                out = dual * 3
            (p, t) = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertFalse(t.requires_grad)
            self.assertEqual(p, foo * 3)
            self.assertEqual(t, bar * 3)
            dual = dual.clone()
            dual.detach_()
            out = dual * 2
            (p, t) = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertEqual(p, foo * 2)
            self.assertIsNone(t)

    def test_view_inplace_non_differentiable_views(self):
        if False:
            print('Hello World!')
        original_foo = torch.rand(2, dtype=torch.double)
        original_bar = torch.ones(2, dtype=torch.double)
        foo = original_foo.clone()
        bar = original_bar.clone()
        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)
            dual *= 2
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            self.assertEqual(bar, original_bar * 2)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 2)
            self.assertEqual(foo, original_foo * 2)
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 2)
            (dual_primal, dual_tangent) = fwAD.unpack_dual(dual)
            self.assertIsNone(fwAD.unpack_dual(dual_primal)[1])
            self.assertIsNone(fwAD.unpack_dual(dual_tangent)[1])
            dual_primal *= 2
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 4)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 2)
            dual_tangent *= 2
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 4)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 4)

    def test_view_inplace_differentiable_views(self):
        if False:
            while True:
                i = 10
        original_foo = torch.rand(2)
        original_bar = torch.ones(2)
        foo = original_foo.clone()
        bar = original_bar.clone()
        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)
            view = dual.narrow(0, 0, 1)
            view *= 2
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            self.assertEqual(fwAD.unpack_dual(dual)[1], torch.tensor([2.0, 1.0]))
            self.assertEqual(fwAD.unpack_dual(view)[1], torch.tensor([2.0]))
            baz = torch.rand(2)
            baz += dual
            self.assertEqual(fwAD.unpack_dual(baz)[1], fwAD.unpack_dual(dual)[1])
            baz = torch.rand(2)
            baz[0] = dual[0]
            self.assertEqual(fwAD.unpack_dual(baz)[1][0], fwAD.unpack_dual(dual)[1][0])
            self.assertEqual(fwAD.unpack_dual(baz)[1][1], 0.0)
            baz = torch.rand(2)
            view = baz.detach()
            view += dual
            self.assertIsNone(fwAD.unpack_dual(baz)[1])

    def test_view_inplace_always_creates_a_view(self):
        if False:
            while True:
                i = 10
        inplace_binary_ops = (lambda x, y: x.add_(y), lambda x, y: x.mul_(y), lambda x, y: x.copy_(y))
        for inplace_binary_op in inplace_binary_ops:
            base = torch.randn(2, 2)
            view = base.transpose(0, 1)
            primal = torch.randn(2, 2)
            tangent = torch.randn(2, 2)
            with fwAD.dual_level():
                dual = fwAD.make_dual(primal, tangent)
                inplace_binary_op(view, dual)
                (p, t) = fwAD.unpack_dual(base)
                p_clone = p.clone()
                t_clone = t.clone()
                view *= 2
                (p, t) = fwAD.unpack_dual(base)
                self.assertTrue(torch.allclose(p_clone * 2, p))
                self.assertTrue(torch.allclose(t_clone * 2, t))

    def test_grad_cleanup(self):
        if False:
            i = 10
            return i + 15
        foo = torch.rand(2)
        bar = torch.rand(2)
        baz = torch.rand(2)
        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            self.assertIs(fwAD.unpack_dual(dual)[1], bar)
        self.assertIsNone(fwAD.unpack_dual(dual)[1])
        with fwAD.dual_level():
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            new_dual = fwAD.make_dual(foo, baz)
            (dual_primal, dual_tangent) = fwAD.unpack_dual(dual)
            (new_dual_primal, new_dual_tangent) = fwAD.unpack_dual(new_dual)
            self.assertEqual(dual_primal, new_dual_primal)
            self.assertIsNone(dual_tangent)
            self.assertEqual(new_dual_tangent, baz)

    def test_detach_view_tracking(self):
        if False:
            for i in range(10):
                print('nop')
        foo = torch.rand(2)
        foo_weak = torch._C._WeakTensorRef(foo)
        out = foo.detach()
        del foo
        self.assertTrue(foo_weak.expired())

    def test_out_variant(self):
        if False:
            for i in range(10):
                print('nop')
        with fwAD.dual_level():
            foo = fwAD.make_dual(torch.rand(2), torch.rand(2))
            bar = torch.rand(2)
            with self.assertRaisesRegex(RuntimeError, 'out= function'):
                torch.add(bar, bar, out=foo)
            with self.assertRaisesRegex(RuntimeError, 'out= function'):
                torch.add(foo, bar, out=bar)

    def test_non_differentiable(self):
        if False:
            while True:
                i = 10
        with fwAD.dual_level():
            foo = fwAD.make_dual(torch.rand(2), torch.rand(2))
            bar = torch.rand(2)
            eq = foo == bar
            foo.eq_(bar)

    def test_create_new_zeros_with_same_meta(self):
        if False:
            return 10
        new_zeroes_fn = torch.ops.aten._new_zeros_with_same_feature_meta

        def check(a, b):
            if False:
                while True:
                    i = 10

            def assert_same_meta(t, target):
                if False:
                    return 10
                for num_bdim in range(t.dim()):
                    result = new_zeroes_fn(t, target, self_num_batch_dims=num_bdim)
                    self.assertEqual(result.dim(), target.dim() + num_bdim)
                    for i in range(num_bdim, result.dim()):
                        self.assertEqual(result.size()[i], target.size()[i - num_bdim])
                        self.assertEqual(result.stride()[i], target.stride()[i - num_bdim])
                    if target.is_contiguous():
                        self.assertTrue(result.is_contiguous())
                    self.assertEqual(result.storage_offset(), target.storage_offset())
                    prod_of_t_bdims = reduce(operator.mul, t.size()[:num_bdim], 1)
                    self.assertEqual(len(result.storage()), len(target.storage()) * prod_of_t_bdims)
                    self.assertEqual(result.dtype, target.dtype)
            assert_same_meta(a, b)
            assert_same_meta(b, a)
        a = torch.randn(5, dtype=torch.float)
        b = torch.randn(2, 3, 4, dtype=torch.double)
        check(a, b)
        a = torch.randn(2, 3, 4).transpose(0, 1).contiguous().transpose(0, 1)
        b = torch.randn(2, 3, 4)
        check(a, b)
        a = torch.randn(5).narrow(0, 1, 2)
        b = torch.randn(2)
        check(a, b)
        a = torch.randn(5).resize_(4)
        b = torch.randn(4)
        check(a, b)
        a = torch.randn(1, 0, 2)
        b = torch.randn(1, 2)
        check(a, b)
        a = torch.tensor(1.0)
        b = torch.randn(1, 2)
        check(a, b)

    def test_backward_graph_destruction(self):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                return 10
            a = torch.rand(10, requires_grad=True)
            da = fwAD.make_dual(torch.rand_like(a), a)
            db = da.exp()
        with fwAD.dual_level():
            fn()

class TestAutogradDeviceType(TestCase):

    def test_min_max_median_backprops_to_all_values(self, device):
        if False:
            return 10
        for f in [torch.min, torch.max, torch.median, torch.nanmedian]:
            x1 = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], device=device, requires_grad=True)
            x2 = torch.tensor([float('nan'), float('nan'), float('nan')], requires_grad=True)
            for x in [x1, x2]:
                y = f(x)
                y.backward()
                self.assertEqual(x.grad.sum(), 1.0)
                self.assertEqual((x.grad == 1 / 3).sum(), 3)

    def test_scatter_index_reduce_amin_amax_backprops_to_all_values(self, device):
        if False:
            print('Hello World!')
        fns = (torch.scatter_reduce, torch.index_reduce)
        reduces = ('amin', 'amax')
        for (fn, reduction) in product(fns, reduces):
            input = torch.randn((2, 3), device=device, dtype=torch.float64, requires_grad=True)
            src = input.clone().detach_().requires_grad_(True)
            idx = torch.arange(2).to(dtype=torch.long, device=device)
            if fn == torch.scatter_reduce:
                idx = idx.unsqueeze(-1).expand((2, 3))
            gradcheck(fn, (input, 0, idx, src, reduction), check_batched_grad=False)

    def test_scatter_index_reduce_prod_gradgrad_error(self, device):
        if False:
            while True:
                i = 10
        input = torch.tensor([1.0], device=device, dtype=torch.float64, requires_grad=True)
        src = torch.tensor([0.0, 0.0], device=device, dtype=torch.float64, requires_grad=True)
        idx = torch.tensor([0, 0], device=device, dtype=torch.long)
        for fn in (torch.scatter_reduce, torch.index_reduce):
            gradcheck(fn, (input, 0, idx, src, 'prod'), check_batched_grad=False)
            with self.assertRaisesRegex(RuntimeError, 'Double backward is unsupported for'):
                gradgradcheck(fn, (input, 0, idx, src, 'prod'))

    @skipIfMps
    def test_parameter_resize(self, device):
        if False:
            i = 10
            return i + 15
        asd = torch.nn.Parameter(torch.ones(16, dtype=torch.double, device=device))
        for i in range(2):
            with torch.no_grad():
                asd.set_(asd[1:])
                asd.grad = None
            m = torch.cat((asd, asd))
            m.sum().backward()

    @skipIfMps
    @dtypes(torch.double, torch.cdouble)
    def test_sparse_ctor_getter_backward(self, device, dtype):
        if False:
            print('Hello World!')

        def _test(size, sparse_dim, nnz, device):
            if False:
                print('Hello World!')
            v_size = [nnz] + list(size[sparse_dim:])
            i = torch.rand(sparse_dim, nnz)
            i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
            i = i.to(torch.long)
            inp = torch.randn(v_size, dtype=torch.double, device=device, requires_grad=True)
            other = self.genSparseTensor(size, sparse_dim, nnz, is_uncoalesced=True, device=device, dtype=dtype)[0]

            def fn(v):
                if False:
                    while True:
                        i = 10
                x = torch.sparse_coo_tensor(i, v, size, dtype=dtype, device=device)
                y = (x + other).coalesce()
                yv = y.values()
                new_v = yv.tanh()
                z = torch.sparse_coo_tensor(y.indices(), new_v, y.size())
                return z.coalesce().values()
            gradcheck(fn, (inp,), check_batched_grad=False)
            with self.assertRaisesRegex(RuntimeError, 'does not have a grad_fn'):
                other.detach().requires_grad_()._values().backward(torch.ones_like(other._values()))
        for (empty_i, empty_v, empty_nnz) in product([True, False], repeat=3):
            sparse_size = [] if empty_i else [2, 1]
            dense_size = [1, 0, 2] if empty_v else [1, 2]
            nnz = 0 if empty_nnz else 5
            _test(sparse_size + dense_size, len(sparse_size), nnz, device)

    @skipMeta
    @skipIfMps
    @dtypes(torch.double, torch.cdouble)
    def test_sparse_backward(self, device, dtype):
        if False:
            i = 10
            return i + 15

        class FixedGradientFunction(Function):

            @staticmethod
            def forward(ctx, x, grad_x):
                if False:
                    i = 10
                    return i + 15
                ctx.save_for_backward(grad_x)
                return x

            @staticmethod
            def backward(ctx, grad_x):
                if False:
                    return 10
                (saved_grad_x,) = ctx.saved_tensors
                return (saved_grad_x, None)
        size = torch.Size([6, 3, 2])
        i1 = torch.tensor([[0, 3, 4], [0, 2, 2]], dtype=torch.long)
        v1 = make_tensor([3, 2], dtype=dtype, device=device)
        sparse_grad1 = torch.sparse_coo_tensor(i1, v1, size, dtype=dtype, device=device)
        i2 = torch.tensor([[0, 1, 3, 4], [0, 1, 2, 2]], dtype=torch.long)
        v2 = make_tensor([4, 2], dtype=dtype, device=device)
        sparse_grad2 = torch.sparse_coo_tensor(i2, v2, size, dtype=dtype, device=device)
        dense_grad = torch.rand(size, device=device, dtype=dtype)
        fn = FixedGradientFunction
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        (fn.apply(x, sparse_grad1) + fn.apply(x, dense_grad) + fn.apply(x, sparse_grad2)).sum().abs().backward()
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        (fn.apply(x, dense_grad) + fn.apply(x, sparse_grad1) + fn.apply(x, sparse_grad2)).sum().abs().backward()
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        (fn.apply(x, sparse_grad1) + fn.apply(x, sparse_grad2)).sum().abs().backward()
        self.assertEqual(x.grad, sparse_grad1 + sparse_grad2)

    @skipIfMps
    def test_sparse_mask_autograd(self, device):
        if False:
            return 10
        tensor = torch.randn(3, requires_grad=True, device=device)
        mask = torch.ones(3, device=device)
        mask[1] = 0
        mask = mask.to_sparse()
        converted = tensor.sparse_mask(mask).to_dense()
        converted.sum().backward()
        self.assertEqual(tensor.grad, mask.to_dense())

    @skipIfMps
    def test_pyscalar_conversions(self, device):
        if False:
            for i in range(10):
                print('nop')

        def _test_pyscalar_conversions(t, integral_conv):
            if False:
                i = 10
                return i + 15
            l = t(torch.zeros(1, 1, 1, dtype=torch.long))
            pyscalar = -12345
            l[0] = pyscalar
            self.assertEqual(integral_conv(l), pyscalar)
            f = Variable(t(torch.randn(1, 1, dtype=torch.double)))
            pyscalar = -12345.1
            f[0] = pyscalar
            self.assertEqual(float(f), pyscalar)
            f[0] = nan
            self.assertTrue(math.isnan(float(f)))
            f[0] = inf
            self.assertEqual(float(f), inf)
            f[0] = -inf
            self.assertEqual(float(f), -inf)
            pyscalar = 1234567890123456789
            self.assertNotEqual(pyscalar, integral_conv(float(pyscalar)))
            l[0] = pyscalar
            self.assertEqual(float(l), float(pyscalar))
            f[0] = nan
            self.assertRaises(ValueError, lambda : integral_conv(f[0]))
            f[0] = inf
            self.assertRaises(OverflowError, lambda : integral_conv(f[0]))
            f[0] = -inf
            self.assertRaises(OverflowError, lambda : integral_conv(f[0]))
            f[0] = sys.float_info.max
            self.assertEqual(integral_conv(f), sys.float_info.max)

            def test_nonzero(tensor, value, expected):
                if False:
                    return 10
                tensor[0] = value
                self.assertEqual(expected, bool(tensor))
                self.assertEqual(expected, True if tensor else False)
            test_nonzero(l, 0, False)
            test_nonzero(l, -2, True)
            test_nonzero(f, 0.0, False)
            test_nonzero(f, sys.float_info.min, True)
            test_nonzero(f, nan, bool(nan))
            test_nonzero(f, inf, bool(inf))
            test_nonzero(f, -inf, bool(-inf))
        _test_pyscalar_conversions(lambda x: x.to(device), lambda x: int(x))

    @dtypesIfMPS(torch.float32)
    @dtypesIfCUDA(torch.half, torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64)
    @dtypes(torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_set_requires_grad_only_for_floats(self, device, dtype):
        if False:
            print('Hello World!')

        def f1():
            if False:
                print('Hello World!')
            a = torch.ones(1, dtype=dtype, device=device)
            a.requires_grad_()

        def f2():
            if False:
                i = 10
                return i + 15
            a = torch.ones(1, dtype=dtype, device=device)
            a.requires_grad = True

        def f3():
            if False:
                print('Hello World!')
            torch.ones(1, dtype=dtype, device=device, requires_grad=True)
        a = torch.ones(1, dtype=dtype, device=device)
        a.requires_grad = False
        a.requires_grad_(False)
        for f in [f1, f2, f3]:
            if dtype.is_floating_point:
                f()
            else:
                with self.assertRaisesRegex(RuntimeError, 'floating point', msg=f'dt: {a.dtype} device: {a.device}'):
                    f()

    @onlyCUDA
    def test_advanced_indexing_backwards_large(self, device):
        if False:
            while True:
                i = 10
        n = 1 << 16
        x = torch.rand(n, 1, device=device, requires_grad=True)
        a = x[:, [0]]
        a.sum().backward()
        self.assertEqual(x.grad, torch.ones(n, 1, device=device))

    def test_advanced_indexing_backwards_memory_format(self, device):
        if False:
            return 10
        shape = (2, 8, 1, 2)
        i = torch.randint(1, shape, device=device).contiguous(memory_format=torch.channels_last)
        x = torch.randn(shape, requires_grad=True, device=device)
        x[i].sum().backward()

    def _test_reentrant_parent_error_on_cpu(self, device):
        if False:
            for i in range(10):
                print('nop')
        t1 = torch.rand([3, 3], requires_grad=True)
        t2 = torch.rand([3, 3], device=device, requires_grad=True)
        t3 = torch.rand([3, 3], device=device, requires_grad=True)
        t4 = t1 * t1
        t5 = TestAutograd.SimulateBackwardError.apply(t4)
        prev = t2 * t2
        for i in range(10):
            prev = prev * t2
        reentrant_root = prev

        class ReentrantFunc(Function):

            @staticmethod
            def forward(ctx, inp):
                if False:
                    i = 10
                    return i + 15
                return inp.clone()

            @staticmethod
            def backward(ctx, grad):
                if False:
                    return 10
                reentrant_root.backward()
                return grad
        t6 = ReentrantFunc.apply(t3)
        t7 = t6 * t6
        with self.assertRaisesRegex(Exception, 'Simulate error'):
            torch.autograd.backward([t5.sum(), t7.sum()])
        self.assertIsNone(t2.grad)
        self.assertIsNone(t1.grad)
        self.assertIsNone(t3.grad)

    @onlyCUDA
    def test_reentrant_parent_error_on_cpu(self, device):
        if False:
            while True:
                i = 10

        def _get_cuda_memory_usage():
            if False:
                return 10
            num_devices = torch.cuda.device_count()
            gc.collect()
            return tuple((torch.cuda.memory_allocated(i) for i in range(num_devices)))
        before = _get_cuda_memory_usage()
        self._test_reentrant_parent_error_on_cpu(device)
        after = _get_cuda_memory_usage()
        start = time.time()
        while before != after and time.time() - start < 30:
            time.sleep(0.1)
            after = _get_cuda_memory_usage()
        self.assertEqual(before, after)

    @skipIfMps
    def test_where_functional(self, device):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        cond = mask_not_all_zeros((5, 5)).to(device=device)

        def where(cond, x, y):
            if False:
                return 10
            return torch.where(cond, x, y)
        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [torch.randn(5, 5, device=device)])
        x = torch.randn(5, 1, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.randn(5, 5, 1, dtype=torch.double, device=device, requires_grad=True)
        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [torch.randn(5, 5, 5, device=device)])

    @skipIfMps
    def test_where_scalar(self, device):
        if False:
            while True:
                i = 10
        x = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        scalar = 4.0
        cond = mask_not_all_zeros((5, 5)).to(device=device)

        def where_scalar_first(cond, x):
            if False:
                while True:
                    i = 10
            return torch.where(cond, scalar, x)

        def where_scalar_second(cond, x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.where(cond, x, scalar)
        gradcheck(where_scalar_first, (cond, x))
        gradgradcheck(where_scalar_first, (cond, x))
        gradcheck(where_scalar_second, (cond, x))
        gradgradcheck(where_scalar_second, (cond, x))

    @onlyCUDA
    def test_free_unneeded_tensor(self, device):
        if False:
            print('Hello World!')
        x = torch.randn(2, 3, 10, 10, device=device, requires_grad=True)
        m = torch.randn(1, 3, 1, 1, device=device)
        z = x.sum()
        base_mem = torch.cuda.memory_allocated()
        z = ((x + 2) * m).sum()
        end_mem = torch.cuda.memory_allocated()
        self.assertEqual(base_mem, end_mem)

    @onlyCUDA
    def test_pin_memory(self, device):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        self.assertEqual(x, x.pin_memory())
        self.assertIsNot(x, x.pin_memory())
        self.assertTrue(x.pin_memory().requires_grad)
        gradcheck(lambda x: x.pin_memory(), [x])
        gradgradcheck(lambda x: x.pin_memory(), [x])

    @onlyCUDA
    def test_profiler_emit_nvtx(self, device):
        if False:
            while True:
                i = 10
        a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
        with torch.cuda.profiler.profile():
            with emit_nvtx():
                a.add(1.0)

    @onlyCUDA
    def test_rnn_backward_to_input_but_not_parameters(self, device):
        if False:
            while True:
                i = 10
        l = torch.nn.LSTM(2, 3).to(device)
        for p in l.parameters():
            p.requires_grad = False
        s = torch.randn(1, 1, 2, requires_grad=True, device=device)
        (out, _) = l(s)
        out.sum().backward()
        self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    @unittest.skipIf(not torch.profiler.itt.is_available(), 'ITT is required')
    def test_profiler_emit_itt(self, device):
        if False:
            i = 10
            return i + 15
        a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
        with emit_itt():
            a.add(1.0)

    @skipIfMps
    @deviceCountAtLeast(1)
    def test_grad_assignment(self, devices):
        if False:
            print('Hello World!')
        x = torch.randn(5, 5, device=devices[0])
        with self.assertRaisesRegex(TypeError, 'expected to be a Tensor or None'):
            x.grad = 0
        with self.assertRaises(RuntimeError):
            x.grad = torch.randn(2, 2, device=devices[0])
        with self.assertRaises(RuntimeError):
            x.grad = torch.randn(5, 5, dtype=torch.long, device=devices[0])
        with self.assertRaises(RuntimeError):
            x.grad = x
        if self.device_type != 'cpu':
            with self.assertRaises(RuntimeError):
                t_cpu = torch.rand(5, 5)
                t_cpu.grad = torch.randn(5, 5, device=devices[0])
        if self.device_type == 'cuda':
            x = x.to(dtype=torch.half, device=devices[0])
            x.grad = torch.zeros_like(x)
        if len(devices) > 1:
            x = torch.randn(5, 5, device=devices[0])
            with self.assertRaises(RuntimeError):
                x.grad = torch.randn(5, 5, device=devices[1])

    @dtypesIfMPS(torch.float32)
    @deviceCountAtLeast(1)
    @dtypes(torch.float, torch.double)
    def test_requires_grad_factory(self, devices, dtype):
        if False:
            return 10
        fns = [torch.ones_like, torch.randn_like]
        x = torch.randn(2, 3, dtype=dtype, device=devices[0])
        for fn in fns:
            for requires_grad in [True, False]:
                output = fn(x, dtype=dtype, device=devices[0], requires_grad=requires_grad)
                self.assertEqual(requires_grad, output.requires_grad)
                self.assertIs(dtype, output.dtype)
                self.assertEqual(devices[0], str(x.device))

    @deviceCountAtLeast(2)
    def test_unused_output_device(self, devices):
        if False:
            for i in range(10):
                print('nop')
        from torch.nn.parallel._functions import Broadcast
        x = torch.randn(5, 5, dtype=torch.float, device=devices[0], requires_grad=True)
        outputs = Broadcast.apply(list(range(len(devices))), x)
        y = outputs[-1] * 2
        y.sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5) * 2)

    @deviceCountAtLeast(2)
    def test_backward_device(self, devices):
        if False:
            for i in range(10):
                print('nop')
        device = [None]

        class Identity(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    print('Hello World!')
                device[0] = grad_output.device
                return grad_output.clone()
        v = torch.randn(1, device=devices[1], requires_grad=True)
        Identity.apply(v).backward()
        self.assertEqual(str(device[0]), devices[1])

    @deviceCountAtLeast(2)
    def test_inputbuffer_add_multidevice(self, devices):
        if False:
            print('Hello World!')
        input = torch.randn(1, device=devices[0], requires_grad=True)
        output = input.to(device=devices[1]) + input.to(device=devices[1])
        output.backward()

    @onlyCPU
    def test_copy_(self, device):
        if False:
            return 10
        x = torch.randn(10, device=device, requires_grad=True)
        floating_dt = floating_types_and(torch.half, torch.bfloat16)
        for dt in floating_dt:
            y = torch.empty(10, device=device, dtype=dt)
            y.copy_(x)
            self.assertTrue(y.requires_grad)
            z = x.to(torch.bfloat16)
            self.assertTrue(z.requires_grad)

    def test_copy_forward_ad_broadcasting(self, device):
        if False:
            return 10
        primal = torch.rand(3, 3, device=device)
        tangent = torch.rand(3, 3, device=device)
        non_dual = torch.rand(1, 3, 3, device=device)
        with fwAD.dual_level():
            dual = fwAD.make_dual(primal, tangent)
            non_dual.copy_(dual)

    def test_copy_forward_ad_same_layout_copies_grad(self, device):
        if False:
            for i in range(10):
                print('nop')
        primal = torch.tensor([[3.0], [4.0]], device=device)
        tangent = torch.tensor([[5.0], [6.0]], device=device)
        with fwAD.dual_level():
            x_dual = fwAD.make_dual(primal, tangent)
            non_dual = torch.tensor([[1.0], [2.0]])
            non_dual.copy_(x_dual)
            self.assertTrue(fwAD.unpack_dual(non_dual).tangent is not tangent)

    @onlyCUDA
    def test_simple_reentrant_cross_device(self, device):
        if False:
            print('Hello World!')

        class ReentrantFunc(Function):
            _cpu_mode = True

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                return x * (x + 2)

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    return 10
                with torch.enable_grad():
                    if ReentrantFunc._cpu_mode:
                        new_param = torch.randn(2, 2, requires_grad=True)
                        (new_param ** 2).sum().backward()
                    else:
                        new_param = torch.randn(2, 2, device=device, requires_grad=True)
                        (new_param ** 2).sum().backward()
                return grad_output
        x = torch.randn(2, 2, device=device, requires_grad=True)
        out = ReentrantFunc.apply(x)
        out.sum().backward()
        x = torch.randn(2, 2, requires_grad=True)
        ReentrantFunc._cpu_mode = False
        out = ReentrantFunc.apply(x)
        out.sum().backward()
        x = torch.randn(2, 2, device=device, requires_grad=True)
        ReentrantFunc._cpu_mode = True
        out = ReentrantFunc.apply(x)
        out.sum().backward()

    @onlyCUDA
    def test_cross_device_reentrant_autograd(self, device):
        if False:
            while True:
                i = 10

        def fn_on_gpu(inp):
            if False:
                while True:
                    i = 10
            dummy = inp * 2 * 2 * 2 * 2
            return inp.to(device=device)

        def parent_on_cpu(inp):
            if False:
                while True:
                    i = 10
            branch1 = inp.to(device=device)
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            branch2 = checkpoint(fn_on_gpu, inp, use_reentrant=True)
            out = branch2 + branch1
            return out
        inp = torch.rand(2, requires_grad=True)
        out = parent_on_cpu(inp)
        out.sum().backward()

    def test_inplace_on_view_backprop_base(self, device):
        if False:
            return 10
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v1.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.tolist(), [[2, 2], [1, 1]])

    def test_inplace_on_view_backprop_view_of_view(self, device):
        if False:
            while True:
                i = 10
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = x.narrow(0, 0, 1)
        v1.mul_(2)
        v2.sum().backward()
        self.assertEqual(root.grad.tolist(), [[2, 2], [0, 0]])

    def test_inplace_on_view_of_view(self, device):
        if False:
            while True:
                i = 10
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = v1.narrow(1, 1, 1)
        v2.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.tolist(), [[1, 2], [1, 1]])

    @skipIfMps
    def test_inplace_on_view_then_no_grad(self, device):
        if False:
            print('Hello World!')
        a = torch.ones(3, 1, dtype=torch.double, device=device, requires_grad=True)
        b = a * 2
        c = b.view_as(b)
        c[0][0] = 3
        with torch.no_grad():
            c.grad_fn
        c.sum().backward()

    @skipIfMps
    def test_inplace_on_view_gradcheck(self, device):
        if False:
            for i in range(10):
                print('nop')
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=True)
        b = torch.randn(2, 2, dtype=torch.double, device=device, requires_grad=True)

        def func(root, b):
            if False:
                for i in range(10):
                    print('nop')
            x = root.clone()
            x.narrow(1, 2, 2).narrow(0, 1, 2).mul_(b)
            x.narrow(1, 0, 2).narrow(0, 1, 2).mul_(b)
            return x
        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(a.size(), dtype=torch.double, device=device, requires_grad=True)
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_on_view_multiple_outputs(self, device):
        if False:
            for i in range(10):
                print('nop')
        root = torch.arange(9.0, dtype=torch.double).reshape(3, 3).requires_grad_()
        x = root.clone()
        v1 = x.unbind()
        with self.assertRaises(RuntimeError):
            v1[0].mul_(2)

    @skipIfMps
    def test_inplace_on_view_of_multiple_output_view(self, device):
        if False:
            print('Hello World!')
        a = torch.rand(10, dtype=torch.double, device=device, requires_grad=True).clone()
        b = a.unbind(0)
        c = b[0].view_as(b[0])
        with self.assertRaises(RuntimeError):
            c.mul_(2)

    @skipIfMps
    def test_inplace_multiple_output_view_of_view(self, device):
        if False:
            while True:
                i = 10
        a = torch.rand(10, dtype=torch.double, device=device, requires_grad=True).clone()
        b = a.view_as(a)
        c = b.unbind(0)
        with self.assertRaises(RuntimeError):
            c[0].mul_(2)

    @skipIfMps
    def test_inplace_on_view_makes_base_require_grad(self, device):
        if False:
            print('Hello World!')
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=False)
        b = torch.randn(4, 2, dtype=torch.double, device=device, requires_grad=True)

        def func(root, b):
            if False:
                return 10
            x = root.clone()
            self.assertFalse(x.requires_grad)
            x.narrow(1, 2, 2).mul_(b)
            self.assertTrue(x.requires_grad)
            return x
        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(a.size(), dtype=torch.double, device=device, requires_grad=True)
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_on_view_backprop_view(self, device):
        if False:
            return 10
        a = torch.tensor([2.0, 5.0], device=device, requires_grad=False)
        b = torch.tensor([3.0], device=device, requires_grad=True)
        res = a.narrow(0, 1, 1).mul_(b)
        res.sum().backward()
        self.assertEqual(b.grad.tolist(), [5])
        self.assertIsNone(a.grad)

    @skipIfMps
    def test_inplace_on_view_modify_base(self, device):
        if False:
            print('Hello World!')
        r = torch.ones(1, dtype=torch.double, device=device, requires_grad=True)

        def fn(r):
            if False:
                for i in range(10):
                    print('nop')
            x = torch.ones(5, dtype=torch.double, device=device)
            v = x.select(0, 1)
            self.assertFalse(v.requires_grad)
            self.assertIsNone(v.grad_fn)
            x.add_(r)
            self.assertTrue(v.requires_grad)
            return v
        gradcheck(fn, [r])
        gradgradcheck(fn, [r])

    @skipIfMps
    def test_inplace_on_view_python(self, device):
        if False:
            return 10
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=True)
        b = torch.randn(2, 2, dtype=torch.double, device=device, requires_grad=True)

        class PyAdd(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x, y):
                if False:
                    i = 10
                    return i + 15
                ctx.mark_dirty(x)
                x.add_(y)
                return x

            @staticmethod
            def backward(ctx, grad):
                if False:
                    return 10
                return (grad, grad)

        def func(root, b):
            if False:
                return 10
            x = root.clone()
            PyAdd.apply(x.narrow(1, 2, 2).narrow(0, 1, 2), b)
            PyAdd.apply(x.narrow(1, 0, 2).narrow(0, 1, 2), b)
            return x
        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(a.size(), dtype=torch.double, device=device, requires_grad=True)
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_on_view_non_contig(self, device):
        if False:
            return 10
        root = torch.ones(2, 3, 2, device=device).select(2, 1).t().requires_grad_(True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = v1.narrow(1, 1, 1)
        v2.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.tolist(), [[1, 2], [1, 1], [1, 1]])

    def test_inplace_on_view_multi_output_unsafe(self, device):
        if False:
            i = 10
            return i + 15
        for f in [lambda t: t.unsafe_split(1), lambda t: t.unsafe_split_with_sizes((1, 1, 1)), lambda t: t.unsafe_chunk(3)]:
            a = torch.randn(3, 3, device=device, requires_grad=True)
            b = a + a
            (s1, s2, s3) = f(b)
            s1.mul_(s2)
            s1.sum().backward()

    def test_inplace_on_view_multi_output_safe(self, device):
        if False:
            return 10
        for f in [lambda t: t.split(1), lambda t: t.split_with_sizes((1, 1, 1)), lambda t: t.chunk(3)]:
            a = torch.randn(3, 3, device=device, requires_grad=True)
            b = a + a
            (s1, s2, s3) = f(b)
            error_msg = 'This view is the output of a function that returns multiple views.'
            with self.assertRaisesRegex(RuntimeError, error_msg):
                s1.mul_(s2)

    def test_inplace_on_view_undefined_grad_output(self, device):
        if False:
            while True:
                i = 10
        a = torch.tensor([1.0], requires_grad=True)
        c = a.clone()
        v = c[:]
        b = torch.tensor(1.0, requires_grad=True)

        class InplaceFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x, other):
                if False:
                    print('Hello World!')
                ctx.mark_dirty(x)
                return x.mul_(2)

            @staticmethod
            def backward(ctx, grad):
                if False:
                    for i in range(10):
                        print('nop')
                return (grad * 2, None)
        out = InplaceFunc.apply(v, b)
        out.backward()
        self.assertIsNone(b.grad)
        self.assertEqual(a.grad.item(), 2)

    @skipIfMps
    def test_mv_grad_stride_0(self, device):
        if False:
            return 10
        mat = torch.randn(2, 2, dtype=torch.double, device=device)
        vec = torch.randn(1, dtype=torch.double, device=device).requires_grad_(True)

        def fn(vec):
            if False:
                return 10
            vec = vec.expand(2)
            return (mat @ vec).sum()
        gradcheck(fn, vec)
        gradgradcheck(fn, vec)

    @onlyCUDA
    def test_gradcheck_input_output_different_device(self, device):
        if False:
            print('Hello World!')
        x = torch.ones((1,), dtype=torch.double, device='cuda', requires_grad=True)
        gradcheck(lambda x: x.to('cpu'), (x,))
        x = torch.ones((1,), dtype=torch.double, device='cpu', requires_grad=True)
        gradcheck(lambda x: x.to('cuda'), (x,))

    def test_strided_leaf_grad_layout(self, device):
        if False:
            for i in range(10):
                print('nop')
        for fmt_a in (torch.contiguous_format, torch.channels_last):
            for fmt_b in (torch.contiguous_format, torch.channels_last):
                a = torch.rand((2, 3, 4, 5), device=device).to(memory_format=fmt_a)
                b = torch.rand((2, 3, 4, 5), device=device).to(memory_format=fmt_b)
                a.requires_grad_()
                b.requires_grad_()
                a.sum().backward()
                self.assertEqual(a.grad.stride(), a.stride())
                b.sum().backward()
                self.assertEqual(b.grad.stride(), b.stride())
                a.grad = None
                b.grad = None
                (a * b).sum().backward()
                self.assertEqual(a.grad.stride(), a.stride())
                self.assertEqual(b.grad.stride(), b.stride())
        c = torch.empty_strided((2, 2), (4, 2), device=device).copy_(torch.rand((2, 2), device=device))
        c.requires_grad_()
        d = torch.rand((2, 2), device=device)
        c.sum().backward()
        self.assertEqual(c.grad.stride(), (2, 1))
        c.grad = None
        (c * d).sum().backward()
        self.assertEqual(c.grad.stride(), (2, 1))

    @skipIfMps
    def test_copy_r_to_c(self, device):
        if False:
            return 10
        out_c = torch.empty(3, 2, dtype=torch.cdouble, device=device)
        inp_r = torch.randn(3, 2, dtype=torch.double, device=device, requires_grad=True)

        def do_test():
            if False:
                return 10
            out_c.copy_(inp_r)
            out_c_inter = out_c.sum()
            out_c_inter.abs().backward()
            with torch.no_grad():
                self.assertEqual(inp_r.grad, torch.ones_like(inp_r) * torch.sgn(out_c_inter).real)
        self.assertNotWarn(do_test)

    def test_to_r_to_c(self, device):
        if False:
            print('Hello World!')

        def do_test():
            if False:
                return 10
            inp_r = torch.randn(3, 2, dtype=torch.double, device=device, requires_grad=True)
            out = inp_r.to(torch.complex128)
            out_inter = out.sum()
            out_inter.abs().backward()
            with torch.no_grad():
                self.assertEqual(inp_r.grad, torch.ones_like(inp_r) * torch.sgn(out_inter).real)
        self.assertNotWarn(do_test)

    def test_non_differentiable_ops(self, device):
        if False:
            i = 10
            return i + 15
        x = torch.tensor([[1, 2], [3, 4.0]], requires_grad=True, device=device)
        out = torch.isin(x, torch.tensor([2, 3], device=device))
        self.assertFalse(out.requires_grad)
        x = torch.randn(3, 3, requires_grad=True)
        out = torch.signbit(x)
        self.assertFalse(out.requires_grad)

    def test_warning_in_backward(self, device):
        if False:
            while True:
                i = 10
        a = torch.zeros((), device=device, requires_grad=True)
        b = torch._C._nn._test_warn_in_autograd(a)
        with self.assertWarnsRegex(UserWarning, 'Warn from backward'):
            b.backward()

    def test_complex_scalar_backward(self, device):
        if False:
            i = 10
            return i + 15
        a = torch.zeros(1, device=device, requires_grad=True)
        b = a * 0.5j
        msg = 'grad can be implicitly created only for real scalar outputs'
        with self.assertRaisesRegex(RuntimeError, msg):
            b.backward()
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.autograd.grad(b, a)

    def test_pow_real_negative_base_complex_exponent(self, device):
        if False:
            print('Hello World!')
        base = -torch.ones(2, device=device, dtype=torch.double)
        exponent = torch.randn(2, device=device, dtype=torch.cdouble, requires_grad=True)

        def fn(exponent):
            if False:
                return 10
            return torch.pow(base, exponent)
        torch.autograd.gradcheck(fn, (exponent,))

        def fn(exponent):
            if False:
                return 10
            return torch.pow(-1, exponent)
        torch.autograd.gradcheck(fn, (exponent,))

    def test_resize_version_bump(self, device):
        if False:
            return 10
        x = torch.rand((1,), device=device)
        y = torch.randn((3,), device=device)
        x.resize_((1, 2))
        self.assertEqual(x._version, 1)
        x.resize_as_(y)
        self.assertEqual(x._version, 2)
        x.resize_((3,))
        self.assertEqual(x._version, 2)
        x.resize_as_(y)
        self.assertEqual(x._version, 2)

class TestAllowMutationOnSaved(TestCase):

    def assertClonedLenEqual(self, ctx, n):
        if False:
            return 10
        self.assertEqual(len(list(ctx.cloned.items())), n)

    def assertTIDMapLenEqual(self, ctx, n):
        if False:
            print('Hello World!')
        self.assertEqual(len(list(ctx.tid_to_weakhandle.items())), n)

    def test_basic(self):
        if False:
            return 10
        a = torch.rand(2, 3, requires_grad=True)

        def fn(a):
            if False:
                while True:
                    i = 10
            b = a.clone()
            out = (b ** 2).sum()
            b.sin_()
            out.sum().backward()
            return a.grad
        msg = 'variables needed for gradient computation has been modified by an inplace'
        with self.assertRaisesRegex(RuntimeError, msg):
            fn(a)
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            da = fn(a)
        self.assertTrue(torch.allclose(a * 2, da))
        self.assertClonedLenEqual(ctx, 0)

    def test_views(self):
        if False:
            i = 10
            return i + 15
        a = torch.rand(2, 3, requires_grad=True)

        def fn(a):
            if False:
                return 10
            b = a.clone()
            c = b.view_as(b)
            out = (b ** 2).sum()
            c.sin_()
            out.sum().backward()
            return a.grad
        msg = 'variables needed for gradient computation has been modified by an inplace'
        with self.assertRaisesRegex(RuntimeError, msg):
            fn(a)
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            da = fn(a)
        self.assertClonedLenEqual(ctx, 0)
        self.assertTrue(torch.allclose(a * 2, da))

    def test_save_base_and_modify_view(self):
        if False:
            for i in range(10):
                print('nop')
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.rand(2, 3, requires_grad=True)
            b = a.clone()
            c = b[:1]
            out = b ** 2
            c *= 10
            out.sum().backward()
            self.assertClonedLenEqual(ctx, 0)
        self.assertClonedLenEqual(ctx, 0)
        self.assertTrue(torch.allclose(a * 2, a.grad))

    def test_save_view_modify_base(self):
        if False:
            print('Hello World!')
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.rand(2, 3, requires_grad=True)
            b = a.clone()
            c = b[:]
            out = (c ** 2).sum()
            b *= 2
            out.backward()
            self.assertTrue(torch.allclose(a * 2, a.grad))

    def test_double_backward(self):
        if False:
            return 10
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.rand(2, 3, requires_grad=True)
            b = a.clone()
            out = (b ** 2).sum()
            b.sin_()
            torch.autograd.grad(out, a, create_graph=True)
            (da,) = torch.autograd.grad(out, a, create_graph=True)
            (d2a,) = torch.autograd.grad(da.sum(), a)
        self.assertTrue(torch.allclose(torch.ones_like(a) * 2, d2a))
        self.assertClonedLenEqual(ctx, 0)

    def test_saved_but_not_anymore(self):
        if False:
            for i in range(10):
                print('nop')
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.randn(2, 3, requires_grad=True).clone()
            out = (a ** 2).sum()
            self.assertTIDMapLenEqual(ctx, 1)
            self.assertClonedLenEqual(ctx, 0)
            out.backward()
            a.sin_()
            self.assertClonedLenEqual(ctx, 0)
            out = (a ** 2).sum()
            a.sin_()
            self.assertClonedLenEqual(ctx, 1)
            del out
            self.assertClonedLenEqual(ctx, 0)

    def test_saved_same_tensor_many_times(self):
        if False:
            while True:
                i = 10
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.randn(2, 3, requires_grad=True).clone()
            b = a ** 2
            c = a ** 2
            a.sin_()
            self.assertClonedLenEqual(ctx, 1)
            del b, c
            self.assertClonedLenEqual(ctx, 0)

    def test_saved_same_tensor_different_versions(self):
        if False:
            print('Hello World!')
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.randn(2, 3, requires_grad=True).clone()
            b = a ** 2
            a.sin_()
            c = a ** 2
            a.sin_()
            self.assertClonedLenEqual(ctx, 2)
            del b
            self.assertClonedLenEqual(ctx, 1)
            del c
            self.assertClonedLenEqual(ctx, 0)

    def test_with_math_views(self):
        if False:
            for i in range(10):
                print('nop')
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.tensor([1 + 1j], requires_grad=True).clone()
            b = a.conj()
            out = (b ** 2).sum()
            a.sin_()
            out.abs().backward()
            a = torch.tensor([1 + 1j], requires_grad=True).clone()
            b = a.conj()
            out = (b ** 2).sum()
            b.sin_()
            out.abs().backward()

    def test_with_out_variant(self):
        if False:
            while True:
                i = 10
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.tensor([1.0], requires_grad=True)
            b = torch.tensor([1.0])
            c = torch.tensor([2.0])
            out = a * b
            self.assertTIDMapLenEqual(ctx, 1)
            torch.sin(c, out=b)
            self.assertClonedLenEqual(ctx, 1)
            out.backward()
            self.assertClonedLenEqual(ctx, 0)

    def test_backward_out_of_context(self):
        if False:
            i = 10
            return i + 15
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.rand(2, 3, requires_grad=True)
            out = (a ** 2).sum()
        msg = "Trying to backward outside of the 'allow_mutation_on_saved_tensors' context"
        with self.assertRaisesRegex(AssertionError, msg):
            out.backward()
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            a = torch.rand(2, 3, requires_grad=True)
            out = (a ** 2).sum()
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            with self.assertRaisesRegex(AssertionError, msg):
                out.backward()

    def test_disallow_nesting(self):
        if False:
            i = 10
            return i + 15
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            msg = 'allow_mutation_on_saved_tensors contexts cannot be nested'
            with self.assertRaisesRegex(RuntimeError, msg):
                with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
                    pass

class TestAutogradInferenceMode(TestCase):

    def _is_inference_tensor(self, tensor):
        if False:
            for i in range(10):
                print('nop')
        try:
            err_msg = 'Inference tensors do not track version counter'
            with self.assertRaisesRegex(RuntimeError, err_msg):
                tensor._version
            return True
        except AssertionError as e:
            return False

    def test_inference_mode_context_manager(self):
        if False:
            while True:
                i = 10
        self.assertFalse(torch.is_inference_mode_enabled())
        with torch.inference_mode():
            self.assertTrue(torch.is_inference_mode_enabled())
            with torch.inference_mode(False):
                self.assertFalse(torch.is_inference_mode_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
        self.assertFalse(torch.is_inference_mode_enabled())

    def test_inference_mode_decorator(self):
        if False:
            for i in range(10):
                print('nop')

        def func(x):
            if False:
                print('Hello World!')
            self.assertEqual(torch.is_inference_mode_enabled(), mode)
            return x * x
        for (mode, use_kwarg) in product((True, False, None), (True, False)):
            if mode is None:
                if use_kwarg:
                    decorated = torch.inference_mode(mode=func)
                else:
                    decorated = torch.inference_mode(func)
                mode = True
            elif use_kwarg:
                decorated = torch.inference_mode(mode=mode)(func)
            else:
                decorated = torch.inference_mode(mode)(func)
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)
                d = decorated(c)
                self.assertTrue(not mode or torch.is_inference(d))
                self.assertEqual(d.requires_grad, requires_grad and (not mode))

    def test_inference_mode_tensor_creation(self):
        if False:
            print('Hello World!')
        with torch.inference_mode():
            c = torch.ones(1, 2, 3)
            self.assertFalse(c.requires_grad)
            self.assertTrue(torch.is_inference(c))
            tmp = torch.ones(1, 2, 3, requires_grad=True)
            self.assertTrue(tmp.requires_grad)
            self.assertTrue(torch.is_inference(tmp))
            tmp = torch.ones(1, 2, 3).requires_grad_(False)
            self.assertFalse(tmp.requires_grad)
            self.assertTrue(torch.is_inference(tmp))

    def test_inference_mode_existing_autograd_session(self):
        if False:
            return 10
        s = torch.ones(1, 2, 3, requires_grad=True)
        a = s.clone()
        out = a * a
        with torch.inference_mode():
            a.add_(2)
        self.assertFalse(torch.is_inference(a))
        err_msg = 'one of the variables needed for gradient computation has been modified by an inplace operation'
        with self.assertRaisesRegex(RuntimeError, err_msg):
            out.backward(torch.ones_like(out))

    def test_inference_mode_inf_tensor_in_inf_mode_functional_op(self):
        if False:
            i = 10
            return i + 15

        def functional_op(x):
            if False:
                print('Hello World!')
            return x * x
        with torch.inference_mode():
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)
                func_out = functional_op(c)
                self.assertTrue(torch.is_inference(func_out))
                self.assertFalse(func_out.requires_grad)

    def test_inference_mode_inf_tensor_in_inf_mode_inplace_op(self):
        if False:
            while True:
                i = 10

        @torch.inference_mode()
        def run_test(fn):
            if False:
                print('Hello World!')
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)
                fn(c)
                self.assertTrue(torch.is_inference(c))
                self.assertEqual(c.requires_grad, requires_grad)
        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))
        run_test(lambda x: x.resize_(1, 2))
        run_test(lambda x: x.resize_as_(torch.ones(1, 2)))
        run_test(lambda x: x.copy_(torch.ones(1, 2, 3)))

    def test_inference_mode_inf_tensor_in_inf_mode_view_op(self):
        if False:
            return 10
        with torch.inference_mode():
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)
                view_out = c.view(-1)
                self.assertTrue(torch.is_inference(view_out))
                self.assertFalse(view_out.requires_grad)

    def test_inference_mode_inf_tensor_in_normal_mode_functional_op(self):
        if False:
            print('Hello World!')

        def functional_op(x):
            if False:
                for i in range(10):
                    print('nop')
            return x * x
        for requires_grad in (True, False):
            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)
        func_out = functional_op(c)
        self.assertFalse(torch.is_inference(func_out))
        self.assertFalse(func_out.requires_grad)
        self.assertTrue(func_out.is_leaf)

    def test_inference_mode_inf_tensor_in_normal_mode_inplace_op(self):
        if False:
            return 10

        def run_test(fn):
            if False:
                return 10
            for requires_grad in (False, True):
                with torch.inference_mode():
                    c = torch.ones(1, 2, 3, requires_grad=requires_grad)
                if requires_grad:
                    pass
                else:
                    err_msg = 'Inplace update to inference tensor outside InferenceMode'
                    with self.assertRaisesRegex(RuntimeError, err_msg):
                        fn(c)
        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

    def test_inference_mode_inf_tensor_in_normal_mode_view_op(self):
        if False:
            i = 10
            return i + 15
        for requires_grad in (True, False):
            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)
            out = c.view(-1)
            self.assertTrue(torch.is_inference(out))
            self.assertFalse(out.requires_grad)
            self.assertFalse(out._is_view())
            self.assertTrue(out.is_leaf)

    def test_normal_tensor_inplace_output_in_inference_mode(self):
        if False:
            i = 10
            return i + 15

        def run_test(fn):
            if False:
                return 10
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()
                with torch.inference_mode():
                    fn(a)
                    self.assertFalse(torch.is_inference(a))
                    self.assertEqual(a.requires_grad, requires_grad)
                    fn(a)
                    self.assertFalse(torch.is_inference(a))
                    self.assertEqual(a.requires_grad, requires_grad)
                    view_out = a.view(-1)
                    self.assertFalse(torch.is_inference(view_out))
                    self.assertEqual(view_out.requires_grad, requires_grad)
        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

    def test_normal_tensor_inplace_output_in_normal_mode(self):
        if False:
            while True:
                i = 10

        def run_test(fn):
            if False:
                while True:
                    i = 10
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()
                with torch.inference_mode():
                    fn(a)
                    self.assertFalse(torch.is_inference(a))
                    self.assertEqual(a.requires_grad, requires_grad)
                fn(a)
                self.assertFalse(torch.is_inference(a))
                self.assertEqual(a.requires_grad, requires_grad)
                fn(a)
                self.assertFalse(torch.is_inference(a))
                self.assertEqual(a.requires_grad, requires_grad)
                view_out = a.view(-1)
                self.assertFalse(torch.is_inference(view_out))
                self.assertEqual(view_out.requires_grad, requires_grad)
            run_test(lambda x: x.add_(2))
            run_test(lambda x: x.transpose_(0, 1))

    def test_normal_tensor_view_output_in_inference_mode(self):
        if False:
            i = 10
            return i + 15
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()
            with torch.inference_mode():
                out = a.view(-1)
                self.assertFalse(torch.is_inference(out))
                self.assertEqual(out.requires_grad, requires_grad)
                self.assertTrue(out._is_view())
                tmp = out.view(-1)
                self.assertFalse(torch.is_inference(tmp))
                self.assertEqual(tmp.requires_grad, requires_grad)
                self.assertTrue(tmp._is_view())
                self.assertTrue(tmp.is_leaf)
                self.assertTrue(torch.is_inference_mode_enabled())
                tmp.add_(2)
                self.assertFalse(torch.is_inference(tmp))
                self.assertEqual(tmp.requires_grad, requires_grad)
                self.assertEqual(a._version, tmp._version)

    def test_normal_tensor_view_output_in_normal_mode(self):
        if False:
            print('Hello World!')

        def functional_op(x):
            if False:
                for i in range(10):
                    print('nop')
            return x * x
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()
            with torch.inference_mode():
                out = a.view(-1)
                self.assertFalse(torch.is_inference(out))
                self.assertEqual(out.requires_grad, requires_grad)
                self.assertTrue(out._is_view())
                self.assertTrue(out.is_leaf)
            tmp = functional_op(out)
            self.assertFalse(torch.is_inference(tmp))
            self.assertEqual(tmp.requires_grad, requires_grad)
            if requires_grad:
                err_msg = 'A view was created in inference mode and is being modified inplace'
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    out.add_(2)
                pass
            else:
                out.add_(2)
            tmp = out.view(2, 3)
            self.assertFalse(torch.is_inference(tmp))
            self.assertEqual(tmp.requires_grad, requires_grad)

    def test_mix_inference_and_normal_tensor_functional_op(self):
        if False:
            while True:
                i = 10
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)
            out = c.add(s)
            self.assertFalse(torch.is_inference(out))
            self.assertEqual(out.requires_grad, requires_grad)
            if requires_grad:
                out.backward(torch.ones_like(out))
                self.assertEqual(c.grad, torch.ones_like(c))
            if requires_grad:
                err_msg = 'Inference tensors cannot be saved for backward'
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    c * s

    def test_mix_inference_and_normal_tensor_inplace_op(self):
        if False:
            while True:
                i = 10
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()
            with torch.inference_mode():
                c = torch.ones(1, 2, 3)
            self.assertTrue(torch.is_inference(c))
            if requires_grad:
                err_msg = 'Inference tensors cannot be saved for backward'
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    a.mul_(c)
                err_msg = "out=... arguments don't support automatic differentiation, but one of the arguments requires grad"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    torch.mul(s, s, out=c)
            else:
                a.mul_(c)
                err_msg = 'Inplace update to inference tensor outside InferenceMode is not allowed'
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    torch.mul(s, s, out=c)

    def test_mix_inference_and_normal_tensor_view_op(self):
        if False:
            while True:
                i = 10
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            with torch.inference_mode():
                c = torch.ones(1, 2, 3)
            tmp1 = c.view_as(s)
            self.assertTrue(torch.is_inference(tmp1))
            self.assertFalse(tmp1.requires_grad)
            tmp2 = s.view_as(c)
            self.assertFalse(torch.is_inference(tmp2))
            self.assertEqual(tmp2.requires_grad, requires_grad)

    def test_inference_mode_handle_direct_view_on_rebase(self):
        if False:
            while True:
                i = 10

        def run_test(fn):
            if False:
                return 10
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()
                with torch.inference_mode():
                    view_out = a.view_as(a)
                if requires_grad:
                    err_msg = 'A view was created in inference mode and is being modified inplace'
                    with self.assertRaisesRegex(RuntimeError, err_msg):
                        fn(view_out)
                    pass
                else:
                    fn(view_out)
        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

    def test_inference_mode_handle_indirect_view_on_rebase(self):
        if False:
            while True:
                i = 10

        def run_test(fn):
            if False:
                while True:
                    i = 10
            for requires_grad in (True, False):
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                a = s.clone()
                with torch.inference_mode():
                    view_out = a.view(-1)
                fn(a)
                if requires_grad:
                    err_msg = 'A view was created in inference mode and its base or another view '
                    with self.assertRaisesRegex(RuntimeError, err_msg):
                        view_out.grad_fn
                    pass
                else:
                    view_out.grad_fn
        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

class TestMultithreadAutograd(TestCase):

    def _run_py_multithread_fn(self, fn, args=(), num_threads=10, kwargs=None, pass_idx=False):
        if False:
            return 10

        class PropagatingThread(threading.Thread):
            """Helper class to propagate exception from child
            thread to main thread on join.

            Reference: https://stackoverflow.com/a/31614591/5602957
            """

            def run(self):
                if False:
                    return 10
                self.exception = None
                try:
                    self.ret = super().run()
                except Exception as e:
                    self.exception = e

            def join(self, timeout=None):
                if False:
                    return 10
                super().join(timeout)
                if self.exception:
                    raise self.exception from self.exception
                return self.ret
        threads = []
        for idx in range(num_threads):
            p = PropagatingThread(target=fn, args=(idx, *args) if pass_idx else args)
            p.start()
            threads.append(p)
        for p in threads:
            p.join()

    def test_multithreaded_exception_propagation(self):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                i = 10
                return i + 15
            self.assertTrue(False)
        with self.assertRaises(AssertionError):
            self._run_py_multithread_fn(fn)

    def test_simple_backward(self):
        if False:
            i = 10
            return i + 15

        def train_fn():
            if False:
                for i in range(10):
                    print('nop')
            x = torch.ones(5, 5, requires_grad=True)
            y = (x + 3) * (x + 4) * 0.5
            y.sum().backward()
            self.assertEqual(x.grad, x + 3.5)
        self._run_py_multithread_fn(train_fn)

    def test_simple_backward_same_input(self):
        if False:
            print('Hello World!')

        def train_fn_backward(x):
            if False:
                print('Hello World!')
            y = (x + 3) * (x + 4) * 0.5
            y.sum().backward()
        x = torch.ones(5, 5, requires_grad=True)
        self._run_py_multithread_fn(train_fn_backward, (x,))
        self.assertEqual(x.grad, 10 * (x + 3.5))

        def train_fn_grad(x):
            if False:
                i = 10
                return i + 15
            y = (x + 3) * (x + 4) * 0.5
            grads = torch.autograd.grad(y.sum(), x)
            self.assertEqual(len(grads), 1)
            self.assertEqual(grads[0], x + 3.5)
        self._run_py_multithread_fn(train_fn_grad, (x,))

    def test_multi_grad_hooks(self):
        if False:
            return 10
        t1 = torch.rand(2, requires_grad=True)
        t2 = torch.rand(2, requires_grad=True)
        t3 = torch.rand(2, requires_grad=True)
        t4 = torch.rand(2, requires_grad=True)
        res = None
        count = [0]

        def hook(grads):
            if False:
                return 10
            nonlocal res
            count[0] += 1
            grad_is_none = [g is not None for g in grads]
            if res is None:
                res = grad_is_none
            else:
                self.assertEqual(res, grad_is_none)
        torch.autograd.graph.register_multi_grad_hook((t1, t2, t3, t4), hook)
        out = (t2 * t3).sum()

        def backward_retain_graph(out, t2, t3):
            if False:
                while True:
                    i = 10
            out.backward(inputs=(t2, t3), retain_graph=True)
        self._run_py_multithread_fn(backward_retain_graph, (out, t2, t3), num_threads=5)
        self.assertEqual(count[0], 5)
        self.assertEqual(res, [False, True, True, False])
        res = None
        count = [0]
        err_count = [0]
        bw_count = [0]

        class Func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    while True:
                        i = 10
                return x

            @staticmethod
            def backward(ctx, gO):
                if False:
                    return 10
                bw_count[0] += 1
                if bw_count[0] == 1:
                    raise RuntimeError('error message')
                else:
                    return gO
        out = (Func.apply(t2) * t3).sum()

        def backward_retain_graph(out, t2, t3):
            if False:
                while True:
                    i = 10
            try:
                out.backward(inputs=(t2, t3), retain_graph=True)
            except RuntimeError:
                err_count[0] += 1
        self._run_py_multithread_fn(backward_retain_graph, (out, t2, t3), num_threads=5)
        self.assertEqual(count[0], 4)
        self.assertEqual(err_count[0], 1)
        self.assertEqual(res, [False, True, True, False])

    def test_dataparallel_saved_tensors_hooks(self):
        if False:
            for i in range(10):
                print('nop')

        def pack(x):
            if False:
                while True:
                    i = 10
            warnings.warn('pack')
            return x
        _self = self

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                with warnings.catch_warnings(record=True) as w:
                    y = x * x
                    if torch.cuda.device_count() >= 2:
                        _self.assertEqual(len(w), 0)
                    else:
                        _self.assertGreater(len(w), 0)
        x = torch.ones(5, 5, requires_grad=True)
        model = torch.nn.DataParallel(Model())
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            model(x)
            with warnings.catch_warnings(record=True) as w:
                y = x * x
                _self.assertGreater(len(w), 0)

    def test_python_thread_in_middle(self):
        if False:
            i = 10
            return i + 15
        success_vs_raises = [0, 0]

        def train_fn_no_retain_graph(x):
            if False:
                return 10
            y = x + x ** 2
            try:
                y.sum().backward()
                success_vs_raises[0] += 1
            except RuntimeError as error:
                success_vs_raises[1] += 1
                self.assertRegex(str(error), 'Specify retain_graph=True')
        x_no_retain = torch.ones(5, 5, requires_grad=True)
        y_no_retain = x_no_retain + x_no_retain ** 2
        self._run_py_multithread_fn(train_fn_no_retain_graph, (y_no_retain,), num_threads=5)
        self.assertTrue(success_vs_raises[0] >= 1)

        def train_fn_retain_graph(x):
            if False:
                return 10
            y = x + x ** 2
            y.sum().backward(retain_graph=True)
        x_retain = torch.ones(5, 5, requires_grad=True)
        y_retain = x_retain + x_retain ** 2
        self._run_py_multithread_fn(train_fn_retain_graph, (y_retain,), num_threads=5)
        self.assertEqual(x_retain.grad, 5 * (4 * x_retain ** 3 + 6 * x_retain ** 2 + 4 * x_retain + 1))

    def test_fork_join_in_middle(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def train_fn_jit_no_retain(middle, orig_x):
            if False:
                i = 10
                return i + 15
            y = middle + middle ** 2
            return torch.autograd.grad([y.sum()], [orig_x])

        @torch.jit.script
        def train_fn_fork_join_calls_no_retain(x):
            if False:
                print('Hello World!')
            y_no_retain = (x + 3) * (x + 4) * 0.5
            fut = torch.jit._fork(train_fn_jit_no_retain, y_no_retain, x)
            grad_hat = train_fn_jit_no_retain(y_no_retain, x)
            grad = torch.jit._wait(fut)
            return (grad, grad_hat)
        try:
            train_fn_fork_join_calls_no_retain(torch.randn(5, 5, requires_grad=True))
        except RuntimeError as error:
            self.assertRegex(str(error), 'Specify retain_graph=True')

        @torch.jit.script
        def train_fn_jit_retain(middle, orig_x):
            if False:
                for i in range(10):
                    print('nop')
            y = middle + middle ** 2
            return torch.autograd.grad([y.sum()], [orig_x], retain_graph=True)

        @torch.jit.script
        def train_fn_fork_join_calls_retain(x):
            if False:
                i = 10
                return i + 15
            y_retain = (x + 3) * (x + 4) * 0.5
            fut1 = torch.jit._fork(train_fn_jit_retain, y_retain, x)
            fut2 = torch.jit._fork(train_fn_jit_retain, y_retain, x)
            grad = train_fn_jit_retain(y_retain, x)
            grad1 = torch.jit._wait(fut1)
            grad2 = torch.jit._wait(fut2)
            return (grad, grad1, grad2)
        (grad, grad1, grad2) = train_fn_fork_join_calls_retain(torch.randn(5, 5, requires_grad=True))
        self.assertEqual(grad, grad1)
        self.assertEqual(grad, grad2)

    def test_preserve_backtrace(self):
        if False:
            return 10

        class Foo(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input):
                if False:
                    return 10
                return input

            @staticmethod
            def backward(ctx, *grad):
                if False:
                    for i in range(10):
                        print('nop')
                raise ValueError('something')
        t = torch.rand(10, requires_grad=True)
        try:
            Foo.apply(t).sum().backward()
        except Exception:
            import traceback
            tb = sys.exc_info()[2]
            tb_str = '\n'.join(traceback.format_tb(tb))
            self.assertTrue('raise ValueError("something")' in tb_str)

    def test_cat_stack_r_to_c(self):
        if False:
            while True:
                i = 10
        inp_c = torch.rand(3, 2, dtype=torch.cdouble, requires_grad=True)
        inp_r = torch.randn(3, 2, dtype=torch.double, requires_grad=True)

        def fn(x1, x2):
            if False:
                while True:
                    i = 10
            return torch.cat((x1, x2), dim=-1)

        def fn2(x1, x2):
            if False:
                while True:
                    i = 10
            return torch.stack((x1, x2), dim=-1)
        torch.autograd.gradcheck(fn, [inp_r, inp_c], check_forward_ad=True)
        torch.autograd.gradcheck(fn, [inp_c, inp_r], check_forward_ad=True)
        torch.autograd.gradcheck(fn2, [inp_r, inp_c], check_forward_ad=True)
        torch.autograd.gradcheck(fn2, [inp_c, inp_r], check_forward_ad=True)

    def test_set_multithreading_enabled_as_context_manager_and_function(self):
        if False:
            i = 10
            return i + 15
        with torch.autograd.set_multithreading_enabled(False):
            self.assertFalse(torch.autograd.is_multithreading_enabled())
        self.assertTrue(torch.autograd.is_multithreading_enabled())
        with torch.autograd.set_multithreading_enabled(True):
            self.assertTrue(torch.autograd.is_multithreading_enabled())
        self.assertTrue(torch.autograd.is_multithreading_enabled())
        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.set_multithreading_enabled(True)
            self.assertTrue(torch.autograd.is_multithreading_enabled())
        self.assertTrue(torch.autograd.is_multithreading_enabled())
        torch.autograd.set_multithreading_enabled(False)
        self.assertFalse(torch.autograd.is_multithreading_enabled())
        torch.autograd.set_multithreading_enabled(True)
        self.assertTrue(torch.autograd.is_multithreading_enabled())

class TestNestedCheckpoint(TestCase):

    @staticmethod
    def grad(fn):
        if False:
            return 10

        def wrapper(x):
            if False:
                print('Hello World!')
            with torch.enable_grad():
                out = fn(x)
                (grad_input,) = torch.autograd.grad(out, inputs=(x,), create_graph=True)
            return grad_input
        return wrapper

    @staticmethod
    def sum(fn):
        if False:
            print('Hello World!')

        def wrapped(x):
            if False:
                while True:
                    i = 10
            return fn(x).sum()
        return wrapped

    @staticmethod
    def checkpoint(fn):
        if False:
            while True:
                i = 10

        def wrapped(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, **kwargs)
        return wrapped

    def get_tests(self, fn):
        if False:
            i = 10
            return i + 15
        (grad, c) = (self.grad, self.checkpoint)
        tests = ((fn, (c(fn), c(c(fn)))), (grad(fn), (grad(c(fn)), grad(c(c(fn))))), (grad(grad(fn)), (grad(c(grad(fn))), c(grad(grad(c(fn)))), grad(c(grad(c(fn)))))), (grad(grad(grad(fn))), (grad(c(grad(grad(c(fn))))), grad(c(grad(c(grad(c(fn)))))))))
        return tests

    def check_graph_dies(self, fn):
        if False:
            print('Hello World!')

        def iter_graph(roots):
            if False:
                i = 10
                return i + 15
            if not roots:
                return
            seen = set()
            q = collections.deque()
            for node in roots:
                if node is not None:
                    seen.add(node)
                    q.append(node)
            while q:
                node = q.popleft()
                for (fn, _idx) in node.next_functions:
                    if fn in seen or fn is None:
                        continue
                    seen.add(fn)
                    q.append(fn)
                yield node

        class Handle:
            __slot__ = ['node_name']

            def __init__(self, node_name):
                if False:
                    for i in range(10):
                        print('nop')
                self.node_name = node_name

        def scope():
            if False:
                print('Hello World!')
            a = torch.randn((), requires_grad=True)
            out = fn(a)
            refs = []
            for node in iter_graph([out.grad_fn]):
                handle = Handle(node.name())
                refs.append(weakref.ref(handle))
                node.metadata['blah'] = handle
            return refs
        refs = scope()
        node_names = [ref().node_name for ref in refs if ref() is not None]
        if len(node_names) > 0:
            print('Nodes still alive:', node_names)
        self.assertEqual(len(node_names), 0)

    @parametrize('early_stop', [True, False])
    def test_nested_checkpoint(self, early_stop):
        if False:
            i = 10
            return i + 15
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            x = torch.randn((), requires_grad=True)

            def f(x):
                if False:
                    i = 10
                    return i + 15
                out = x.sin().exp().sin()
                return out

            def g(x):
                if False:
                    for i in range(10):
                        print('nop')
                a = x.sin().exp().sin()
                b = x.sin().exp().sin()
                (ga,) = torch.autograd.grad(a, x)
                (gb,) = torch.autograd.grad(b, x)
                return x.sin()
            for fn in (f, g):
                for (expected_fn, actual_fns) in self.get_tests(fn):
                    expected = expected_fn(x)
                    for actual_fn in actual_fns:
                        actual = actual_fn(x)
                        self.assertTrue(torch.allclose(expected, actual))
                        self.check_graph_dies(actual_fn)

    @parametrize('early_stop', [True, False])
    def test_nested_checkpoint_two_children(self, early_stop):
        if False:
            i = 10
            return i + 15
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            (grad, sum, c) = (self.grad, self.sum, self.checkpoint)

            def f(x):
                if False:
                    for i in range(10):
                        print('nop')
                return x.sin().exp().sin()

            def g(x):
                if False:
                    i = 10
                    return i + 15
                return x.cos().sin().exp()

            def hc(x):
                if False:
                    return 10
                return c(g)(c(f)(x))

            def h(x):
                if False:
                    i = 10
                    return i + 15
                return g(f(x))
            a = torch.randn(3, 3, requires_grad=True)
            expected = grad(sum(grad(sum(h))))(a)
            actual = grad(sum(grad(sum(c(hc)))))(a)
            self.assertTrue(torch.allclose(expected, actual))
            actual = grad(sum(c(grad(sum(c(hc))))))(a)
            self.assertTrue(torch.allclose(expected, actual))
            self.check_graph_dies(grad(c(hc)))
            self.check_graph_dies(grad(sum(grad(sum(c(hc))))))
            self.check_graph_dies(grad(sum(c(grad(sum(c(hc)))))))

    @parametrize('early_stop', [True, False])
    def test_nested_checkpoint_non_tensor_inputs_and_outputs(self, early_stop):
        if False:
            print('Hello World!')

        def fn(k, a, b, f):
            if False:
                while True:
                    i = 10
            return (f(k * a * b.exp()), 1, 'abcd')
        k = 3
        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)

        def f(x):
            if False:
                print('Hello World!')
            return x.sin()
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            (out, _unused1, _unused2) = checkpoint(fn, k, a, b, f, use_reentrant=False)
        actual_grads = torch.autograd.grad(out, (a, b))
        (out, _unused1, _unused2) = fn(k, a, b, f)
        expected_grads = torch.autograd.grad(out, (a, b))
        for (actual, expected) in zip(actual_grads, expected_grads):
            self.assertTrue(torch.allclose(actual, expected))

    @parametrize('early_stop', [True, False])
    def test_nested_checkpoint_kwargs(self, early_stop):
        if False:
            return 10

        def fn(a, blah=None):
            if False:
                for i in range(10):
                    print('nop')
            out = a.sin().exp()
            if blah is not None:
                out = out * blah
            return out.sin().exp()
        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            out = checkpoint(fn, a, blah=b, use_reentrant=False)
            actual_grads = torch.autograd.grad(out, (a, b))
            out = fn(a, blah=b)
            expected_grads = torch.autograd.grad(out, (a, b))
            for (actual, expected) in zip(actual_grads, expected_grads):
                self.assertTrue(torch.allclose(actual, expected))

    @parametrize('early_stop', [True, False])
    def test_nested_checkpoint_same_graph(self, early_stop):
        if False:
            i = 10
            return i + 15
        counter = [0]

        def hook(*_unused_args):
            if False:
                print('Hello World!')
            counter[0] += 1

        def fn(a):
            if False:
                return 10
            return a.sin().cos().sin()
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            out = checkpoint(fn, a, use_reentrant=False)
        out.grad_fn.next_functions[0][0].register_hook(hook)
        out.backward()
        self.assertEqual(counter[0], 1)

    @parametrize('early_stop', [True, False])
    def test_nested_checkpoint_reentrant_backwards(self, early_stop):
        if False:
            return 10

        def fn(a):
            if False:
                i = 10
                return i + 15
            x = a.sin().cos()
            out = x.sin()
            return (x, out)

        def hook(*_unused_args):
            if False:
                return 10
            x.backward(retain_graph=True)
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            (x, out) = checkpoint(fn, a, use_reentrant=False)
        out.grad_fn.register_hook(hook)
        out.backward(retain_graph=True)

    def test_nested_checkpoint_set_early_stop(self):
        if False:
            while True:
                i = 10
        counter = [0]

        def clone(x):
            if False:
                i = 10
                return i + 15
            counter[0] += 1
            return x.clone()

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return clone(x.sin().cos())
        a = torch.tensor(1.0, requires_grad=True)
        out = checkpoint(fn, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 1)
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            out = checkpoint(fn, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 2)

    def test_nested_checkpoint_set_early_stop_no_recompution_needed(self):
        if False:
            return 10
        python_dispatch_counter = [0]
        counter = [0]

        class SinCounterMode(TorchDispatchMode):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    i = 10
                    return i + 15
                kwargs = {} if kwargs is None else kwargs
                if func is torch.ops.aten.sin.default:
                    self.count += 1
                return func(*args, **kwargs)

        def fn(x):
            if False:
                print('Hello World!')
            counter[0] += 1
            return x.sin()
        a = torch.tensor(1.0, requires_grad=True)
        with SinCounterMode() as python_dispatch_counter:
            out = checkpoint(fn, a, use_reentrant=False)
            out.backward()
        self.assertEqual(counter[0], 2)
        self.assertEqual(python_dispatch_counter.count, 1)
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with SinCounterMode() as python_dispatch_counter:
            with torch.utils.checkpoint.set_checkpoint_early_stop(False):
                out = checkpoint(fn, a, use_reentrant=False)
            out.backward()
        self.assertEqual(counter[0], 2)
        self.assertEqual(python_dispatch_counter.count, 2)
        counter = [0]

        def fn2(x):
            if False:
                i = 10
                return i + 15
            counter[0] += 1
            return x.clone()
        a = torch.tensor(1.0, requires_grad=True)
        out = checkpoint(fn2, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 1)
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            out = checkpoint(fn2, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 1)

class TestAutogradMultipleDispatch(TestCase):

    def test_autograd_multiple_dispatch_registrations(self, device):
        if False:
            while True:
                i = 10
        t = torch.randn(3, 3, device=device, requires_grad=True)
        out = torch._test_autograd_multiple_dispatch(t)
        grad = torch.randn(3, 3, device=device)
        out.backward(grad)
        if 'cuda' not in device:
            self.assertEqual(t.grad, grad + 1)
        else:
            self.assertEqual(t.grad, grad * 2)
        a = torch.arange(6, dtype=torch.float, device=device).reshape(2, 3).requires_grad_(True)
        b = torch.arange(8, dtype=torch.float, device=device).reshape(2, 4).requires_grad_(True)
        nt = torch.nested.as_nested_tensor([a, b], dtype=torch.float, device=device)
        nt_out = torch._test_autograd_multiple_dispatch(nt)
        c = torch.randn(2, 3, device=device)
        d = torch.randn(2, 4, device=device)
        nt_grad = torch.nested.nested_tensor([c, d], dtype=torch.float, device=device)
        nt_out.backward(nt_grad)
        self.assertEqual(a.grad, c * c)
        self.assertEqual(b.grad, d * d)

    def test_autograd_composite_implicit_and_dispatch_registration(self, device):
        if False:
            print('Hello World!')
        t = torch.randn(3, 3, device=device, requires_grad=True)
        out = torch._test_autograd_multiple_dispatch(t, True)
        grad = torch.randn(3, 3, device=device)
        out.backward(grad)
        self.assertEqual(t.grad, grad)
        a = torch.arange(6, dtype=torch.float, device=device).reshape(2, 3).requires_grad_(True)
        b = torch.arange(8, dtype=torch.float, device=device).reshape(2, 4).requires_grad_(True)
        nt = torch.nested.as_nested_tensor([a, b], dtype=torch.float, device=device)
        nt_out = torch._test_autograd_multiple_dispatch(nt, True)
        c = torch.randn(2, 3, device=device)
        d = torch.randn(2, 4, device=device)
        nt_grad = torch.nested.nested_tensor([c, d], dtype=torch.float, device=device)
        nt_out.backward(nt_grad)
        self.assertEqual(a.grad, c * c + c)
        self.assertEqual(b.grad, d * d + d)

    def test_foward_mode_AD(self, device):
        if False:
            i = 10
            return i + 15
        primal = torch.randn(3, device=device)
        tangent = torch.randn(3, device=device)
        with fwAD.dual_level():
            dual_input = fwAD.make_dual(primal, tangent)
            err_msg = 'Trying to use forward AD with .* that does not support it'
            hint_msg = 'Running forward AD for an OP that does not implement it should raise a NotImplementedError'
            if 'cuda' in device:
                with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                    torch._test_autograd_multiple_dispatch(dual_input)
            else:
                torch._test_autograd_multiple_dispatch(dual_input)

    def test_view_copy(self, device):
        if False:
            return 10
        t = torch.randn(2, 2, device=device, requires_grad=True)
        t_ref = t.clone().detach().requires_grad_()
        t_view = torch._test_autograd_multiple_dispatch_view(t_ref)
        t_view_copy = torch._test_autograd_multiple_dispatch_view_copy(t)
        grad = torch.randn(4, device=device)
        t_view_copy.backward(grad)
        t_view.backward(grad.clone())
        self.assertEqual(t_view_copy, t_view)
        self.assertEqual(t.grad, t_ref.grad)
        if 'cuda' in device:
            self.assertEqual(t.grad, grad.reshape_as(t) + 1)
        else:
            self.assertEqual(t.grad, grad.reshape_as(t))

    @onlyCPU
    def test_per_dispatch_key_input_saving(self, device):
        if False:
            print('Hello World!')

        def foo(x):
            if False:
                while True:
                    i = 10
            x = x.clone()
            res = x.sum(-1, keepdim=True)
            x.add_(x)
            return res
        inp = torch.rand(2, device=device, requires_grad=True)
        foo(inp).backward()
        nt = torch.nested.nested_tensor([torch.rand(2), torch.rand(2)], device=device, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'):
            foo(nt).backward(torch.nested.nested_tensor([torch.rand(1), torch.rand(1)], device=device))

    @onlyCUDA
    def test_backward_single_threaded(self):
        if False:
            print('Hello World!')
        threads_eq = None

        class TestFn(Function):

            @staticmethod
            def forward(ctx, x, self):
                if False:
                    i = 10
                    return i + 15
                ctx.self = self
                ctx.tid = threading.get_ident()
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                if False:
                    while True:
                        i = 10
                nonlocal threads_eq
                threads_eq = ctx.tid == threading.get_ident()
                return (gO, None)
        inp = torch.rand(10, device='cuda', requires_grad=True)
        with torch.autograd.set_multithreading_enabled(False):
            TestFn.apply(inp, None).sum().backward()
        self.assertTrue(threads_eq)
        TestFn.apply(inp, None).sum().backward()
        self.assertFalse(threads_eq)

    @onlyCUDA
    def test_backward_tls_stash(self):
        if False:
            print('Hello World!')
        local = threading.local()
        local.my_obj = {}
        local.my_obj[10] = 10
        test_self = self
        torch._C._stash_obj_in_tls('my_obj', local.my_obj)

        class TestFn(Function):

            @staticmethod
            def forward(ctx, x, self):
                if False:
                    print('Hello World!')
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                if False:
                    return 10
                test_self.assertTrue(torch._C._is_key_in_tls('my_obj'))
                test_self.assertTrue(torch._C._get_obj_in_tls('my_obj')[10] == 10)
                torch._C._get_obj_in_tls('my_obj')[10] = 5
                return (gO, None)
        inp = torch.rand(10, device='cuda', requires_grad=True)
        TestFn.apply(inp, None).sum().backward()
        self.assertEqual(local.my_obj[10], 5)
from autograd.test_complex import TestAutogradComplex
from autograd.test_functional import TestAutogradFunctional
instantiate_device_type_tests(TestAutogradDeviceType, globals(), except_for=None)
instantiate_device_type_tests(TestAutogradMultipleDispatch, globals(), only_for=('cpu', 'cuda'))
instantiate_parametrized_tests(TestAutograd)
instantiate_parametrized_tests(TestNestedCheckpoint)
if __name__ == '__main__':
    run_tests()