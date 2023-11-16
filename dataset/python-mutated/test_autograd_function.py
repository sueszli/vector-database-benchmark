import copy
import math
import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils

class CustomFunc1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, foo):
        if False:
            print('Hello World!')
        return foo + foo

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            while True:
                i = 10
        return grad_output

class CustomFunc3(torch.autograd.Function):

    @staticmethod
    def forward(ctx, foo):
        if False:
            i = 10
            return i + 15
        result = foo + foo
        torch._dynamo.graph_break()
        result = result + foo
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            print('Hello World!')
        (result,) = ctx.saved_tensors
        return grad_output * math.sqrt(result.numel())

class Module1(torch.nn.Module):

    def forward(self, foo):
        if False:
            print('Hello World!')
        return CustomFunc1().apply(foo)

class Module2(torch.nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.fn = CustomFunc1.apply

    def forward(self, foo):
        if False:
            for i in range(10):
                print('nop')
        return self.fn(foo)

class Module3(torch.nn.Module):

    def forward(self, foo):
        if False:
            i = 10
            return i + 15
        return CustomFunc1().apply(foo)

class Module4(torch.nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.fn = CustomFunc1.apply

    def forward(self, foo):
        if False:
            while True:
                i = 10
        return self.fn(foo)

class Module5(torch.nn.Module):

    def forward(self, foo):
        if False:
            for i in range(10):
                print('nop')
        return CustomFunc3().apply(foo)

class Module6(torch.nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.fn = CustomFunc3.apply

    def forward(self, foo):
        if False:
            for i in range(10):
                print('nop')
        return self.fn(foo)

class LinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(input, weight, bias):
        if False:
            while True:
                i = 10
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        if False:
            print('Hello World!')
        (input, weight, bias) = inputs
        ctx.save_for_backward(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            i = 10
            return i + 15
        (input, weight, bias) = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return (grad_input, grad_weight, grad_bias)

class ModuleLinear(torch.nn.Module):

    def forward(self, input, weight, bias=None):
        if False:
            return 10
        return LinearFunction.apply(input, weight, bias)

class MaterializingGradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if False:
            while True:
                i = 10
        ctx.set_materialize_grads(False)
        return (x.clone(), x.clone())

    @staticmethod
    def backward(ctx, grad_out1, grad_out2):
        if False:
            i = 10
            return i + 15
        return (grad_out1, grad_out2)

class MaterializingGradModule(torch.nn.Module):

    def forward(self, x):
        if False:
            print('Hello World!')
        return MaterializingGradFunction.apply(x)

class CustomFuncBwdPrintGraphBreak(torch.autograd.Function):

    @staticmethod
    def forward(ctx, foo):
        if False:
            for i in range(10):
                print('nop')
        return torch.add(foo, foo)

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            while True:
                i = 10
        print('graph break!')
        return grad_output

class CustomFuncBwdPrintModule(torch.nn.Module):

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return CustomFuncBwdPrintGraphBreak.apply(x)

class CustomFuncStrideBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, foo):
        if False:
            return 10
        return torch.add(foo, foo)

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            return 10
        return grad_output.stride()

class CustomFuncStrideModule(torch.nn.Module):

    def forward(self, x):
        if False:
            print('Hello World!')
        return CustomFuncStrideBwd.apply(x)

class CustomFuncSaveForBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, foo):
        if False:
            while True:
                i = 10
        result = foo + foo
        result = result + foo
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            return 10
        (result,) = ctx.saved_tensors
        return grad_output * math.sqrt(result.numel())

class SaveForBwdModule(torch.nn.Module):

    def forward(self, foo):
        if False:
            print('Hello World!')
        return CustomFuncSaveForBwd().apply(foo)

class ContextSaveAndMark(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if False:
            print('Hello World!')
        with torch.no_grad():
            ctx.save_for_backward(x)
            ctx.mark_non_differentiable(x)
            return x

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            print('Hello World!')
        return grad_output

class ContextMarkAndSave(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if False:
            i = 10
            return i + 15
        with torch.no_grad():
            ctx.mark_non_differentiable(x)
            ctx.save_for_backward(x)
            return x

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            i = 10
            return i + 15
        return grad_output

class ModuleWithGradFunc(torch.nn.Module):

    def __init__(self, func):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.f = func.apply

    def forward(self, x):
        if False:
            return 10
        return self.f(x)

class AutogradFunctionTests(torch._dynamo.test_case.TestCase):

    def test_autograd_function_equivalence(self):
        if False:
            for i in range(10):
                print('nop')
        for grad in [True, False]:
            for i in range(1, 5):
                torch._dynamo.reset()
                model = globals()[f'Module{i}']()
                opt_model = torch._dynamo.optimize('eager')(model)
                self.assertTrue(torch.allclose(opt_model(torch.ones(2, 3, requires_grad=grad)), torch.tensor([2.0], requires_grad=grad)))

    def test_autograd_function_has_graph_break(self):
        if False:
            print('Hello World!')
        for grad in [True, False]:
            x = torch.randn(10, requires_grad=grad)
            for model in [Module5(), Module6()]:
                torch._dynamo.reset()
                cnts = torch._dynamo.testing.CompileCounter()
                opt_model = torch._dynamo.optimize(cnts)(model)
                for _ in range(3):
                    ref = model(x)
                    res = opt_model(x)
                    self.assertTrue(torch.allclose(ref, res))
                self.assertEqual(cnts.frame_count, 2)

    def test_linear_setup_context(self):
        if False:
            return 10
        model = ModuleLinear()
        opt_model = torch._dynamo.optimize('eager')(model)
        input = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        weight = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
        optim_result = opt_model(input, weight)
        eager_result = model(input, weight)
        self.assertEqual(optim_result, eager_result)

    def test_materialize_grad(self):
        if False:
            return 10
        model = MaterializingGradModule()
        opt_model = torch._dynamo.optimize('eager')(model)
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        optim_result = opt_model(x)
        eager_result = model(x)
        self.assertEqual(optim_result, eager_result)

    def test_print_in_bwd(self):
        if False:
            print('Hello World!')
        model = CustomFuncBwdPrintModule()
        opt_model = torch._dynamo.optimize('eager', nopython=True)(model)
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, '.*BuiltinVariable\\(print\\).*'):
            opt_model(x)

    def test_stride_in_bwd(self):
        if False:
            print('Hello World!')
        model = CustomFuncStrideModule()
        opt_model = torch._dynamo.optimize('eager', nopython=True)(model)
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, 'Illegal getattr invocation stride in strict mod'):
            opt_model(x)

    def test_save_for_bwd(self):
        if False:
            i = 10
            return i + 15
        model = SaveForBwdModule()
        opt_model = torch._dynamo.optimize('eager', nopython=True)(model)
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        opt_model(x)

    def test_classmethod(self):
        if False:
            i = 10
            return i + 15

        class Shake(torch.autograd.Function):

            @classmethod
            def forward(cls, ctx, foo):
                if False:
                    i = 10
                    return i + 15
                return foo + foo

            @classmethod
            def backward(cls, ctx, grad_output):
                if False:
                    for i in range(10):
                        print('nop')
                return grad_output

        def f(x):
            if False:
                while True:
                    i = 10
            return Shake.apply(x)
        x = torch.randn(4, 4, 4, 4, requires_grad=True)
        opt_m = torch.compile(backend='eager')(f)
        opt_m(x)

    def test_function_context_save_and_mark(self):
        if False:
            print('Hello World!')
        mod = ModuleWithGradFunc(ContextSaveAndMark)
        (args, kwargs) = ([torch.rand([1])], {})
        before = mod(*args, **kwargs)
        torch._dynamo.reset()
        compiled_model = torch._dynamo.optimize('eager')(mod)
        after = compiled_model(*args, **kwargs)
        self.assertEqual(before, after)

    def test_function_context_mark_and_save(self):
        if False:
            for i in range(10):
                print('nop')
        mod = ModuleWithGradFunc(ContextMarkAndSave)
        (args, kwargs) = ([torch.rand([1])], {})
        before = mod(*args, **kwargs)
        torch._dynamo.reset()
        compiled_model = torch._dynamo.optimize('eager')(mod)
        after = compiled_model(*args, **kwargs)
        self.assertEqual(before, after)

    def test_multi_output(self):
        if False:
            print('Hello World!')
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        class Foo(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                return (x.clone(), x.clone())

            @staticmethod
            def backward(ctx, grad1, grad2):
                if False:
                    i = 10
                    return i + 15
                return grad1 + grad2

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            if False:
                while True:
                    i = 10
            return Foo.apply(x)
        x = torch.randn(3, requires_grad=True)
        result = f(x)
        self.assertEqual(result, Foo.apply(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_graph_break_if_lifted_free_variable(self):
        if False:
            while True:
                i = 10
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()
        delta = torch.randn(3)

        class Foo(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    i = 10
                    return i + 15
                return (x.clone(), (x + delta).clone())

            @staticmethod
            def backward(ctx, grad1, grad2):
                if False:
                    print('Hello World!')
                return grad1 + grad2

        @torch.compile(backend=cnt)
        def f(x):
            if False:
                return 10
            return Foo.apply(x)
        x = torch.randn(3, requires_grad=True)
        result = f(x)
        self.assertEqual(result, Foo.apply(x))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(list(torch._dynamo.utils.counters['graph_break'].values()), [1])

    def test_function_with_bound_free_variable(self):
        if False:
            while True:
                i = 10

        class LowerBound(torch.autograd.Function):

            @staticmethod
            def forward(ctx, inputs, bound):
                if False:
                    i = 10
                    return i + 15
                ctx.save_for_backward(inputs, inputs.new_ones(1) * bound)
                return inputs.clamp(min=bound)

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    i = 10
                    return i + 15
                (inputs, bound) = ctx.saved_tensors
                return ((inputs >= bound) * grad_output, None)

        class MyMod(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.gamma = torch.nn.Parameter(torch.rand([4, 128, 32, 32]))

            def forward(self, x):
                if False:
                    return 10
                gamma = LowerBound.apply(self.gamma, 1)
                return x + gamma
        mod = MyMod()
        (args, kwargs) = ([torch.rand([4, 128, 32, 32])], {})
        before = mod(*args, **kwargs)
        compiled_model = torch._dynamo.optimize('eager')(mod)
        after = compiled_model(*args, **kwargs)
        self.assertEqual(before, after)

    def test_smoke_from_test_autograd(self):
        if False:
            return 10

        class Func(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    i = 10
                    return i + 15
                out0 = x.clone()
                out1 = x.clone()
                ctx.mark_non_differentiable(out1)
                ctx._materialize_non_diff_grads = False
                return (out0, out1)

            @staticmethod
            def backward(ctx, g0, g1):
                if False:
                    i = 10
                    return i + 15
                assert g1 is None
                return g0

        def mult1(x):
            if False:
                print('Hello World!')
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
                    for i in range(10):
                        print('nop')
                (x, y) = ctx.saved_tensors
                return (grad_output * y)[:, None, None] / x
        mult2 = Mult.apply

        class Double(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    for i in range(10):
                        print('nop')
                y = x ** 2
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    return 10
                (x, _) = ctx.saved_tensors
                return grad_output * 2 * x

        class Double2(Double):

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    i = 10
                    return i + 15
                (x, y) = ctx.saved_tensors
                return grad_output * 2 * y / x
        double = Double.apply
        double2 = Double2.apply

        class Identity(torch.autograd.Function):

            @staticmethod
            def forward(ctx, a, b):
                if False:
                    return 10
                return (a, a + b)

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                if False:
                    while True:
                        i = 10
                return (grad_a + grad_b, grad_b)

        class MyFunc2(torch.autograd.Function):

            @staticmethod
            def forward(ctx, inp):
                if False:
                    while True:
                        i = 10
                return inp.clone()

            @staticmethod
            def backward(ctx, gO):
                if False:
                    print('Hello World!')
                return torch.tensor(float('nan')).expand(10, 10)

        def run_fn(a):
            if False:
                print('Hello World!')
            out = MyFunc2.apply(a)
            return out.sum()

        class MyFn(torch.autograd.Function):

            @staticmethod
            def forward(ctx, inp):
                if False:
                    return 10
                return inp.view_as(inp)

            @staticmethod
            def backward(ctx, grad):
                if False:
                    return 10
                return grad

        class MyAdder(torch.autograd.Function):

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
                    while True:
                        i = 10
                return (grad, grad)

        class InplaceMul(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    i = 10
                    return i + 15
                result = x.mul_(2)
                ctx.mark_dirty(result)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    return 10
                pass

            @staticmethod
            def jvp(ctx, x_t):
                if False:
                    for i in range(10):
                        print('nop')
                if jvp_err:
                    return x_t
                else:
                    return x_t.mul_(2)

        class MyFn2(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return (x + y, x)

            @staticmethod
            def vjp(ctx, gO1, gO2):
                if False:
                    return 10
                return (gO1 + gO2, gO1)

            @staticmethod
            def jvp(ctx, x_t, y_t):
                if False:
                    for i in range(10):
                        print('nop')
                return (x_t + y_t, fn(x_t))

        class MyFn3(torch.autograd.Function):

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
                    for i in range(10):
                        print('nop')
                return (grad, None)

        def test():
            if False:
                for i in range(10):
                    print('nop')
            a = torch.tensor(1.0, requires_grad=True)
            out = Func.apply(a)[0]
            out.backward()
            x = torch.ones(2, 4, 4).requires_grad_()
            mult2(x)
            x = torch.tensor(2).double().requires_grad_()
            double(x)
            double2(x)
            x = torch.randn(5, 5, requires_grad=True)
            y = torch.randn(5, 5, requires_grad=True)
            (q, p) = Identity.apply(x, y)
            a = torch.rand(1, 2)
            b = torch.rand(1, requires_grad=True)
            view_a = MyFn.apply(a)
            a = torch.ones(2, requires_grad=True)
            b = torch.ones(2, requires_grad=True)
            c = MyAdder.apply(a.clone(), b)
            c.sum().backward()
            z = torch.tensor(1.0, requires_grad=True)
            x = z.clone()
            y = InplaceMul.apply(x)
            a = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
            b = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
            c = torch.tensor(1.0, dtype=torch.double)
            d = torch.tensor(1.0, dtype=torch.double)
            MyFn2.apply(a, b)
            MyFn2.apply(c, d)
            base = torch.rand(10, requires_grad=True)
            foo = MyFn3.apply(base, False)
        test()
        opt_test = torch._dynamo.optimize('eager')(test)
        opt_test()

    def test_tensor_subclass_intermediary_input(self):
        if False:
            return 10

        class FooTensor(torch.Tensor):

            @staticmethod
            def __new__(cls, data, config, scale):
                if False:
                    return 10
                self = torch.Tensor._make_wrapper_subclass(cls, config[0], strides=config[1], storage_offset=config[2], dtype=config[3], layout=config[4], requires_grad=config[5], device=data.device)
                self._data = data
                self._config = config
                self._scale = scale
                return self

            def __repr__(self):
                if False:
                    while True:
                        i = 10
                return 'FooTensor'

            def __tensor_flatten__(self):
                if False:
                    i = 10
                    return i + 15
                return (('_data',), (self._config, self._scale))

            @staticmethod
            def __tensor_unflatten__(tensors, metadatas):
                if False:
                    i = 10
                    return i + 15
                return FooTensor(tensors['_data'], metadatas[0], metadatas[1])

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs=None):
                if False:
                    i = 10
                    return i + 15
                if func == torch.ops.aten.clone.default:
                    return FooTensor(args[0]._data.clone(), args[0]._config, args[0]._scale)
                elif func == torch.ops.aten.view.default:
                    new_data = args[0]._data.view(*args[1:])
                    return FooTensor(new_data, args[0]._config, args[0]._scale)
                raise NotImplementedError()
            __torch_function__ = torch._C._disabled_torch_function_impl

        class foo_autograd_fn(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                x2 = x._data + 1.0
                x3 = FooTensor(x2, x._config, x._scale)
                return x3._data

            @staticmethod
            def backward(ctx, g):
                if False:
                    return 10
                return g
        x_ref = torch.randn(4, 4).requires_grad_(True)
        x = copy.deepcopy(x_ref)
        scale = torch.tensor(1.0)
        torch._dynamo.allow_in_graph(FooTensor)

        def foo(x, scale):
            if False:
                while True:
                    i = 10
            config = (x.size(), x.stride(), x.storage_offset(), x.dtype, x.layout, x.requires_grad)
            x = FooTensor(x, config, scale)
            x = foo_autograd_fn.apply(x)
            return x
        y_ref = foo(x_ref, scale)
        y_ref.sum().backward()
        foo_opt = torch.compile(foo, backend='eager')
        y = foo_opt(x, scale)
        y.sum().backward()
        self.assertEqual(y, y_ref)
        self.assertEqual(x.grad, x_ref.grad)

    def test_smuggle_symint_issue_111031(self):
        if False:
            for i in range(10):
                print('nop')
        from torch.autograd import Function

        class Foo(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    i = 10
                    return i + 15
                ctx.x0 = x.size(0)
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                if False:
                    while True:
                        i = 10
                return grad_out * ctx.x0
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True, dynamic=True)
        def foo(x):
            if False:
                i = 10
                return i + 15
            return Foo.apply(x)
        foo(torch.randn(2, requires_grad=True))
        self.assertEqual(cnts.frame_count, 1)

    def test_smuggle_tensor_and_complex_structures(self):
        if False:
            print('Hello World!')
        from torch.autograd import Function

        class Foo(Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                ctx.x0 = x
                ctx.x1 = [1, 2, 3]
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                if False:
                    while True:
                        i = 10
                x0mul = grad_out * ctx.x0
                for i in ctx.x1:
                    x0mul = x0mul * i + x0mul
                return x0mul
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True, dynamic=True)
        def foo(x):
            if False:
                i = 10
                return i + 15
            return Foo.apply(x)
        foo(torch.randn(2, requires_grad=True))
        self.assertEqual(cnts.frame_count, 1)
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()