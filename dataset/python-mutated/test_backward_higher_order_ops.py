import functools
import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
from torch import _inductor as inductor
from torch._dynamo import compiled_autograd
from torch._dynamo._trace_wrapped_higher_order_op import trace_wrapped
from torch._dynamo.testing import normalize_gm
from torch._dynamo.utils import counters
from torch.fx.experimental.proxy_tensor import make_fx

def _multiply(x):
    if False:
        i = 10
        return i + 15
    return x * x

def _multiply_invoke(grad):
    if False:
        while True:
            i = 10
    return trace_wrapped(grad, fn=_multiply)

class BackwardHigherOrderOpTests(torch._dynamo.test_case.TestCase):

    def test_invoke_in_eager(self):
        if False:
            return 10
        x = torch.tensor([0.5, 0.5], requires_grad=True)
        y = torch.tensor([0.5, 0.5], requires_grad=True)

        def fn(x, y):
            if False:
                print('Hello World!')
            x.register_hook(_multiply_invoke)
            return x * y
        out = fn(x, y)
        grad_out = torch.tensor([2.0, 2.0])
        out.backward(grad_out)
        self.assertEqual(x.grad, y * grad_out)

    def test_invoke_in_pt2(self):
        if False:
            while True:
                i = 10
        for backend in ['eager', 'aot_eager', 'inductor']:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            def fn(x, y):
                if False:
                    i = 10
                    return i + 15
                x.register_hook(_multiply_invoke)
                return x * y
            fn = torch._dynamo.optimize(backend)(fn)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            out.backward(grad_out)
            self.assertEqual(x.grad, grad_out * y)

    def test_invoke_make_fx_forward_contrived(self):
        if False:
            while True:
                i = 10
        x = torch.tensor([0.5, 0.5], requires_grad=True)
        out = make_fx(_multiply_invoke)(x)
        self.assertEqual(out(x), torch.tensor([0.25, 0.25]))
        actual = normalize_gm(out.print_readable(False))
        expected = 'class _multiply_invoke(torch.nn.Module):\n    def forward(self, grad_1: "f32[2]"):\n        trace_wrapped: "f32[2]" = torch__dynamo__trace_wrapped_higher_order_op_self_invoke(grad_1);  grad_1 = None\n        assert_1: "f32[2]" = torch__dynamo__trace_wrapped_higher_order_op__assert_meta(trace_wrapped, (2,), (1,), torch.float32);  trace_wrapped = None\n        detach: "f32[2]" = torch.ops.aten.detach.default(assert_1);  assert_1 = None\n        detach_1: "f32[2]" = torch.ops.aten.detach.default(detach);  detach = None\n        detach_2: "f32[2]" = torch.ops.aten.detach.default(detach_1);  detach_1 = None\n        detach_3: "f32[2]" = torch.ops.aten.detach.default(detach_2);  detach_2 = None\n        return detach_3\n'
        self.assertExpectedInline(actual, expected)

    def test_invoke_make_bw(self):
        if False:
            i = 10
            return i + 15
        x = torch.tensor([0.5, 0.5], requires_grad=True)

        def fwd(x):
            if False:
                return 10
            z = x * x
            return z + z
        res = fwd(x)
        res.backward(torch.tensor([1.0, 1.0]))
        out = make_fx(_multiply_invoke)(x.grad)
        self.assertEqual(out(x.grad), torch.tensor([4.0, 4.0]))
        actual = normalize_gm(out.print_readable(False))
        expected = 'class _multiply_invoke(torch.nn.Module):\n    def forward(self, grad_1: "f32[2]"):\n        trace_wrapped: "f32[2]" = torch__dynamo__trace_wrapped_higher_order_op_self_invoke(grad_1);  grad_1 = None\n        assert_1: "f32[2]" = torch__dynamo__trace_wrapped_higher_order_op__assert_meta(trace_wrapped, (2,), (1,), torch.float32);  trace_wrapped = None\n        return assert_1\n'
        self.assertExpectedInline(actual, expected)

    def test_invoke_in_pt2_compiled_autograd(self):
        if False:
            i = 10
            return i + 15
        graph = None

        def compiler_fn(gm):
            if False:
                while True:
                    i = 10

            def inner_compiler(gm_, example_inputs_):
                if False:
                    print('Hello World!')
                nonlocal graph
                self.assertEqual(graph, None)
                graph = gm_
                return inductor.compile(gm_, example_inputs_)
            return torch.compile(gm, backend=inner_compiler, fullgraph=True, dynamic=True)
        for backend in ['eager', 'aot_eager', 'inductor']:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            def fn(x, y):
                if False:
                    print('Hello World!')
                x.register_hook(_multiply_invoke)
                return x + y
            fn = torch._dynamo.optimize(backend)(fn)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            with compiled_autograd.enable(compiler_fn):
                out.backward(grad_out)
            actual = normalize_gm(graph.print_readable(False))
            self.assertEqual(x.grad, grad_out * grad_out)
            expected = 'class GraphModule(torch.nn.Module):\n    def forward(self, s0 : torch.SymInt, L_inputs_0_ : torch.Tensor, L_inputs_1_ : torch.Tensor, L_inputs_2_ : torch.Tensor):\n        getitem = L_inputs_0_\n        getitem_1 = L_inputs_1_\n        getitem_2 = L_inputs_2_\n\n        accumulate_grad__default = torch.ops.inductor.accumulate_grad_.default(getitem_1, getitem);  getitem_1 = None\n\n        call_hook = getitem * getitem;  getitem = None\n\n        accumulate_grad__default_1 = torch.ops.inductor.accumulate_grad_.default(getitem_2, call_hook);  getitem_2 = call_hook = None\n        return ()\n'
            self.assertExpectedInline(actual, expected)
            graph = None

    def test_invoke_in_pt2_compiled_autograd_side_effect(self):
        if False:
            return 10

        def _side_effect_stateful_fn2(x, obj):
            if False:
                return 10
            obj.counter = obj.counter + 1
            return _multiply(x)

        def _side_effectful_invoke2(grad, fn):
            if False:
                return 10
            return trace_wrapped(grad, fn=fn)
        graph = None

        def compiler_fn(gm):
            if False:
                return 10

            def inner_compiler(gm_, example_inputs_):
                if False:
                    while True:
                        i = 10
                nonlocal graph
                self.assertEqual(graph, None)
                graph = gm_
                return inductor.compile(gm_, example_inputs_)
            return torch.compile(gm, backend=inner_compiler, fullgraph=True, dynamic=True)
        for backend in ['eager', 'aot_eager', 'inductor']:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            class MyObj:

                def __init__(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.counter = 0
            obj = MyObj()
            inner_fn = functools.partial(_side_effect_stateful_fn2, obj=obj)
            hook_fn = functools.partial(_side_effectful_invoke2, fn=inner_fn)
            x.register_hook(hook_fn)

            def fn(x, y):
                if False:
                    print('Hello World!')
                return x + y
            fn = torch._dynamo.optimize(backend, nopython=True)(fn)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            with compiled_autograd.enable(compiler_fn):
                out.backward(grad_out)
            actual = normalize_gm(graph.print_readable(False))
            self.assertEqual(obj.counter, 1)
            self.assertEqual(x.grad, grad_out + grad_out)
            self.assertExpectedInline(actual, 'class GraphModule(torch.nn.Module):\n    def forward(self, s0 : torch.SymInt, L_inputs_0_ : torch.Tensor, L_inputs_1_ : torch.Tensor, L_inputs_2_ : torch.Tensor):\n        getitem = L_inputs_0_\n        getitem_1 = L_inputs_1_\n        getitem_2 = L_inputs_2_\n\n        accumulate_grad__default = torch.ops.inductor.accumulate_grad_.default(getitem_1, getitem);  getitem_1 = None\n\n        call_hook = getitem * getitem;  getitem = None\n\n        accumulate_grad__default_1 = torch.ops.inductor.accumulate_grad_.default(getitem_2, call_hook);  getitem_2 = call_hook = None\n        return ()\n')
            out = fn(x, y)
            out.backward(grad_out)
            self.assertEqual(obj.counter, 2)
            out = fn(x, y)
            out.backward(grad_out)
            self.assertEqual(obj.counter, 3)
            graph = None

    def test_invoke_in_pt2_compiled_autograd_graph_breaks(self):
        if False:
            for i in range(10):
                print('nop')

        def _graph_breaking_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            print('Boo!')
            return _multiply(x)

        def _graph_break_invoke(grad):
            if False:
                while True:
                    i = 10
            return trace_wrapped(grad, fn=_graph_breaking_fn)

        def compiler_fn(gm):
            if False:
                for i in range(10):
                    print('nop')
            return torch.compile(gm, backend='inductor', fullgraph=True, dynamic=True)
        for backend in ['eager', 'aot_eager', 'inductor']:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            def fn(x, y):
                if False:
                    print('Hello World!')
                x.register_hook(_graph_break_invoke)
                return x + y
            fn = torch._dynamo.optimize(backend, nopython=True)(fn)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, 'print'):
                with compiled_autograd.enable(compiler_fn):
                    out.backward(grad_out)
            graph = None
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()