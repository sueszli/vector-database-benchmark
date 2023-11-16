from typing import Union, Callable, List, Any, Optional, Dict
from unittest.mock import patch
from torch.testing._internal.common_utils import TestCase, run_tests, IS_ARM64, IS_MACOS, IS_X86, compare_equal_outs_and_grads, outs_and_grads, skipIfRocm
from torch.testing._internal.two_tensor import TwoTensor, TwoTensorMode
import copy
import torch
import torch.nn as nn
import torch.utils._pytree as pytree
import unittest
import warnings
import itertools
from contextlib import nullcontext
from functools import partial
from torch.nn.utils.rnn import PackedSequence
from torch.testing._internal.common_device_type import instantiate_device_type_tests, toleranceOverride, tol
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_modules import module_db, modules
from torch.testing._internal.control_flow_opinfo_db import control_flow_opinfo_db
from torch.testing._internal.optests import _test_aot_autograd_forwards_backwards_helper, aot_autograd_check
from functorch import grad, vjp, vmap, jacrev, make_fx
from torch._functorch.aot_autograd import aot_module_simplified, aot_export_module, aot_export_joint_simple
from functorch.compile import nnc_jit, compiled_function, compiled_module, min_cut_rematerialization_partition, aot_function, aot_module, nop, default_partition, default_decompositions, memory_efficient_fusion, get_aot_compilation_context, make_boxed_compiler
from torch._decomp import decomposition_table
from torch.testing._internal.common_device_type import ops
from common_utils import decorate, xfail, skip, skipOps, decorateForModules
from torch._subclasses.fake_tensor import DynamicOutputShapeException, FakeTensorMode
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import ShapeEnv, GuardOnDataDependentSymNode
USE_TORCHVISION = False
try:
    import torchvision
    USE_TORCHVISION = True
except ImportError:
    warnings.warn("Couldn't import torchvision. Some of our tests use it, try to install it with commands from pytorch.org, post-fixed with `--no-deps` to avoid overwriting the pytorch installation", UserWarning)
USE_NETWORKX = False
try:
    import networkx
    USE_NETWORKX = True
except ImportError:
    warnings.warn('Some tests use networkx but it was not installed', UserWarning)

class AOTTestCase(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()

class TestPythonKey(AOTTestCase):

    def test_make_fx(self, device):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                i = 10
                return i + 15
            return torch.sin(x)
        inp = torch.randn(3)
        fx_f = make_fx(f)(inp)
        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_make_fx_grad(self, device):
        if False:
            return 10

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sin(x).sum()
        inp = torch.randn(3)
        f = grad(f)
        fx_f = make_fx(f)(inp)
        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_scalar_device(self, device):
        if False:
            print('Hello World!')

        def f(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return a + b
        inps = [torch.randn(3, device=device), torch.tensor(5)]
        fx_f = make_fx(f)(*inps)
        self.assertEqual(fx_f(*inps), f(*inps))

    def test_make_fx_vmap(self, device):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sin(x)
        inp = torch.randn(5, 3)
        f = vmap(f)
        fx_f = make_fx(f)(inp)
        new_inp = torch.randn(5, 3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_make_fx_jacrev(self, device):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.sin().sum()
        inp = torch.randn(3)
        f = jacrev(jacrev(f))
        fx_f = make_fx(f)(inp)
        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_make_fx_vjp(self, device):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                print('Hello World!')
            return torch.sin(x).sum()
        primals = torch.randn(3)
        (_, vjp_fn) = vjp(f, primals)
        cotangent = torch.randn(())
        fx_f = make_fx(vjp_fn)(cotangent, True, True)
        new_cotangent = torch.randn(())
        self.assertEqual(fx_f(new_cotangent, True, True), vjp_fn(new_cotangent))

    def test_make_fx_functionalize(self, device):
        if False:
            print('Hello World!')
        from functorch.experimental import functionalize

        def fn(a):
            if False:
                for i in range(10):
                    print('nop')
            a = a * 2
            a.relu_()
            return a
        a = torch.randn(3, device=device)
        symbolic_gm = torch.fx.symbolic_trace(fn)
        includes_method_relu_ = any((str(n.target) == 'relu_' for n in symbolic_gm.graph.nodes))
        self.assertTrue(includes_method_relu_)
        gm = make_fx(functionalize(symbolic_gm))(a)
        includes_aten_relu = any((n.target == torch.ops.aten.relu.default for n in gm.graph.nodes))
        self.assertTrue(includes_aten_relu)

    def test_make_fx_no_decompose(self, device):
        if False:
            for i in range(10):
                print('nop')
        return self.skipTest('error: maximum recursion reached')

        def f(x):
            if False:
                print('Hello World!')
            return torch.tanh(x).sum()
        fx_f = make_fx(grad(f))(torch.randn(5))
        ops = {i.target for i in fx_f.graph.nodes}
        self.assertEqual(torch.ops.aten.tanh_backward in ops, True)
        fx_f = make_fx(grad(f), decomposition_table)(torch.randn(5))
        ops = {i.target for i in fx_f.graph.nodes}
        self.assertEqual(torch.ops.aten.tanh_backward in ops, False)

    def test_nnc_jit(self, device):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                return 10
            return torch.sin(x)
        jit_f = nnc_jit(f)
        inp = torch.randn(3)
        self.assertEqual(jit_f(inp), f(inp))

    def test_nnc_scalar(self, device):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                print('Hello World!')
            return torch.sin(x)
        jit_f = nnc_jit(f)
        inp = torch.randn(())
        self.assertEqual(jit_f(inp), f(inp))

    def test_nnc_pytrees(self, device):
        if False:
            return 10

        def f(x):
            if False:
                i = 10
                return i + 15
            return [torch.sin(x[0])]
        jit_f = nnc_jit(f)
        inp = [torch.randn(3)]
        self.assertEqual(jit_f(inp), f(inp))

    def test_external_calls(self, device):
        if False:
            while True:
                i = 10

        def f(a, b):
            if False:
                i = 10
                return i + 15
            return torch.mv(a, b)
        jit_f = nnc_jit(f)
        inp = [torch.randn(3, 3), torch.randn(3)]
        self.assertEqual(jit_f(*inp), f(*inp))

    def test_nnc_passthrough(self, device):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                i = 10
                return i + 15
            return (x + y, y)
        inp = (torch.randn(3), torch.randn(3))
        jit_f = nnc_jit(f)
        self.assertEqual(jit_f(*inp), f(*inp))

        def f(x):
            if False:
                return 10
            x['a'] = x['a'] * 2
            return x
        inp = ({'a': torch.randn(3), 'b': torch.randn(3)},)
        jit_f = nnc_jit(f)
        self.assertEqual(jit_f(*inp), f(*inp))

    @unittest.skipIf(not USE_TORCHVISION, 'test requires torchvision')
    def test_resnet18_backward_trace(self, device):
        if False:
            print('Hello World!')
        mod = torchvision.models.resnet18()

        def f(x):
            if False:
                return 10
            out = mod(x)
            out.sum().backward()
            return [a.grad for a in mod.parameters()]
        inp = torch.randn(3, 3, 250, 250, requires_grad=True)
        grads = f(inp)
        mod.zero_grad()
        mod(inp).sum().backward()
        grads2 = [a.grad for a in mod.parameters()]
        self.assertEqual(grads, grads2)

def get_base(t):
    if False:
        while True:
            i = 10
    return t._base if t._is_view() else t

def is_in_base(t, maybe_tensors):
    if False:
        for i in range(10):
            print('nop')
    t_base = get_base(t)
    for maybe_tensor in maybe_tensors:
        if isinstance(maybe_tensor, torch.Tensor):
            if t_base is get_base(maybe_tensor):
                return True
    return False

class TestAOTAutograd(AOTTestCase):

    @patch('functorch.compile.config.debug_assert', True)
    def verify_aot_autograd(self, f, inp_: Union[Callable, List[Any]], *, test_mutation: bool=False, decompositions: Optional[Dict]=None, dynamic: bool=False, make_inputs_subclasses: bool=False):
        if False:
            for i in range(10):
                print('nop')
        for keep_input_mutations in [True, False]:
            if isinstance(inp_, Callable):
                inp_callable = inp_
                with TwoTensorMode() if make_inputs_subclasses else nullcontext():
                    (inp_copy, graph_inps_copy) = inp_callable()
                    (inp, graph_inps) = inp_callable()
            else:
                inp_copy = []
                inp = []
                dupes_map = {}
                for (i, x) in enumerate(inp_):
                    if x in dupes_map:
                        x_dupe_idx = dupes_map[x]
                        inp_copy.append(inp_copy[x_dupe_idx])
                        inp.append(inp[x_dupe_idx])
                    else:
                        dupes_map[x] = i
                        if not isinstance(x, torch.Tensor):
                            x_copy = x
                            x_copy2 = x
                        else:
                            x_copy = x.clone().detach().requires_grad_(x.requires_grad)
                            x_copy2 = x.clone().detach().requires_grad_(x.requires_grad)
                            if x.requires_grad and (not x.is_leaf):
                                x_copy = x_copy.clone()
                                x_copy2 = x_copy2.clone()
                        inp_copy.append(x_copy)
                        inp.append(x_copy2)
                if test_mutation:
                    graph_inps = [x.add(1) for x in inp]
                    graph_inps_copy = [x.add(1) for x in inp_copy]
                else:
                    graph_inps = inp
                    graph_inps_copy = inp_copy
            fw_graph_cell = [None]
            if isinstance(f, nn.Module):
                compiled_f = aot_module(f, fw_compiler=make_boxed_compiler(partial(extract_graph, graph_cell=fw_graph_cell)), bw_compiler=nop, decompositions=decompositions, keep_inference_input_mutations=keep_input_mutations, dynamic=dynamic)
            else:
                compiled_f = aot_function(f, fw_compiler=make_boxed_compiler(partial(extract_graph, graph_cell=fw_graph_cell)), bw_compiler=nop, decompositions=decompositions, keep_inference_input_mutations=keep_input_mutations, dynamic=dynamic)
            (ref_out, ref_grad) = outs_and_grads(f, graph_inps, inp)
            (test_out, test_grad) = outs_and_grads(compiled_f, graph_inps_copy, inp_copy)
            self.assertEqual(ref_grad, test_grad)
            if isinstance(ref_out, torch.Tensor):
                self.assertTrue(isinstance(test_out, torch.Tensor))
                (ref_out, test_out) = ([ref_out], [test_out])
            for (ref_o, test_o) in zip(ref_out, test_out):
                if isinstance(ref_o, torch.Tensor):
                    self.assertEqual(ref_o.requires_grad, test_o.requires_grad)
                    self.assertEqual(ref_o.is_leaf, test_o.is_leaf)
                    ref_is_view_of_non_interm = is_in_base(ref_o, graph_inps) or is_in_base(ref_o, ref_out)
                    test_is_view_of_non_interm = is_in_base(test_o, graph_inps_copy) or is_in_base(test_o, test_out)
                    self.assertEqual(ref_is_view_of_non_interm, test_is_view_of_non_interm)
                    self.assertEqual(ref_o, test_o)
                    if test_mutation:
                        ref_o.mul_(2)
                        test_o.mul_(2)
                        self.assertEqual(ref_o, test_o)
            for (ref_i, test_i) in zip(inp, inp_copy):
                if isinstance(ref_i, torch.Tensor):
                    self.assertEqual(ref_i.requires_grad, test_i.requires_grad)
                self.assertEqual(ref_i, test_i)
        return fw_graph_cell[0]

    def test_non_tensor_and_none_inputs(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            return a * c
        inp = [2, None, torch.ones(3, 3, dtype=torch.float32, requires_grad=True)]
        self.verify_aot_autograd(f, inp)
        inp = [2, None, torch.ones(3, 3, dtype=torch.float32, requires_grad=False)]
        self.verify_aot_autograd(f, inp)

    def test_single_output(self):
        if False:
            i = 10
            return i + 15

        def f(a, b):
            if False:
                return 10
            return a + b
        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)
        inp = [torch.randn(3, 3, requires_grad=False), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)

    def test_multi_output(self):
        if False:
            i = 10
            return i + 15

        def f(a, b):
            if False:
                i = 10
                return i + 15
            return (a + b, a - b)
        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)
        inp = [torch.randn(3, 3, requires_grad=False), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)

    def test_multi_output_list(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return [a + b, a - b]
        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)
        inp = [torch.randn(3, 3, requires_grad=False), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)

    def test_squeeze_mutation(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                return 10
            b = a.clone().squeeze(-1)
            b.add_(1.0)
            return a + b
        inp = [torch.randn(3, 1, requires_grad=True)]
        self.verify_aot_autograd(f, inp, dynamic=True)
        inp = [torch.randn(3, 1, requires_grad=False)]
        self.verify_aot_autograd(f, inp, dynamic=True)

    def test_complex_linear(self):
        if False:
            print('Hello World!')
        inp = [torch.randn(1, 10, 10, dtype=torch.complex64)]

        class F(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.linear = nn.Linear(10, 10, dtype=torch.complex64)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.linear(x).sum().abs()
        self.verify_aot_autograd(F(), inp)

    def test_embedding_bag_view_dynamic(self):
        if False:
            i = 10
            return i + 15

        class F(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.emb = torch.nn.EmbeddingBag(100, 8, sparse=True)

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                return self.emb(x, y).view(-1)
        x = torch.arange(3)
        y = torch.arange(3)
        self.verify_aot_autograd(F(), [x, y], dynamic=False)
        self.verify_aot_autograd(F(), [x, y], dynamic=True)

    def test_input_mutation_simple(self):
        if False:
            while True:
                i = 10

        def f(a):
            if False:
                print('Hello World!')
            a.mul_(2)
            return a * 3
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None\n    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None\n    mul_1 = torch.ops.aten.mul.Tensor(mul, 3)\n    return [mul, mul_1]')

    def test_input_mutation_simple_with_none_and_nontensor(self):
        if False:
            i = 10
            return i + 15

        def f(a, b, c):
            if False:
                while True:
                    i = 10
            return a * c
        f_compiled = aot_function(f, nop)
        for req_grad in [True, False]:
            inp = [torch.ones(3, 3, requires_grad=req_grad), None, 3]
            out_ref = f(*inp)
            out_test = f_compiled(*inp)
            self.assertEqual(out_ref, out_test)

    def test_mutates_input_noncontiguous(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                i = 10
                return i + 15
            a.add_(1)
            return ()
        f_compiled = aot_function(f, nop)
        ref = torch.ones(4, requires_grad=True) + 0
        ref_view = ref[0::2]
        test = torch.ones(4, requires_grad=True) + 0
        test_view = test[0::2]
        out_ref = f(ref_view)
        out_test = f_compiled(test_view)
        print(ref)
        print(test)
        self.assertEqual(ref, test)

    def test_input_mutation_modifies_autograd_meta_of_aliases(self):
        if False:
            return 10

        def f(a):
            if False:
                i = 10
                return i + 15
            a.mul_(2)
            out = a + 1
            return out.detach()
        x_ref = torch.ones(3, 3, requires_grad=True).clone()
        x_ref_view = x_ref.view(3, 3)
        x_test = torch.ones(3, 3, requires_grad=True).clone()
        x_test_view = x_test.view(3, 3)
        f_compiled = aot_function(f, nop, keep_inference_input_mutations=True)
        f(x_ref)
        f_compiled(x_test)
        self.assertEqual(x_ref_view, x_test_view)
        self.assertEqual(x_ref_view._version, x_test_view._version)
        self.assertEqual(x_ref_view.grad_fn.__class__, x_test_view.grad_fn.__class__)
        (x_ref * x_ref_view).sum().backward()
        (x_test * x_test_view).sum().backward()
        self.assertEqual(x_ref.grad, x_test.grad)
        self.assertEqual(x_ref_view.grad, x_test_view.grad)

    def test_outputs_are_aliased(self):
        if False:
            return 10

        def f(a):
            if False:
                print('Hello World!')
            b = a.mul(2)
            c = b.view(-1)
            return (b, c)
        f_compiled = aot_function(f, nop)
        for req_grad in [True, False]:
            inp = torch.ones(3, requires_grad=req_grad)
            out_ref = f(inp)
            out_test = f_compiled(inp)
            self.assertEqual(out_ref[0], out_test[0])
            self.assertEqual(out_ref[1], out_test[1])
            out_ref[0].mul_(3)
            out_test[0].mul_(3)
            self.assertEqual(out_ref[0], out_test[0])
            self.assertEqual(out_ref[1], out_test[1])

    def test_input_mutation_is_output(self):
        if False:
            i = 10
            return i + 15

        def f(a):
            if False:
                while True:
                    i = 10
            a.mul_(2)
            return a
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None\n    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None\n    return [mul, mul]')

    def test_input_mutation_multiple(self):
        if False:
            while True:
                i = 10

        def f(a, b, c):
            if False:
                while True:
                    i = 10
            a.mul_(2)
            c.mul_(2)
            return a + b + c

        def create_inp(req_grad):
            if False:
                for i in range(10):
                    print('nop')
            return [torch.ones(3, 3, requires_grad=req_grad), torch.ones(3, 3, requires_grad=req_grad), torch.ones(3, 3, requires_grad=req_grad)]
        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1, primals_2, primals_3):\n    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None\n    clone_1 = torch.ops.aten.clone.default(primals_3);  primals_3 = None\n    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None\n    mul_1 = torch.ops.aten.mul.Tensor(clone_1, 2);  clone_1 = None\n    add = torch.ops.aten.add.Tensor(mul, primals_2);  primals_2 = None\n    add_1 = torch.ops.aten.add.Tensor(add, mul_1);  add = None\n    return [mul, mul_1, add_1]')

    def test_input_mutation_metadata(self):
        if False:
            i = 10
            return i + 15

        def f(a, b):
            if False:
                while True:
                    i = 10
            a.transpose_(1, 0)
            return a + b

        def create_inp(req_grad):
            if False:
                print('Hello World!')
            return [torch.ones(3, 3, requires_grad=req_grad), torch.ones(3, 3, requires_grad=req_grad)]
        self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)

    def test_input_output_aliase_custom_autograd_function(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    return 10
                return x

            @staticmethod
            def backward(ctx, gx):
                if False:
                    print('Hello World!')
                return gx * 0.5

        def f(x):
            if False:
                while True:
                    i = 10
            return Foo.apply(x)
        inp = [torch.ones(2, 2, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=False)

    def test_input_mutation_requires_grad_detach(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                return 10
            a.detach().mul_(2)
            return a + 3
        inp = [torch.ones(4, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=False)
        inp = [torch.ones(4, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_mutation_requires_grad_no_grad(self):
        if False:
            return 10

        def f(a):
            if False:
                return 10
            with torch.no_grad():
                a.mul_(2)
            return a + 3
        inp = [torch.ones(4, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=False)

    def test_input_mutation_requires_grad_no_grad_detach_mixed(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a):
            if False:
                i = 10
                return i + 15
            a.detach().mul_(2)
            a.mul_(3)
            with torch.no_grad():
                a.mul_(4)
            return a + 5
        inp = [torch.ones(4, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_mutation_metadata2(self):
        if False:
            while True:
                i = 10

        def f(a):
            if False:
                while True:
                    i = 10
            a.transpose_(1, 0)
            a.mul_(2)
            return a + 1
        inp = [torch.ones(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_mutation_batchnorm(self):
        if False:
            for i in range(10):
                print('nop')

        def f(inpt, weight, bias, running_mean, running_var):
            if False:
                for i in range(10):
                    print('nop')
            return torch._native_batch_norm_legit(inpt, weight, bias, running_mean, running_var, True, 0.5, 1e-05)

        def create_inp(req_grad):
            if False:
                while True:
                    i = 10
            return [torch.ones(2, 5, 5, 5, requires_grad=req_grad), torch.ones(5, requires_grad=req_grad), torch.ones(5, requires_grad=req_grad), torch.ones(5), torch.ones(5)]
        from torch._decomp import get_decompositions
        decompositions = get_decompositions([torch.ops.aten._native_batch_norm_legit_functional, torch.ops.aten.native_batch_norm_backward])
        self.verify_aot_autograd(f, create_inp(True), test_mutation=True, decompositions=decompositions)
        self.verify_aot_autograd(f, create_inp(False), test_mutation=True, decompositions=decompositions)

    def test_batchnorm_inference(self):
        if False:
            while True:
                i = 10
        inp = [torch.ones(2, 5, 5, 5, requires_grad=True), torch.ones(5, requires_grad=True), torch.ones(5, requires_grad=True), torch.ones(5), torch.ones(5)]
        m = torch.nn.BatchNorm2d(4, 4)
        m.eval()
        fw_graph_cell = [None]
        inp = torch.ones(4, 4, 4, 4)
        fw_graph_cell = [None]
        compiled_m = aot_module(m, fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell), bw_compiler=nop, keep_inference_input_mutations=True)
        inp = torch.ones(4, 4, 4, 4)
        with torch.no_grad():
            out = compiled_m(inp)
        code = fw_graph_cell[0].code.strip()
        self.assertTrue('copy_' not in str(code))

    def test_input_output_view_simple(self):
        if False:
            i = 10
            return i + 15

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            return a.view(-1)
        inp = [torch.ones(2, 2, requires_grad=False).add(1)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(2, 2, requires_grad=True).add(1)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    view = torch.ops.aten.view.default(primals_1, [-1]);  primals_1 = None\n    return [view]')

    def test_input_output_view_mutate_multiple(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            a.mul_(2)
            c.mul_(3)
            return (b.view(2, 2), c.view(2, 2))

        def create_inp(req_grad):
            if False:
                print('Hello World!')
            return [torch.ones(2, 2, requires_grad=req_grad).add(1), torch.ones(2, 2, requires_grad=req_grad).add(1), torch.ones(2, 2, requires_grad=req_grad).add(1)]
        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1, primals_2, primals_3):\n    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None\n    clone_1 = torch.ops.aten.clone.default(primals_3);  primals_3 = None\n    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None\n    mul_1 = torch.ops.aten.mul.Tensor(clone_1, 3);  clone_1 = None\n    view = torch.ops.aten.view.default(primals_2, [2, 2]);  primals_2 = None\n    view_2 = torch.ops.aten.view.default(mul_1, [2, 2])\n    return [mul, mul_1, view, view_2]')

    def test_input_output_view_metadata_mutate_multiple(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b, c):
            if False:
                while True:
                    i = 10
            b.mul_(3)
            c.t_()
            return (a.view(2, 2), b.view(2, 2), c.view(2, 2))

        def create_inp(req_grad):
            if False:
                for i in range(10):
                    print('nop')
            return [torch.ones(2, 2, requires_grad=req_grad).add(1), torch.ones(2, 2, requires_grad=req_grad).add(1), torch.ones(2, 2, requires_grad=req_grad).add(1)]
        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1, primals_2, primals_3):\n    clone = torch.ops.aten.clone.default(primals_2);  primals_2 = None\n    view = torch.ops.aten.view.default(primals_3, [2, 2]);  primals_3 = None\n    mul = torch.ops.aten.mul.Tensor(clone, 3);  clone = None\n    t = torch.ops.aten.t.default(view);  view = None\n    view_1 = torch.ops.aten.view.default(primals_1, [2, 2]);  primals_1 = None\n    view_3 = torch.ops.aten.view.default(t, [2, 2])\n    view_4 = torch.ops.aten.view.default(mul, [2, 2])\n    return [mul, t, view_1, view_4, view_3]')

    def test_input_mutation_and_output_view(self):
        if False:
            i = 10
            return i + 15

        def f(a):
            if False:
                while True:
                    i = 10
            a.add_(1)
            return a.view(-1)
        inp = [torch.ones(2, 2, requires_grad=False).add(1)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(2, 2, requires_grad=True).add(1)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None\n    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None\n    view_1 = torch.ops.aten.view.default(add, [-1])\n    return [add, view_1]')

    def test_input_mutation_output_view_multiple(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b, c, d):
            if False:
                return 10
            b.transpose_(1, 0)
            c.add_(1)
            return (d + 1, b.diagonal(), a + c)

        def create_inp(req_grad):
            if False:
                return 10
            return [torch.arange(4, requires_grad=req_grad, dtype=torch.float32).view(2, 2).add(1), torch.arange(4, requires_grad=req_grad, dtype=torch.float32).view(2, 2).add(1), torch.ones(2, 2, requires_grad=req_grad).add(1), torch.ones(2, 2, requires_grad=req_grad).add(1)]
        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1, primals_2, primals_3, primals_4):\n    view = torch.ops.aten.view.default(primals_2, [2, 2]);  primals_2 = None\n    clone = torch.ops.aten.clone.default(primals_3);  primals_3 = None\n    transpose = torch.ops.aten.transpose.int(view, 1, 0);  view = None\n    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None\n    add_1 = torch.ops.aten.add.Tensor(primals_4, 1);  primals_4 = None\n    diagonal = torch.ops.aten.diagonal.default(transpose)\n    add_2 = torch.ops.aten.add.Tensor(primals_1, add);  primals_1 = None\n    return [transpose, add, add_1, diagonal, add_2]')

    def test_output_aliases_intermediate_single(self):
        if False:
            while True:
                i = 10

        def f(a):
            if False:
                return 10
            out = torch.mul(a, 3)
            return out.view(-1)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None\n    view = torch.ops.aten.view.default(mul, [-1]);  mul = None\n    return [view]')

    def test_output_aliases_input_multi_output_view_should_raise_autograd_error(self):
        if False:
            print('Hello World!')

        def f1(a):
            if False:
                i = 10
                return i + 15
            return list(a.unbind(0))
        f1_compiled = aot_function(f1, nop)
        inp1 = torch.ones(3, 3, requires_grad=True).clone()
        inp2 = torch.ones(3, 3, requires_grad=True).clone()
        inp3 = torch.ones(3, 3, requires_grad=True).clone()
        with self.assertRaisesRegex(RuntimeError, 'Such functions do not allow the output views'):
            out_test1 = f1_compiled(inp1)
            out_test1[0].mul_(2)
        with self.assertRaisesRegex(RuntimeError, 'Such functions do not allow the output views'):
            out_test2 = f1_compiled(inp2)
            inp2.mul_(2)
            grad_fn = out_test2[0].grad_fn
        with self.assertRaisesRegex(RuntimeError, 'Such functions do not allow the output views'):
            out_test3 = f1_compiled(inp3)
            out_test1[0].detach().mul_(2)
            grad_fn = out_test2[0].grad_fn

    def test_output_aliases_input_multi_output_view(self):
        if False:
            i = 10
            return i + 15

        def f1(a):
            if False:
                while True:
                    i = 10
            return list(a.unbind(0))
        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f1_compiled = aot_function(f1, nop)
        out_ref = f1(inp_ref)
        out_test = f1_compiled(inp)
        self.assertTrue(all(('CompiledFunctionBackward' in str(o.grad_fn) for o in out_test)))
        sum(out_ref).sum().backward()
        sum(out_test).sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        def f3(a):
            if False:
                while True:
                    i = 10
            return (*list(a.unbind(0)), a.view(a.shape))
        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f3_compiled = aot_function(f3, nop)
        inp_ref_clone = inp_ref.clone()
        inp_clone = inp.clone()
        out_ref = f3(inp_ref_clone)
        out_test = f3_compiled(inp_clone)
        self.assertTrue(all(('AsStridedBackward' in str(o.grad_fn) for o in out_test[:3])))
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)
        inp_ref_clone.view(-1).mul_(3)
        inp_clone.view(-1).mul_(3)
        (inp_ref + out_ref[-1]).sum().backward()
        (inp + out_test[-1]).sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

    def test_output_aliases_intermediate_multi_output_view(self):
        if False:
            print('Hello World!')

        def f1(a):
            if False:
                for i in range(10):
                    print('nop')
            out = torch.mul(a, 3)
            return list(out.unbind(0))
        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f1_compiled = aot_function(f1, nop)
        out_ref = f1(inp_ref)
        out_test = f1_compiled(inp)
        self.assertTrue(all(('CompiledFunctionBackward' in str(o.grad_fn) for o in out_test)))
        sum(out_ref).sum().backward()
        sum(out_test).sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        def f2(a):
            if False:
                print('Hello World!')
            out = torch.mul(a, 3)
            return (*list(out.unbind(0)), out)
        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f2_compiled = aot_function(f2, nop)
        out_ref = f2(inp_ref)
        out_test = f2_compiled(inp)
        self.assertTrue(all(('CompiledFunctionBackward' in str(o.grad_fn) for o in out_test)))
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)
        out_ref[-1].sum().backward()
        out_test[-1].sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        def f3(a):
            if False:
                for i in range(10):
                    print('nop')
            out = torch.mul(a, 3)
            return (*list(out.unbind(0)), out.view(out.shape))
        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f3_compiled = aot_function(f3, nop)
        out_ref = f3(inp_ref)
        out_test = f3_compiled(inp)
        self.assertTrue(all(('CompiledFunctionBackward' in str(o.grad_fn) for o in out_test)))
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)
        out_ref[-1].sum().backward()
        out_test[-1].sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        def f4(a):
            if False:
                while True:
                    i = 10
            out = torch.mul(a, 3)
            return (*list(out.unbind(0)), out, out.view(out.shape))
        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f4_compiled = aot_function(f4, nop)
        out_ref = f4(inp_ref)
        out_test = f4_compiled(inp)
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)
        out_ref_sum = out_ref[-1] + out_ref[-2]
        out_test_sum = out_test[-1] + out_test[-2]
        out_ref_sum.sum().backward()
        out_test_sum.sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

    def test_output_aliases_intermediate_mutation_linear(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                return 10
            return (x + 1).view(-1)
        inp = [torch.ones(3, 3, requires_grad=True)]
        from torch._inductor.decomposition import decompositions
        f_compiled = aot_function(f, nop, decompositions=decompositions)
        out_ref = f(*inp)
        out_test = f_compiled(*inp)
        out_ref.mul_(2)
        out_test.mul_(2)
        self.assertEqual(out_ref, out_test)

    def test_output_aliases_intermediate_no_grad(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                i = 10
                return i + 15
            out = torch.mul(a, 3)
            return (out.view(-1), b.add(1))
        inp = [torch.ones(3, 3), torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3), torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1, primals_2):\n    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None\n    view = torch.ops.aten.view.default(mul, [-1]);  mul = None\n    add = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None\n    return [view, add]')

    def test_output_aliases_intermediate_returned_multiple_times(self):
        if False:
            while True:
                i = 10

        def f(a):
            if False:
                while True:
                    i = 10
            out = torch.mul(a, 3)
            out_view = out.view(-1)
            return (out, out_view, out)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_output_aliases_intermediate_multiple(self):
        if False:
            i = 10
            return i + 15

        def f(a):
            if False:
                i = 10
                return i + 15
            out = torch.mul(a, 3)
            return (out.view(-1), out.view(-1))
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None\n    view = torch.ops.aten.view.default(mul, [-1])\n    view_1 = torch.ops.aten.view.default(mul, [-1])\n    return [view, view_1, mul]')

    def test_output_aliases_intermediate_and_returned(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a):
            if False:
                return 10
            out = torch.mul(a, 3)
            return (out.view(-1), out)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None\n    view = torch.ops.aten.view.default(mul, [-1])\n    return [view, mul]')

    def test_output_aliases_intermediate_and_returned_flipped(self):
        if False:
            i = 10
            return i + 15

        def f(a):
            if False:
                i = 10
                return i + 15
            out = torch.mul(a, 3)
            return (out, out.view(-1))
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None\n    view = torch.ops.aten.view.default(mul, [-1])\n    return [mul, view]')

    def test_output_aliases_intermediate_and_returned_different_grad(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                while True:
                    i = 10
            out = torch.mul(a, 3)
            return (out.view(-1), out, out[0].detach())
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None\n    view = torch.ops.aten.view.default(mul, [-1])\n    select = torch.ops.aten.select.int(mul, 0, 0)\n    detach = torch.ops.aten.detach.default(select);  select = None\n    detach_1 = torch.ops.aten.detach.default(detach);  detach = None\n    return [view, mul, detach_1]')

    def test_output_aliases_intermediate_inplace_view(self):
        if False:
            while True:
                i = 10

        def f(a):
            if False:
                while True:
                    i = 10
            out = torch.mul(a, 3)
            out.t_()
            return out
        inp = [torch.ones(2, 4, requires_grad=True)]

    def test_output_aliases_intermediate_inplace_view_with_detach(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                return 10
            out = torch.mul(a, 3)
            out.t_()
            out.detach_()
            return (out, a + 1)
        inp = [torch.ones(2, 4, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(2, 4, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    mul = torch.ops.aten.mul.Tensor(primals_1, 3)\n    t = torch.ops.aten.t.default(mul);  mul = None\n    add = torch.ops.aten.add.Tensor(primals_1, 1);  primals_1 = None\n    return [t, add]')

    def test_output_aliases_intermediate_inplace_view_and_view(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                return 10
            out = torch.mul(a, 3)
            out_view = out.unsqueeze(0)
            out.t_()
            out_view2 = out.unsqueeze(0)
            return (out_view, out, out_view2)
        inp = [torch.ones(2, 4, requires_grad=True)]

    def test_output_aliases_intermediate_multiple_mixed(self):
        if False:
            i = 10
            return i + 15

        def f(a):
            if False:
                print('Hello World!')
            out1 = torch.mul(a, 3)
            out2 = torch.mul(a, 4)
            return (out1.view(-1), out2.transpose(1, 0), out1.transpose(1, 0))
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    mul = torch.ops.aten.mul.Tensor(primals_1, 3)\n    mul_1 = torch.ops.aten.mul.Tensor(primals_1, 4);  primals_1 = None\n    view = torch.ops.aten.view.default(mul, [-1])\n    transpose = torch.ops.aten.transpose.int(mul_1, 1, 0);  mul_1 = None\n    transpose_1 = torch.ops.aten.transpose.int(mul, 1, 0)\n    return [view, transpose, transpose_1, mul]')

    def test_output_all_alias_types(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                i = 10
                return i + 15
            a.transpose_(1, 0)
            tmp = a.mul(2)
            return (tmp.squeeze(), tmp.transpose(1, 0), a.unsqueeze(0))

        def inp_callable(req_grad):
            if False:
                while True:
                    i = 10
            x = torch.ones(1, 2, 4, requires_grad=req_grad).clone()
            return [(x,), (x,)]
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True)
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True, make_inputs_subclasses=True)
        with self.assertRaisesRegex(AssertionError, 'which is currently unsupported in the subclass use case'):
            self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True, make_inputs_subclasses=True)
        fw_graph = self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    view = torch.ops.aten.view.default(primals_1, [1, 2, 4]);  primals_1 = None\n    transpose = torch.ops.aten.transpose.int(view, 1, 0);  view = None\n    mul = torch.ops.aten.mul.Tensor(transpose, 2)\n    squeeze = torch.ops.aten.squeeze.default(mul)\n    transpose_1 = torch.ops.aten.transpose.int(mul, 1, 0)\n    unsqueeze = torch.ops.aten.unsqueeze.default(transpose, 0)\n    return [transpose, squeeze, transpose_1, unsqueeze, mul]')

    def test_input_data_and_metadata_mutation(self):
        if False:
            i = 10
            return i + 15

        def f(a):
            if False:
                print('Hello World!')
            a.t_()
            a[0].mul_(2)
            return a.view(a.shape)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None\n    t = torch.ops.aten.t.default(clone)\n    select = torch.ops.aten.select.int(t, 0, 0);  t = None\n    mul = torch.ops.aten.mul.Tensor(select, 2);  select = None\n    t_1 = torch.ops.aten.t.default(clone);  clone = None\n    select_scatter = torch.ops.aten.select_scatter.default(t_1, mul, 0, 0);  t_1 = mul = None\n    t_2 = torch.ops.aten.t.default(select_scatter);  select_scatter = None\n    t_4 = torch.ops.aten.t.default(t_2)\n    t_6 = torch.ops.aten.t.default(t_2);  t_2 = None\n    view_1 = torch.ops.aten.view.default(t_6, [3, 3]);  t_6 = None\n    return [t_4, view_1]')

    def test_view_and_inplace_view(self):
        if False:
            while True:
                i = 10

        def f(a, b):
            if False:
                while True:
                    i = 10
            a.t_()
            return (b.view(b.shape), a.view(a.shape))

        def create_inp(req_grad):
            if False:
                i = 10
                return i + 15
            return [torch.ones(3, 3, requires_grad=req_grad), torch.ones(3, 3, requires_grad=req_grad)]
        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1, primals_2):\n    view = torch.ops.aten.view.default(primals_1, [3, 3]);  primals_1 = None\n    t = torch.ops.aten.t.default(view);  view = None\n    view_1 = torch.ops.aten.view.default(primals_2, [3, 3]);  primals_2 = None\n    view_2 = torch.ops.aten.view.default(t, [3, 3])\n    return [t, view_1, view_2]')

    def test_view_detach(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a):
            if False:
                i = 10
                return i + 15
            tmp = a.detach()
            a.mul_(2)
            return (a, tmp)
        inp = [torch.ones(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_inplace_requires_grad_true(self):
        if False:
            print('Hello World!')

        def f(a, b):
            if False:
                print('Hello World!')
            a.requires_grad_(True)
            return (a.mul(3), b.mul(4))
        inp = [torch.ones(3, 3, requires_grad=False), torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1, primals_2):\n    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None\n    mul_1 = torch.ops.aten.mul.Tensor(primals_2, 4);  primals_2 = None\n    return [mul, mul_1]')

    def test_input_data_and_metadata_mutation_aliases_other_input(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                return 10
            a.mul_(2)
            b.t_()
            return a.mul(b)

        def inp_callable(req_grad):
            if False:
                while True:
                    i = 10
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            inp1 = x[0]
            inp2 = x[0]
            return ([base], [inp1, inp2])
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True)
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True)
        with self.assertRaisesRegex(RuntimeError, 'Encountered aliased inputs that are mutated in the graph, but'):
            self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True, make_inputs_subclasses=True)
        with self.assertRaisesRegex(RuntimeError, 'Encountered aliased inputs that are mutated in the graph, but'):
            self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True, make_inputs_subclasses=True)

    def test_input_mutation_noncontiguous(self):
        if False:
            i = 10
            return i + 15

        def f(a):
            if False:
                print('Hello World!')
            a.mul_(2)
            return a + 1

        def inp_callable(req_grad):
            if False:
                return 10
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            inp = x[:, 0]
            return ([base], [inp])
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True)
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True)
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True, make_inputs_subclasses=True)
        with self.assertRaisesRegex(AssertionError, 'attempted to compile the backward with incorrect subclass metadata'):
            self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True, make_inputs_subclasses=True)

    def test_input_mutation_false_aliasing(self):
        if False:
            print('Hello World!')

        def f(a, b):
            if False:
                while True:
                    i = 10
            a.mul_(3)
            b.mul_(2)
            return a + b

        def inp_callable1(req_grad):
            if False:
                print('Hello World!')
            base = torch.ones(4, 4, requires_grad=req_grad)
            x = base.add(1)
            a = x[0:2]
            b = x[2:4]
            return ([base], [a, b])
        fw_graph = self.verify_aot_autograd(f, partial(inp_callable1, req_grad=False), test_mutation=True)
        self.verify_aot_autograd(f, partial(inp_callable1, req_grad=True), test_mutation=True)
        self.verify_aot_autograd(f, partial(inp_callable1, req_grad=False), test_mutation=True, make_inputs_subclasses=True)
        with self.assertRaisesRegex(AssertionError, 'attempted to compile the backward with incorrect subclass metadata'):
            self.verify_aot_autograd(f, partial(inp_callable1, req_grad=True), test_mutation=True, make_inputs_subclasses=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, arg0_1, arg1_1):\n    mul = torch.ops.aten.mul.Tensor(arg0_1, 3);  arg0_1 = None\n    mul_1 = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None\n    add = torch.ops.aten.add.Tensor(mul, mul_1)\n    return (mul, mul_1, add)')

        def inp_callable2(req_grad):
            if False:
                while True:
                    i = 10
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            a = x.as_strided((4, 4), (8, 1), storage_offset=0)
            b = x.as_strided((4, 4), (8, 1), storage_offset=28)
            return ([base], [a, b])

        def inp_callable3(req_grad):
            if False:
                print('Hello World!')
            base = torch.ones(4, 4, requires_grad=req_grad)
            x = base.add(1)
            a = x[:, 0:2]
            b = x[:, 2:4]
            return ([base], [a, b])

        def inp_callable4(req_grad):
            if False:
                i = 10
                return i + 15
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            a = x.as_strided((4, 4), (9, 1), storage_offset=0)
            b = x.as_strided((4, 4), (9, 1), storage_offset=22)
            return ([base], [a, b])

        def inp_callable5(req_grad):
            if False:
                for i in range(10):
                    print('nop')
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            a = x.as_strided((4, 4), (9, 1), storage_offset=0)
            b = x.as_strided((4, 4), (9, 1), storage_offset=23)
            return ([base], [a, b])

        def inp_callable_overlap1(req_grad):
            if False:
                for i in range(10):
                    print('nop')
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            a = x.as_strided((4, 4), (9, 1), storage_offset=0)
            b = x.as_strided((4, 4), (9, 1), storage_offset=24)
            return ([base], [a, b])

        def inp_callable_overlap2(req_grad):
            if False:
                print('Hello World!')
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            a = x.as_strided((4, 4), (9, 1), storage_offset=0)
            b = x.as_strided((4, 4), (9, 1), storage_offset=25)
            return ([base], [a, b])
        fw_graph2 = self.verify_aot_autograd(f, partial(inp_callable2, req_grad=False), test_mutation=True)
        fw_graph3 = self.verify_aot_autograd(f, partial(inp_callable3, req_grad=False), test_mutation=True)
        fw_graph4 = self.verify_aot_autograd(f, partial(inp_callable4, req_grad=False), test_mutation=True)
        fw_graph5 = self.verify_aot_autograd(f, partial(inp_callable5, req_grad=False), test_mutation=True)
        fw_graph_overlap1 = self.verify_aot_autograd(f, partial(inp_callable_overlap2, req_grad=False), test_mutation=True)
        fw_graph_overlap2 = self.verify_aot_autograd(f, partial(inp_callable_overlap1, req_grad=False), test_mutation=True)
        self.assertEqual(str(fw_graph.code), str(fw_graph2.code))
        self.assertEqual(str(fw_graph.code), str(fw_graph3.code))
        self.assertEqual(str(fw_graph.code), str(fw_graph4.code))
        self.assertEqual(str(fw_graph.code), str(fw_graph5.code))
        self.assertNotEqual(str(fw_graph.code), str(fw_graph_overlap1.code))
        self.assertNotEqual(str(fw_graph.code), str(fw_graph_overlap2.code))
        self.assertTrue('as_strided_scatter' in str(fw_graph_overlap1.code))
        self.assertTrue('as_strided_scatter' in str(fw_graph_overlap2.code))

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is unavailable')
    def test_mem_leak_from_save_for_bw(self):
        if False:
            return 10

        def f(a, b):
            if False:
                i = 10
                return i + 15
            add = a + a
            split = torch.functional.split(add, [4, 4], dim=1)
            getitem_2 = split[1]
            unsqueeze = getitem_2.unsqueeze(-1)
            mul = unsqueeze * b
            return (getitem_2, mul)
        f_compiled = aot_function(f, nop)
        inps = [torch.ones(8, 8, device='cuda', requires_grad=True), torch.ones(1, 4, 1, device='cuda', requires_grad=True)]
        mem_before = torch.cuda.memory_allocated()
        f_compiled(*inps)
        mem_after = torch.cuda.memory_allocated()
        self.assertTrue(mem_after == mem_before)

    def test_output_aliases_multiple_inputs_get_correct_one(self):
        if False:
            print('Hello World!')

        def f(a, b):
            if False:
                print('Hello World!')
            return (a.view(a.shape), b.view(b.shape))

        def inp_callable(req_grad):
            if False:
                for i in range(10):
                    print('nop')
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.mul(2)
            inp1 = x.view(-1)
            inp2 = x[0]
            return ([base], [inp1, inp2])
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True)
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True)
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True, make_inputs_subclasses=True)
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True, make_inputs_subclasses=True)

    def test_input_mutation_aliases_other_input(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                for i in range(10):
                    print('nop')
            a.add_(1)
            return a + b

        def inp_callable(req_grad):
            if False:
                print('Hello World!')
            base = torch.ones(4, 2, requires_grad=req_grad)
            x = base.add(1)
            inp1 = x[0]
            inp2 = x[0]
            return ([base], [inp1, inp2])
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None\n    as_strided = torch.ops.aten.as_strided.default(clone, [2], [1], 0)\n    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None\n    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [2], [1], 0);  clone = add = None\n    as_strided_2 = torch.ops.aten.as_strided.default(as_strided_scatter, [2], [1], 0)\n    as_strided_3 = torch.ops.aten.as_strided.default(as_strided_scatter, [2], [1], 0)\n    add_1 = torch.ops.aten.add.Tensor(as_strided_2, as_strided_3);  as_strided_2 = as_strided_3 = None\n    return [as_strided_scatter, add_1]')

    def test_input_mutation_aliases_other_input2(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                return 10
            a.add_(1)
            return a + b

        def inp_callable(req_grad):
            if False:
                print('Hello World!')
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            inp1 = x[0]
            inp2 = x
            return ([base], [inp1, inp2])
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None\n    as_strided = torch.ops.aten.as_strided.default(clone, [2], [1], 0)\n    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None\n    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [2], [1], 0);  clone = add = None\n    as_strided_2 = torch.ops.aten.as_strided.default(as_strided_scatter, [2], [1], 0)\n    as_strided_3 = torch.ops.aten.as_strided.default(as_strided_scatter, [2, 2], [2, 1], 0)\n    add_1 = torch.ops.aten.add.Tensor(as_strided_2, as_strided_3);  as_strided_2 = as_strided_3 = None\n    return [as_strided_scatter, add_1]')

    def test_input_mutation_aliases_and_output_alias(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                return 10
            a.add_(1)
            return b.view(b.shape)

        def inp_callable(req_grad):
            if False:
                while True:
                    i = 10
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            return ([base], [x.view(-1), x.view(-1)])
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1):\n    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None\n    as_strided = torch.ops.aten.as_strided.default(clone, [4], [1], 0)\n    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None\n    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [4], [1], 0);  clone = add = None\n    as_strided_6 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)\n    view_1 = torch.ops.aten.view.default(as_strided_6, [4]);  as_strided_6 = None\n    return [as_strided_scatter, view_1]')

    def test_input_aliased_with_mutation_output_alias(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b, c):
            if False:
                while True:
                    i = 10
            c.mul_(2)
            return (b.add(1), c.view(-1))

        def inp_callable(req_grad):
            if False:
                return 10
            base1 = torch.ones(2, 2, requires_grad=req_grad)
            base2 = torch.ones(2, 2, requires_grad=req_grad)
            x = base1.add(1)
            y = base2.add(1)
            return ([base1, base2], [x.view(-1), y, x.view(-1)])
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1, primals_2):\n    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None\n    as_strided_1 = torch.ops.aten.as_strided.default(clone, [4], [1], 0)\n    mul = torch.ops.aten.mul.Tensor(as_strided_1, 2);  as_strided_1 = None\n    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, mul, [4], [1], 0);  clone = mul = None\n    add = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None\n    as_strided_6 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)\n    view_1 = torch.ops.aten.view.default(as_strided_6, [-1]);  as_strided_6 = None\n    return [as_strided_scatter, add, view_1]')

    def test_input_metadata_mutation_aliases(self):
        if False:
            while True:
                i = 10

        def f(a, b):
            if False:
                print('Hello World!')
            a.t_()
            return a + b

        def inp_callable(req_grad):
            if False:
                for i in range(10):
                    print('nop')
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            return ([base], [x.view(-1), x.view(-1)])
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1, primals_2):\n    view = torch.ops.aten.view.default(primals_1, [4]);  primals_1 = None\n    t = torch.ops.aten.t.default(view);  view = None\n    add = torch.ops.aten.add.Tensor(t, primals_2);  primals_2 = None\n    return [t, add]')

    def test_input_mutation_aliases_and_none_require_gradients(self):
        if False:
            i = 10
            return i + 15

        def f(a, b, c):
            if False:
                while True:
                    i = 10
            a.mul_(2)
            return (b + 1, c + 1)

        def inp_callable(req_grad):
            if False:
                for i in range(10):
                    print('nop')
            base = torch.ones(2, 2)
            c_arg = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            return ([base, c_arg], [x.view(-1), x.view(-1), c_arg])
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True)
        with self.assertRaisesRegex(RuntimeError, 'is a tensor subclass. This is not supported today'):
            self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True, make_inputs_subclasses=True)
        fw_graph = self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1, primals_2):\n    as_strided = torch.ops.aten.as_strided.default(primals_1, [4], [1], 0)\n    mul = torch.ops.aten.mul.Tensor(as_strided, 2);  as_strided = None\n    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(primals_1, mul, [4], [1], 0);  primals_1 = mul = None\n    as_strided_3 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)\n    add = torch.ops.aten.add.Tensor(as_strided_3, 1);  as_strided_3 = None\n    add_1 = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None\n    return [as_strided_scatter, add, add_1]')

    def test_input_mutation_aliases_bases_out_of_order(self):
        if False:
            i = 10
            return i + 15

        def f(a, b, c, d):
            if False:
                print('Hello World!')
            b.add_(1)
            d.t_()
            return (a + c + d, b.view(-1))

        def inp_callable(req_grad):
            if False:
                return 10
            base1 = torch.ones(2, 2, requires_grad=req_grad)
            base2 = torch.ones(2, 2, requires_grad=req_grad)
            x1 = base1.add(1)
            x2 = base2.add(1)
            return ([base1, base2], [x1.view(-1), x2.view(-1), x1.view(-1), x2.view(-1)])
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True)
        with self.assertRaisesRegex(RuntimeError, 'is a tensor subclass. This is not supported today'):
            self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True, make_inputs_subclasses=True)
        fw_graph = self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1, primals_2, primals_3):\n    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None\n    as_strided = torch.ops.aten.as_strided.default(clone, [4], [1], 0)\n    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None\n    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [4], [1], 0);  clone = add = None\n    add_1 = torch.ops.aten.add.Tensor(primals_2, primals_3);  primals_2 = primals_3 = None\n    as_strided_3 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)\n    t_1 = torch.ops.aten.t.default(as_strided_3);  as_strided_3 = None\n    add_2 = torch.ops.aten.add.Tensor(add_1, t_1);  add_1 = None\n    as_strided_11 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)\n    view_1 = torch.ops.aten.view.default(as_strided_11, [-1]);  as_strided_11 = None\n    return [as_strided_scatter, add_2, view_1, t_1]')

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is unavailable')
    def test_synthetic_base_base_attribute_is_none(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                for i in range(10):
                    print('nop')
            a.add_(1)
            return a + b

        def inp_callable():
            if False:
                print('Hello World!')
            base = torch.ones(4, 4, device='cuda')
            a = base[0].detach()
            b = base[1].detach()
            base2 = torch.ones(2, 2, requires_grad=True)
            return ([base], [a, b])
        self.verify_aot_autograd(f, inp_callable, test_mutation=True)

    def test_input_mutation_alias_everything(self):
        if False:
            i = 10
            return i + 15

        def f(a, b, c):
            if False:
                print('Hello World!')
            c.mul_(2)
            b.t_()
            tmp = a + c
            out1 = tmp.view(-1)
            out2 = b.t()
            out3 = out1.unsqueeze(0)
            return (out1, out2, out3)

        def inp_callable(req_grad):
            if False:
                return 10
            base1 = torch.ones(2, 2, requires_grad=req_grad)
            base2 = torch.ones(2, 2, requires_grad=req_grad)
            base1_ = base1.add(1)
            base2_ = base2.add(1)
            a = base1_.view(-1)
            b = base2_
            c = base1_.view(-1)
            return ([base1, base2], [a, b, c])
        self.verify_aot_autograd(f, partial(inp_callable, req_grad=False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, partial(inp_callable, req_grad=True), test_mutation=True)
        self.assertExpectedInline(fw_graph.code.strip(), 'def forward(self, primals_1, primals_2):\n    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None\n    view = torch.ops.aten.view.default(primals_2, [2, 2]);  primals_2 = None\n    as_strided_1 = torch.ops.aten.as_strided.default(clone, [4], [1], 0)\n    mul = torch.ops.aten.mul.Tensor(as_strided_1, 2);  as_strided_1 = None\n    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, mul, [4], [1], 0);  clone = mul = None\n    as_strided_2 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)\n    t = torch.ops.aten.t.default(view);  view = None\n    as_strided_3 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)\n    add = torch.ops.aten.add.Tensor(as_strided_3, as_strided_2);  as_strided_3 = as_strided_2 = None\n    view_1 = torch.ops.aten.view.default(add, [-1])\n    t_1 = torch.ops.aten.t.default(t)\n    unsqueeze = torch.ops.aten.unsqueeze.default(view_1, 0)\n    return [as_strided_scatter, t, view_1, t_1, unsqueeze, add]')

    def test_dynamic_shape_output_not_in_bw_graph(self):
        if False:
            return 10

        def f(x):
            if False:
                return 10
            return [x + 1, x.shape[0]]
        inp = torch.ones(5, requires_grad=True)
        bw_graph_cell = [None]
        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell), decompositions={}, keep_inference_input_mutations=False, dynamic=True)
        out = compiled_f(inp)
        out[0].sum().backward()
        self.assertExpectedInline(bw_graph_cell[0].code.strip(), 'def forward(self, tangents_1):\n    return [tangents_1]')

    def test_no_grad_input_output(self):
        if False:
            print('Hello World!')

        def f(a, b):
            if False:
                print('Hello World!')
            return (a.cos(), b.cos(), a * b)
        inp_thunks = [lambda : torch.randn(5, requires_grad=True), lambda : torch.randn(5, requires_grad=False)]
        for inps in itertools.product(inp_thunks, repeat=2):
            inps = [i() for i in inps]
            self.verify_aot_autograd(f, inps)

    def test_some_output_requires_grad_input_doesnt(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                i = 10
                return i + 15
            a_view = a.view(-1)
            a_view.requires_grad_(True)
            return a_view
        inp = [torch.randn(3, 3), torch.randn(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp)

    def test_some_outputs_dont_require_grad_view(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                while True:
                    i = 10
            return (a.detach(), b)
        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp)

    def test_some_outputs_dont_require_grad_non_view(self):
        if False:
            while True:
                i = 10

        def f(a, b):
            if False:
                return 10
            return (a.add(1).detach(), b)
        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp)

    def test_inner_grad(self):
        if False:
            print('Hello World!')

        def foo(x):
            if False:
                return 10
            y = torch.exp(x)
            z = torch.autograd.grad(y, x)
            return z
        inps = [torch.randn((), requires_grad=True)]
        self.verify_aot_autograd(foo, inps)

    def test_grad_context(self):
        if False:
            print('Hello World!')

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return x * 2
        inps = [torch.randn((), requires_grad=True)]
        graph_size = None

        def get_graph_size(fx_g, _):
            if False:
                i = 10
                return i + 15
            nonlocal graph_size
            graph_size = len(fx_g.graph.nodes)
            return fx_g
        f = aot_function(foo, nop, get_graph_size)
        with torch.set_grad_enabled(False):
            f(*inps)
        self.assertIsNone(graph_size)
        f = aot_function(foo, nop, get_graph_size)
        with torch.set_grad_enabled(True):
            out = f(*inps)
            self.assertIsNone(graph_size)
            out.sum().backward()
            self.assertTrue(graph_size > 2)

    def test_output_dict(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                print('Hello World!')
            return {'a': x, 'b': x}
        inp = [torch.randn(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp)

        def f(x, y):
            if False:
                return 10
            return {'a': x, 'b': y + x}
        inp = [torch.randn(3, requires_grad=True), torch.randn(3)]
        self.verify_aot_autograd(f, inp)

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            new_d = {}
            for k in x:
                new_d[k] = x[k] * 2
            return new_d
        a = torch.randn(3, requires_grad=True)
        b = torch.randn(3, requires_grad=True)

        def inp_callable():
            if False:
                for i in range(10):
                    print('nop')
            inps = [{'a': a, 'b': b}]
            return (inps, inps)
        self.verify_aot_autograd(f, inp_callable)

    def test_module(self):
        if False:
            i = 10
            return i + 15
        mod = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        compiled_mod = compiled_module(mod, nop, nop)
        inp = torch.randn(32, 32)
        ref_out = mod(inp)
        ref_out.sum().backward()
        ref_grads = sorted([(name, p.grad) for (name, p) in mod.named_parameters()])
        out = compiled_mod(inp)
        out.sum().backward()
        grads = sorted([(name, p.grad) for (name, p) in mod.named_parameters()])
        self.assertEqual((out, grads), (ref_out, ref_grads))

    def test_batchnorm(self):
        if False:
            print('Hello World!')
        mod = compiled_module(nn.BatchNorm2d(4), nop, nop)
        x = torch.ones(1, 4, 2, 2)
        mod(x).sum().backward()

    def test_list_codegen(self):
        if False:
            while True:
                i = 10

        def list_nop(f, _):
            if False:
                for i in range(10):
                    print('nop')

            def g(inps):
                if False:
                    for i in range(10):
                        print('nop')
                return f(*inps)
            g._boxed_call = True
            return g

        def f(a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            return a.sin() * b.cos() * c.sin()
        f = aot_function(f, list_nop)
        inp = [torch.randn(5, requires_grad=True) for _ in range(3)]
        f(*inp).sum().backward()

    @patch('torch._functorch.aot_autograd.AOT_COUNTER', new_callable=itertools.count)
    def test_compilation_context(self, counter):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                print('Hello World!')
            return x.sin().sin()
        count = []

        def compiler(fx_g, _):
            if False:
                return 10
            context = get_aot_compilation_context()
            count.append((context[0], len(fx_g.graph.nodes)))
            return fx_g
        f = aot_function(f, compiler)
        out = f(torch.randn(5, requires_grad=True))
        f = aot_function(f, compiler)
        f(torch.randn(5))
        out.sum().backward()
        self.assertExpectedInline(str(count), "[(['0_forward'], 4), (['1_inference'], 4), (['0_backward'], 8)]")

    def test_dupe_arg(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x, y):
            if False:
                while True:
                    i = 10
            return x + y
        x = torch.randn(3, 3, requires_grad=True)
        self.verify_aot_autograd(f, [x, x])

    def test_dupe_arg_torture(self):
        if False:
            i = 10
            return i + 15

        def f(x, y):
            if False:
                return 10
            x.t_()
            y.t_()
            return x + y
        x = torch.randn(3, 3, requires_grad=True).clone()
        self.verify_aot_autograd(f, [x, x])

    def test_dupe_arg_returned_as_output(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b, a_):
            if False:
                for i in range(10):
                    print('nop')
            a[0].add_(1)
            return a_
        f_compiled = aot_function(f, nop)
        a = torch.ones(2)
        b = torch.ones(2)
        out_ref = f(a, b, a)
        a2 = torch.ones(2)
        b2 = torch.ones(2)
        out_test = f_compiled(a2, b2, a2)
        self.assertEqual(out_ref, out_test)
        self.assertEqual(a, a2)

    @patch('torch._functorch.aot_autograd.AOT_COUNTER', new_callable=itertools.count)
    @patch('torch._functorch.config.debug_assert', True)
    def test_invalid_dupe_left_bias(self, counter):
        if False:
            print('Hello World!')

        class F(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                x.t_()
                return (x + y,)
        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True)
        self.verify_aot_autograd(F(), [x, x])
        fxx = aot_module_simplified(F(), (x, x), nop)
        self.assertExpectedRaisesInline(AssertionError, lambda : fxx(x, y), 'At compilation time, graph 2 was compiled under the assumption that input 1 would be a duplicate of input 0, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch.')

    @patch('torch._functorch.aot_autograd.AOT_COUNTER', new_callable=itertools.count)
    @patch('torch._functorch.config.debug_assert', True)
    def test_invalid_dupe(self, counter):
        if False:
            i = 10
            return i + 15
        self._test_invalid_dupe(counter, fake=False)

    @patch('torch._functorch.aot_autograd.AOT_COUNTER', new_callable=itertools.count)
    @patch('torch._functorch.config.debug_assert', True)
    def test_invalid_dupe_fake(self, counter):
        if False:
            print('Hello World!')
        self._test_invalid_dupe(counter, fake=True)

    def _test_invalid_dupe(self, counter, fake):
        if False:
            print('Hello World!')

        class F(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                x.t_()
                y.t_()
                return (x + y,)
        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True).clone()
        if fake:
            shape_env = ShapeEnv()
            fake_mode = FakeTensorMode(shape_env=shape_env)
            fake_x = fake_mode.from_tensor(x)
            fake_y = fake_mode.from_tensor(y)
        if fake:
            fxy = aot_module_simplified(F(), (fake_x, fake_y), nop)
        else:
            fxy = aot_module_simplified(F(), (x, y), nop)
        fxy(x, y)
        fxy(x, x)
        if fake:
            fxx = aot_module_simplified(F(), (fake_x, fake_x), nop)
        else:
            fxx = aot_module_simplified(F(), (x, x), nop)
        fxx(x, x)
        self.assertExpectedRaisesInline(AssertionError, lambda : fxx(x, y), 'At compilation time, graph 1 was compiled under the assumption that input 1 would be a duplicate of input 0, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch.')

    @patch('torch._functorch.aot_autograd.AOT_COUNTER', new_callable=itertools.count)
    @patch('torch._functorch.config.debug_assert', True)
    def test_invalid_requires_grad(self, counter):
        if False:
            while True:
                i = 10
        self._test_invalid_requires_grad(counter, fake=False)

    @patch('torch._functorch.aot_autograd.AOT_COUNTER', new_callable=itertools.count)
    @patch('torch._functorch.config.debug_assert', True)
    def test_invalid_requires_grad_fake(self, counter):
        if False:
            return 10
        self._test_invalid_requires_grad(counter, fake=True)

    def _test_invalid_requires_grad(self, counter, fake):
        if False:
            while True:
                i = 10

        class F(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                return (x + y,)
        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        z = torch.randn(3, 3, requires_grad=False)
        if fake:
            shape_env = ShapeEnv()
            fake_mode = FakeTensorMode(shape_env=shape_env)
            fake_x = fake_mode.from_tensor(x)
            fake_y = fake_mode.from_tensor(y)
            fake_z = fake_mode.from_tensor(z)
        if fake:
            fxy = aot_module_simplified(F(), (fake_x, fake_y), nop)
        else:
            fxy = aot_module_simplified(F(), (x, y), nop)
        compare_equal_outs_and_grads(self, F(), fxy, (x, y))
        compare_equal_outs_and_grads(self, F(), fxy, (x, z))
        if fake:
            fxz = aot_module_simplified(F(), (fake_x, fake_z), nop)
        else:
            fxz = aot_module_simplified(F(), (x, z), nop)
        compare_equal_outs_and_grads(self, F(), fxz, (x, z))
        self.assertExpectedRaisesInline(AssertionError, lambda : fxz(x, y), 'At compilation time, graph 1 was compiled under the assumption that input 1 would not require grad, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch.')

    def test_custom_autograd(self):
        if False:
            print('Hello World!')

        class CustomFn(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    print('Hello World!')
                return grad_output + 1

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return CustomFn.apply(x)
        self.verify_aot_autograd(f, [torch.randn(3)])

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is unavailable')
    def test_autocast_disable_guard(self):
        if False:
            for i in range(10):
                print('nop')
        with torch._C._DisableAutocast():
            x = torch.rand([4, 4]).cuda()
            y = x @ x
            self.assertEqual(y.dtype, torch.float32)

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is unavailable')
    def test_nonidempotent_amp(self):
        if False:
            return 10

        def f(self_s_emb, add_3):
            if False:
                print('Hello World!')
            einsum_2 = torch.functional.einsum('ah,th->t', self_s_emb, add_3)
            log_softmax_2 = einsum_2.log_softmax(-1)
            return (log_softmax_2,)
        args = [torch.rand((1, 256), dtype=torch.float32, device='cuda'), torch.rand((30, 256), dtype=torch.float16, device='cuda')]
        with torch.cuda.amp.autocast(enabled=True):
            self.verify_aot_autograd(f, args)
        args = [e.requires_grad_(True) for e in args]
        with torch.cuda.amp.autocast(enabled=True):
            self.verify_aot_autograd(f, args)

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is unavailable')
    @unittest.skipIf(not torch.backends.cudnn.is_available(), 'CUDNN is unavailable')
    @skipIfRocm
    def test_batch_norm_amp(self):
        if False:
            while True:
                i = 10
        device = 'cuda'
        input_dtype = torch.float16
        param_dtype = torch.float32
        (weight, bias) = (torch.ones(64, device=device, dtype=param_dtype, requires_grad=True) for _ in range(2))
        (running_mean, running_var) = (torch.ones(64, device=device, dtype=param_dtype) for _ in range(2))

        def bn(x):
            if False:
                while True:
                    i = 10
            return torch.ops.aten.cudnn_batch_norm(x, weight, bias, running_mean, running_var, False, 0.1, 1e-05)
        inp = torch.ones(torch.Size([16, 64, 112, 112]), dtype=input_dtype, device=device)
        ref = bn(inp)
        cudnn_batch_norm_decomp = torch._decomp.get_decompositions({torch.ops.aten.cudnn_batch_norm})
        aot_fn = make_fx(bn, decomposition_table=cudnn_batch_norm_decomp)(inp)
        res = aot_fn(inp)
        for (a, b) in zip(ref, res):
            assert torch.allclose(a, b)

    def test_output_op_depending_on_symint(self):
        if False:
            while True:
                i = 10
        "\n        It won't be obvious from reading this test what it's testing for.  We should probably make it into a more\n        focused unit test.\n\n        An issue with the following program was the expand op would end up depending on a symint whose proxy was\n        incorrectly associated with one of the grad tensors rather than input tensors.  It broke partitioner logic\n        and the net result was aot_function failed to produce a function and threw an exception instead.\n        "
        inp = torch.randn(5, requires_grad=True)

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.expand(x.shape)
        af = aot_function(f, nop, partition_fn=partial(min_cut_rematerialization_partition, compiler='inductor'), dynamic=True)
        out = af(inp)
        self.assertEqual(out, f(inp))

    def test_inference_mode(self):
        if False:
            for i in range(10):
                print('nop')
        m = torch.nn.Linear(4, 4)
        inp = torch.randn(4, 4)
        aot_mod = aot_module(m, fw_compiler=nop)
        with torch.inference_mode():
            out_ref = m(inp)
            out_test = aot_mod(inp)
        self.assertEqual(out_ref, out_test)

    def test_default_partitioner_saves_symints_not_tensors_for_bw(self):
        if False:
            while True:
                i = 10
        '\n        In this test, the important thing is that primals_1 is **only** needed in the backward\n        in order to grab its sizes.\n        We need to assert that what we save for the backward are the tensor\'s sizes, and not the tensor itself.\n\n        The way this test is set up, it will actually fail if we try to save the input tensor for backward.\n        Why?\n        b.masked_fill_(c, 0) has a backward that requires knowing a\'s sizes\n        b.masked_fill_(c, 0) **also** mutates a (because b and a are aliased)\n        The autograd engine yells at us if we save "a" for backward, and then try to mutate it.\n        '
        inp = torch.randn(2, 2, requires_grad=True)

        def f(a):
            if False:
                print('Hello World!')
            b = a[0]
            c = torch.ones_like(b, dtype=torch.bool)
            d = b.masked_fill_(c, 0)
            return d
        compiled_f = aot_function(f, nop, dynamic=True)
        inp_ref = torch.ones(2, 2, requires_grad=True)
        inp_test = torch.ones(2, 2, requires_grad=True)
        out_ref = f(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())
        self.assertEqual(out_ref, out_test)
        out_ref.sum().backward()
        out_test.sum().backward()
        self.assertEqual(inp_ref.grad, inp_test.grad)

    def test_buffer_copied_in_graph(self):
        if False:
            return 10

        class MyModel(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.register_buffer('buf', torch.zeros(1))
                self.w1 = torch.nn.Parameter(torch.zeros(1))
                self.w2 = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                if False:
                    return 10
                self.buf.add_(1)
                return (self.w1 * x * self.w2).sum() + self.buf.sum()
        model_for_eager = MyModel()
        model_for_compile = copy.deepcopy(model_for_eager)
        fw_graph_cell = [None]
        compiled_f = aot_module(model_for_compile, fw_compiler=make_boxed_compiler(partial(extract_graph, graph_cell=fw_graph_cell)), bw_compiler=nop, keep_inference_input_mutations=True)
        inp_ref = torch.ones(1, requires_grad=True)
        inp_test = torch.ones(1, requires_grad=True)
        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())
        self.assertExpectedInline(fw_graph_cell[0].code.strip(), 'def forward(self, primals_1, primals_2, primals_3, primals_4):\n    add = torch.ops.aten.add.Tensor(primals_3, 1)\n    mul = torch.ops.aten.mul.Tensor(primals_1, primals_4)\n    mul_1 = torch.ops.aten.mul.Tensor(mul, primals_2)\n    sum_1 = torch.ops.aten.sum.default(mul_1);  mul_1 = None\n    sum_2 = torch.ops.aten.sum.default(add)\n    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None\n    copy_ = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = None\n    return [add_1, primals_1, primals_2, primals_4, mul]')
        self.assertEqual(out_ref, out_test)
        out_ref.sum().backward()
        out_test.sum().backward()
        eager_grads = [p.grad for (_, p) in model_for_eager.named_parameters()]
        compile_grads = [p.grad for (_, p) in model_for_compile.named_parameters()]
        self.assertEqual(eager_grads, compile_grads)
        self.assertEqual(inp_ref.grad, inp_test.grad)

    def test_buffer_copied_in_graph_with_different_shapes(self):
        if False:
            return 10

        class MyModel(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.register_buffer('buf', torch.ones(4, 4))
                self.w = torch.nn.Parameter(torch.Tensor([[4, 5], [1, 2], [6, 7], [8, 9]]))

            def forward(self, x):
                if False:
                    return 10
                self.buf.add_(1)
                return (self.w @ x).sum() + self.buf.sum()
        model_for_eager = MyModel()
        model_for_compile = copy.deepcopy(model_for_eager)
        fw_graph_cell = [None]
        compiled_f = aot_module(model_for_compile, fw_compiler=make_boxed_compiler(partial(extract_graph, graph_cell=fw_graph_cell)), bw_compiler=nop, keep_inference_input_mutations=True)
        inp_ref = torch.ones(2, 4, requires_grad=True)
        inp_test = torch.ones(2, 4, requires_grad=True)
        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())
        self.assertExpectedInline(fw_graph_cell[0].code.strip(), 'def forward(self, primals_1, primals_2, primals_3):\n    add = torch.ops.aten.add.Tensor(primals_2, 1)\n    mm = torch.ops.aten.mm.default(primals_1, primals_3)\n    sum_1 = torch.ops.aten.sum.default(mm);  mm = None\n    sum_2 = torch.ops.aten.sum.default(add)\n    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None\n    copy_ = torch.ops.aten.copy_.default(primals_2, add);  primals_2 = add = None\n    return [add_1, primals_1, primals_3]')
        self.assertEqual(out_ref, out_test)
        out_ref.sum().backward()
        out_test.sum().backward()
        eager_grads = [p.grad for (_, p) in model_for_eager.named_parameters()]
        compile_grads = [p.grad for (_, p) in model_for_compile.named_parameters()]
        self.assertEqual(eager_grads, compile_grads)
        self.assertEqual(inp_ref.grad, inp_test.grad)

    def test_buffer_batch_norm(self):
        if False:
            i = 10
            return i + 15

        class MyModel(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.m = torch.nn.BatchNorm1d(100)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.m(x)
        model_for_eager = MyModel()
        model_for_compile = copy.deepcopy(model_for_eager)
        fw_graph_cell = [None]
        bw_graph_cell = [None]
        compiled_f = aot_module(model_for_compile, fw_compiler=make_boxed_compiler(partial(extract_graph, graph_cell=fw_graph_cell)), bw_compiler=make_boxed_compiler(partial(extract_graph, graph_cell=bw_graph_cell)), keep_inference_input_mutations=True)
        inp_ref = torch.ones(20, 100, requires_grad=True)
        inp_test = torch.ones(20, 100, requires_grad=True)
        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())
        self.assertExpectedInline(fw_graph_cell[0].code.strip(), 'def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6):\n    add = torch.ops.aten.add.Tensor(primals_5, 1)\n    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(primals_6, primals_1, primals_2, primals_3, primals_4, True, 0.1, 1e-05);  primals_2 = None\n    getitem = _native_batch_norm_legit_functional[0]\n    getitem_1 = _native_batch_norm_legit_functional[1]\n    getitem_2 = _native_batch_norm_legit_functional[2]\n    getitem_3 = _native_batch_norm_legit_functional[3]\n    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None\n    copy_ = torch.ops.aten.copy_.default(primals_3, getitem_3);  primals_3 = None\n    copy__1 = torch.ops.aten.copy_.default(primals_4, getitem_4);  primals_4 = None\n    copy__2 = torch.ops.aten.copy_.default(primals_5, add);  primals_5 = add = None\n    return [getitem, primals_1, primals_6, getitem_1, getitem_2, getitem_3, getitem_4]')
        self.assertEqual(out_ref, out_test)
        out_ref.sum().backward()
        out_test.sum().backward()
        eager_grads = [p.grad for (_, p) in model_for_eager.named_parameters()]
        compile_grads = [p.grad for (_, p) in model_for_compile.named_parameters()]
        self.assertEqual(eager_grads, compile_grads)
        self.assertExpectedInline(bw_graph_cell[0].code.strip(), 'def forward(self, primals_1, primals_6, getitem_1, getitem_2, getitem_3, getitem_4, tangents_1):\n    native_batch_norm_backward = torch.ops.aten.native_batch_norm_backward.default(tangents_1, primals_6, primals_1, getitem_3, getitem_4, getitem_1, getitem_2, True, 1e-05, [True, True, True]);  tangents_1 = primals_6 = primals_1 = getitem_3 = getitem_4 = getitem_1 = getitem_2 = None\n    getitem_5 = native_batch_norm_backward[0]\n    getitem_6 = native_batch_norm_backward[1]\n    getitem_7 = native_batch_norm_backward[2];  native_batch_norm_backward = None\n    return [getitem_6, getitem_7, None, None, None, getitem_5]')
        self.assertEqual(inp_ref.grad, inp_test.grad)

    def test_new_inp_requires_grad_now(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                return 10
            return x.add_(y)
        fw_graph_cell = [None]
        bw_graph_cell = [None]
        compiled_f = aot_function(f, fw_compiler=make_boxed_compiler(partial(extract_graph, graph_cell=fw_graph_cell)), bw_compiler=make_boxed_compiler(partial(extract_graph, graph_cell=bw_graph_cell)), keep_inference_input_mutations=True)
        inp_ref = (torch.ones(20, 100, requires_grad=False), torch.ones(20, 100, requires_grad=True))
        inp_test = (torch.ones(20, 100, requires_grad=False), torch.ones(20, 100, requires_grad=True))
        out_ref = f(*inp_ref)
        out_test = compiled_f(*inp_test)
        self.assertExpectedInline(fw_graph_cell[0].code.strip(), 'def forward(self, primals_1, primals_2):\n    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None\n    add = torch.ops.aten.add.Tensor(clone, primals_2);  clone = primals_2 = None\n    return [add, add]')
        self.assertEqual(out_ref, out_test)
        out_ref.sum().backward()
        out_test.sum().backward()
        self.assertExpectedInline(bw_graph_cell[0].code.strip(), 'def forward(self, tangents_1):\n    return [None, tangents_1]')

    def test_real_weights_in_symbolic_mode(self):
        if False:
            while True:
                i = 10
        from functorch.experimental import functionalize

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.linear(x)
                return x
        m = M().eval()
        inp = torch.randn(2, 5)
        gm = make_fx(m, tracing_mode='symbolic', _allow_non_fake_inputs=True)(inp)
        self.assertEqual(gm(torch.ones(2, 5)), m(torch.ones(2, 5)))
        gm_functionalized = make_fx(functionalize(gm), tracing_mode='symbolic', _allow_non_fake_inputs=True)(inp)
        self.assertEqual(gm_functionalized(torch.ones(2, 5)), m(torch.ones(2, 5)))
        inp_count = 0
        for node in gm.graph.nodes:
            if node.op == 'placeholder':
                inp_count += 1
        self.assertEqual(inp_count, 1)
        inp_count = 0
        for node in gm_functionalized.graph.nodes:
            if node.op == 'placeholder':
                inp_count += 1
        self.assertEqual(inp_count, 1)
        with self.assertRaisesRegex(Exception, 'Please convert all Tensors to FakeTensors'):
            make_fx(m, tracing_mode='symbolic', _allow_non_fake_inputs=False)(torch.randn(2, 5))

    def test_real_weights_in_symbolic_mode_with_inplace_ops(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.register_buffer('buffer', torch.ones(4, 5))

            def forward(self, x):
                if False:
                    print('Hello World!')
                y = self.buffer.add_(3)
                y.resize_([20])
                assert y.shape == self.buffer.shape
                return x.sum() + self.buffer.sum()
        m = M().eval()
        inp = torch.randn(2, 5)
        with self.assertRaisesRegex(Exception, "Can't call metadata"):
            make_fx(m, tracing_mode='symbolic', _allow_non_fake_inputs=True)(inp)

def extract_graph(fx_g, _, graph_cell):
    if False:
        i = 10
        return i + 15
    graph_cell[0] = fx_g
    return fx_g

def get_ins_outs(fx_g):
    if False:
        i = 10
        return i + 15
    ins = []
    outs = []
    for n in fx_g.graph.nodes:
        if n.op == 'placeholder':
            ins.append(n)
        elif n.op == 'output':
            outs = tuple(n.args[0])
    return (ins, outs)

def get_num_ins_outs(fx_g):
    if False:
        while True:
            i = 10
    return tuple((len(i) for i in get_ins_outs(fx_g)))

def get_fw_bw_graph(f, inps, partitioner=min_cut_rematerialization_partition, dynamic=False):
    if False:
        return 10
    fw_graph_cell = [None]
    bw_graph_cell = [None]
    aot_function(f, fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell), bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell), partition_fn=partitioner, decompositions=default_decompositions, dynamic=dynamic)(*inps).sum().backward()
    return (fw_graph_cell[0], bw_graph_cell[0])

class TestMod(torch.nn.Module):

    def __init__(self, fn):
        if False:
            while True:
                i = 10
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(2, requires_grad=True))
        self.fn = fn

    def forward(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self.fn(self.p, *args)

class TestAOTExport(AOTTestCase):

    def test_aot_export_module_joint(self):
        if False:
            print('Hello World!')

        class ConvBatchnormRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 3, 1, 1)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv(x)
                x = self.bn(x)
                user_out = torch.nn.functional.relu(x)
                loss = user_out.sum()
                return (loss, user_out.detach())
        mod = ConvBatchnormRelu()
        mod.train()
        inp = torch.randn(1, 1, 3, 3)
        o_ref = mod(inp)
        (fx_g, signature) = aot_export_module(mod, [inp], trace_joint=True, output_loss_index=0)
        self.assertExpectedInline(fx_g.print_readable(print_output=False), 'class <lambda>(torch.nn.Module):\n    def forward(self, arg0_1: "f32[3, 1, 1, 1]", arg1_1: "f32[3]", arg2_1: "f32[3]", arg3_1: "f32[3]", arg4_1: "f32[3]", arg5_1: "f32[3]", arg6_1: "i64[]", arg7_1: "f32[1, 1, 3, 3]"):\n        # No stacktrace found for following nodes\n        convolution: "f32[1, 3, 3, 3]" = torch.ops.aten.convolution.default(arg7_1, arg0_1, arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1_1 = None\n        add: "i64[]" = torch.ops.aten.add.Tensor(arg6_1, 1);  arg6_1 = None\n        _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(convolution, arg2_1, arg3_1, arg4_1, arg5_1, True, 0.1, 1e-05);  arg3_1 = arg4_1 = arg5_1 = None\n        getitem: "f32[1, 3, 3, 3]" = _native_batch_norm_legit_functional[0]\n        getitem_1: "f32[3]" = _native_batch_norm_legit_functional[1]\n        getitem_2: "f32[3]" = _native_batch_norm_legit_functional[2]\n        getitem_3: "f32[3]" = _native_batch_norm_legit_functional[3]\n        getitem_4: "f32[3]" = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None\n        relu: "f32[1, 3, 3, 3]" = torch.ops.aten.relu.default(getitem);  getitem = None\n        detach: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(relu)\n        detach_1: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(relu)\n        detach_2: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(detach_1);  detach_1 = None\n        sum_1: "f32[]" = torch.ops.aten.sum.default(relu)\n        detach_3: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(relu);  relu = None\n        detach_4: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(detach_3);  detach_3 = None\n        detach_5: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(detach_4);  detach_4 = None\n        detach_6: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(detach_5);  detach_5 = None\n        ones_like: "f32[]" = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format)\n        expand: "f32[1, 3, 3, 3]" = torch.ops.aten.expand.default(ones_like, [1, 3, 3, 3]);  ones_like = None\n        detach_7: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(detach_2);  detach_2 = None\n        detach_8: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(detach_7);  detach_7 = None\n        threshold_backward: "f32[1, 3, 3, 3]" = torch.ops.aten.threshold_backward.default(expand, detach_8, 0);  expand = detach_8 = None\n        native_batch_norm_backward = torch.ops.aten.native_batch_norm_backward.default(threshold_backward, convolution, arg2_1, getitem_3, getitem_4, getitem_1, getitem_2, True, 1e-05, [True, True, True]);  threshold_backward = convolution = arg2_1 = getitem_1 = getitem_2 = None\n        getitem_5: "f32[1, 3, 3, 3]" = native_batch_norm_backward[0]\n        getitem_6: "f32[3]" = native_batch_norm_backward[1]\n        getitem_7: "f32[3]" = native_batch_norm_backward[2];  native_batch_norm_backward = None\n        convolution_backward = torch.ops.aten.convolution_backward.default(getitem_5, arg7_1, arg0_1, [3], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  getitem_5 = arg7_1 = arg0_1 = None\n        getitem_8 = convolution_backward[0]\n        getitem_9: "f32[3, 1, 1, 1]" = convolution_backward[1]\n        getitem_10: "f32[3]" = convolution_backward[2];  convolution_backward = None\n        return (getitem_3, getitem_4, add, sum_1, detach_6, getitem_9, getitem_10, getitem_6, getitem_7)\n        ')
        self.assertExpectedInline(str(signature.parameters), "['conv.weight', 'conv.bias', 'bn.weight', 'bn.bias']")
        self.assertExpectedInline(str(signature.buffers), "['bn.running_mean', 'bn.running_var', 'bn.num_batches_tracked']")
        self.assertExpectedInline(str(signature.user_inputs), "['arg7_1']")
        self.assertExpectedInline(str(signature.inputs_to_parameters), "{'arg0_1': 'conv.weight', 'arg1_1': 'conv.bias', 'arg2_1': 'bn.weight', 'arg3_1': 'bn.bias'}")
        self.assertExpectedInline(str(signature.inputs_to_buffers), "{'arg4_1': 'bn.running_mean', 'arg5_1': 'bn.running_var', 'arg6_1': 'bn.num_batches_tracked'}")
        self.assertExpectedInline(str(signature.buffers_to_mutate), "{'getitem_3': 'bn.running_mean', 'getitem_4': 'bn.running_var', 'add': 'bn.num_batches_tracked'}")
        self.assertExpectedInline(str(signature.backward_signature.gradients_to_parameters), "{'getitem_9': 'conv.weight', 'getitem_10': 'conv.bias', 'getitem_6': 'bn.weight', 'getitem_7': 'bn.bias'}")
        self.assertExpectedInline(str(signature.backward_signature.gradients_to_user_inputs), '{}')
        self.assertExpectedInline(str(signature.backward_signature.loss_output), 'getitem_3')
        (fx_g_inference, signature_inference) = aot_export_module(mod, [inp], trace_joint=False)
        self.assertExpectedInline(fx_g_inference.print_readable(print_output=False), 'class <lambda>(torch.nn.Module):\n    def forward(self, arg0_1: "f32[3, 1, 1, 1]", arg1_1: "f32[3]", arg2_1: "f32[3]", arg3_1: "f32[3]", arg4_1: "f32[3]", arg5_1: "f32[3]", arg6_1: "i64[]", arg7_1: "f32[1, 1, 3, 3]"):\n        # No stacktrace found for following nodes\n        convolution: "f32[1, 3, 3, 3]" = torch.ops.aten.convolution.default(arg7_1, arg0_1, arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg7_1 = arg0_1 = arg1_1 = None\n        add: "i64[]" = torch.ops.aten.add.Tensor(arg6_1, 1);  arg6_1 = None\n        _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(convolution, arg2_1, arg3_1, arg4_1, arg5_1, True, 0.1, 1e-05);  convolution = arg2_1 = arg3_1 = arg4_1 = arg5_1 = None\n        getitem: "f32[1, 3, 3, 3]" = _native_batch_norm_legit_functional[0]\n        getitem_3: "f32[3]" = _native_batch_norm_legit_functional[3]\n        getitem_4: "f32[3]" = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None\n        relu: "f32[1, 3, 3, 3]" = torch.ops.aten.relu.default(getitem);  getitem = None\n        sum_1: "f32[]" = torch.ops.aten.sum.default(relu)\n        detach: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(relu);  relu = None\n        detach_1: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(detach);  detach = None\n        return (getitem_3, getitem_4, add, sum_1, detach_1)\n        ')

    def test_aot_export_simplified_basic(self):
        if False:
            print('Hello World!')

        def f(x, y):
            if False:
                print('Hello World!')
            return (x * y, y * y.detach())
        x = torch.randn(2, requires_grad=True)
        y = torch.randn(2, requires_grad=True)
        f_graph_fw = aot_export_joint_simple(f, [x, y], trace_joint=False)
        out_ref = f(x, y)
        out_test = f_graph_fw(x, y)
        self.assertEqual(out_ref, out_test)
        x = torch.randn(2, requires_grad=True)
        y = torch.randn(2, requires_grad=True)
        x2 = x.clone().detach().requires_grad_(True)
        y2 = y.clone().detach().requires_grad_(True)
        x3 = x.clone().detach().requires_grad_(True)
        y3 = y.clone().detach().requires_grad_(True)
        f_graph_joint = aot_export_joint_simple(f, [x, y], trace_joint=True)
        num_fw_outputs = 2
        (fw_g, bw_g) = default_partition(f_graph_joint, [x, y], num_fwd_outputs=num_fw_outputs)
        out_ref2 = f(x2, y2)
        fw_outs = fw_g(x3, y3)
        (out_test2, activations) = (fw_outs[:num_fw_outputs], fw_outs[num_fw_outputs:])
        self.assertEqual(out_ref2, out_test2)
        grad_outs = [torch.ones_like(x) for x in out_ref2]
        grads_ref = torch.autograd.grad(out_ref2, [x2, y2], grad_outputs=grad_outs)
        grads_test = bw_g(*activations, *grad_outs)
        for (g_ref, g_test) in zip(grads_ref, grads_test):
            self.assertEqual(g_ref, g_test)

    def test_aot_export_metadata_mutation_banned(self):
        if False:
            while True:
                i = 10

        def fn(p, x):
            if False:
                return 10
            x.t_()
            return (x * 2,)
        mod = TestMod(fn)
        inp = torch.randn(2)
        with self.assertRaisesRegex(RuntimeError, 'Found an input that received a metadata mutation'):
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=False)
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=True)
            aot_export_module(mod, [inp], trace_joint=False)

    def test_aot_export_forward_mutation_no_buffer_mut_banned(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.register_buffer('buffer1', torch.ones(6, 4))

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x.add_(4)
                return (x.cos().sum() + self.buffer1.sum(),)
        with self.assertRaisesRegex(RuntimeError, 'Found following user inputs located at \\[0\\] are mutated'):
            aot_export_module(M(), [torch.ones(6, 4)], trace_joint=False)

    def test_aot_export_forward_mutation_multiple_mut_banned(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.register_buffer('buffer1', torch.ones(6, 4))

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                y.add_(4)
                self.buffer1.add_(5)
                return (x.cos().sum() + y.sin().sum(), self.buffer1.sum())
        with self.assertRaisesRegex(RuntimeError, 'Found following user inputs located at \\[1\\] are mutated'):
            aot_export_module(M(), [torch.ones(6, 4), torch.zeros(6, 4)], trace_joint=False)

    def test_aot_export_input_mutation_on_parameter_banned(self):
        if False:
            return 10

        def fn(p, x):
            if False:
                while True:
                    i = 10
            p.mul_(2)
            return (p + x,)
        mod = TestMod(fn)
        inp = torch.randn(2)
        with self.assertRaisesRegex(RuntimeError, 'Found a graph input that requires gradients, and received a mutation'):
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=False)
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=True)
            aot_export_module(mod, [inp], trace_joint=False)

    def test_aot_export_synthetic_bases_banned(self):
        if False:
            i = 10
            return i + 15

        def fn(p, x, y):
            if False:
                i = 10
                return i + 15
            x.mul_(2)
            return (x + y,)
        mod = TestMod(fn)
        inp = torch.randn(2)
        inp2 = inp.view(-1)
        with self.assertRaisesRegex(RuntimeError, 'Encountered aliased inputs that are mutated'):
            aot_export_joint_simple(fn, [mod.p, inp, inp2], trace_joint=False)
            aot_export_joint_simple(fn, [mod.p, inp, inp2], trace_joint=True)
            aot_export_module(mod, [inp, inp2], trace_joint=False)

    def test_aot_export_input_dupes_banned(self):
        if False:
            return 10

        def fn(p, x, y):
            if False:
                return 10
            x.mul_(2)
            return (x + y,)
        mod = TestMod(fn)
        inp = torch.randn(2)
        with self.assertRaisesRegex(RuntimeError, 'Encountered duplicated inputs that are mutated in the graph'):
            aot_export_joint_simple(fn, [mod.p, inp, inp], trace_joint=False)
            aot_export_joint_simple(fn, [mod.p, inp, inp], trace_joint=True)
            aot_export_module(mod, [inp, inp], trace_joint=False)

    def test_aot_export_multiple_outputs_require_grad_banned(self):
        if False:
            return 10

        def fn(p, x):
            if False:
                print('Hello World!')
            out = p * x
            return (out, out.sum())
        mod = TestMod(fn)
        inp = torch.randn(2)
        with self.assertRaisesRegex(RuntimeError, 'Found an output of the forward that requires gradients, that was not'):
            aot_export_module(mod, [inp], trace_joint=True, output_loss_index=1)

    def test_aot_export_simplified_input_mutations_banned(self):
        if False:
            while True:
                i = 10

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            x.mul_(2)
            return (x + x,)
        inp = torch.randn(2)
        with self.assertRaisesRegex(RuntimeError, 'Found following user inputs located at \\[0\\] are mutated'):
            aot_export_joint_simple(fn, [inp], trace_joint=False)
            aot_export_joint_simple(fn, [inp], trace_joint=True)

    def test_aot_export_simplified_pytrees_banned(self):
        if False:
            while True:
                i = 10

        def fn(inps):
            if False:
                print('Hello World!')
            return (inps[0] + inps[1],)
        inp1 = torch.randn(2)
        inp2 = torch.randn(2)
        inps = [inp1, inp2]
        with self.assertRaisesRegex(RuntimeError, 'aot_export_joint_simple requires individual inputs not to be pytrees'):
            aot_export_joint_simple(fn, [inps], trace_joint=False)
            aot_export_joint_simple(fn, [inps], trace_joint=True)

    def test_aot_export_functionalized_rng_banned(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(p, x):
            if False:
                for i in range(10):
                    print('nop')
            return (p + x,)
        mod = TestMod(fn)
        inp = torch.randn(2)
        with patch('functorch.compile.config.functionalize_rng_ops', True), self.assertRaisesRegex(RuntimeError, 'Functionalized RNG is not currently supported in the aot_export'):
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=False)
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=True)
            aot_export_module(mod, [inp], trace_joint=False)

class TestPartitioning(AOTTestCase):

    @unittest.skipIf(not USE_NETWORKX, 'networkx not available')
    def test_recompute_partitioning(self):
        if False:
            return 10

        def fn(a, b):
            if False:
                i = 10
                return i + 15
            return torch.sin(torch.sin(a)) + b
        ref_a = torch.rand(10, 10, requires_grad=True)
        ref_b = torch.rand(10, 10, requires_grad=True)
        ref = fn(ref_a, ref_b)
        ref.sum().backward()
        res_a = ref_a.clone().detach().requires_grad_(True)
        res_b = ref_b.clone().detach().requires_grad_(True)

        def compile_fn(x, _):
            if False:
                while True:
                    i = 10
            return x
        compiled_fn = compiled_function(fn, compile_fn, compile_fn, min_cut_rematerialization_partition)
        res = compiled_fn(res_a, res_b)
        res.sum().backward()
        assert torch.allclose(ref, res, atol=0.001, rtol=0.001)
        assert torch.allclose(ref_a.grad, res_a.grad, atol=0.001, rtol=0.001)
        assert torch.allclose(ref_b.grad, res_b.grad, atol=0.001, rtol=0.001)

    def test_meta_tensor_inplace_op(self):
        if False:
            return 10

        class MockModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(3072, 768, requires_grad=True))
                self.bias = torch.nn.Parameter(torch.randn(3072, requires_grad=True))

            def forward(self, add_4):
                if False:
                    print('Hello World!')
                linear_4 = torch.nn.functional.linear(add_4, self.weight, bias=self.bias)
                gelu = torch.nn.functional.gelu(linear_4)
                return gelu

        def check_meta_tensor(fx_g, _):
            if False:
                for i in range(10):
                    print('nop')
            for node in fx_g.graph.nodes:
                if node.op != 'output':
                    assert 'tensor_meta' in node.meta
            return fx_g
        inp0 = torch.randn(16, 128, 768, requires_grad=True)
        inputs = [inp0]
        mod = MockModule().to(device='cpu')
        aot_mod = aot_module(mod, fw_compiler=check_meta_tensor)
        aot_mod(*inputs)

    def test_default_partitioner_getitem(self):
        if False:
            i = 10
            return i + 15
        mod = nn.LayerNorm([10])

        def f(x, mod_weight, mod_bias):
            if False:
                while True:
                    i = 10
            return torch.nn.functional.layer_norm(x, [10], mod_weight, mod_bias, eps=1e-06)
        (fw_graph, bw_graph) = get_fw_bw_graph(f, [torch.randn(3, 10, requires_grad=True), mod.weight, mod.bias], partitioner=default_partition)
        self.assertEqual(get_num_ins_outs(fw_graph), (3, 6))
        self.assertEqual(get_num_ins_outs(bw_graph), (6, 3))

    @unittest.skipIf(not USE_NETWORKX, 'networkx not available')
    def test_min_cut_partitioner_save_shape(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                return 10
            s = x.sum(dim=1)
            return s
        inp = [torch.ones([10, 10], requires_grad=True)]
        (fw_graph, bw_graph) = get_fw_bw_graph(f, inp, dynamic=True)
        (_, fw_output) = get_ins_outs(fw_graph)
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 3))
        self.assertEqual(get_num_ins_outs(bw_graph), (3, 1))
        self.assertEqual(str(fw_output[0]), 'sum_1')
        self.assertEqual(str(fw_output[1]), 'sym_size_int')
        self.assertEqual(str(fw_output[2]), 'sym_size_int_1')
        inp = [torch.randn(10, requires_grad=True), torch.randn((3, 10), requires_grad=True), torch.randn((2, 10), requires_grad=True)]

        def f(a, b, c):
            if False:
                return 10
            sb = torch.ops.aten.sym_size(b)
            sc = c.size()
            x = sb[0] + sc[0]
            a_sz = (x, a.size(0))
            return torch.cat([a.expand(a_sz), b, c])
        (fw_graph, bw_graph) = get_fw_bw_graph(f, inp, dynamic=True)
        self.assertEqual(get_num_ins_outs(fw_graph), (3, 4))
        self.assertEqual(get_num_ins_outs(bw_graph), (4, 3))
        (_, outs) = get_ins_outs(fw_graph)
        self.assertTrue(all((is_sym_node(n) for n in outs[1:])))

    def test_default_partitioner_output_tensor_shape_tensor(self):
        if False:
            for i in range(10):
                print('nop')
        inp = [torch.randn(10, requires_grad=True), torch.randn((3, 10), requires_grad=True), torch.randn((2, 10), requires_grad=True), torch.randn((10, 1), requires_grad=True)]

        def f(a, b, c, d):
            if False:
                for i in range(10):
                    print('nop')
            sb = b.size()
            sc = c.size()
            x = sb[0] + sc[0]
            a_sz = (x, a.size(0))
            cat = torch.cat([a.expand(a_sz), b, c])
            mm = torch.mm(cat, d)
            mm2 = torch.mm(mm, a.view(mm.size(1), a.size(0)))
            return (cat, sb, c, mm2)
        fw_graph_cell = [None]
        bw_graph_cell = [None]
        compiled_outs = aot_function(f, fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell), bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell), partition_fn=default_partition, decompositions=default_decompositions, dynamic=True)(*inp)
        fw_graph = fw_graph_cell[0]
        (compiled_outs[0].sum() + compiled_outs[2].sum()).backward()
        bw_graph = bw_graph_cell[0]
        self.assertEqual(get_num_ins_outs(fw_graph), (4, 13))
        self.assertEqual(get_num_ins_outs(bw_graph), (10, 4))
        (_, fw_graph_out_nodes) = get_ins_outs(fw_graph)
        self.assertEqual([False, True, True, False, False] + [False] * 4 + [True] * 4, [is_sym_node(n) for n in fw_graph_out_nodes])
        real_outs = f(*inp)
        self.assertEqual(compiled_outs, real_outs)
        self.assertTrue(isinstance(real_outs[1], torch.Size))
        self.assertFalse(isinstance(compiled_outs[1], torch.Size))

    @unittest.skipIf(not USE_NETWORKX, 'networkx not available')
    def test_min_cut_partitioner_output_tensor_shape_tensor(self):
        if False:
            print('Hello World!')
        inp = [torch.randn(10, requires_grad=True), torch.randn((3, 10), requires_grad=True), torch.randn((2, 10), requires_grad=True), torch.randn((10, 1), requires_grad=True)]

        def f(a, b, c, d):
            if False:
                while True:
                    i = 10
            sb = b.size()
            sc = c.size()
            x = sb[0] + sc[0]
            a_sz = (x, a.size(0))
            cat = torch.cat([a.expand(a_sz), b, c])
            mm = torch.mm(cat, d)
            mm2 = torch.mm(mm, a.view(mm.size(1), a.size(0)))
            return (cat, sb, c, mm2)
        fw_graph_cell = [None]
        bw_graph_cell = [None]
        compiled_outs = aot_function(f, fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell), bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell), partition_fn=min_cut_rematerialization_partition, decompositions=default_decompositions, dynamic=True)(*inp)
        fw_graph = fw_graph_cell[0]
        (compiled_outs[0].sum() + compiled_outs[2].sum()).backward()
        bw_graph = bw_graph_cell[0]
        self.assertEqual(get_num_ins_outs(fw_graph), (4, 12))
        self.assertEqual(get_num_ins_outs(bw_graph), (9, 4))
        (_, fw_graph_out_nodes) = get_ins_outs(fw_graph)
        self.assertEqual([False, True, True, False, False] + [False] * 4 + [True] * 3, [is_sym_node(n) for n in fw_graph_out_nodes])
        real_outs = f(*inp)
        self.assertEqual(compiled_outs, real_outs)
        self.assertTrue(isinstance(real_outs[1], torch.Size))
        self.assertFalse(isinstance(compiled_outs[1], torch.Size))

    @unittest.skipIf(not USE_NETWORKX, 'networkx not available')
    def test_min_cut_partitioner(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                print('Hello World!')
            return x.cos().cos().cos()
        (fw_graph, bw_graph) = get_fw_bw_graph(f, [torch.randn(3, requires_grad=True)])
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 2))
        self.assertEqual(get_num_ins_outs(bw_graph), (2, 1))

        def f(a, b, c, d):
            if False:
                i = 10
                return i + 15
            x = a + b + c + d
            return x.cos().cos()
        (fw_graph, bw_graph) = get_fw_bw_graph(f, [torch.randn(3, requires_grad=True) for _ in range(4)])
        self.assertEqual(get_num_ins_outs(fw_graph), (4, 2))
        self.assertEqual(get_num_ins_outs(bw_graph), (2, 4))

    @unittest.skipIf(not USE_NETWORKX, 'networkx not available')
    def test_min_cut_partitioner_recomputable_ops(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x * x * x
        recomputable_ops = []
        partition_fn = partial(min_cut_rematerialization_partition, recomputable_ops=recomputable_ops)
        (fw_graph, bw_graph) = get_fw_bw_graph(f, [torch.randn(3, requires_grad=True)], partition_fn)
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 3))
        self.assertEqual(get_num_ins_outs(bw_graph), (3, 1))
        recomputable_ops = [torch.ops.aten.mul]
        partition_fn = partial(min_cut_rematerialization_partition, recomputable_ops=recomputable_ops)
        (fw_graph, bw_graph) = get_fw_bw_graph(f, [torch.randn(3, requires_grad=True)], partition_fn)
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 2))
        self.assertEqual(get_num_ins_outs(bw_graph), (2, 1))

    def test_contiguous(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                while True:
                    i = 10
            return x.view(2, 3).t()
        inp = torch.randn(6, requires_grad=True)
        out = aot_function(f, nop)(inp)
        torch.autograd.grad(out, inp, torch.randn(3, 2))

    def test_preserve_random(self):
        if False:
            i = 10
            return i + 15

        def fn(x):
            if False:
                return 10
            return torch.nn.functional.dropout(x, 0.5) + x
        x = torch.randn(4)
        torch.manual_seed(0)
        ref = fn(x)
        torch.manual_seed(0)
        aot_fn = aot_function(fn, nop)
        res = aot_fn(x)
        assert torch.allclose(ref, res)

    def test_generate_gives_inference_graph(self):
        if False:
            for i in range(10):
                print('nop')

        def generate(x):
            if False:
                while True:
                    i = 10
            with torch.no_grad():
                return torch.mul(x, x)
        inference_graph_cell = [None]
        inference_compiler = make_boxed_compiler(partial(extract_graph, graph_cell=inference_graph_cell))
        aot_fn = aot_function(generate, nop, inference_compiler=inference_compiler)
        x = torch.randn(4, requires_grad=True)
        res = aot_fn(x)
        self.assertTrue(inference_graph_cell[0] is not None)

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is unavailable')
    @unittest.skipIf(not USE_TORCHVISION, 'test requires torchvision')
    def test_autocast(self):
        if False:
            return 10
        mod = torchvision.models.resnet18().cuda()
        mod.train()
        x = torch.randn(16, 3, 32, 32, device='cuda')
        aot_mod = memory_efficient_fusion(mod)
        with torch.cuda.amp.autocast(True):
            res = aot_mod(x)
        res.sum().backward()

class TestAOTDispatch(AOTTestCase):

    def test_aot_dispatch_simple(self):
        if False:
            while True:
                i = 10

        def f(a, b):
            if False:
                return 10
            aa = torch.mul(a, 6)
            bb = torch.div(b, 2)
            return aa + bb
        a1_ref = torch.ones(3, 3, requires_grad=True)
        a2_ref = torch.ones(3, 3, requires_grad=True)
        a_ref = TwoTensor(a1_ref, a2_ref)
        b_ref = torch.ones(3, 3, requires_grad=True)
        a1_test = a1_ref.clone().detach().requires_grad_(True)
        a2_test = a2_ref.clone().detach().requires_grad_(True)
        a_test = TwoTensor(a1_test, a2_test)
        b_test = b_ref.clone().detach().requires_grad_(True)
        fw_graph_cell = [None]
        bw_graph_cell = [None]
        compiled_f = aot_function(f, fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell), bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell), partition_fn=min_cut_rematerialization_partition)
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)
        out_ref.sum().backward()
        out_test.sum().backward()
        self.assertEqual(a_ref.grad.a, a_test.grad.a)
        self.assertEqual(a_ref.grad.b, a_test.grad.b)
        self.assertEqual(b_ref.grad.a, b_test.grad.a)
        self.assertEqual(b_ref.grad.b, b_test.grad.b)
        self.assertExpectedInline(fw_graph_cell[0].code.strip(), 'def forward(self, primals_1, primals_2, primals_3):\n    mul = torch.ops.aten.mul.Tensor(primals_1, 6);  primals_1 = None\n    mul_1 = torch.ops.aten.mul.Tensor(primals_2, 6);  primals_2 = None\n    div = torch.ops.aten.div.Tensor(primals_3, 2);  primals_3 = None\n    add = torch.ops.aten.add.Tensor(mul, div);  mul = None\n    add_1 = torch.ops.aten.add.Tensor(mul_1, div);  mul_1 = div = None\n    return [add, add_1]')
        self.assertExpectedInline(bw_graph_cell[0].code.strip(), 'def forward(self, tangents_1, tangents_2):\n    div_1 = torch.ops.aten.div.Tensor(tangents_1, 2)\n    div_2 = torch.ops.aten.div.Tensor(tangents_2, 2)\n    mul_2 = torch.ops.aten.mul.Tensor(tangents_1, 6);  tangents_1 = None\n    mul_3 = torch.ops.aten.mul.Tensor(tangents_2, 6);  tangents_2 = None\n    return [mul_2, mul_3, div_1, div_2]')

    def test_aot_dispatch_inference(self):
        if False:
            while True:
                i = 10

        def f(a, b):
            if False:
                print('Hello World!')
            aa = torch.mul(a, 6)
            bb = torch.div(b, 2)
            return aa + bb
        a1_ref = torch.ones(3, 3)
        a2_ref = torch.ones(3, 3)
        a_ref = TwoTensor(a1_ref, a2_ref)
        b_ref = torch.ones(3, 3)
        a1_test = a1_ref.clone()
        a2_test = a2_ref.clone()
        a_test = TwoTensor(a1_test, a2_test)
        b_test = b_ref.clone()
        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop, partition_fn=min_cut_rematerialization_partition)
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)

    def test_aot_dispatch_incorrect_backward(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                i = 10
                return i + 15
            aa = torch.mul(a, 2)
            bb = torch.add(b, 3)
            out_subclass = torch.div(aa, bb)
            out_reg = torch.add(b, b)
            return (out_subclass, out_reg)
        a1_ref = torch.ones(3, 3, requires_grad=True)
        a2_ref = torch.ones(3, 3, requires_grad=True)
        a_ref = TwoTensor(a1_ref, a2_ref)
        b_ref = torch.ones(3, 3, requires_grad=True)
        a1_test = a1_ref.clone().detach().requires_grad_(True)
        a2_test = a2_ref.clone().detach().requires_grad_(True)
        a_test = TwoTensor(a1_test, a2_test)
        b_test = b_ref.clone().detach().requires_grad_(True)
        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop, partition_fn=min_cut_rematerialization_partition)
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        self.assertEqual(out_ref[0].a, out_test[0].a)
        self.assertEqual(out_ref[0].b, out_test[0].b)
        self.assertEqual(out_ref[1], out_test[1])
        with self.assertRaisesRegex(AssertionError, 'incorrectly attempted to compile the backward with incorrect subclass metadata'):
            (out_test[0] + out_test[1]).sum().backward()

    def test_aot_dispatch_output_alias(self):
        if False:
            print('Hello World!')

        def f(a, b):
            if False:
                while True:
                    i = 10
            return (b.view(b.shape), a * b)
        b1_ref = torch.ones(3, 3, requires_grad=True)
        b2_ref = torch.ones(3, 3, requires_grad=True)
        b_ref = TwoTensor(b1_ref, b2_ref)
        a_ref = torch.ones(3, 3, requires_grad=True)
        b1_test = b1_ref.clone().detach().requires_grad_(True)
        b2_test = b2_ref.clone().detach().requires_grad_(True)
        b_test = TwoTensor(b1_test, b2_test)
        a_test = a_ref.clone().detach().requires_grad_(True)
        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop, partition_fn=min_cut_rematerialization_partition)
        (out_ref1, out_ref2) = f(a_ref, b_ref)
        (out_test1, out_test2) = compiled_f(a_test, b_test)
        self.assertEqual(out_ref1, out_test1)
        self.assertEqual(out_ref2.a, out_test2.a)
        self.assertEqual(out_ref2.b, out_test2.b)
        (out_ref1 + out_ref2).sum().backward()
        (out_test1 + out_test2).sum().backward()
        self.assertEqual(a_ref.grad.a, a_test.grad.a)
        self.assertEqual(a_ref.grad.b, a_test.grad.b)
        self.assertEqual(b_ref.grad.a, b_test.grad.a)
        self.assertEqual(b_ref.grad.b, b_test.grad.b)

    def test_aot_dispatch_input_mutation(self):
        if False:
            i = 10
            return i + 15

        def f(a, b):
            if False:
                while True:
                    i = 10
            a.mul_(2)
            b.mul_(3)
            return a + b
        b1_ref = torch.ones(3, 3, requires_grad=True)
        b2_ref = torch.ones(3, 3, requires_grad=True)
        b_ref_base = TwoTensor(b1_ref, b2_ref)
        a_ref_base = torch.ones(3, 3, requires_grad=True)
        b_ref = b_ref_base + 1
        a_ref = a_ref_base + 1
        b1_test = b1_ref.clone().detach().requires_grad_(True)
        b2_test = b2_ref.clone().detach().requires_grad_(True)
        b_test_base = TwoTensor(b1_test, b2_test)
        a_test_base = a_ref_base.clone().detach().requires_grad_(True)
        b_test = b_test_base + 1
        a_test = a_test_base + 1
        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop, partition_fn=min_cut_rematerialization_partition)
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)
        self.assertEqual(a_test, a_ref)
        self.assertEqual(b_test.a, b_ref.a)
        self.assertEqual(b_test.b, b_ref.b)
        (b_ref * out_ref).sum().backward()
        (b_test * out_test).sum().backward()
        self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
        self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)
        self.assertEqual(b_ref_base.grad.a, b_test_base.grad.a)
        self.assertEqual(b_ref_base.grad.b, b_test_base.grad.b)

    def test_aot_dispatch_input_metadata_mutation(self):
        if False:
            print('Hello World!')

        def f(a, b):
            if False:
                print('Hello World!')
            a.t_()
            b.t_()
            return a + b
        b1_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b2_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b_ref_base = TwoTensor(b1_ref, b2_ref)
        a_ref_base = torch.arange(9, dtype=torch.float32).reshape(3, 3).detach().requires_grad_(True)
        b_ref = b_ref_base + 1
        a_ref = a_ref_base + 1
        b1_test = b1_ref.clone().detach().requires_grad_(True)
        b2_test = b2_ref.clone().detach().requires_grad_(True)
        b_test_base = TwoTensor(b1_test, b2_test)
        a_test_base = a_ref_base.clone().detach().requires_grad_(True)
        b_test = b_test_base + 1
        a_test = a_test_base + 1
        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop, partition_fn=min_cut_rematerialization_partition)
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)
        self.assertEqual(a_test, a_ref)
        self.assertEqual(b_test.a, b_ref.a)
        self.assertEqual(b_test.b, b_ref.b)
        (b_ref * out_ref).sum().backward()
        (b_test * out_test).sum().backward()
        self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
        self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)
        self.assertEqual(b_ref_base.grad.a, b_test_base.grad.a)
        self.assertEqual(b_ref_base.grad.b, b_test_base.grad.b)

    def test_aot_dispatch_input_data_and_metadata_mutation(self):
        if False:
            print('Hello World!')

        def f(a, b):
            if False:
                i = 10
                return i + 15
            a.t_()
            b.t_()
            a.mul_(2)
            b.mul_(3)
            return a + b
        b1_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b2_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b_ref_base = TwoTensor(b1_ref, b2_ref)
        a_ref_base = torch.arange(9, dtype=torch.float32).reshape(3, 3).detach().requires_grad_(True)
        b_ref = b_ref_base + 1
        a_ref = a_ref_base + 1
        b1_test = b1_ref.clone().detach().requires_grad_(True)
        b2_test = b2_ref.clone().detach().requires_grad_(True)
        b_test_base = TwoTensor(b1_test, b2_test)
        a_test_base = a_ref_base.clone().detach().requires_grad_(True)
        b_test = b_test_base + 1
        a_test = a_test_base + 1
        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop, partition_fn=min_cut_rematerialization_partition)
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)
        self.assertEqual(a_test, a_ref)
        self.assertEqual(b_test.a, b_ref.a)
        self.assertEqual(b_test.b, b_ref.b)
        (b_ref * out_ref).sum().backward()
        (b_test * out_test).sum().backward()
        self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
        self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)
        self.assertEqual(b_ref_base.grad.a, b_test_base.grad.a)
        self.assertEqual(b_ref_base.grad.b, b_test_base.grad.b)

    def test_aot_dispatch_input_mutation_and_output_alias(self):
        if False:
            return 10

        def f(a, b):
            if False:
                while True:
                    i = 10
            a.mul_(2)
            b.mul_(3)
            return (b.view(b.shape), a + b)
        b1_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b2_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b_ref_base = TwoTensor(b1_ref, b2_ref)
        a_ref_base = torch.arange(9, dtype=torch.float32).reshape(3, 3).detach().requires_grad_(True)
        b_ref = b_ref_base + 1
        a_ref = a_ref_base + 1
        b1_test = b1_ref.clone().detach().requires_grad_(True)
        b2_test = b2_ref.clone().detach().requires_grad_(True)
        b_test_base = TwoTensor(b1_test, b2_test)
        a_test_base = a_ref_base.clone().detach().requires_grad_(True)
        b_test = b_test_base + 1
        a_test = a_test_base + 1
        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop, partition_fn=min_cut_rematerialization_partition)
        (out_ref1, out_ref2) = f(a_ref, b_ref)
        (out_test1, out_test2) = compiled_f(a_test, b_test)
        self.assertEqual(out_ref1.a, out_test1.a)
        self.assertEqual(out_ref1.b, out_test1.b)
        self.assertEqual(out_ref2.a, out_test2.a)
        self.assertEqual(out_ref2.b, out_test2.b)
        self.assertEqual(a_test, a_ref)
        self.assertEqual(b_test.a, b_ref.a)
        self.assertEqual(b_test.b, b_ref.b)
        (out_ref1 * out_ref2).sum().backward()
        (out_test1 * out_test2).sum().backward()
        self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
        self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)

class TestAOTModuleSimplified(AOTTestCase):

    def test_aot_module_simplified(self):
        if False:
            while True:
                i = 10

        class MockModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.linear = torch.nn.Linear(20, 30)

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return (self.linear(x) + y,)
        mod = MockModule()
        mod.zero_grad()
        x = torch.randn(128, 20, requires_grad=True)
        y = torch.randn(128, 30, requires_grad=True)
        inputs = [x, y]
        cloned_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
        ref = mod(*inputs)
        ref[0].sum().backward()
        compiled_f = aot_module_simplified(mod, cloned_inputs, nop)
        mod.zero_grad()
        res = compiled_f(*cloned_inputs)
        res[0].sum().backward()
        assert torch.allclose(ref[0], res[0])
        assert torch.allclose(inputs[0].grad, cloned_inputs[0].grad)
        assert torch.allclose(inputs[1].grad, cloned_inputs[1].grad)

    def test_aot_module_simplified_dynamic(self):
        if False:
            return 10

        class MockModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.linear = torch.nn.Linear(20, 30)

            def forward(self, x, y):
                if False:
                    return 10
                return (self.linear(x) + y,)
        mod = MockModule()
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        x = torch.randn(128, 20, requires_grad=True)
        y = torch.randn(128, 30, requires_grad=True)
        inputs = [x, y]
        fake_inputs = [fake_mode.from_tensor(x) for x in inputs]
        compiled_f = aot_module_simplified(mod, fake_inputs, nop)
        ref = mod(*inputs)
        ref[0].sum().backward()
        cloned_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
        res = compiled_f(*cloned_inputs)
        res[0].sum().backward()
        self.assertExpectedInline(shape_env.format_guards(), ' - Eq(s1, 20)\n - Eq(s2, 30)')
        assert torch.allclose(ref[0], res[0])
        assert torch.allclose(inputs[0].grad, cloned_inputs[0].grad)
        assert torch.allclose(inputs[1].grad, cloned_inputs[1].grad)

    def test_lift_fresh_copy_in_graph(self):
        if False:
            while True:
                i = 10

        class MyMod(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                _tensor_constant0 = torch.tensor([1])
                lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0)
                y = x.mul(lift_fresh_copy)
                return (y,)
        mod = MyMod()
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        x = torch.ones(4, requires_grad=True)
        inputs = [x]
        fake_inputs = [fake_mode.from_tensor(x) for x in inputs]
        compiled_f = aot_module_simplified(mod, fake_inputs, nop)
        out_ref = mod(x)
        out_test = compiled_f(x)
        self.assertEqual(out_ref[0].detach(), out_test[0].detach())

    def test_inference_python_dispatcher(self):
        if False:
            while True:
                i = 10

        class MockModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            def forward(self, x):
                if False:
                    return 10
                return (self.upsample(x),)
        mod = MockModule()
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        x = torch.randn(2, 512, 40, 59)
        inputs = [x]
        fake_inputs = [fake_mode.from_tensor(x) for x in inputs]
        compiled_f = aot_module_simplified(mod, fake_inputs, nop)

    def test_aot_module_simplified_preserves_stack_trace(self):
        if False:
            return 10

        class MockModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.linear = torch.nn.Linear(20, 30)

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                z = self.linear(x)
                z = z + y
                z = z.relu()
                return (z,)
        tracer = torch.fx.Tracer()
        tracer.record_stack_traces = True
        graph = tracer.trace(MockModule())
        mod = torch.fx.GraphModule(tracer.root, graph)
        for node in mod.graph.nodes:
            if node.op == 'output':
                continue
            self.assertTrue(node.stack_trace is not None)
            assert 'test_aotdispatch.py' in node.stack_trace

        def assert_compiler(gm: torch.fx.GraphModule, _):
            if False:
                while True:
                    i = 10
            for node in gm.graph.nodes:
                if node.op == 'output' or node.op == 'placeholder':
                    continue
                self.assertTrue(node.stack_trace is not None)
                assert 'test_aotdispatch.py' in node.stack_trace
            return gm.forward
        x = torch.randn(128, 20, requires_grad=True)
        y = torch.randn(128, 30, requires_grad=True)
        inputs = [x, y]
        compiled_f = aot_module_simplified(mod, inputs, fw_compiler=assert_compiler, bw_compiler=assert_compiler)
        res = compiled_f(*inputs)
        res[0].sum().backward()

    def test_aot_module_simplified_fake_tensor_gm_raises(self):
        if False:
            while True:
                i = 10
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        real_x = torch.randn(4, requires_grad=True)
        fake_x = fake_mode.from_tensor(real_x)
        real_z = torch.randn(4)
        fake_z = fake_mode.from_tensor(real_z)

        class MockModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return (x + fake_z,)
        with self.assertRaisesRegex(AssertionError, 'Unexpected fake'):
            aot_module_simplified(MockModule(), (fake_x,), nop)
aot_autograd_failures = {xfail('cov'), xfail('nn.functional.gaussian_nll_loss'), xfail('tensor_split'), xfail('corrcoef'), xfail('quantile'), xfail('nanquantile'), xfail('narrow'), xfail('istft'), xfail('linalg.eig'), skip('as_strided_scatter'), skip('as_strided', 'partial_views'), skip('max_pool2d_with_indices_backward'), skip('nn.functional.nll_loss', ''), xfail('to_sparse'), xfail('corrcoef'), xfail('cov'), xfail('chalf'), xfail('sparse.sampled_addmm'), xfail('sparse.mm', 'reduce'), skip('nn.functional.binary_cross_entropy_with_logits'), skip('nn.functional.margin_ranking_loss'), skip('linalg.lu_solve'), decorate('matmul', decorator=unittest.skipIf(IS_ARM64, 'flaky')), decorate('__rmatmul__', decorator=unittest.skipIf(IS_ARM64, 'flaky')), decorate('svd_lowrank', decorator=toleranceOverride({torch.float32: tol(atol=0.0001, rtol=1e-05)})), decorate('linalg.householder_product', decorator=unittest.skipIf(IS_MACOS and IS_X86, 'flaky')), decorate('linalg.pinv', 'singular', decorator=toleranceOverride({torch.float32: tol(atol=1e-05, rtol=1e-05)})), decorate('nn.functional.interpolate', 'bicubic', decorator=toleranceOverride({torch.float32: tol(atol=0.0001, rtol=1e-05)})), decorate('nn.functional.conv2d', decorator=unittest.skipIf(IS_ARM64, 'flaky'))}
symbolic_aot_autograd_failures = {xfail('block_diag', ''), xfail('combinations', ''), xfail('frexp', ''), xfail('i0', ''), xfail('index_fill', ''), xfail('kthvalue', ''), xfail('linalg.eigvals', ''), xfail('linalg.lstsq', ''), xfail('linalg.lstsq', 'grad_oriented'), xfail('linalg.lu_solve', ''), skip('nn.functional.batch_norm', ''), xfail('nn.functional.binary_cross_entropy', ''), xfail('nn.functional.cross_entropy', ''), xfail('nn.functional.ctc_loss', ''), xfail('nn.functional.embedding_bag', ''), xfail('nn.functional.fractional_max_pool2d', ''), xfail('nn.functional.fractional_max_pool3d', ''), xfail('nn.functional.group_norm', ''), xfail('nn.functional.interpolate', 'linear'), xfail('nn.functional.interpolate', 'trilinear'), xfail('nn.functional.nll_loss', ''), xfail('nn.functional.pixel_shuffle', ''), xfail('nn.functional.pixel_unshuffle', ''), xfail('_segment_reduce', 'lengths'), xfail('_segment_reduce', 'offsets'), xfail('special.i1', ''), xfail('trace', ''), xfail('_upsample_bilinear2d_aa'), decorate('linalg.householder_product', decorator=unittest.skipIf(IS_MACOS and IS_X86, 'flaky')), xfail('fft.fft', ''), xfail('fft.hfft2', ''), xfail('fft.hfft', ''), xfail('fft.hfftn', ''), xfail('fft.ifft', ''), xfail('fft.ihfft2', ''), xfail('fft.ihfft', ''), xfail('fft.ihfftn', ''), xfail('fft.irfft2', ''), xfail('fft.irfft', ''), xfail('fft.irfftn', ''), xfail('fft.rfft2', ''), xfail('fft.rfft', ''), xfail('fft.rfftn', ''), xfail('stft', '')}

def _test_aot_autograd_helper(self, device, dtype, op, dynamic=False):
    if False:
        print('Hello World!')
    if not op.supports_autograd:
        self.skipTest('Op does not support autograd')
    cant_check_data_specialization = set({'nn.functional.max_unpool1d', 'nn.functional.max_unpool2d', 'nn.functional.max_unpool3d'})
    try_check_data_specialization = op.name not in cant_check_data_specialization
    sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
    for sample_input in sample_inputs_itr:
        t_args = [sample_input.input] + list(sample_input.args)
        t_kwargs = sample_input.kwargs
        try:
            aot_autograd_check(op.op, t_args, t_kwargs, dynamic, self.assertRaisesRegex, self.assertEqual, check_gradients=True, try_check_data_specialization=try_check_data_specialization)
        except DynamicOutputShapeException:
            self.skipTest('Dynamic output shape operation in trace')
        except GuardOnDataDependentSymNode:
            if op.name == '__getitem__':
                self.skipTest('Dynamic output shape operation in trace')
            else:
                raise

def _test_aot_autograd_module_helper(self, device, dtype, training, module_info, *, dynamic=False):
    if False:
        print('Hello World!')
    module_cls = module_info.module_cls
    module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype, requires_grad=True, training=training)
    for module_input in module_inputs:
        if module_input.forward_input is None:
            continue
        (args, kwargs) = (module_input.constructor_input.args, module_input.constructor_input.kwargs)
        m = module_cls(*args, **kwargs)
        m.to(device).to(dtype)
        m.train(training)
        (args, kwargs) = (module_input.forward_input.args, module_input.forward_input.kwargs)
        (flat_args, args_spec) = pytree.tree_flatten((args, kwargs))
        if any(tuple((isinstance(flat_arg, PackedSequence) for flat_arg in flat_args))):
            continue
        if issubclass(module_info.module_cls, torch.nn.modules.lazy.LazyModuleMixin):
            with torch.no_grad():
                m(*args, **kwargs)
        sentinel_val = -42
        is_tensor_spec = [sentinel_val if isinstance(arg, torch.Tensor) else arg for arg in flat_args]
        args = [arg for arg in flat_args if isinstance(arg, torch.Tensor)]

        def f(params_buffers_args):
            if False:
                print('Hello World!')
            (named_params, named_buffers, args) = params_buffers_args
            cur_flat_args = list(is_tensor_spec)
            args = iter(args)
            for (idx, v) in enumerate(cur_flat_args):
                if v == sentinel_val:
                    cur_flat_args[idx] = next(args)
            (c_args, c_kwargs) = pytree.tree_unflatten(cur_flat_args, args_spec)
            params_and_buffers = {**named_params, **named_buffers}
            return torch.func.functional_call(m, params_and_buffers, c_args, c_kwargs)
        named_params = dict(m.named_parameters(remove_duplicate=False))
        named_buffers = dict(m.named_buffers(remove_duplicate=False))
        num_params_buffers = len(named_params) + len(named_buffers)
        compiled_f = aot_function(f, nop, num_params_buffers=num_params_buffers, dynamic=dynamic)
        params_buffers_args = [named_params, named_buffers, args]
        _test_aot_autograd_forwards_backwards_helper(f, compiled_f, params_buffers_args, self.assertRaisesRegex, self.assertEqual, True)

class TestEagerFusionOpInfo(AOTTestCase):

    @ops(op_db + control_flow_opinfo_db, allowed_dtypes=(torch.float,))
    @skipOps('TestEagerFusionOpInfo', 'test_aot_autograd_exhaustive', aot_autograd_failures)
    def test_aot_autograd_exhaustive(self, device, dtype, op):
        if False:
            while True:
                i = 10
        _test_aot_autograd_helper(self, device, dtype, op)

    @ops(op_db + control_flow_opinfo_db, allowed_dtypes=(torch.float,))
    @patch('functorch.compile.config.debug_assert', True)
    @skipOps('TestEagerFusionOpInfo', 'test_aot_autograd_symbolic_exhaustive', aot_autograd_failures | symbolic_aot_autograd_failures)
    def test_aot_autograd_symbolic_exhaustive(self, device, dtype, op):
        if False:
            i = 10
            return i + 15
        _test_aot_autograd_helper(self, device, dtype, op, dynamic=True)
aot_autograd_module_failures = set({torch.nn.GaussianNLLLoss, torch.nn.TransformerEncoder, torch.nn.Transformer})
symbolic_aot_autograd_module_failures = {torch.nn.Transformer, torch.nn.TransformerEncoder, torch.nn.GaussianNLLLoss, torch.nn.GroupNorm, torch.nn.FractionalMaxPool2d, torch.nn.FractionalMaxPool3d}

class TestEagerFusionModuleInfo(AOTTestCase):

    @modules(module_db, allowed_dtypes=(torch.float,))
    @decorateForModules(unittest.expectedFailure, aot_autograd_module_failures)
    def test_aot_autograd_module_exhaustive(self, device, dtype, training, module_info):
        if False:
            while True:
                i = 10
        _test_aot_autograd_module_helper(self, device, dtype, training, module_info)

    @modules(module_db, allowed_dtypes=(torch.float,))
    @decorateForModules(unittest.expectedFailure, aot_autograd_module_failures | symbolic_aot_autograd_module_failures)
    def test_aot_autograd_symbolic_module_exhaustive(self, device, dtype, training, module_info):
        if False:
            while True:
                i = 10
        _test_aot_autograd_module_helper(self, device, dtype, training, module_info, dynamic=True)
only_for = 'cpu'
instantiate_device_type_tests(TestPythonKey, globals(), only_for=only_for)
instantiate_device_type_tests(TestEagerFusionOpInfo, globals(), only_for=only_for)
instantiate_device_type_tests(TestEagerFusionModuleInfo, globals(), only_for=only_for)
if __name__ == '__main__':
    run_tests()