import torch
from contextlib import nullcontext
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfTorchDynamo, TEST_WITH_TORCHDYNAMO, IS_WINDOWS, xfail_inherited_tests
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode, dispatch_functionalize
from torch.testing._internal.logging_tensor import LoggingTensor, capture_logs
from torch.utils._pytree import tree_map_only
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.reinplace import reinplace
from torch._dispatch.python import enable_crossref_functionalize, enable_python_dispatcher
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _pytree as pytree
import unittest

def are_aliased(x, y):
    if False:
        for i in range(10):
            print('nop')
    x_storage = StorageWeakRef(x.storage())
    y_storage = StorageWeakRef(y.storage())
    return x_storage == y_storage

def _functionalize(f, *, reapply_views: bool, crossref: bool, skip_input_mutations: bool=False):
    if False:
        i = 10
        return i + 15

    def to_fun(t: torch.Tensor):
        if False:
            print('Hello World!')
        func_t = torch._to_functional_tensor(t)
        func_t.requires_grad = t.requires_grad
        return func_t

    def wrapped(*inputs):
        if False:
            while True:
                i = 10
        ctx = nullcontext()
        if crossref:
            ctx = enable_crossref_functionalize()
        with ctx:
            inputs_functional = tree_map_only(torch.Tensor, to_fun, inputs)
            torch._enable_functionalization(reapply_views=reapply_views)
            try:
                out = f(*inputs_functional)
            finally:
                torch._disable_functionalization()
            flat_inputs = pytree.tree_leaves(inputs)
            flat_inputs_functional = pytree.tree_leaves(inputs_functional)
            for (inpt, input_functional) in zip(flat_inputs, flat_inputs_functional):
                torch._sync(input_functional)
                inpt_new = torch._from_functional_tensor(input_functional)
                if inpt_new is not inpt and (not skip_input_mutations):
                    if inpt_new.shape == inpt.shape:
                        inpt.copy_(inpt_new)
            tree_map_only(torch.Tensor, torch._sync, out)
            out_unwrapped = tree_map_only(torch.Tensor, torch._from_functional_tensor, out)
            return out_unwrapped
    return wrapped

@unittest.skipIf(TEST_WITH_TORCHDYNAMO, 'https://github.com/pytorch/pytorch/issues/81457')
class TestFunctionalization(TestCase):
    crossref = False

    def get_logs(self, func, *inpts, reapply_views=False, run_reinplace=False):
        if False:
            print('Hello World!')
        inpts_clone = tree_map_only(torch.Tensor, torch.clone, inpts)
        traced_f = make_fx(_functionalize(func, reapply_views=reapply_views, crossref=self.crossref))(*inpts)
        if run_reinplace:
            traced_f = reinplace(traced_f, *inpts_clone)
        return traced_f.code

    def assert_functionalization(self, func, *inpts, reapply_views=False, mutated_input_metadata=False):
        if False:
            while True:
                i = 10
        clones1 = tree_map_only(torch.Tensor, torch.clone, inpts)
        clones2 = tree_map_only(torch.Tensor, torch.clone, inpts)
        clones3 = tree_map_only(torch.Tensor, torch.clone, inpts)
        out_ref = func(*inpts)
        out_functional = _functionalize(func, reapply_views=reapply_views, crossref=self.crossref)(*clones1)
        functional_func = make_fx(_functionalize(func, reapply_views=True, crossref=self.crossref))(*clones2)
        reinplace_func = reinplace(functional_func, *clones2)
        out_reinplace = reinplace_func(*clones3)
        if not mutated_input_metadata:
            flat_inpts = pytree.tree_leaves(inpts)
            flat_clones1 = pytree.tree_leaves(clones1)
            flat_clones3 = pytree.tree_leaves(clones3)
            for (inpt, input_clone, input_clone3) in zip(flat_inpts, flat_clones1, flat_clones3):
                self.assertEqual(inpt, input_clone)
                self.assertEqual(inpt, input_clone3)
        if isinstance(out_ref, tuple):
            (out_refs, out_functionals, out_reinplaces) = (list(out_ref), list(out_functional), list(out_reinplace))
        else:
            (out_refs, out_functionals, out_reinplaces) = ([out_ref], [out_functional], [out_reinplace])
        for (out_ref_, out_functional_, out_reinplace_) in zip(out_refs, out_functionals, out_reinplaces):
            self.assertEqual(out_ref_, out_functional_)
            self.assertEqual(out_ref_, out_reinplace_)

    def test_save_for_backwards_segfault(self):
        if False:
            return 10
        inp = torch._to_functional_tensor(LoggingTensor(torch.randn(2, 2))).requires_grad_(True)
        inp.exp()

    def test_multiple_views_of_same_base(self):
        if False:
            return 10

        def f(x):
            if False:
                while True:
                    i = 10
            y = x.view(-1)
            z = x.view(-1)
            x.add_(1)
            y2 = y + 1
            z2 = z + 1
            return z2
        self.assert_functionalization(f, torch.ones(4))

    def test_freeze(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                while True:
                    i = 10
            y = x.clone()
            z = y[0]
            torch._freeze_functional_tensor(y)
            x.add_(1)
            self.assertRaises(RuntimeError, lambda : y.add_(1))
            self.assertRaises(RuntimeError, lambda : z.add_(1))
            return z
        _functionalize(f, reapply_views=True, crossref=self.crossref)(torch.ones(3, 3))

    def test_copy_stride_mismatch(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                return 10
            y = torch.empty_strided((2, 2), (5, 1))
            y.copy_(x)
            return y
        r = _functionalize(f, reapply_views=True, crossref=self.crossref)(torch.ones(2, 2))
        self.assertEqual(r.stride(), (5, 1))

    def test_set_(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                i = 10
                return i + 15
            y = torch.ones(2)
            y.set_(x.storage())
            return y
        r = _functionalize(f, reapply_views=True, crossref=False)(torch.ones(2))
        self.assertEqual(str(r.device), 'cpu')

    def test_advanced_indexing(self):
        if False:
            while True:
                i = 10

        def f():
            if False:
                while True:
                    i = 10
            x = torch.zeros(3, 3)
            idx = torch.tensor([0])
            val = torch.ones(3, 1)
            x[:, idx] = val
            return x
        self.assert_functionalization(f)

    def test_view_clone_view_inplace(self):
        if False:
            print('Hello World!')

        def f(input):
            if False:
                return 10
            shape = [1, 1024, 128, 128]
            input_reshaped = input.view(shape)
            out = input_reshaped.clone()
            r = out.view(input.shape)
            r.relu_()
            return r

        def g(x):
            if False:
                return 10
            loss = f(x).sum()
            from torch._functorch.aot_autograd import setup_stacktrace_preservation_hooks
            import torch.fx.traceback as fx_traceback
            setup_stacktrace_preservation_hooks([loss.grad_fn])
            with fx_traceback.preserve_node_meta():
                loss.backward()
            return x.grad
        with torch.autograd.detect_anomaly(check_nan=False):
            logs = self.get_logs(g, torch.ones(16, 64, 128, 128, requires_grad=True))
        self.assertExpectedInline(logs, '\n\n\ndef forward(self, arg0_1):\n    view_copy = torch.ops.aten.view_copy.default(arg0_1, [1, 1024, 128, 128]);  arg0_1 = None\n    clone = torch.ops.aten.clone.default(view_copy);  view_copy = None\n    view_copy_1 = torch.ops.aten.view_copy.default(clone, [16, 64, 128, 128])\n    relu = torch.ops.aten.relu.default(view_copy_1);  view_copy_1 = None\n    view_copy_2 = torch.ops.aten.view_copy.default(relu, [1, 1024, 128, 128]);  relu = None\n    view_copy_3 = torch.ops.aten.view_copy.default(view_copy_2, [16, 64, 128, 128]);  view_copy_2 = None\n    view_copy_4 = torch.ops.aten.view_copy.default(clone, [16, 64, 128, 128]);  clone = None\n    sum_1 = torch.ops.aten.sum.default(view_copy_3)\n    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None\n    expand_copy = torch.ops.aten.expand_copy.default(ones_like, [16, 64, 128, 128]);  ones_like = None\n    view_copy_5 = torch.ops.aten.view_copy.default(expand_copy, [1, 1024, 128, 128]);  expand_copy = None\n    new_empty_strided = torch.ops.aten.new_empty_strided.default(view_copy_5, [1, 1024, 128, 128], [16777216, 16384, 128, 1])\n    copy = torch.ops.aten.copy.default(new_empty_strided, view_copy_5);  new_empty_strided = view_copy_5 = None\n    view_copy_6 = torch.ops.aten.view_copy.default(copy, [16, 64, 128, 128])\n    view_copy_7 = torch.ops.aten.view_copy.default(copy, [16, 64, 128, 128])\n    clone_1 = torch.ops.aten.clone.default(view_copy_7, memory_format = torch.contiguous_format)\n    threshold_backward = torch.ops.aten.threshold_backward.default(clone_1, view_copy_3, 0);  clone_1 = view_copy_3 = None\n    copy_1 = torch.ops.aten.copy.default(view_copy_7, threshold_backward);  view_copy_7 = threshold_backward = None\n    view_copy_8 = torch.ops.aten.view_copy.default(copy_1, [1, 1024, 128, 128]);  copy_1 = None\n    view_copy_9 = torch.ops.aten.view_copy.default(view_copy_8, [16, 64, 128, 128])\n    view_copy_10 = torch.ops.aten.view_copy.default(copy, [16, 64, 128, 128]);  copy = None\n    detach_copy = torch.ops.aten.detach_copy.default(view_copy_10);  view_copy_10 = None\n    view_copy_11 = torch.ops.aten.view_copy.default(view_copy_8, [16, 64, 128, 128]);  view_copy_8 = None\n    detach_copy_1 = torch.ops.aten.detach_copy.default(view_copy_11);  view_copy_11 = None\n    return detach_copy_1\n    ')

    def test_simple(self):
        if False:
            return 10

        def f(x):
            if False:
                return 10
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            y.add_(tmp)
            z = x * x
            return y
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)\n    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2])\n    add = torch.ops.aten.add.Tensor(view_copy, ones);  view_copy = ones = None\n    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2]);  add = None\n    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [4, 2])\n    mul = torch.ops.aten.mul.Tensor(view_copy_1, view_copy_1)\n    copy_ = torch.ops.aten.copy_.default(arg0_1, view_copy_1);  arg0_1 = view_copy_1 = None\n    return view_copy_2\n    ")
        reinplaced_logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, "\n\n\ndef forward(self, arg0_1):\n    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)\n    view = torch.ops.aten.view.default(arg0_1, [4, 2])\n    add = torch.ops.aten.add.Tensor(view, ones);  view = ones = None\n    view_1 = torch.ops.aten.view.default(add, [4, 2]);  add = None\n    view_2 = torch.ops.aten.view.default(view_1, [4, 2])\n    mul = torch.ops.aten.mul.Tensor(view_1, view_1)\n    copy_ = torch.ops.aten.copy_.default(arg0_1, view_1);  arg0_1 = view_1 = None\n    return view_2\n    ")

    def test_simple_out(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                while True:
                    i = 10
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            z = torch.empty(())
            torch.add(y, tmp, out=z)
            w = z * z
            return w
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)\n    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2]);  arg0_1 = None\n    empty = torch.ops.aten.empty.memory_format([], device = device(type='cpu'), pin_memory = False)\n    add = torch.ops.aten.add.Tensor(view_copy, ones);  view_copy = ones = None\n    mul = torch.ops.aten.mul.Tensor(add, add);  add = None\n    return mul\n    ")
        reinplaced_logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, "\n\n\ndef forward(self, arg0_1):\n    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)\n    view = torch.ops.aten.view.default(arg0_1, [4, 2]);  arg0_1 = None\n    empty = torch.ops.aten.empty.memory_format([], device = device(type='cpu'), pin_memory = False)\n    add = torch.ops.aten.add.Tensor(view, ones);  view = ones = None\n    mul = torch.ops.aten.mul.Tensor(add, add);  add = None\n    return mul\n    ")

    def test_multi_out(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                while True:
                    i = 10
            out_min = torch.empty(4)
            out_max = torch.empty(4)
            torch.aminmax(x, dim=0, out=(out_max, out_min))
            return out_max
        self.assert_functionalization(f, torch.arange(8, dtype=torch.float32))
        logs = self.get_logs(f, torch.arange(8, dtype=torch.float32))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    empty = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False)\n    empty_1 = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False)\n    aminmax = torch.ops.aten.aminmax.default(arg0_1, dim = 0);  arg0_1 = None\n    getitem = aminmax[0]\n    getitem_1 = aminmax[1];  aminmax = None\n    return getitem\n    ")
        reinplaced_logs = self.get_logs(f, torch.arange(8, dtype=torch.float32), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, "\n\n\ndef forward(self, arg0_1):\n    empty = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False)\n    empty_1 = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False)\n    aminmax = torch.ops.aten.aminmax.default(arg0_1, dim = 0);  arg0_1 = None\n    getitem = aminmax[0]\n    getitem_1 = aminmax[1];  aminmax = None\n    return getitem\n    ")

    def test_tensor_ctr(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                return 10
            y = torch.tensor((1, 2, 3))
            z = y.view(-1)
            z.add_(1)
            return y
        inpt = torch.arange(3, dtype=torch.float32)
        self.assert_functionalization(f, inpt)
        logs = self.get_logs(f, inpt)
        self.assertExpectedInline(logs, '\n\n\ndef forward(self, arg0_1):\n    _tensor_constant0 = self._tensor_constant0\n    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None\n    view_copy = torch.ops.aten.view_copy.default(lift_fresh_copy, [-1]);  lift_fresh_copy = None\n    add = torch.ops.aten.add.Tensor(view_copy, 1);  view_copy = None\n    view_copy_1 = torch.ops.aten.view_copy.default(add, [3]);  add = None\n    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [-1])\n    return view_copy_1\n    ')
        reinplaced_logs = self.get_logs(f, inpt, reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, '\n\n\ndef forward(self, arg0_1):\n    _tensor_constant0 = self._tensor_constant0\n    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None\n    view = torch.ops.aten.view.default(lift_fresh_copy, [-1]);  lift_fresh_copy = None\n    add = torch.ops.aten.add_.Tensor(view, 1)\n    view_1 = torch.ops.aten.view.default(view, [3]);  view = None\n    view_2 = torch.ops.aten.view.default(view_1, [-1])\n    return view_1\n    ')

    def test_advanced_indexing_correct_strides(self):
        if False:
            while True:
                i = 10

        def f(a):
            if False:
                return 10
            b = a.clone()[:, 1]
            c = torch.ones_like(b, dtype=torch.bool)
            d = b.masked_fill_(c, 0)
            return d
        self.assert_functionalization(f, torch.ones(2, 2), reapply_views=True)

    def test_tensor_list_mixed_functional_nonfunctional(self):
        if False:
            return 10
        nonfunctional_tensor = torch.ones(2, dtype=torch.long)

        def f(x):
            if False:
                i = 10
                return i + 15
            functional_tensor = torch.ones(2, dtype=torch.long)
            out = x[functional_tensor, nonfunctional_tensor]
            return out
        out = f(torch.ones(2, 2))
        out_functional = _functionalize(f, reapply_views=True, crossref=self.crossref)(torch.ones(2, 2))
        self.assertEqual(out, out_functional)

    def test_inplace_on_non_view(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                i = 10
                return i + 15
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            x.add_(tmp)
            return y
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)\n    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2])\n    add = torch.ops.aten.add.Tensor(arg0_1, ones);  ones = None\n    copy_ = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = None\n    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2]);  add = None\n    return view_copy_1\n    ")
        reinplaced_logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, "\n\n\ndef forward(self, arg0_1):\n    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)\n    view = torch.ops.aten.view.default(arg0_1, [4, 2])\n    add = torch.ops.aten.add.Tensor(arg0_1, ones);  ones = None\n    copy_ = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = None\n    view_1 = torch.ops.aten.view.default(add, [4, 2]);  add = None\n    return view_1\n    ")

    def test_mutable_op_not_inplace_or_other(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                return 10
            return torch._fused_moving_avg_obs_fq_helper(x, x, x, x, x, x, x, 1.0, 0, 1, 0)
        logs = self.get_logs(f, torch.ones(1))
        self.assertExpectedInline(logs, '\n\n\ndef forward(self, arg0_1):\n    _fused_moving_avg_obs_fq_helper_functional = torch.ops.aten._fused_moving_avg_obs_fq_helper_functional.default(arg0_1, arg0_1, arg0_1, arg0_1, arg0_1, arg0_1, arg0_1, 1.0, 0, 1, 0)\n    getitem = _fused_moving_avg_obs_fq_helper_functional[0]\n    getitem_1 = _fused_moving_avg_obs_fq_helper_functional[1]\n    getitem_2 = _fused_moving_avg_obs_fq_helper_functional[2]\n    getitem_3 = _fused_moving_avg_obs_fq_helper_functional[3]\n    getitem_4 = _fused_moving_avg_obs_fq_helper_functional[4]\n    getitem_5 = _fused_moving_avg_obs_fq_helper_functional[5];  _fused_moving_avg_obs_fq_helper_functional = None\n    copy_ = torch.ops.aten.copy_.default(arg0_1, getitem_5);  arg0_1 = getitem_5 = None\n    return (getitem, getitem_1)\n    ')

    def test_as_strided(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                i = 10
                return i + 15
            y = x.as_strided((2,), (2,), 1)
            y.add_(1)
            return x
        self.assert_functionalization(f, torch.ones(9))
        logs = self.get_logs(f, torch.ones(9))
        self.assertExpectedInline(logs, '\n\n\ndef forward(self, arg0_1):\n    as_strided_copy = torch.ops.aten.as_strided_copy.default(arg0_1, [2], [2], 1)\n    add = torch.ops.aten.add.Tensor(as_strided_copy, 1);  as_strided_copy = None\n    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(arg0_1, add, [2], [2], 1);  add = None\n    as_strided_copy_1 = torch.ops.aten.as_strided_copy.default(as_strided_scatter, [2], [2], 1)\n    copy_ = torch.ops.aten.copy_.default(arg0_1, as_strided_scatter);  arg0_1 = None\n    return as_strided_scatter\n    ')

    def test_tensor_list_composite(self):
        if False:
            return 10

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            y = torch.block_diag(x, x)
            return y
        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(logs, '\n\n\ndef forward(self, arg0_1):\n    block_diag = torch.ops.aten.block_diag.default([arg0_1, arg0_1]);  arg0_1 = None\n    return block_diag\n    ')

    def test_cat(self):
        if False:
            return 10

        def f(x):
            if False:
                while True:
                    i = 10
            out = torch.empty(0)
            torch.cat((x,), out=out)
            return out
        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    empty = torch.ops.aten.empty.memory_format([0], device = device(type='cpu'), pin_memory = False)\n    cat = torch.ops.aten.cat.default([arg0_1]);  arg0_1 = None\n    return cat\n    ")
        reinplaced_logs = self.get_logs(f, torch.ones(2, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, "\n\n\ndef forward(self, arg0_1):\n    empty = torch.ops.aten.empty.memory_format([0], device = device(type='cpu'), pin_memory = False)\n    cat = torch.ops.aten.cat.default([arg0_1]);  arg0_1 = None\n    return cat\n    ")

    def test_diagonal(self):
        if False:
            return 10

        def f(x):
            if False:
                return 10
            tmp = torch.ones(2)
            y = x.clone().diagonal()
            y.add_(tmp)
            z = x * x
            return z
        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)\n    clone = torch.ops.aten.clone.default(arg0_1)\n    diagonal_copy = torch.ops.aten.diagonal_copy.default(clone)\n    add = torch.ops.aten.add.Tensor(diagonal_copy, ones);  diagonal_copy = ones = None\n    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(clone, add);  clone = add = None\n    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter);  diagonal_scatter = None\n    mul = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None\n    return mul\n    ")
        reinplaced_logs = self.get_logs(f, torch.ones(2, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, "\n\n\ndef forward(self, arg0_1):\n    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)\n    clone = torch.ops.aten.clone.default(arg0_1)\n    diagonal = torch.ops.aten.diagonal.default(clone)\n    add = torch.ops.aten.add_.Tensor(diagonal, ones);  diagonal = ones = None\n    diagonal_1 = torch.ops.aten.diagonal.default(clone);  clone = None\n    mul = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None\n    return mul\n    ")

    def test_diagonal_mutated_input(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                return 10
            tmp = torch.ones(2)
            y = x.diagonal()
            y.add_(tmp)
            return x
        x = torch.ones(2, 2)
        self.assert_functionalization(f, x)
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)\n    diagonal_copy = torch.ops.aten.diagonal_copy.default(arg0_1)\n    add = torch.ops.aten.add.Tensor(diagonal_copy, ones);  diagonal_copy = ones = None\n    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(arg0_1, add);  add = None\n    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter)\n    copy_ = torch.ops.aten.copy_.default(arg0_1, diagonal_scatter);  arg0_1 = None\n    return diagonal_scatter\n    ")

    def test_channels_last_contiguous(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                print('Hello World!')
            return x.contiguous(memory_format=torch.channels_last)
            tmp = torch.ones(2)
            y = x.diagonal()
            y.add_(tmp)
            return x
        x = torch.randn(4, 8, 8, 3).permute(0, 3, 1, 2)
        self.assert_functionalization(f, x)
        logs = self.get_logs(f, x).strip()
        self.assertExpectedInline(logs, 'def forward(self, arg0_1):\n    return arg0_1')

    def test_split(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            tmp = torch.ones(2)
            (y1, y2) = x.split(2)
            y3 = y2.diagonal()
            y3.add_(tmp)
            z = x * x
            return y3
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)\n    split_copy = torch.ops.aten.split_copy.Tensor(arg0_1, 2)\n    getitem = split_copy[0]\n    getitem_1 = split_copy[1];  split_copy = None\n    diagonal_copy = torch.ops.aten.diagonal_copy.default(getitem_1);  getitem_1 = None\n    add = torch.ops.aten.add.Tensor(diagonal_copy, ones);  diagonal_copy = ones = None\n    split_copy_1 = torch.ops.aten.split_copy.Tensor(arg0_1, 2)\n    getitem_2 = split_copy_1[0]\n    getitem_3 = split_copy_1[1];  split_copy_1 = None\n    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(getitem_3, add);  getitem_3 = add = None\n    slice_scatter = torch.ops.aten.slice_scatter.default(arg0_1, diagonal_scatter, 0, 2, 4);  diagonal_scatter = None\n    split_copy_2 = torch.ops.aten.split_copy.Tensor(slice_scatter, 2)\n    getitem_4 = split_copy_2[0]\n    getitem_5 = split_copy_2[1];  split_copy_2 = None\n    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(getitem_5);  getitem_5 = None\n    mul = torch.ops.aten.mul.Tensor(slice_scatter, slice_scatter)\n    copy_ = torch.ops.aten.copy_.default(arg0_1, slice_scatter);  arg0_1 = slice_scatter = None\n    return diagonal_copy_1\n    ")

    def test_view_inplace(self):
        if False:
            return 10

        def f(x):
            if False:
                while True:
                    i = 10
            tmp = torch.ones(4)
            x.transpose_(1, 0)
            y = x[0]
            y.add_(tmp)
            return x
        self.assert_functionalization(f, torch.ones(4, 2), mutated_input_metadata=True)
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    ones = torch.ops.aten.ones.default([4], device = device(type='cpu'), pin_memory = False)\n    transpose_copy = torch.ops.aten.transpose_copy.int(arg0_1, 1, 0)\n    select_copy = torch.ops.aten.select_copy.int(transpose_copy, 0, 0);  transpose_copy = None\n    add = torch.ops.aten.add.Tensor(select_copy, ones);  select_copy = ones = None\n    transpose_copy_1 = torch.ops.aten.transpose_copy.int(arg0_1, 1, 0);  arg0_1 = None\n    select_scatter = torch.ops.aten.select_scatter.default(transpose_copy_1, add, 0, 0);  transpose_copy_1 = add = None\n    transpose_copy_2 = torch.ops.aten.transpose_copy.int(select_scatter, 1, 0);  select_scatter = None\n    transpose_copy_3 = torch.ops.aten.transpose_copy.int(transpose_copy_2, 1, 0)\n    select_copy_1 = torch.ops.aten.select_copy.int(transpose_copy_3, 0, 0);  transpose_copy_3 = None\n    transpose_copy_4 = torch.ops.aten.transpose_copy.int(transpose_copy_2, 1, 0);  transpose_copy_2 = None\n    return transpose_copy_4\n    ")

    def test_optional_tensor_list(self):
        if False:
            return 10

        def f(x):
            if False:
                i = 10
                return i + 15
            y = x.view(8)
            indices = torch.arange(4)
            values = torch.arange(4, dtype=y.dtype)
            y.index_put_((indices,), values, accumulate=False)
            return y
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    view_copy = torch.ops.aten.view_copy.default(arg0_1, [8])\n    arange = torch.ops.aten.arange.default(4, device = device(type='cpu'), pin_memory = False)\n    arange_1 = torch.ops.aten.arange.default(4, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)\n    index_put = torch.ops.aten.index_put.default(view_copy, [arange], arange_1);  view_copy = arange = arange_1 = None\n    view_copy_1 = torch.ops.aten.view_copy.default(index_put, [4, 2]);  index_put = None\n    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [8])\n    copy_ = torch.ops.aten.copy_.default(arg0_1, view_copy_1);  arg0_1 = view_copy_1 = None\n    return view_copy_2\n    ")

    def test_scalars(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                i = 10
                return i + 15
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            y.add_(1)
            z = 2 * y
            z.div_(1)
            return z
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)\n    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2])\n    add = torch.ops.aten.add.Tensor(view_copy, 1);  view_copy = None\n    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2]);  add = None\n    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [4, 2])\n    mul = torch.ops.aten.mul.Tensor(view_copy_2, 2);  view_copy_2 = None\n    div = torch.ops.aten.div.Tensor(mul, 1);  mul = None\n    copy_ = torch.ops.aten.copy_.default(arg0_1, view_copy_1);  arg0_1 = view_copy_1 = None\n    return div\n    ")

    @skipIfTorchDynamo('Test does not work with TorchDynamo')
    def test_metadata_change(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                while True:
                    i = 10
            y = x.clone()
            out = y.ge_(0)
            return out
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, '\n\n\ndef forward(self, arg0_1):\n    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None\n    ge = torch.ops.aten.ge.Scalar(clone, 0);  clone = None\n    _to_copy = torch.ops.aten._to_copy.default(ge, dtype = torch.float32, layout = torch.strided);  ge = None\n    return _to_copy\n    ')
        reinplaced_logs = self.get_logs(f, torch.ones(2, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, '\n\n\ndef forward(self, arg0_1):\n    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None\n    ge = torch.ops.aten.ge.Scalar(clone, 0);  clone = None\n    _to_copy = torch.ops.aten._to_copy.default(ge, dtype = torch.float32, layout = torch.strided);  ge = None\n    return _to_copy\n    ')

    @skipIfTorchDynamo('Test does not work with TorchDynamo')
    def test_metadata_change_out_op(self):
        if False:
            i = 10
            return i + 15

        def f(t, y):
            if False:
                print('Hello World!')
            out_1 = torch.ones(1)
            return torch.add(t, y, out=out_1)
        (inpt1, inpt2) = (torch.tensor([1]), torch.tensor([1]))
        (inpt1_func, inpt2_func) = (torch._to_functional_tensor(inpt1), torch._to_functional_tensor(inpt2))
        out_ref = f(inpt1, inpt2)
        torch._enable_functionalization(reapply_views=True)
        try:
            out_functional = f(inpt1_func, inpt2_func)
        finally:
            torch._disable_functionalization()
        self.assertEqual(out_ref, torch._from_functional_tensor(out_functional))

    def test_only_one_view(self):
        if False:
            return 10

        def f(x):
            if False:
                while True:
                    i = 10
            return x.view(4, 2)
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, '\n\n\ndef forward(self, arg0_1):\n    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2]);  arg0_1 = None\n    return view_copy\n    ')

    def test_everything(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                i = 10
                return i + 15
            tmp = torch.ones(2, 2)
            x2 = x + x
            y = x2.view(8)
            z0 = y.reshape(2, 4)
            z1 = z0.transpose(1, 0)
            z1.unsqueeze_(0)
            z1.squeeze_()
            (z2, z3) = z1.split(2)
            z2.add_(tmp)
            z4 = z0[0] + z2.reshape(4)
            return z2
        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    ones = torch.ops.aten.ones.default([2, 2], device = device(type='cpu'), pin_memory = False)\n    add = torch.ops.aten.add.Tensor(arg0_1, arg0_1);  arg0_1 = None\n    view_copy = torch.ops.aten.view_copy.default(add, [8])\n    view_copy_1 = torch.ops.aten.view_copy.default(view_copy, [2, 4]);  view_copy = None\n    transpose_copy = torch.ops.aten.transpose_copy.int(view_copy_1, 1, 0)\n    unsqueeze_copy = torch.ops.aten.unsqueeze_copy.default(transpose_copy, 0);  transpose_copy = None\n    squeeze_copy = torch.ops.aten.squeeze_copy.default(unsqueeze_copy);  unsqueeze_copy = None\n    split_copy = torch.ops.aten.split_copy.Tensor(squeeze_copy, 2);  squeeze_copy = None\n    getitem = split_copy[0]\n    getitem_1 = split_copy[1];  split_copy = None\n    add_1 = torch.ops.aten.add.Tensor(getitem, ones);  getitem = ones = None\n    view_copy_2 = torch.ops.aten.view_copy.default(add, [8]);  add = None\n    view_copy_3 = torch.ops.aten.view_copy.default(view_copy_2, [2, 4]);  view_copy_2 = None\n    transpose_copy_1 = torch.ops.aten.transpose_copy.int(view_copy_3, 1, 0);  view_copy_3 = None\n    unsqueeze_copy_1 = torch.ops.aten.unsqueeze_copy.default(transpose_copy_1, 0);  transpose_copy_1 = None\n    squeeze_copy_1 = torch.ops.aten.squeeze_copy.default(unsqueeze_copy_1);  unsqueeze_copy_1 = None\n    slice_scatter = torch.ops.aten.slice_scatter.default(squeeze_copy_1, add_1, 0, 0, 2);  squeeze_copy_1 = add_1 = None\n    unsqueeze_copy_2 = torch.ops.aten.unsqueeze_copy.default(slice_scatter, 0);  slice_scatter = None\n    squeeze_copy_2 = torch.ops.aten.squeeze_copy.dim(unsqueeze_copy_2, 0);  unsqueeze_copy_2 = None\n    transpose_copy_2 = torch.ops.aten.transpose_copy.int(squeeze_copy_2, 1, 0);  squeeze_copy_2 = None\n    view_copy_4 = torch.ops.aten.view_copy.default(transpose_copy_2, [8]);  transpose_copy_2 = None\n    view_copy_5 = torch.ops.aten.view_copy.default(view_copy_4, [4, 2]);  view_copy_4 = None\n    view_copy_6 = torch.ops.aten.view_copy.default(view_copy_5, [8])\n    view_copy_7 = torch.ops.aten.view_copy.default(view_copy_6, [2, 4]);  view_copy_6 = None\n    transpose_copy_3 = torch.ops.aten.transpose_copy.int(view_copy_7, 1, 0);  view_copy_7 = None\n    unsqueeze_copy_3 = torch.ops.aten.unsqueeze_copy.default(transpose_copy_3, 0);  transpose_copy_3 = None\n    squeeze_copy_3 = torch.ops.aten.squeeze_copy.default(unsqueeze_copy_3);  unsqueeze_copy_3 = None\n    split_copy_1 = torch.ops.aten.split_copy.Tensor(squeeze_copy_3, 2);  squeeze_copy_3 = None\n    getitem_2 = split_copy_1[0]\n    getitem_3 = split_copy_1[1];  split_copy_1 = None\n    select_copy = torch.ops.aten.select_copy.int(view_copy_1, 0, 0);  view_copy_1 = None\n    view_copy_8 = torch.ops.aten.view_copy.default(getitem_2, [4])\n    view_copy_9 = torch.ops.aten.view_copy.default(view_copy_5, [8])\n    view_copy_10 = torch.ops.aten.view_copy.default(view_copy_9, [2, 4]);  view_copy_9 = None\n    select_copy_1 = torch.ops.aten.select_copy.int(view_copy_10, 0, 0);  view_copy_10 = None\n    view_copy_11 = torch.ops.aten.view_copy.default(view_copy_5, [8]);  view_copy_5 = None\n    view_copy_12 = torch.ops.aten.view_copy.default(view_copy_11, [2, 4]);  view_copy_11 = None\n    transpose_copy_4 = torch.ops.aten.transpose_copy.int(view_copy_12, 1, 0);  view_copy_12 = None\n    unsqueeze_copy_4 = torch.ops.aten.unsqueeze_copy.default(transpose_copy_4, 0);  transpose_copy_4 = None\n    squeeze_copy_4 = torch.ops.aten.squeeze_copy.default(unsqueeze_copy_4);  unsqueeze_copy_4 = None\n    split_copy_2 = torch.ops.aten.split_copy.Tensor(squeeze_copy_4, 2);  squeeze_copy_4 = None\n    getitem_4 = split_copy_2[0]\n    getitem_5 = split_copy_2[1];  split_copy_2 = None\n    view_copy_13 = torch.ops.aten.view_copy.default(getitem_4, [4]);  getitem_4 = None\n    add_2 = torch.ops.aten.add.Tensor(select_copy_1, view_copy_13);  select_copy_1 = view_copy_13 = None\n    return getitem_2\n    ")
        reinplaced_logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, "\n\n\ndef forward(self, arg0_1):\n    ones = torch.ops.aten.ones.default([2, 2], device = device(type='cpu'), pin_memory = False)\n    add = torch.ops.aten.add.Tensor(arg0_1, arg0_1);  arg0_1 = None\n    view = torch.ops.aten.view.default(add, [8])\n    view_1 = torch.ops.aten.view.default(view, [2, 4]);  view = None\n    transpose = torch.ops.aten.transpose.int(view_1, 1, 0)\n    unsqueeze = torch.ops.aten.unsqueeze.default(transpose, 0);  transpose = None\n    squeeze = torch.ops.aten.squeeze.default(unsqueeze);  unsqueeze = None\n    split = torch.ops.aten.split.Tensor(squeeze, 2);  squeeze = None\n    getitem = split[0]\n    getitem_1 = split[1];  split = None\n    add_1 = torch.ops.aten.add_.Tensor(getitem, ones);  getitem = ones = None\n    view_2 = torch.ops.aten.view.default(add, [8]);  add = None\n    view_3 = torch.ops.aten.view.default(view_2, [2, 4]);  view_2 = None\n    transpose_1 = torch.ops.aten.transpose.int(view_3, 1, 0);  view_3 = None\n    unsqueeze_1 = torch.ops.aten.unsqueeze.default(transpose_1, 0);  transpose_1 = None\n    squeeze_1 = torch.ops.aten.squeeze.default(unsqueeze_1);  unsqueeze_1 = None\n    unsqueeze_2 = torch.ops.aten.unsqueeze.default(squeeze_1, 0);  squeeze_1 = None\n    squeeze_2 = torch.ops.aten.squeeze.dim(unsqueeze_2, 0);  unsqueeze_2 = None\n    transpose_2 = torch.ops.aten.transpose.int(squeeze_2, 1, 0);  squeeze_2 = None\n    view_4 = torch.ops.aten.view.default(transpose_2, [8]);  transpose_2 = None\n    view_5 = torch.ops.aten.view.default(view_4, [4, 2]);  view_4 = None\n    view_6 = torch.ops.aten.view.default(view_5, [8])\n    view_7 = torch.ops.aten.view.default(view_6, [2, 4]);  view_6 = None\n    transpose_3 = torch.ops.aten.transpose.int(view_7, 1, 0);  view_7 = None\n    unsqueeze_3 = torch.ops.aten.unsqueeze.default(transpose_3, 0);  transpose_3 = None\n    squeeze_3 = torch.ops.aten.squeeze.default(unsqueeze_3);  unsqueeze_3 = None\n    split_1 = torch.ops.aten.split.Tensor(squeeze_3, 2);  squeeze_3 = None\n    getitem_2 = split_1[0]\n    getitem_3 = split_1[1];  split_1 = None\n    select = torch.ops.aten.select.int(view_1, 0, 0);  view_1 = None\n    clone = torch.ops.aten.clone.default(getitem_2, memory_format = torch.contiguous_format)\n    _unsafe_view = torch.ops.aten._unsafe_view.default(clone, [4]);  clone = None\n    view_8 = torch.ops.aten.view.default(view_5, [8]);  view_5 = None\n    view_9 = torch.ops.aten.view.default(view_8, [2, 4]);  view_8 = None\n    select_1 = torch.ops.aten.select.int(view_9, 0, 0);  view_9 = None\n    add_2 = torch.ops.aten.add.Tensor(select_1, _unsafe_view);  select_1 = _unsafe_view = None\n    return getitem_2\n    ")

    def test_reapply_views_simple(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            y.add_(tmp)
            z = x * x
            return y
        self.assert_functionalization(f, torch.ones(4, 2), reapply_views=True)
        logs = self.get_logs(f, torch.ones(4, 2), reapply_views=True)
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)\n    view = torch.ops.aten.view.default(arg0_1, [4, 2])\n    add = torch.ops.aten.add.Tensor(view, ones);  view = ones = None\n    view_1 = torch.ops.aten.view.default(add, [4, 2]);  add = None\n    view_2 = torch.ops.aten.view.default(view_1, [4, 2])\n    mul = torch.ops.aten.mul.Tensor(view_1, view_1)\n    copy_ = torch.ops.aten.copy_.default(arg0_1, view_1);  arg0_1 = view_1 = None\n    return view_2\n    ")

    def test_aliases_maintained_after_pass_when_reapplying_views(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                print('Hello World!')
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            z = x.view(4, 2)
            y.add_(tmp)
            return (y, z)
        input_functional = torch._to_functional_tensor(torch.ones(4, 2))
        torch._enable_functionalization(reapply_views=True)
        try:
            (y, z) = f(input_functional)
            torch._sync(y)
            torch._sync(z)
        finally:
            torch._disable_functionalization()
        _y = torch._from_functional_tensor(y)
        _z = torch._from_functional_tensor(z)
        self.assertTrue(are_aliased(_y, _z))

    def test_copy_(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                return 10
            tmp = torch.zeros(2, 2)
            tmp_slice = tmp.diagonal()
            y = tmp_slice.copy_(x)
            z = y.add_(x)
            return z
        logs = self.get_logs(f, torch.ones(2))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)\n    diagonal_copy = torch.ops.aten.diagonal_copy.default(zeros)\n    copy = torch.ops.aten.copy.default(diagonal_copy, arg0_1);  diagonal_copy = None\n    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(zeros, copy);  zeros = copy = None\n    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter)\n    add = torch.ops.aten.add.Tensor(diagonal_copy_1, arg0_1);  diagonal_copy_1 = arg0_1 = None\n    diagonal_scatter_1 = torch.ops.aten.diagonal_scatter.default(diagonal_scatter, add);  diagonal_scatter = add = None\n    diagonal_copy_2 = torch.ops.aten.diagonal_copy.default(diagonal_scatter_1);  diagonal_scatter_1 = None\n    return diagonal_copy_2\n    ")
        reinplaced_logs = self.get_logs(f, torch.ones(2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, "\n\n\ndef forward(self, arg0_1):\n    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)\n    diagonal = torch.ops.aten.diagonal.default(zeros)\n    copy = torch.ops.aten.copy_.default(diagonal, arg0_1);  diagonal = None\n    diagonal_1 = torch.ops.aten.diagonal.default(zeros)\n    add = torch.ops.aten.add_.Tensor(diagonal_1, arg0_1);  diagonal_1 = arg0_1 = None\n    diagonal_2 = torch.ops.aten.diagonal.default(zeros);  zeros = None\n    return diagonal_2\n    ")
        self.assert_functionalization(f, torch.ones(1))
        logs = self.get_logs(f, torch.ones(1))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)\n    diagonal_copy = torch.ops.aten.diagonal_copy.default(zeros)\n    copy = torch.ops.aten.copy.default(diagonal_copy, arg0_1);  diagonal_copy = None\n    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(zeros, copy);  zeros = copy = None\n    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter)\n    add = torch.ops.aten.add.Tensor(diagonal_copy_1, arg0_1);  diagonal_copy_1 = arg0_1 = None\n    diagonal_scatter_1 = torch.ops.aten.diagonal_scatter.default(diagonal_scatter, add);  diagonal_scatter = add = None\n    diagonal_copy_2 = torch.ops.aten.diagonal_copy.default(diagonal_scatter_1);  diagonal_scatter_1 = None\n    return diagonal_copy_2\n    ")
        reinplaced_logs = self.get_logs(f, torch.ones(1), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, "\n\n\ndef forward(self, arg0_1):\n    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)\n    diagonal = torch.ops.aten.diagonal.default(zeros)\n    copy = torch.ops.aten.copy_.default(diagonal, arg0_1);  diagonal = None\n    diagonal_1 = torch.ops.aten.diagonal.default(zeros)\n    add = torch.ops.aten.add_.Tensor(diagonal_1, arg0_1);  diagonal_1 = arg0_1 = None\n    diagonal_2 = torch.ops.aten.diagonal.default(zeros);  zeros = None\n    return diagonal_2\n    ")
        self.assert_functionalization(f, torch.ones(2, dtype=torch.long))
        logs = self.get_logs(f, torch.ones(2, dtype=torch.long))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)\n    diagonal_copy = torch.ops.aten.diagonal_copy.default(zeros)\n    copy = torch.ops.aten.copy.default(diagonal_copy, arg0_1);  diagonal_copy = None\n    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(zeros, copy);  zeros = copy = None\n    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter)\n    add = torch.ops.aten.add.Tensor(diagonal_copy_1, arg0_1);  diagonal_copy_1 = arg0_1 = None\n    diagonal_scatter_1 = torch.ops.aten.diagonal_scatter.default(diagonal_scatter, add);  diagonal_scatter = add = None\n    diagonal_copy_2 = torch.ops.aten.diagonal_copy.default(diagonal_scatter_1);  diagonal_scatter_1 = None\n    return diagonal_copy_2\n    ")
        reinplaced_logs = self.get_logs(f, torch.ones(2, dtype=torch.long), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, "\n\n\ndef forward(self, arg0_1):\n    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)\n    diagonal = torch.ops.aten.diagonal.default(zeros)\n    copy = torch.ops.aten.copy_.default(diagonal, arg0_1);  diagonal = None\n    diagonal_1 = torch.ops.aten.diagonal.default(zeros)\n    add = torch.ops.aten.add_.Tensor(diagonal_1, arg0_1);  diagonal_1 = arg0_1 = None\n    diagonal_2 = torch.ops.aten.diagonal.default(zeros);  zeros = None\n    return diagonal_2\n    ")
        self.assert_functionalization(f, torch.ones(1, dtype=torch.long))
        logs = self.get_logs(f, torch.ones(1, dtype=torch.long))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)\n    diagonal_copy = torch.ops.aten.diagonal_copy.default(zeros)\n    copy = torch.ops.aten.copy.default(diagonal_copy, arg0_1);  diagonal_copy = None\n    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(zeros, copy);  zeros = copy = None\n    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter)\n    add = torch.ops.aten.add.Tensor(diagonal_copy_1, arg0_1);  diagonal_copy_1 = arg0_1 = None\n    diagonal_scatter_1 = torch.ops.aten.diagonal_scatter.default(diagonal_scatter, add);  diagonal_scatter = add = None\n    diagonal_copy_2 = torch.ops.aten.diagonal_copy.default(diagonal_scatter_1);  diagonal_scatter_1 = None\n    return diagonal_copy_2\n    ")
        reinplaced_logs = self.get_logs(f, torch.ones(1, dtype=torch.long), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, "\n\n\ndef forward(self, arg0_1):\n    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)\n    diagonal = torch.ops.aten.diagonal.default(zeros)\n    copy = torch.ops.aten.copy_.default(diagonal, arg0_1);  diagonal = None\n    diagonal_1 = torch.ops.aten.diagonal.default(zeros)\n    add = torch.ops.aten.add_.Tensor(diagonal_1, arg0_1);  diagonal_1 = arg0_1 = None\n    diagonal_2 = torch.ops.aten.diagonal.default(zeros);  zeros = None\n    return diagonal_2\n    ")

    def test_expand_symint(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.expand(x.size(0), x.size(1))
        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(logs, '\n\n\ndef forward(self, arg0_1):\n    expand_copy = torch.ops.aten.expand_copy.default(arg0_1, [2, 2]);  arg0_1 = None\n    return expand_copy\n    ')

    def test_fill_(self):
        if False:
            return 10

        def f(x):
            if False:
                return 10
            y = x + x
            z = y.diagonal()
            z.fill_(0)
            return y
        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(logs, '\n\n\ndef forward(self, arg0_1):\n    add = torch.ops.aten.add.Tensor(arg0_1, arg0_1);  arg0_1 = None\n    diagonal_copy = torch.ops.aten.diagonal_copy.default(add)\n    fill = torch.ops.aten.fill.Scalar(diagonal_copy, 0);  diagonal_copy = None\n    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(add, fill);  add = fill = None\n    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter)\n    return diagonal_scatter\n    ')
        reinplaced_logs = self.get_logs(f, torch.ones(2, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, '\n\n\ndef forward(self, arg0_1):\n    add = torch.ops.aten.add.Tensor(arg0_1, arg0_1);  arg0_1 = None\n    diagonal = torch.ops.aten.diagonal.default(add)\n    fill = torch.ops.aten.fill_.Scalar(diagonal, 0);  diagonal = None\n    diagonal_1 = torch.ops.aten.diagonal.default(add)\n    return add\n    ')

    def test_resize_smaller(self):
        if False:
            return 10

        def f(w):
            if False:
                return 10
            x = w + 1
            y = x.view(4, 4)
            y.resize_(3, 3)
            y2 = y.view(-1)
            y2.add_(1)
            z = y + 1
            return z
        self.assert_functionalization(f, torch.ones(8, 2))
        logs = self.get_logs(f, torch.ones(8, 2))
        self.assertExpectedInline(logs, '\n\n\ndef forward(self, arg0_1):\n    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None\n    view_copy = torch.ops.aten.view_copy.default(add, [4, 4])\n    resize = torch.ops.aten.resize.default(view_copy, [3, 3])\n    as_strided_copy = torch.ops.aten.as_strided_copy.default(view_copy, [3, 3], [3, 1]);  view_copy = None\n    view_copy_1 = torch.ops.aten.view_copy.default(as_strided_copy, [-1]);  as_strided_copy = None\n    add_1 = torch.ops.aten.add.Tensor(view_copy_1, 1);  view_copy_1 = None\n    view_copy_2 = torch.ops.aten.view_copy.default(add, [4, 4]);  add = None\n    as_strided_copy_1 = torch.ops.aten.as_strided_copy.default(view_copy_2, [3, 3], [3, 1])\n    view_copy_3 = torch.ops.aten.view_copy.default(add_1, [3, 3]);  add_1 = None\n    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(view_copy_2, view_copy_3, [3, 3], [3, 1]);  view_copy_2 = view_copy_3 = None\n    view_copy_4 = torch.ops.aten.view_copy.default(as_strided_scatter, [8, 2]);  as_strided_scatter = None\n    view_copy_5 = torch.ops.aten.view_copy.default(view_copy_4, [4, 4])\n    as_strided_copy_2 = torch.ops.aten.as_strided_copy.default(view_copy_5, [3, 3], [3, 1]);  view_copy_5 = None\n    view_copy_6 = torch.ops.aten.view_copy.default(as_strided_copy_2, [-1]);  as_strided_copy_2 = None\n    view_copy_7 = torch.ops.aten.view_copy.default(view_copy_4, [4, 4]);  view_copy_4 = None\n    as_strided_copy_3 = torch.ops.aten.as_strided_copy.default(view_copy_7, [3, 3], [3, 1]);  view_copy_7 = None\n    add_2 = torch.ops.aten.add.Tensor(as_strided_copy_3, 1);  as_strided_copy_3 = None\n    return add_2\n    ')
        reinplaced_logs = self.get_logs(f, torch.ones(8, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, '\n\n\ndef forward(self, arg0_1):\n    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None\n    view = torch.ops.aten.view.default(add, [4, 4])\n    resize = torch.ops.aten.resize.default(view, [3, 3])\n    as_strided = torch.ops.aten.as_strided.default(view, [3, 3], [3, 1]);  view = None\n    view_1 = torch.ops.aten.view.default(as_strided, [-1]);  as_strided = None\n    add_1 = torch.ops.aten.add_.Tensor(view_1, 1)\n    view_2 = torch.ops.aten.view.default(add, [4, 4]);  add = None\n    as_strided_1 = torch.ops.aten.as_strided.default(view_2, [3, 3], [3, 1])\n    view_3 = torch.ops.aten.view.default(view_1, [3, 3]);  view_1 = None\n    view_4 = torch.ops.aten.view.default(view_2, [8, 2]);  view_2 = None\n    view_5 = torch.ops.aten.view.default(view_4, [4, 4])\n    as_strided_2 = torch.ops.aten.as_strided.default(view_5, [3, 3], [3, 1]);  view_5 = None\n    view_6 = torch.ops.aten.view.default(as_strided_2, [-1]);  as_strided_2 = None\n    view_7 = torch.ops.aten.view.default(view_4, [4, 4]);  view_4 = None\n    as_strided_3 = torch.ops.aten.as_strided.default(view_7, [3, 3], [3, 1]);  view_7 = None\n    add_2 = torch.ops.aten.add_.Tensor(as_strided_3, 1)\n    return as_strided_3\n    ')

    def test_resize_same_size_diff_rank(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                return 10
            y = x.clone()
            y.resize_(25, 5)
            return y
        self.assert_functionalization(f, torch.ones(5, 5, 5))

    def test_resize_larger_valid(self):
        if False:
            return 10

        def f(x):
            if False:
                while True:
                    i = 10
            y = x + 1
            y.resize_(5, 5)
            y2 = y.view(25)
            y2.fill_(1)
            out = y + 1
            return (y, out)
        self.assert_functionalization(f, torch.ones(8, 2))
        logs = self.get_logs(f, torch.ones(8, 2))
        self.assertExpectedInline(logs, '\n\n\ndef forward(self, arg0_1):\n    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None\n    resize = torch.ops.aten.resize.default(add, [5, 5]);  add = None\n    view_copy = torch.ops.aten.view_copy.default(resize, [25]);  resize = None\n    fill = torch.ops.aten.fill.Scalar(view_copy, 1);  view_copy = None\n    view_copy_1 = torch.ops.aten.view_copy.default(fill, [5, 5]);  fill = None\n    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [25])\n    add_1 = torch.ops.aten.add.Tensor(view_copy_1, 1)\n    return (view_copy_1, add_1)\n    ')
        reinplaced_logs = self.get_logs(f, torch.ones(8, 2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, '\n\n\ndef forward(self, arg0_1):\n    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None\n    resize = torch.ops.aten.resize_.default(add, [5, 5])\n    view = torch.ops.aten.view.default(add, [25]);  add = None\n    fill = torch.ops.aten.fill_.Scalar(view, 1)\n    view_1 = torch.ops.aten.view.default(view, [5, 5]);  view = None\n    view_2 = torch.ops.aten.view.default(view_1, [25])\n    add_1 = torch.ops.aten.add.Tensor(view_1, 1)\n    return (view_1, add_1)\n    ')

    def test_resize_larger_invalid(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                print('Hello World!')
            y = x + 1
            z = y.view(4, 4)
            z.resize_(5, 5)
            z2 = z.view(25)
            z2.fill_(1)
            out = z + 1
            return (y, out)
        with self.assertRaisesRegex(RuntimeError, 'Attempted to resize a view tensor to a larger size. This is not allowed in the functionalization pass'):
            self.assert_functionalization(f, torch.ones(8, 2))

    def test_nested_functions_propagate_updates(self):
        if False:
            print('Hello World!')

        def g(x):
            if False:
                i = 10
                return i + 15
            y = x[0]
            y.add_(1)

        def f(x):
            if False:
                while True:
                    i = 10
            g(x)
            y = x + x
            return y
        self.assert_functionalization(f, torch.ones(2, 2))

    def test_mixed_wrappers_valid(self):
        if False:
            print('Hello World!')

        def f(x, y):
            if False:
                print('Hello World!')
            z = x + y
            z.add_(1)
            return z
        x1_not_functional = LoggingTensor(torch.ones(4))
        x2_functional = torch._to_functional_tensor(LoggingTensor(torch.ones(4)))
        with capture_logs() as logs:
            y = f(x1_not_functional, x2_functional)
        self.assertExpectedInline('\n'.join(logs), '$2: f32[4] = torch._ops.aten.add.Tensor($0, $1)\n$3: f32[4] = torch._ops.aten.add.Tensor($2, 1)')

    def test_mixed_wrappers_invalid(self):
        if False:
            i = 10
            return i + 15
        x1_not_functional = torch.ones(4)
        x2_functional = torch._to_functional_tensor(torch.ones(4))
        with self.assertRaises(RuntimeError):
            x1_not_functional.add_(x2_functional)

    def test_index_mutation_on_non_input(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                i = 10
                return i + 15
            tmp = torch.zeros(10)
            tmp[5].fill_(1)
            return tmp
        self.assert_functionalization(f, torch.ones(2))
        logs = self.get_logs(f, torch.ones(2))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1):\n    zeros = torch.ops.aten.zeros.default([10], device = device(type='cpu'), pin_memory = False)\n    select_copy = torch.ops.aten.select_copy.int(zeros, 0, 5)\n    fill = torch.ops.aten.fill.Scalar(select_copy, 1);  select_copy = None\n    select_scatter = torch.ops.aten.select_scatter.default(zeros, fill, 0, 5);  zeros = fill = None\n    select_copy_1 = torch.ops.aten.select_copy.int(select_scatter, 0, 5)\n    return select_scatter\n    ")
        reinplaced_logs = self.get_logs(f, torch.ones(2), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, "\n\n\ndef forward(self, arg0_1):\n    zeros = torch.ops.aten.zeros.default([10], device = device(type='cpu'), pin_memory = False)\n    select = torch.ops.aten.select.int(zeros, 0, 5)\n    fill = torch.ops.aten.fill_.Scalar(select, 1);  select = None\n    select_1 = torch.ops.aten.select.int(zeros, 0, 5)\n    return zeros\n    ")

    def test_instance_norm(self):
        if False:
            print('Hello World!')
        size = 100

        def f(x, running_mean, running_var):
            if False:
                for i in range(10):
                    print('nop')
            with enable_python_dispatcher():
                return torch.instance_norm(x, None, None, running_mean, running_var, use_input_stats=True, momentum=0.1, eps=1e-05, cudnn_enabled=False)
        self.assert_functionalization(f, torch.randn(20, size, 35, 45), torch.zeros(size), torch.ones(size))
        if not IS_WINDOWS:
            logs = self.get_logs(f, torch.randn(20, size, 35, 45), torch.zeros(size), torch.ones(size))
            self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1, arg1_1, arg2_1):\n    repeat = torch.ops.aten.repeat.default(arg1_1, [20])\n    repeat_1 = torch.ops.aten.repeat.default(arg2_1, [20])\n    view_copy = torch.ops.aten.view_copy.default(arg0_1, [1, 2000, 35, 45]);  arg0_1 = None\n    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))\n    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(view_copy, None, None, repeat, repeat_1, True, 0.1, 1e-05);  view_copy = repeat = repeat_1 = None\n    getitem = _native_batch_norm_legit_functional[0]\n    getitem_1 = _native_batch_norm_legit_functional[1]\n    getitem_2 = _native_batch_norm_legit_functional[2]\n    getitem_3 = _native_batch_norm_legit_functional[3]\n    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None\n    alias_copy = torch.ops.aten.alias_copy.default(arg1_1)\n    view_copy_1 = torch.ops.aten.view_copy.default(getitem_3, [20, 100])\n    view_copy_2 = torch.ops.aten.view_copy.default(getitem_3, [20, 100]);  getitem_3 = None\n    mean = torch.ops.aten.mean.dim(view_copy_2, [0]);  view_copy_2 = None\n    copy = torch.ops.aten.copy.default(alias_copy, mean);  alias_copy = mean = None\n    alias_copy_1 = torch.ops.aten.alias_copy.default(copy);  copy = None\n    alias_copy_2 = torch.ops.aten.alias_copy.default(alias_copy_1)\n    alias_copy_3 = torch.ops.aten.alias_copy.default(arg2_1)\n    view_copy_3 = torch.ops.aten.view_copy.default(getitem_4, [20, 100])\n    view_copy_4 = torch.ops.aten.view_copy.default(getitem_4, [20, 100]);  getitem_4 = None\n    mean_1 = torch.ops.aten.mean.dim(view_copy_4, [0]);  view_copy_4 = None\n    copy_1 = torch.ops.aten.copy.default(alias_copy_3, mean_1);  alias_copy_3 = mean_1 = None\n    alias_copy_4 = torch.ops.aten.alias_copy.default(copy_1);  copy_1 = None\n    alias_copy_5 = torch.ops.aten.alias_copy.default(alias_copy_4)\n    view_copy_5 = torch.ops.aten.view_copy.default(getitem, [20, 100, 35, 45]);  getitem = None\n    copy_ = torch.ops.aten.copy_.default(arg1_1, alias_copy_1);  arg1_1 = alias_copy_1 = None\n    copy__1 = torch.ops.aten.copy_.default(arg2_1, alias_copy_4);  arg2_1 = alias_copy_4 = None\n    return view_copy_5\n    ")
            reinplaced_logs = self.get_logs(f, torch.randn(20, size, 35, 45), torch.zeros(size), torch.ones(size), reapply_views=True, run_reinplace=True)
            self.assertExpectedInline(reinplaced_logs, "\n\n\ndef forward(self, arg0_1, arg1_1, arg2_1):\n    repeat = torch.ops.aten.repeat.default(arg1_1, [20])\n    repeat_1 = torch.ops.aten.repeat.default(arg2_1, [20])\n    view = torch.ops.aten.view.default(arg0_1, [1, 2000, 35, 45]);  arg0_1 = None\n    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))\n    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(view, None, None, repeat, repeat_1, True, 0.1, 1e-05);  view = repeat = repeat_1 = None\n    getitem = _native_batch_norm_legit_functional[0]\n    getitem_1 = _native_batch_norm_legit_functional[1]\n    getitem_2 = _native_batch_norm_legit_functional[2]\n    getitem_3 = _native_batch_norm_legit_functional[3]\n    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None\n    alias = torch.ops.aten.alias.default(arg1_1)\n    view_1 = torch.ops.aten.view.default(getitem_3, [20, 100])\n    view_2 = torch.ops.aten.view.default(getitem_3, [20, 100]);  getitem_3 = None\n    mean = torch.ops.aten.mean.dim(view_2, [0]);  view_2 = None\n    copy = torch.ops.aten.copy.default(alias, mean);  alias = mean = None\n    alias_1 = torch.ops.aten.alias.default(copy);  copy = None\n    alias_2 = torch.ops.aten.alias.default(alias_1)\n    alias_3 = torch.ops.aten.alias.default(arg2_1)\n    view_3 = torch.ops.aten.view.default(getitem_4, [20, 100])\n    view_4 = torch.ops.aten.view.default(getitem_4, [20, 100]);  getitem_4 = None\n    mean_1 = torch.ops.aten.mean.dim(view_4, [0]);  view_4 = None\n    copy_1 = torch.ops.aten.copy.default(alias_3, mean_1);  alias_3 = mean_1 = None\n    alias_4 = torch.ops.aten.alias.default(copy_1);  copy_1 = None\n    alias_5 = torch.ops.aten.alias.default(alias_4)\n    view_5 = torch.ops.aten.view.default(getitem, [20, 100, 35, 45]);  getitem = None\n    copy_ = torch.ops.aten.copy_.default(arg1_1, alias_1);  arg1_1 = alias_1 = None\n    copy__1 = torch.ops.aten.copy_.default(arg2_1, alias_4);  arg2_1 = alias_4 = None\n    return view_5\n    ")

    def test_mutation_overlapping_mem(self):
        if False:
            i = 10
            return i + 15

        def fn(x):
            if False:
                i = 10
                return i + 15
            t1 = torch.add(x, x)
            t2 = t1.unfold(1, 3, 2)
            t3 = t2.abs_()
            return t3
        with self.assertRaisesRegex(RuntimeError, 'encountered a tensor being mutated that has internal overlap'):
            x = torch.ones(1, 5)
            out = _functionalize(fn, reapply_views=True, crossref=False)(x)

    def test_batch_norm(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x, running_mean, running_var):
            if False:
                print('Hello World!')
            with enable_python_dispatcher():
                return torch.batch_norm(x, None, None, running_mean, running_var, True, 0.1, 1e-05, False)
        self.assert_functionalization(f, torch.randn(20, 100, 35, 45), torch.zeros(100), torch.ones(100))
        logs = self.get_logs(f, torch.randn(20, 100, 35, 45), torch.zeros(100), torch.ones(100))
        self.assertExpectedInline(logs, "\n\n\ndef forward(self, arg0_1, arg1_1, arg2_1):\n    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))\n    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(arg0_1, None, None, arg1_1, arg2_1, True, 0.1, 1e-05);  arg0_1 = None\n    getitem = _native_batch_norm_legit_functional[0]\n    getitem_1 = _native_batch_norm_legit_functional[1]\n    getitem_2 = _native_batch_norm_legit_functional[2]\n    getitem_3 = _native_batch_norm_legit_functional[3]\n    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None\n    copy_ = torch.ops.aten.copy_.default(arg1_1, getitem_3);  arg1_1 = getitem_3 = None\n    copy__1 = torch.ops.aten.copy_.default(arg2_1, getitem_4);  arg2_1 = getitem_4 = None\n    return getitem\n    ")
        reinplaced_logs = self.get_logs(f, torch.randn(20, 100, 35, 45), torch.zeros(100), torch.ones(100), reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(reinplaced_logs, "\n\n\ndef forward(self, arg0_1, arg1_1, arg2_1):\n    empty = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='cpu'))\n    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(arg0_1, None, None, arg1_1, arg2_1, True, 0.1, 1e-05);  arg0_1 = None\n    getitem = _native_batch_norm_legit_functional[0]\n    getitem_1 = _native_batch_norm_legit_functional[1]\n    getitem_2 = _native_batch_norm_legit_functional[2]\n    getitem_3 = _native_batch_norm_legit_functional[3]\n    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None\n    copy_ = torch.ops.aten.copy_.default(arg1_1, getitem_3);  arg1_1 = getitem_3 = None\n    copy__1 = torch.ops.aten.copy_.default(arg2_1, getitem_4);  arg2_1 = getitem_4 = None\n    return getitem\n    ")

    def test_python_functionalization(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            x_view = x.view(-1)
            x.mul_(2)
            return x_view + 1

        def f_functionalized(x):
            if False:
                for i in range(10):
                    print('nop')
            x_wrapped = FunctionalTensor.to_functional(x)
            maybe_disable = torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize))
            with maybe_disable, FunctionalTensorMode():
                out_wrapped = f(x_wrapped)
            out_unwrapped = out_wrapped.elem
            torch._sync(out_unwrapped)
            return torch._from_functional_tensor(out_unwrapped)
        x = torch.randn(2, requires_grad=True) + 1
        fx_g = make_fx(f_functionalized)(x)
        self.assertExpectedInline(fx_g.code.strip(), 'def forward(self, x_1):\n    view = torch.ops.aten.view.default(x_1, [-1])\n    mul = torch.ops.aten.mul.Tensor(x_1, 2);  x_1 = None\n    view_1 = torch.ops.aten.view.default(mul, [-1]);  mul = None\n    add = torch.ops.aten.add.Tensor(view_1, 1);  view_1 = None\n    return add')

    def test_python_functionalization_zero_tensor(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                return 10
            y = torch.ops.aten._efficientzerotensor([4])
            out = x + y
            out.mul_(2)
            return out
        x = torch.randn(4)
        out_ref = f(x)
        out_test = dispatch_functionalize(f)(x)
        out_test_cpp = _functionalize(f, reapply_views=True, crossref=False, skip_input_mutations=True)(x)
        self.assertEqual(out_ref, out_test)
        self.assertEqual(out_ref, out_test_cpp)
        fx_g = make_fx(dispatch_functionalize(f))(x)
        fx_g_cpp = make_fx(_functionalize(f, reapply_views=True, crossref=False, skip_input_mutations=True))(x)
        self.assertEqual(fx_g_cpp.code.strip(), fx_g.code.strip())

    def test_python_functionalization_is_conj(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                return 10
            out = x.conj()
            return (out, out.is_conj())
        x = torch.randn(4, dtype=torch.complex64)
        out_ref = f(x)
        out_test = dispatch_functionalize(f)(x)
        out_test_cpp = _functionalize(f, reapply_views=True, crossref=False)(x)
        self.assertEqual(out_ref[0], out_test[0])
        self.assertEqual(out_ref[1], out_test[1])
        self.assertEqual(out_ref[0], out_test_cpp[0])
        self.assertEqual(out_ref[1], out_test_cpp[1])

    def test_python_functionalization_is_neg(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                while True:
                    i = 10
            out = x.neg()
            return (out, out.is_neg())
        x = torch.randn(4, dtype=torch.complex64)
        out_ref = f(x)
        out_test = dispatch_functionalize(f)(x)
        out_test_cpp = _functionalize(f, reapply_views=True, crossref=False)(x)
        self.assertEqual(out_ref[0], out_test[0])
        self.assertEqual(out_ref[1], out_test[1])
        self.assertEqual(out_ref[0], out_test_cpp[0])
        self.assertEqual(out_ref[1], out_test_cpp[1])

    def test_python_functionalization_conj(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                print('Hello World!')
            y = x.clone().conj()
            y.mul_(2)
            return torch.view_as_real(y.resolve_conj())
        x = torch.randn(4, dtype=torch.complex64)
        out_ref = f(x)
        out_test = dispatch_functionalize(f)(x)
        out_test_cpp = _functionalize(f, reapply_views=True, crossref=False, skip_input_mutations=True)(x)
        self.assertEqual(out_ref, out_test)
        self.assertEqual(out_test, out_test_cpp)
        fx_g = make_fx(dispatch_functionalize(f))(x)
        fx_g_cpp = make_fx(_functionalize(f, reapply_views=True, crossref=False, skip_input_mutations=True))(x)
        self.assertExpectedInline(fx_g.code.strip(), 'def forward(self, arg0_1):\n    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None\n    _conj = torch.ops.aten._conj.default(clone);  clone = None\n    clone_1 = torch.ops.aten.clone.default(_conj)\n    mul = torch.ops.aten.mul.Tensor(clone_1, 2);  clone_1 = None\n    clone_2 = torch.ops.aten.clone.default(_conj);  _conj = None\n    copy = torch.ops.aten.copy.default(clone_2, mul);  clone_2 = mul = None\n    _conj_1 = torch.ops.aten._conj.default(copy);  copy = None\n    _conj_2 = torch.ops.aten._conj.default(_conj_1);  _conj_1 = None\n    clone_3 = torch.ops.aten.clone.default(_conj_2);  _conj_2 = None\n    view_as_real = torch.ops.aten.view_as_real.default(clone_3);  clone_3 = None\n    return view_as_real')
        self.assertEqual(fx_g_cpp.code.strip(), fx_g.code.strip())

    def test_python_functionalization_neg(self):
        if False:
            return 10

        def f(x):
            if False:
                return 10
            y = x._neg_view()
            z = y.resolve_neg()
            return z + 1
        x = torch.randn(4)
        out_ref = f(x)
        out_test = dispatch_functionalize(f)(x)
        out_test_cpp = _functionalize(f, reapply_views=True, crossref=False, skip_input_mutations=True)(x)
        self.assertEqual(out_ref, out_test)
        self.assertEqual(out_ref, out_test_cpp)
        fx_g = make_fx(dispatch_functionalize(f))(x)
        fx_g_cpp = make_fx(_functionalize(f, reapply_views=True, crossref=False, skip_input_mutations=True))(x)
        self.assertExpectedInline(fx_g.code.strip(), 'def forward(self, arg0_1):\n    _neg_view = torch.ops.aten._neg_view.default(arg0_1);  arg0_1 = None\n    clone = torch.ops.aten.clone.default(_neg_view);  _neg_view = None\n    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None\n    return add')
        self.assertEqual(fx_g_cpp.code.strip(), fx_g.code.strip())

    def test_python_functionalization_lift_fresh_storage(self):
        if False:
            for i in range(10):
                print('nop')
        unlifted = torch.tensor([0.0])
        maybe_disable = torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize))
        with maybe_disable, FunctionalTensorMode():
            lifted = torch.ops.aten.lift_fresh.default(unlifted)
        self.assertNotEqual(unlifted.untyped_storage(), lifted.untyped_storage())

    def test_python_functionalization_lift_fresh(self):
        if False:
            return 10

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            tmp = torch.tensor([0.0])
            return tmp + x
        x = torch.randn(4)
        out_ref = f(x)
        out_test = dispatch_functionalize(f)(x)
        out_test_cpp = _functionalize(f, reapply_views=True, crossref=False, skip_input_mutations=True)(x)
        self.assertEqual(out_ref, out_test)
        self.assertEqual(out_ref, out_test_cpp)
        fx_g = make_fx(dispatch_functionalize(f))(x)
        fx_g_cpp = make_fx(_functionalize(f, reapply_views=True, crossref=False, skip_input_mutations=True))(x)
        self.assertExpectedInline(fx_g.code.strip(), 'def forward(self, arg0_1):\n    _tensor_constant0 = self._tensor_constant0\n    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None\n    add = torch.ops.aten.add.Tensor(lift_fresh_copy, arg0_1);  lift_fresh_copy = arg0_1 = None\n    return add')
        self.assertEqual(fx_g_cpp.code.strip(), fx_g.code.strip())

@xfail_inherited_tests(['test_as_strided', 'test_copy_', 'test_diagonal', 'test_diagonal_mutated_input', 'test_everything', 'test_fill_', 'test_split', 'test_view_clone_view_inplace', 'test_view_inplace'])
@unittest.skipIf(TEST_WITH_TORCHDYNAMO, 'dynamo-ing code with proxy + fake doesnt work well')
class TestCrossRefFunctionalization(TestFunctionalization):
    crossref = True
if __name__ == '__main__':
    run_tests()