import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.fx.passes.reinplace import reinplace
from torch.fx.experimental.proxy_tensor import make_fx
try:
    from functorch.experimental import functionalize
    HAS_FUNCTIONALIZATION = True
except Exception as e:
    HAS_FUNCTIONALIZATION = False

class TestReinplacePass(TestCase):

    def test_reinplace_basic(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                return 10
            a = x.clone()
            b = a.add(1)
            return b
        inpt = torch.ones(2)
        f2 = reinplace(make_fx(f)(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, '\n\n\ndef forward(self, x_1):\n    clone = torch.ops.aten.clone.default(x_1);  x_1 = None\n    add = torch.ops.aten.add_.Tensor(clone, 1)\n    return clone\n    ')

    def test_reinplace_with_view(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                i = 10
                return i + 15
            a = x.clone()
            a_view = a.view(-1)
            b = a.add(1)
            c = a_view.add(1)
            return c
        inpt = torch.ones(2)
        f2 = reinplace(make_fx(f)(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, '\n\n\ndef forward(self, x_1):\n    clone = torch.ops.aten.clone.default(x_1);  x_1 = None\n    view = torch.ops.aten.view.default(clone, [-1])\n    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None\n    add_1 = torch.ops.aten.add_.Tensor(view, 1)\n    return view\n    ')

    def test_reinplace_different_metadata(self):
        if False:
            return 10

        def f(a_):
            if False:
                for i in range(10):
                    print('nop')
            a = a_.clone()
            b = a + 1
            c = torch.ge(b, a)
            return c
        inpt = torch.ones(4)
        f2 = reinplace(make_fx(f)(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, '\n\n\ndef forward(self, a__1):\n    clone = torch.ops.aten.clone.default(a__1);  a__1 = None\n    add = torch.ops.aten.add.Tensor(clone, 1)\n    ge = torch.ops.aten.ge.Tensor(add, clone);  add = clone = None\n    return ge\n    ')

    def test_reinplace_overlapping_memory(self):
        if False:
            print('Hello World!')

        def f(a_):
            if False:
                i = 10
                return i + 15
            a = a_.clone()
            b = a.expand(4, 4)
            c = b.add(1)
            return c
        inpt = torch.ones(1)
        f2 = reinplace(make_fx(f)(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, '\n\n\ndef forward(self, a__1):\n    clone = torch.ops.aten.clone.default(a__1);  a__1 = None\n    expand = torch.ops.aten.expand.default(clone, [4, 4]);  clone = None\n    add = torch.ops.aten.add.Tensor(expand, 1);  expand = None\n    return add\n    ')

    def test_reinplace_scatter_op(self):
        if False:
            return 10

        def f(a_):
            if False:
                for i in range(10):
                    print('nop')
            a = a_.clone()
            e = a.view(-1)
            b = a.view(-1)
            c = b[0]
            d = c.view(-1)
            d.add_(1)
            return a + e
        if not HAS_FUNCTIONALIZATION:
            return
        inpt = torch.ones(4)
        f2 = reinplace(make_fx(functionalize(f))(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, '\n\n\ndef forward(self, a__1):\n    clone = torch.ops.aten.clone.default(a__1);  a__1 = None\n    view = torch.ops.aten.view.default(clone, [-1])\n    view_1 = torch.ops.aten.view.default(clone, [-1])\n    select = torch.ops.aten.select.int(view_1, 0, 0);  view_1 = None\n    view_2 = torch.ops.aten.view.default(select, [-1]);  select = None\n    add = torch.ops.aten.add_.Tensor(view_2, 1)\n    view_3 = torch.ops.aten.view.default(clone, [-1]);  clone = None\n    select_1 = torch.ops.aten.select.int(view_3, 0, 0)\n    view_4 = torch.ops.aten.view.default(view_2, []);  view_2 = None\n    view_5 = torch.ops.aten.view.default(view_3, [4]);  view_3 = None\n    view_6 = torch.ops.aten.view.default(view_5, [-1])\n    select_2 = torch.ops.aten.select.int(view_6, 0, 0);  view_6 = None\n    view_7 = torch.ops.aten.view.default(select_2, [-1]);  select_2 = None\n    view_8 = torch.ops.aten.view.default(view_5, [-1])\n    add_1 = torch.ops.aten.add_.Tensor(view_5, view_8);  view_8 = None\n    return view_5\n    ')

    def test_reinplace_scatter_twice(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a_):
            if False:
                return 10
            a = a_.clone()
            b = a[:, 1]
            c = b[1]
            c.add_(1)
            return a
        if not HAS_FUNCTIONALIZATION:
            return
        inpt = torch.ones(4, 4)
        f2 = reinplace(make_fx(functionalize(f))(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, '\n\n\ndef forward(self, a__1):\n    clone = torch.ops.aten.clone.default(a__1);  a__1 = None\n    slice_1 = torch.ops.aten.slice.Tensor(clone, 0, 0, 9223372036854775807)\n    select = torch.ops.aten.select.int(slice_1, 1, 1);  slice_1 = None\n    select_1 = torch.ops.aten.select.int(select, 0, 1);  select = None\n    add = torch.ops.aten.add_.Tensor(select_1, 1);  select_1 = None\n    slice_2 = torch.ops.aten.slice.Tensor(clone, 0, 0, 9223372036854775807)\n    select_2 = torch.ops.aten.select.int(slice_2, 1, 1);  slice_2 = None\n    slice_3 = torch.ops.aten.slice.Tensor(clone, 0, 0, 9223372036854775807)\n    select_3 = torch.ops.aten.select.int(slice_3, 1, 1);  slice_3 = None\n    select_4 = torch.ops.aten.select.int(select_3, 0, 1);  select_3 = None\n    return clone\n    ')

    def test_reinplace_scatter_twice_with_different_view_op_valid(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a_):
            if False:
                while True:
                    i = 10
            a = a_.clone()
            b = a[:, 1]
            c = b[1]
            c_updated = c.add(1)
            good_mirror_of_b = a.as_strided((4,), (4,), 1)
            b_updated = torch.select_scatter(good_mirror_of_b, c_updated, 0, 1)
            return b_updated
        inpt = torch.ones(4, 4)
        f2 = reinplace(make_fx(f)(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, '\n\n\ndef forward(self, a__1):\n    clone = torch.ops.aten.clone.default(a__1);  a__1 = None\n    slice_1 = torch.ops.aten.slice.Tensor(clone, 0, 0, 9223372036854775807)\n    select = torch.ops.aten.select.int(slice_1, 1, 1);  slice_1 = None\n    select_1 = torch.ops.aten.select.int(select, 0, 1);  select = None\n    add = torch.ops.aten.add_.Tensor(select_1, 1);  select_1 = None\n    as_strided = torch.ops.aten.as_strided.default(clone, [4], [4], 1);  clone = None\n    return as_strided\n    ')

    def test_reinplace_scatter_twice_with_different_view_op_invalid(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a_):
            if False:
                i = 10
                return i + 15
            a = a_.clone()
            b = a[:, 1]
            c = b[1]
            c_updated = c.add(1)
            good_mirror_of_b = a.as_strided((4,), (4,), 1)
            b_updated = torch.select_scatter(good_mirror_of_b, c_updated, 0, 0)
            return b_updated
        inpt = torch.ones(4, 4)
        f2 = reinplace(make_fx(f)(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, '\n\n\ndef forward(self, a__1):\n    clone = torch.ops.aten.clone.default(a__1);  a__1 = None\n    slice_1 = torch.ops.aten.slice.Tensor(clone, 0, 0, 9223372036854775807)\n    select = torch.ops.aten.select.int(slice_1, 1, 1);  slice_1 = None\n    select_1 = torch.ops.aten.select.int(select, 0, 1);  select = None\n    add = torch.ops.aten.add.Tensor(select_1, 1);  select_1 = None\n    as_strided = torch.ops.aten.as_strided.default(clone, [4], [4], 1);  clone = None\n    select_int = torch.ops.aten.select.int(as_strided, 0, 0)\n    copy__default = torch.ops.aten.copy_.default(select_int, add);  select_int = add = None\n    return as_strided\n    ')

    def test_reinplace_scatter_twice_with_different_view_op_invalid2(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a_):
            if False:
                return 10
            a = a_.clone()
            b = a[:, 1]
            c = b[1]
            c_updated = c.add(1)
            bad_mirror_of_b = a.as_strided((4,), (4,), 0)
            b_updated = torch.select_scatter(bad_mirror_of_b, c_updated, 0, 1)
            return b_updated
        inpt = torch.ones(4, 4)
        f2 = reinplace(make_fx(f)(inpt), inpt)
        expected_out = f(inpt)
        actual_out = f2(inpt)
        self.assertExpectedInline(f2.code, '\n\n\ndef forward(self, a__1):\n    clone = torch.ops.aten.clone.default(a__1);  a__1 = None\n    slice_1 = torch.ops.aten.slice.Tensor(clone, 0, 0, 9223372036854775807)\n    select = torch.ops.aten.select.int(slice_1, 1, 1);  slice_1 = None\n    select_1 = torch.ops.aten.select.int(select, 0, 1);  select = None\n    add = torch.ops.aten.add.Tensor(select_1, 1);  select_1 = None\n    as_strided = torch.ops.aten.as_strided.default(clone, [4], [4], 0);  clone = None\n    select_int = torch.ops.aten.select.int(as_strided, 0, 1)\n    copy__default = torch.ops.aten.copy_.default(select_int, add);  select_int = add = None\n    return as_strided\n    ')

    def test_out_node_updated(self):
        if False:
            return 10

        def f():
            if False:
                print('Hello World!')
            x = torch.zeros(2, 2)
            y = x.diagonal()
            y_updated = y.add(1)
            z = torch.diagonal_scatter(x, y_updated)
            return [z]
        if not HAS_FUNCTIONALIZATION:
            return
        f2 = reinplace(make_fx(functionalize(f))())
        expected_out = f()
        actual_out = f2()
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, "\n\n\ndef forward(self):\n    zeros = torch.ops.aten.zeros.default([2, 2], device = device(type='cpu'), pin_memory = False)\n    diagonal = torch.ops.aten.diagonal.default(zeros)\n    add = torch.ops.aten.add_.Tensor(diagonal, 1);  diagonal = None\n    return [zeros]\n    ")

    def test_reinplace_index_mutation(self):
        if False:
            while True:
                i = 10

        def f():
            if False:
                return 10
            a = torch.zeros(4, 4, 4)
            a[:, 2:] = torch.ones(4, 2, 4)
            return a
        if not HAS_FUNCTIONALIZATION:
            return
        f2 = reinplace(make_fx(functionalize(f))())
        expected_out = f()
        actual_out = f2()
        self.assertEqual(actual_out, expected_out)
        self.assertExpectedInline(f2.code, "\n\n\ndef forward(self):\n    zeros = torch.ops.aten.zeros.default([4, 4, 4], device = device(type='cpu'), pin_memory = False)\n    ones = torch.ops.aten.ones.default([4, 2, 4], device = device(type='cpu'), pin_memory = False)\n    slice_1 = torch.ops.aten.slice.Tensor(zeros, 0, 0, 9223372036854775807)\n    slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 2, 9223372036854775807);  slice_1 = None\n    copy = torch.ops.aten.copy_.default(slice_2, ones);  slice_2 = ones = None\n    slice_3 = torch.ops.aten.slice.Tensor(zeros, 0, 0, 9223372036854775807)\n    slice_4 = torch.ops.aten.slice.Tensor(zeros, 0, 0, 9223372036854775807)\n    slice_5 = torch.ops.aten.slice.Tensor(slice_4, 1, 2, 9223372036854775807);  slice_4 = None\n    return zeros\n    ")
if __name__ == '__main__':
    run_tests()