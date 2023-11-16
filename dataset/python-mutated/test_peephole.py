import torch
from torch.testing._internal.jit_utils import JitTestCase, RUN_CUDA, _inline_everything
from torch import nn
from torch.testing import FileCheck
from typing import Callable, List
import unittest
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestPeephole(JitTestCase):

    def test_peephole_with_writes(self):
        if False:
            while True:
                i = 10

        def test_write(x):
            if False:
                while True:
                    i = 10
            s = 0
            s += x
            s += x
            return s
        self.checkScript(test_write, (torch.ones(4, 4),))

    def test_peephole_with_non_output_writes(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.ignore
        def nomnom(x):
            if False:
                i = 10
                return i + 15
            pass

        def test_write(x):
            if False:
                return 10
            t = torch.ones_like(x)
            z = x.clone()
            y = z + 0
            z.add_(t)
            nomnom(z)
            return y + y
        a = torch.ones(4, 4)
        j = self.checkScript(test_write, (a,))

    def test_peephole_no_output_aliasing(self):
        if False:
            i = 10
            return i + 15

        def test_peephole(x):
            if False:
                for i in range(10):
                    print('nop')
            y = x + 0
            return (x, y)
        a = torch.ones(4, 4)
        j = self.checkScript(test_peephole, (a,))
        (r1, r2) = j(a)
        self.assertNotEqual(r1.data_ptr(), r2.data_ptr())

    def test_peephole(self):
        if False:
            for i in range(10):
                print('nop')
        a = torch.tensor([0.4])
        b = torch.tensor([0.7])
        c = torch.tensor([0], dtype=torch.int32)

        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x.type_as(y)
        tf = torch.jit.trace(f, (a, b))
        FileCheck().check('type_as').run(str(tf.graph))
        self.run_pass('peephole', tf.graph)
        FileCheck().check_not('type_as').run(str(tf.graph))
        tf2 = torch.jit.trace(f, (a, c))
        s = str(tf2.graph)
        self.run_pass('peephole', tf2.graph)
        self.assertEqual(s, str(s))

    def test_peephole_dynamic(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                i = 10
                return i + 15
            return x.type_as(y)
        fn = torch.jit.script(f)
        s = str(fn.graph)
        torch._C._jit_pass_peephole(fn.graph)
        self.assertEqual(s, str(fn.graph))

    def test_peephole_list_ops(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo(x, y, z):
            if False:
                i = 10
                return i + 15
            return len([x, y, z])
        self.run_pass('peephole', foo.graph)
        FileCheck().check('value=3').check_next('return').run(foo.graph)

        @torch.jit.script
        def foo(x, y, z):
            if False:
                return 10
            li = [x, y, z]
            for i in range(len(x)):
                li.append(x)
            return len([x, y, z])
        self.run_pass('peephole', foo.graph)
        FileCheck().check_not('aten::len').run(foo.graph)

        @torch.jit.script
        def foo(x, y, z):
            if False:
                print('Hello World!')
            li = [x, y, z]
            return (li[1], li[-2])
        FileCheck().check('aten::__getitem__').run(foo.graph)
        self.run_pass('peephole', foo.graph)
        FileCheck().check_not('aten::__getitem__').run(foo.graph)

        @torch.jit.script
        def foo(x, y, z):
            if False:
                while True:
                    i = 10
            li = [x, y, z]
            return li[-7]
        self.run_pass('peephole', foo.graph)
        FileCheck().check('aten::__getitem__').run(foo.graph)

        @torch.jit.script
        def foo(x, y, z):
            if False:
                i = 10
                return i + 15
            li = [x, y, z]
            for i in range(len(x)):
                li.append(x)
            return li[-2]
        self.run_pass('peephole', foo.graph)
        FileCheck().check('aten::__getitem__').run(foo.graph)

    @unittest.skipIf(not RUN_CUDA, 'cpp tests require CUDA')
    def test_peephole_cuda(self):
        if False:
            while True:
                i = 10
        a = torch.tensor([0.4], device='cpu')
        b = torch.tensor([0.7], device='cuda')
        c = torch.tensor([0.7], device='cuda')

        def f(x, y):
            if False:
                i = 10
                return i + 15
            return x.type_as(y)
        trace = torch.jit.trace(f, (a, c))
        s = str(trace.graph)
        self.run_pass('peephole', trace.graph)
        self.assertEqual(s, str(trace.graph))
        trace = torch.jit.trace(f, (b, c))
        self.run_pass('peephole', trace.graph)
        self.run_pass('dce', trace.graph)
        FileCheck().check_not('type_as').run(str(trace.graph))

    @_inline_everything
    def test_peephole_type_refinements(self):
        if False:
            i = 10
            return i + 15

        def refine(x):
            if False:
                print('Hello World!')
            return x if x is not None else torch.tensor(3)

        @torch.jit.script
        def test():
            if False:
                print('Hello World!')
            return refine(torch.tensor(4))
        FileCheck().check('prim::unchecked_cast').run(test.graph)
        self.run_pass('peephole', test.graph)
        FileCheck().check_not('prim::unchecked_cast').run(test.graph)

        def is_int_tensor(x):
            if False:
                while True:
                    i = 10
            scalar = x.item()
            if isinstance(scalar, int):
                return scalar + 3
            else:
                return 8
        self.checkScript(is_int_tensor, (torch.tensor(2),))
        self.checkScript(is_int_tensor, (torch.tensor(2.5),))
        graph = torch.jit.script(is_int_tensor).graph
        self.run_pass('peephole', graph)
        FileCheck().check('prim::unchecked_cast').run(graph)

    def test_short_circuit_optimization(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def const_expressions(x):
            if False:
                for i in range(10):
                    print('nop')
            return (x == 1 and False, x == 1 or True)
        self.run_pass('constant_propagation', const_expressions.graph)
        FileCheck().check_not('prim::If').check_not('aten::eq').run(const_expressions.graph)
        self.assertEqual(const_expressions(1), (False, True))

        @torch.jit.script
        def redundant_expressions(x):
            if False:
                for i in range(10):
                    print('nop')
            return (x == 1 and True, x == 1 or False)
        self.run_pass('peephole', redundant_expressions.graph)
        self.assertEqual(redundant_expressions(1), (True, True))
        self.assertEqual(redundant_expressions(0), (False, False))
        FileCheck().check('aten::eq').check_not('prim::If').run(redundant_expressions.graph)

    def test_conv_dim_folding(self):
        if False:
            return 10
        modules = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        for mod in modules:

            class ConvDim(torch.nn.Module):

                def __init__(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    super().__init__()
                    self.conv = mod(3, 32, kernel_size=3, stride=2, bias=False)

                def forward(self, x):
                    if False:
                        i = 10
                        return i + 15
                    x = self.conv(x)
                    return x.dim()
            conv_dim = torch.jit.script(ConvDim())
            self.run_pass('inline', conv_dim.graph)
            self.run_pass('peephole', conv_dim.graph)
            FileCheck().check_not('conv').check_not('dim').run(conv_dim.graph)

            class ConvDimMutate(torch.nn.Module):

                def __init__(self):
                    if False:
                        i = 10
                        return i + 15
                    super().__init__()
                    self.conv = mod(3, 32, kernel_size=3, stride=2, bias=False)

                def forward(self, x):
                    if False:
                        print('Hello World!')
                    x = self.conv(x)
                    x.resize_([4, 4])
                    return x.dim()
            conv_dim = torch.jit.script(ConvDimMutate())
            self.run_pass('inline', conv_dim.graph)
            self.run_pass('peephole', conv_dim.graph)
            FileCheck().check('conv').check('dim').run(conv_dim.graph)

    def test_normalized_rsub(self):
        if False:
            i = 10
            return i + 15
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])

        def convertible_rsub(x, y):
            if False:
                while True:
                    i = 10
            return (x - y, torch.rsub(y, x))
        self.checkScript(convertible_rsub, (a, b))
        op_graph = torch.jit.script(convertible_rsub).graph
        FileCheck().check_count('aten::sub', 2, exactly=True).run(op_graph)
        FileCheck().check_count('aten::rsub', 0, exactly=True).run(op_graph)

    def test_normalized_is_op(self):
        if False:
            print('Hello World!')

        def convertible_is_op(x: bool, y: bool):
            if False:
                i = 10
                return i + 15
            return (x is True, False is x, x is y)
        self.checkScript(convertible_is_op, (True, False))
        op_graph = torch.jit.script(convertible_is_op).graph
        FileCheck().check_count('aten::eq', 3, exactly=True).run(op_graph)
        FileCheck().check_count('aten::__is__', 0, exactly=True).run(op_graph)

    def test_normalized_isnot_op(self):
        if False:
            return 10

        def convertible_isnot_op(x: bool, y: bool):
            if False:
                print('Hello World!')
            return (x is not True, False is not x, x is not y)
        self.checkScript(convertible_isnot_op, (True, False))
        op_graph = torch.jit.script(convertible_isnot_op).graph
        FileCheck().check_count('aten::ne', 3, exactly=True).run(op_graph)
        FileCheck().check_count('aten::__isnot__', 0, exactly=True).run(op_graph)

    def test_peephole_list_len(self):
        if False:
            return 10

        def run_peephole_and_check_const_value(graph, const_string):
            if False:
                for i in range(10):
                    print('nop')
            torch._C._jit_pass_peephole_list_idioms(graph, refine_list_len=True)
            self.run_pass('constant_propagation', graph)
            FileCheck().check(const_string).check_next('return').run(graph)

        def gen_li(inp_len: int):
            if False:
                i = 10
                return i + 15
            return [0 for i in range(inp_len)]

        @torch.jit.script
        def foo(x: List[int], y: List[int]):
            if False:
                while True:
                    i = 10
            if len(x) != 4 or len(y) != 5:
                raise Exception('')
            return len(x) + len(y)
        run_peephole_and_check_const_value(foo.graph, 'value=9')
        self.assertEqual(foo(gen_li(4), gen_li(5)), 9)
        with self.assertRaises(Exception):
            foo(2, 4)

        @torch.jit.script
        def foo(x: List[int], y: List[int]):
            if False:
                return 10
            if len(x) == 4 and len(y) == 5:
                pass
            else:
                raise Exception('hi')
            return len(x) + len(y)
        run_peephole_and_check_const_value(foo.graph, 'value=9')
        self.assertEqual(foo(gen_li(4), gen_li(5)), 9)
        with self.assertRaises(Exception):
            foo(2, 4)

        @torch.jit.script
        def foo(x: List[int], y: List[int], z: List[int]):
            if False:
                i = 10
                return i + 15
            if len(x) != 4:
                raise Exception('..')
            elif len(y) != 8:
                raise Exception('...')
            elif len(z) == 3:
                pass
            else:
                raise Exception('...')
            return len(x) + len(y) * len(z)
        run_peephole_and_check_const_value(foo.graph, 'value=28')
        self.assertEqual(foo(gen_li(4), gen_li(8), gen_li(3)), 28)
        with self.assertRaises(Exception):
            foo(1, 2, 3)

        @torch.jit.script
        def foo(x: List[int], cond: bool):
            if False:
                while True:
                    i = 10
            if len(x) == 4:
                if cond:
                    return len(x)
                return 4
            return 4
        run_peephole_and_check_const_value(foo.graph, 'value=4')

        def test_const_tuple_output(graph, const_inputs):
            if False:
                return 10
            tup = graph.findNode('prim::TupleConstruct')
            for (i, elem) in enumerate(tup.inputs()):
                if i in const_inputs:
                    self.assertIsNotNone(elem.toIValue())
                else:
                    self.assertIsNone(elem.toIValue())

        @torch.jit.script
        def foo(x: List[int], b: List[int]):
            if False:
                i = 10
                return i + 15
            if len(x) == 5:
                x1 = True
            else:
                x1 = len(b) != 4
            assert x1 == False
            return (len(x), len(b))
        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        torch._C._jit_pass_constant_propagation(foo.graph)
        test_const_tuple_output(foo.graph, [1])

        @torch.jit.script
        def foo(x: List[int], b: List[int]):
            if False:
                while True:
                    i = 10
            if len(x) == 5:
                x1 = False
            else:
                x1 = len(b) != 4
            assert x1 == False
            return (len(x), len(b))
        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        torch._C._jit_pass_constant_propagation(foo.graph)
        test_const_tuple_output(foo.graph, [])

        @torch.jit.script
        def foo(x: List[int], b: List[int]):
            if False:
                return 10
            if len(x) == 5:
                x1 = True
            else:
                x1 = len(b) == 4
            assert x1 == False
            return (len(x), len(b))
        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        torch._C._jit_pass_constant_propagation(foo.graph)
        test_const_tuple_output(foo.graph, [])

        @torch.jit.script
        def foo(x: List[int], b: List[int]):
            if False:
                print('Hello World!')
            if len(x) == 5:
                x1 = True
            else:
                x1 = len(b) != 4
            assert x1 == False
            return (len(x), len(b))
        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        torch._C._jit_pass_constant_propagation(foo.graph)
        test_const_tuple_output(foo.graph, [1])

        @torch.jit.script
        def foo(x: List[int], b: List[int]):
            if False:
                while True:
                    i = 10
            if len(x) != 5:
                x1 = len(b) != 4
            else:
                x1 = True
            assert x1 == False
            return (len(x), len(b))
        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        torch._C._jit_pass_constant_propagation(foo.graph)
        test_const_tuple_output(foo.graph, [1])

        @torch.jit.script
        def foo(x: List[int], b: List[int]):
            if False:
                while True:
                    i = 10
            if len(x) != 5:
                x1 = len(b) != 4
            else:
                x1 = True
            assert not x1
            return (len(x), len(b))
        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        torch._C._jit_pass_constant_propagation(foo.graph)
        test_const_tuple_output(foo.graph, [1])

        @torch.jit.script
        def foo(x: List[int]):
            if False:
                while True:
                    i = 10
            assert len(x) == 4
            x.append(3)
            return len(x)
        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        self.run_pass('constant_propagation', foo.graph)
        FileCheck().check_count('aten::len', 2).run(foo.graph)

        @torch.jit.script
        def foo(x: List[int], y: List[int]):
            if False:
                for i in range(10):
                    print('nop')
            assert len(x) == 4 or len(y) == 5
            return len(x) + len(y)
        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        self.run_pass('constant_propagation', foo.graph)
        FileCheck().check_count('aten::len', 4).run(foo.graph)

    def test_integer_refinement(self):
        if False:
            i = 10
            return i + 15

        def run_peephole_and_check_const_value(graph, const_string):
            if False:
                print('Hello World!')
            self.run_pass('refine_integer_values', graph)
            self.run_pass('constant_propagation', graph)
            self.run_pass('dce', graph)
            FileCheck().check(const_string).check_next('return').run(graph)

        @torch.jit.script
        def foo(x: int, y: int):
            if False:
                print('Hello World!')
            if x != 4 or y != 5:
                raise Exception('')
            return x + y
        graph = foo.graph
        self.run_pass('refine_integer_values', graph)
        self.run_pass('constant_propagation', graph)
        self.run_pass('dce', graph)
        run_peephole_and_check_const_value(foo.graph, 'value=9')
        self.assertEqual(foo(4, 5), 9)
        with self.assertRaises(Exception):
            foo(2, 4)

        @torch.jit.script
        def foo(x: int, y: int):
            if False:
                i = 10
                return i + 15
            if x == 4 and y == 5:
                pass
            else:
                raise Exception('hi')
            return x + y
        run_peephole_and_check_const_value(foo.graph, 'value=9')
        self.assertEqual(foo(4, 5), 9)
        with self.assertRaises(Exception):
            foo(2, 4)

        @torch.jit.script
        def foo(x: int, y: int, z: int):
            if False:
                print('Hello World!')
            if x != 4:
                raise Exception('..')
            elif y != 8:
                raise Exception('...')
            elif z == 3:
                pass
            else:
                raise Exception('...')
            return x + y * z
        run_peephole_and_check_const_value(foo.graph, 'value=28')
        self.assertEqual(foo(4, 8, 3), 28)
        with self.assertRaises(Exception):
            foo(1, 2, 3)

        @torch.jit.script
        def foo(x: int, cond: bool):
            if False:
                while True:
                    i = 10
            if x == 4:
                if cond:
                    return x
                return 4
            return 4
        run_peephole_and_check_const_value(foo.graph, 'value=4')

        @torch.jit.script
        def foo(x: int, y: int):
            if False:
                for i in range(10):
                    print('nop')
            assert x == 4 or y == 5
            return x + y
        torch._C._jit_pass_peephole_list_idioms(foo.graph, refine_list_len=True)
        self.run_pass('constant_propagation', foo.graph)
        FileCheck().check('aten::add').run(foo.graph)

    def test_optimize_out_comparison_same_value(self):
        if False:
            return 10

        def foo(x: int):
            if False:
                return 10
            return (x == x, x != x)

        def foo2(x: List[int]):
            if False:
                while True:
                    i = 10
            return (x == x, x != x)
        for (func, inp) in zip([foo, foo2], [1, [2, 3]]):
            func_s = torch.jit.script(func)
            self.run_pass('peephole', func_s.graph)
            FileCheck().check_not('aten::eq').check_not('aten::neq').run(func_s.graph)
            self.assertEqual(func(inp), func_s(inp))

    def test_peephole_add_zero(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def foo(x: int):
            if False:
                for i in range(10):
                    print('nop')
            return (x + 0, 0 + x)
        self.run_pass('peephole', foo.graph)
        FileCheck().check_not('aten::add')
        self.assertEqual(foo(3), (3, 3))

    def test_noop_peephole(self):
        if False:
            return 10

        def foo1(x):
            if False:
                return 10
            return x + 0

        def foo2():
            if False:
                i = 10
                return i + 15
            x = torch.zeros([2, 2])
            x.sub_(3)
            return x + 0

        def foo3():
            if False:
                for i in range(10):
                    print('nop')
            x = torch.zeros([2, 2])
            return (x, x + 0)

        def foo4():
            if False:
                print('Hello World!')
            x = torch.zeros([2, 2])
            return x + 0.0
        funcs = (foo1, foo2, foo3, foo4)
        inps = ((torch.ones([2]),), (), (), ())
        for (func, inp) in zip(funcs, inps):
            foo_s = torch.jit.script(func)
            self.run_pass('peephole', foo_s.graph)
            FileCheck().check_count('aten::add', 1, exactly=True).run(foo_s.graph)
            self.assertEqual(func(*inp), foo_s(*inp))

        def func(x):
            if False:
                i = 10
                return i + 15
            return (x + 0) * 1 - 5
        func_s = torch.jit.script(func)
        self.run_pass('peephole', func_s.graph)
        FileCheck().check_not('aten::add').check('aten::mul').run(func_s.graph)
        self.run_pass('peephole', func_s.graph)
        FileCheck().check_not('aten::add').check_not('aten::mul').run(func_s.graph)
        self.assertEqual(func(torch.ones([2, 2])), func_s(torch.ones([2, 2])))

        def func(x):
            if False:
                i = 10
                return i + 15
            return x + 0.0 - 5
        func_s = torch.jit.script(func)
        inp = next(func_s.graph.inputs())
        inp.setType(torch._C.TensorType.create_from_tensor(torch.rand([2, 2])))
        torch._C._jit_pass_peephole(func_s.graph, disable_shape_peepholes=True)
        FileCheck().check('aten::add').run(func_s.graph)
        torch._C._jit_pass_peephole(func_s.graph, disable_shape_peepholes=False)
        FileCheck().check_not('aten::add').run(func_s.graph)

    def test_refine_integer_values(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def foo(x: int):
            if False:
                i = 10
                return i + 15
            y = 1
            if x == 1:
                return y
            else:
                return x
        self.run_pass('refine_integer_values', foo.graph)
        self.run_pass('constant_propagation', foo.graph)
        self.run_pass('dce', foo.graph)
        FileCheck().check('graph').check_next('return').run(foo.graph)
        self.assertEqual(foo(2), 2)
        self.assertEqual(foo(1), 1)

    def test_peephole_len_list(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo(x):
            if False:
                i = 10
                return i + 15
            return len(x.size())
        self.run_pass('peephole', foo.graph)
        FileCheck().check('aten::len').run(foo.graph)
        inputs = list(foo.graph.inputs())
        inputs[0].setType(inputs[0].type().with_sizes([None, None]))
        self.run_pass('peephole', foo.graph)
        FileCheck().check_not('aten::len').run(foo.graph)
        self.assertEqual(2, foo(torch.rand([3, 1])))

        @torch.jit.script
        def foo(x):
            if False:
                while True:
                    i = 10
            li = x.size()
            li.append(4)
            return len(li)
        inputs = list(foo.graph.inputs())
        inputs[0].setType(inputs[0].type().with_sizes([None, None]))
        self.run_pass('peephole', foo.graph)
        FileCheck().check('aten::len').run(foo.graph)
        self.assertEqual(3, foo(torch.rand([3, 1])))

    def test_peephole_optional_refine(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo(z: int, z2: int, cond: bool):
            if False:
                i = 10
                return i + 15
            if cond:
                return z
            else:
                return z2
        out = next(foo.graph.findNode('prim::If').outputs())
        out.setType(torch._C.OptionalType(torch._C.IntType.get()))
        self.run_pass('peephole', foo.graph)
        FileCheck().check_not('int?').run(foo.graph)

    def test_peephole_int(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def foo(x):
            if False:
                i = 10
                return i + 15
            return int(x)
        FileCheck().check('aten::Int').run(foo.graph)
        next(foo.graph.inputs()).setType(torch._C.IntType.get())
        self.run_pass('peephole', foo.graph)
        FileCheck().check_not('aten::Int').run(foo.graph)

    def test_peephole_arith(self):
        if False:
            return 10

        @torch.jit.script
        def foo(input0: int, input1: int, input2: int, input3: int):
            if False:
                for i in range(10):
                    print('nop')
            _1 = torch.add(input1, 2)
            _3 = torch.add(input3, 2)
            _5 = torch.add(1, torch.sub(_1, 3) // 1)
            _6 = torch.add(1 * torch.sub(_3, 3) // 1, 1) / 1
            return [_5, int(_6)]
        FileCheck().check('aten::add').check('aten::sub').check('aten::mul').check('aten::floordiv').check('aten::div').run(foo.graph)
        self.run_pass('peephole', foo.graph)
        FileCheck().check('graph').check('):').check_next('ListConstruct').check_next('return').run(foo.graph)
        self.assertEqual(foo(0, 1, 2, 3), [1, 3])

    def test_peephole_dict_getitem_simple(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo(a: int, b: int):
            if False:
                while True:
                    i = 10
            d = {0: a, 1: b}
            x = d[1]
            y = d[0]
            return (x, y)
        self.run_pass('peephole', foo.graph)
        FileCheck().check_not('DictConstruct').check_not('__getitem__').run(foo.graph)
        self.assertEqual(foo(0, 1), (1, 0))

        @torch.jit.script
        def foo(a: int, b: int):
            if False:
                print('Hello World!')
            d = {'0': a, '1': b}
            x = d['1']
            y = d['0']
            return (x, y)
        self.run_pass('peephole', foo.graph)
        FileCheck().check_not('DictConstruct').check_not('__getitem__').run(foo.graph)
        self.assertEqual(foo(0, 1), (1, 0))

        @torch.jit.script
        def foo(a: int, b: int):
            if False:
                return 10
            d = {0.0: a, 1.0: b}
            x = d[1.0]
            y = d[0.0]
            return (x, y)
        self.run_pass('peephole', foo.graph)
        FileCheck().check_not('DictConstruct').check_not('__getitem__').run(foo.graph)
        self.assertEqual(foo(0, 1), (1, 0))

    def test_peephole_dict_getitem_no_optimization_missing_key(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo():
            if False:
                while True:
                    i = 10
            d = {0: 1}
            return d[2]
        self.run_pass('peephole', foo.graph)
        FileCheck().check('DictConstruct').check('__getitem__').run(foo.graph)

    def test_peephole_dict_getitem_no_optimization_get_input_arg(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo(a: int):
            if False:
                while True:
                    i = 10
            d = {0: 1}
            return d[a]
        self.run_pass('peephole', foo.graph)
        FileCheck().check('DictConstruct').check('__getitem__').run(foo.graph)
        self.assertEqual(foo(0), 1)

    def test_peephole_dict_getitem_no_optimization_dict_modified(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo():
            if False:
                while True:
                    i = 10
            d = {0: 1}
            d[0] = 2
            return d[0]
        self.run_pass('peephole', foo.graph)
        FileCheck().check('DictConstruct').check('__getitem__').run(foo.graph)
        self.assertEqual(foo(), 2)

    def test_peephole_dict_getitem_no_optimization_overlapping_keys(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def foo():
            if False:
                while True:
                    i = 10
            d = {0: 1, 0: 2}
            return d[0]
        self.run_pass('peephole', foo.graph)
        FileCheck().check('DictConstruct').check('__getitem__').run(foo.graph)

    def test_peephole_dict_getitem_no_optimization_keys_might_overlap(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo(x: int):
            if False:
                return 10
            d = {0: 1, x: 2}
            return d[x]
        self.run_pass('peephole', foo.graph)
        FileCheck().check('DictConstruct').check('__getitem__').run(foo.graph)

    def test_peephole_dict_getitem_no_optimization_unsupported_type(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            a = torch.rand((2, 2))
            d = {a: 1}
            return d[a]
        self.run_pass('peephole', foo.graph)
        FileCheck().check('DictConstruct').check('__getitem__').run(foo.graph)
        self.assertEqual(foo(), 1)

    def test_peephole_dict_len(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def foo():
            if False:
                return 10
            d = {0: 1, 1: 2}
            return len(d)
        self.run_pass('peephole', foo.graph)
        FileCheck().check_not('DictConstruct').check_not('len').run(foo.graph)
        self.assertEqual(foo(), 2)

    def test_peephole_dict_len_no_optimization_overlapping_keys(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def foo():
            if False:
                print('Hello World!')
            d = {0: 1, 0: 2}
            return len(d)
        self.run_pass('peephole', foo.graph)
        FileCheck().check('DictConstruct').check('len').run(foo.graph)
        self.assertEqual(foo(), 1)

    def test_peephole_dict_len_no_optimization_keys_might_overlap(self):
        if False:
            return 10

        @torch.jit.script
        def foo(x: int):
            if False:
                while True:
                    i = 10
            d = {0: 1, x: 2}
            return len(d)
        self.run_pass('peephole', foo.graph)
        FileCheck().check('DictConstruct').check('len').run(foo.graph)

    def test_peephole_dict_len_no_optimization_unsupported_type(self):
        if False:
            return 10

        @torch.jit.script
        def foo():
            if False:
                return 10
            a = torch.rand((2, 2))
            d = {a: 1}
            return len(d)
        self.run_pass('peephole', foo.graph)
        FileCheck().check('DictConstruct').check('len').run(foo.graph)
        self.assertEqual(foo(), 1)

    def test_peephole_slice_all_three_args(self):
        if False:
            return 10

        def foo(x: int):
            if False:
                print('Hello World!')
            return [1, 2, x, 4, 5, 6, 7][-5:6:2]
        graph = torch.jit.script(foo).graph
        self.run_pass('peephole', graph)
        FileCheck().check_not('aten::slice').run(graph)
        self.checkScript(foo, (3,))

    def test_peephole_slice_one_empty_arg(self):
        if False:
            i = 10
            return i + 15

        def check_helper(fn: Callable[[int], None]) -> None:
            if False:
                for i in range(10):
                    print('nop')
            graph = torch.jit.script(fn).graph
            self.run_pass('peephole', graph)
            FileCheck().check_not('aten::slice').run(graph)
            self.checkScript(fn, (3,))

        def foo(x: int):
            if False:
                for i in range(10):
                    print('nop')
            return [1, 2, x, 4, 5, 6, 7][1::2]
        check_helper(foo)

        def foo(x: int):
            if False:
                for i in range(10):
                    print('nop')
            return [1, 2, x, 4, 5, 6, 7][:5:3]
        check_helper(foo)

        def foo(x: int):
            if False:
                while True:
                    i = 10
            return [1, 2, x, 4, 5, 6, 7][0:4]
        check_helper(foo)

    def test_peephole_slice_two_empty_args(self):
        if False:
            i = 10
            return i + 15

        def check_helper(fn: Callable[[int], None]) -> None:
            if False:
                while True:
                    i = 10
            graph = torch.jit.script(fn).graph
            self.run_pass('peephole', graph)
            FileCheck().check_not('aten::slice').run(graph)
            self.checkScript(fn, (3,))

        def foo(x: int):
            if False:
                while True:
                    i = 10
            return [1, 2, x, 4, 5, 6, 7][::2]
        check_helper(foo)

        def foo(x: int):
            if False:
                print('Hello World!')
            return [1, 2, x, 4, 5, 6, 7][:5]
        check_helper(foo)

        def foo(x: int):
            if False:
                while True:
                    i = 10
            return [1, 2, x, 4, 5, 6, 7][1:]
        check_helper(foo)

    def test_peephole_slice_optimization_not_applied_list_modified(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo():
            if False:
                while True:
                    i = 10
            li = [1, 2, 3, 4, 5, 6, 7]
            li[0] = 0
            return li[2:5]
        self.run_pass('peephole', foo.graph)
        FileCheck().check('aten::slice').run(foo.graph)

    def test_peephole_slice_optimization_not_applied_non_const_args(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo(x: int, y: int):
            if False:
                return 10
            li = [1, 2, 3, 4, 5, 6, 7]
            return li[x:y]
        self.run_pass('peephole', foo.graph)
        FileCheck().check('aten::slice').run(foo.graph)