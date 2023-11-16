import os
import sys
import torch
from torch.testing import FileCheck
from typing import List
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, freeze_rng_state
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestRemoveMutation(JitTestCase):

    def test_aten_inplace(self):
        if False:
            print('Hello World!')

        def test_not_new_alias(x):
            if False:
                i = 10
                return i + 15
            y = x[0]
            y.add_(2)
            return y
        fn = torch.jit.script(test_not_new_alias)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check('aten::add_').run(graph)
        self.assertEqual(fn(torch.ones([2, 2])), test_not_new_alias(torch.ones([2, 2])))

        def test_no_lowering():
            if False:
                return 10
            x = torch.tensor([2, 2])
            x[0] = 3
            return x
        fn = torch.jit.script(test_no_lowering)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check('aten::copy_').run(graph)
        self.assertEqual(fn(), test_no_lowering())

        def test_move_before_not_valid():
            if False:
                for i in range(10):
                    print('nop')
            y = torch.tensor([2, 2])
            z = y + 2
            y.add_(2)
            return (y, z)
        fn = torch.jit.script(test_move_before_not_valid)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check('aten::add_').run(graph)
        self.assertEqual(fn(), test_move_before_not_valid())

        def test_successful():
            if False:
                i = 10
                return i + 15
            x = torch.tensor([2, 2])
            x.add_(1)
            x.add_(3)
            y = x + 4
            return (x, y)
        fn = torch.jit.script(test_successful)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check_not('aten::add_').run(graph)
        self.assertEqual(test_successful(), fn())

        def test_intermediary_use():
            if False:
                return 10
            x = torch.tensor([2, 2])
            x.add_(1)
            y = x + 4
            x.add_(3)
            return (x, y)
        fn = torch.jit.script(test_intermediary_use)
        graph = fn.graph
        FileCheck().check_count('aten::add_', 2).run(graph)
        self.run_pass('remove_mutation', graph)
        FileCheck().check_count('aten::add_', 1).run(graph)
        self.assertEqual(test_intermediary_use(), fn())

    def test_if_output(self):
        if False:
            while True:
                i = 10

        def foo(x, cond: bool):
            if False:
                for i in range(10):
                    print('nop')
            if cond:
                y = x + 5
            else:
                y = x + 2
            y.add_(4)
            return y
        out_eager = foo(torch.tensor(5), True)
        foo_script = torch.jit.script(foo)
        FileCheck().check('aten::add_').run(foo_script.graph)
        self.run_pass('remove_mutation', foo_script.graph)
        FileCheck().check_not('aten::add_').run(foo_script.graph)
        self.assertEqual(out_eager, foo_script(torch.tensor(5), True))

    def test_if_output_fail(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo(cond: bool):
            if False:
                while True:
                    i = 10
            li = []
            if cond:
                x = torch.tensor(1)
                li.append(x)
            else:
                x = torch.tensor(2)
            y = x.add_(2)
            return (y, li)
        self.run_pass('inline', foo.graph)
        self.run_pass('remove_mutation', foo.graph)
        FileCheck().check('aten::add_').run(foo.graph)

        @torch.jit.script
        def foo(cond: bool, y):
            if False:
                print('Hello World!')
            if cond:
                x = y
            else:
                x = torch.tensor(2)
            z = x.add_(2)
            return z
        self.run_pass('inline', foo.graph)
        self.run_pass('remove_mutation', foo.graph)
        FileCheck().check('aten::add_').run(foo.graph)

    def test_special_mapped_op(self):
        if False:
            while True:
                i = 10

        def test_successful():
            if False:
                for i in range(10):
                    print('nop')
            x = torch.tensor([2, 2])
            y = torch.tensor([2, 4])
            x.zero_()
            y.fill_(3)
            return (x, y)
        fn = torch.jit.script(test_successful)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check_not('aten::zero_').check_not('aten::fill_').run(graph)
        self.assertEqual(test_successful(), fn())

        def test_successful():
            if False:
                while True:
                    i = 10
            x = torch.tensor([2, 2])
            y = torch.tensor([2, 4])
            x.fill_(y)
            return x + x
        fn = torch.jit.script(test_successful)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check_not('aten::fill_').run(graph)

        def normal():
            if False:
                i = 10
                return i + 15
            return torch.rand(2, 1, 3, 4, dtype=torch.float).normal_()
        fn = torch.jit.script(normal)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check_not('normal_').run(graph)
        with freeze_rng_state():
            out_eager = normal()
        with freeze_rng_state():
            out_script = fn()
        self.assertEqual(out_eager, out_script)

    def test_lists_append(self):
        if False:
            for i in range(10):
                print('nop')

        def successful_remove():
            if False:
                while True:
                    i = 10
            return [i for i in range(5)]
        fn = torch.jit.script(successful_remove)
        graph = fn.graph
        self.run_pass('loop_unrolling', graph)
        self.run_pass('remove_mutation', graph)
        self.run_pass('constant_propagation', graph)
        FileCheck().check('graph').check_next('Constant').check_next('return').run(graph)
        self.assertEqual(successful_remove(), successful_remove())

        def intermediary_use():
            if False:
                while True:
                    i = 10
            a = [1, 2]
            b = len(a)
            a.append(3)
            return a
        fn = torch.jit.script(intermediary_use)
        graph = fn.graph
        FileCheck().check('append').run(graph)
        self.run_pass('remove_mutation', graph)
        FileCheck().check_not('append').run(graph)
        self.assertEqual(intermediary_use(), fn())

    def test_lists_insert(self):
        if False:
            while True:
                i = 10

        def successful_remove():
            if False:
                i = 10
                return i + 15
            a: List[int] = []
            a.insert(0, 1)
            a.insert(0, 2)
            a.insert(-10, 3)
            a.insert(-9, 4)
            a.insert(10, 5)
            return a
        fn = torch.jit.script(successful_remove)
        graph = fn.graph
        torch._C._jit_pass_remove_mutation(graph)
        torch._C._jit_pass_constant_propagation(graph)
        FileCheck().check('graph').check_next('Constant').check_next('return').run(graph)
        self.assertEqual(successful_remove(), fn())

    def test_list_indexing_removal(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def out_of_bounds():
            if False:
                for i in range(10):
                    print('nop')
            x = [1, 2]
            x[4] = 3
            return x
        torch._C._jit_pass_remove_mutation(out_of_bounds.graph)
        FileCheck().check('set_item').run(out_of_bounds.graph)

        @torch.jit.script
        def unknown(y: int):
            if False:
                i = 10
                return i + 15
            x = [1, 2]
            x[y] = 3
            return x
        torch._C._jit_pass_remove_mutation(out_of_bounds.graph)
        FileCheck().check('set_item').run(out_of_bounds.graph)

        def successful():
            if False:
                for i in range(10):
                    print('nop')
            x = [1, 2, 3]
            x[0] = 4
            x[-1] = 0
            return x
        scripted_fn = torch.jit.script(successful)
        torch._C._jit_pass_remove_mutation(scripted_fn.graph)
        FileCheck().check_not('set_item').run(scripted_fn.graph)
        self.checkScript(successful, ())

        def successful():
            if False:
                for i in range(10):
                    print('nop')
            x = [1, 2, 3]
            x[0] = 4
            x[-1] = 0
            return x
        scripted_fn = torch.jit.script(successful)
        torch._C._jit_pass_remove_mutation(scripted_fn.graph)
        FileCheck().check_not('set_item').run(scripted_fn.graph)
        self.checkScript(successful, ())

        def successful():
            if False:
                i = 10
                return i + 15
            x = [1]
            x[-1] = 3
            return x
        scripted_fn = torch.jit.script(successful)
        torch._C._jit_pass_remove_mutation(scripted_fn.graph)
        FileCheck().check_not('set_item').run(scripted_fn.graph)
        self.checkScript(successful, ())

    def test_common_pytorch_list_ops(self):
        if False:
            i = 10
            return i + 15
        for op in ['cat', 'stack', 'vstack', 'hstack', 'dstack']:

            class OpMod(torch.nn.Module):

                def __init__(self, op):
                    if False:
                        for i in range(10):
                            print('nop')
                    super().__init__()
                    self.op = torch_op

                def forward(self):
                    if False:
                        while True:
                            i = 10
                    x = torch.tensor([1, 2, 3, 4])
                    x.add_(3)
                    y = [x, x]
                    return self.op(y) + 3
            torch_op = getattr(torch, op)
            mod = OpMod(torch_op)
            mod_script = torch.jit.script(mod)
            self.run_pass('remove_mutation', mod_script.forward.graph)
            FileCheck().check_not('aten::add_').run(mod_script.forward.graph)
            self.assertEqual(mod(), mod_script())
            for inputs in ([torch.rand(2, 2)], [torch.rand(2, 2) for _ in range(2)]):
                result = torch_op(inputs)
                sums = [ten.sum() for ten in result]
                for inp in inputs:
                    inp.fill_(10)
                self.assertEqual(sums, [ten.sum() for ten in result])

        @torch.jit.script
        def test_multiple_uses():
            if False:
                for i in range(10):
                    print('nop')
            x = torch.tensor([1, 2, 3, 4])
            x.add_(3)
            y = [x, x]
            return (torch.cat(y), y)
        self.run_pass('remove_mutation', mod_script.forward.graph)
        FileCheck().check('aten::add_').run(test_multiple_uses.graph)