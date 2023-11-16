import dataclasses
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
import torch
import torch._dynamo as torchdynamo
from functorch.experimental.control_flow import map, cond
from torch import Tensor
from torch.export import Constraint, Dim, export
from torch._export import DEFAULT_EXPORT_DYNAMO_CONFIG, dynamic_dim, capture_pre_autograd_graph, _export
from torch._export.utils import get_buffer, get_param, is_buffer, is_param, register_dataclass_as_pytree_node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._pytree import LeafSpec, tree_flatten, tree_unflatten, TreeSpec, treespec_loads, treespec_dumps

@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestUnflatten(TestCase):

    def compare_outputs(self, eager, unflattened, args):
        if False:
            while True:
                i = 10
        orig_output = eager(*args)
        unflattened_output = unflattened(*args)
        self.assertTrue(torch.allclose(orig_output, unflattened_output))

    def test_unflatten_nested(self):
        if False:
            while True:
                i = 10

        class NestedChild(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x / x

        class Child1(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.nested = NestedChild()
                self.register_parameter('child1param', torch.nn.Parameter(torch.ones(2, 3)))

            def forward(self, x):
                if False:
                    return 10
                x = self.nested(x)
                return x + self.child1param

        class Child2(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.register_buffer('child2buffer', torch.ones(2, 3))

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x - self.child2buffer

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()
                self.register_parameter('rootparam', torch.nn.Parameter(torch.ones(2, 3)))

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = x * self.rootparam
                x = self.foo(x)
                x = self.bar(x)
                return x
        orig_eager = MyModule()
        export_module = export(orig_eager, (torch.rand(2, 3),), {})
        unflattened = export_module.module(flat=False)
        inputs = (torch.rand(2, 3),)
        self.compare_outputs(orig_eager, unflattened, inputs)
        self.compare_outputs(orig_eager.foo, unflattened.foo, inputs)
        self.compare_outputs(orig_eager.bar, unflattened.bar, inputs)
        self.compare_outputs(orig_eager.foo.nested, unflattened.foo.nested, inputs)
        orig_state_dict = orig_eager.state_dict()
        exported_state_dict = unflattened.state_dict()
        for (name, value) in orig_state_dict.items():
            self.assertTrue(torch.allclose(value, exported_state_dict[name]))

    def test_unflatten_buffer_mutation(self):
        if False:
            i = 10
            return i + 15

        class Child(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.register_buffer('child2buffer', torch.ones(2, 3))

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                self.child2buffer.add_(x)
                return x - self.child2buffer

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.foo = Child()
                self.register_parameter('rootparam', torch.nn.Parameter(torch.ones(2, 3)))

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.foo(x)
                return x * self.rootparam
        eager_module = MyModule()
        export_module = export(eager_module, (torch.rand(2, 3),), {})
        unflattened_module = export_module.module(flat=False)
        eager_buffer = eager_module.foo.child2buffer
        unflattened_buffer = unflattened_module.foo.child2buffer
        self.assertTrue(torch.allclose(eager_buffer, unflattened_buffer))
        inputs = (torch.rand(2, 3),)
        eager_module(*inputs)
        unflattened_module(*inputs)
        self.assertTrue(torch.allclose(eager_buffer, unflattened_buffer))

    def test_unflatten_nested_access(self):
        if False:
            return 10

        class Child(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.register_buffer('child2buffer', torch.ones(2, 3))

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x - self.child2buffer

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.foo = Child()
                self.register_parameter('rootparam', torch.nn.Parameter(torch.ones(2, 3)))

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = x + self.foo.child2buffer
                x = self.foo(x)
                return x
        eager_module = MyModule()
        export_module = export(eager_module, (torch.rand(2, 3),), {})
        unflattened_module = export_module.module(flat=False)
        inputs = (torch.rand(2, 3),)
        self.compare_outputs(eager_module, unflattened_module, inputs)

    def test_unflatten_shared_submodule(self):
        if False:
            for i in range(10):
                print('nop')

        class Shared(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                layernorm = torch.nn.LayerNorm(10)
                self.sub_net = torch.nn.Sequential(layernorm, torch.nn.ReLU(), layernorm, torch.nn.ReLU())

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.sub_net(x)
        eager_module = Shared()
        inps = (torch.rand(10),)
        export_module = export(eager_module, inps, {})
        unflattened_module = export_module.module(flat=False)
        self.compare_outputs(eager_module, unflattened_module, inps)
        self.assertTrue(hasattr(unflattened_module, 'sub_net'))
        for i in range(len(eager_module.sub_net)):
            self.assertTrue(hasattr(unflattened_module.sub_net, str(i)))
        self.assertEqual(id(getattr(unflattened_module.sub_net, '0')), id(getattr(unflattened_module.sub_net, '2')))

    def test_unflatten_preserve_signature(self):
        if False:
            for i in range(10):
                print('nop')

        class NestedChild(torch.nn.Module):

            def forward(self, zx, y):
                if False:
                    while True:
                        i = 10
                return {'x': y['key'] + zx[1], 'w': y['key'] * zx[1]}

        class Child1(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.nested = NestedChild()

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                z = torch.ones_like(x)
                xw = self.nested((z, x), y={'key': y})
                return xw['w'] + z - xw['x']

        class Child2(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()

            def forward(self, x):
                if False:
                    return 10
                return x - 1

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()

            def forward(self, x, y):
                if False:
                    return 10
                x = self.foo(x, y)
                x = self.bar(x)
                return x
        orig_eager = MyModule()
        inps = (torch.rand(2, 3), torch.rand(2, 3))
        export_module = export(orig_eager, inps, {}, preserve_module_call_signature=('foo.nested',))
        unflattened = export_module.module(flat=False)
        self.compare_outputs(export_module, unflattened, inps)
        unflattened.foo.nested = NestedChild()
        self.compare_outputs(export_module, unflattened, inps)

    def test_unflatten_param_list_dict(self):
        if False:
            print('Hello World!')

        class Mod(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.param_list = torch.nn.ParameterList()
                self.param_dict = torch.nn.ParameterDict()
                for i in range(2):
                    self.param_list.append(torch.nn.Parameter(torch.randn((2, 3))))
                    self.param_dict[f'key_{i}'] = torch.nn.Parameter(torch.randn((2, 3)))

            def forward(self, x):
                if False:
                    print('Hello World!')
                for i in range(2):
                    x = x + self.param_list[i]
                    x = x + self.param_dict[f'key_{i}']
                return x
        export_module = torch.export.export(Mod(), (torch.randn((2, 3)),))
        unflattened = export_module.module(flat=False)
        self.compare_outputs(export_module, unflattened, (torch.randn((2, 3)),))

    def test_unflatten_wrong_input(self):
        if False:
            for i in range(10):
                print('nop')

        class Mod(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.param_list = torch.nn.ParameterList()
                self.param_dict = torch.nn.ParameterDict()
                for i in range(2):
                    self.param_list.append(torch.nn.Parameter(torch.randn((2, 3))))
                    self.param_dict[f'key_{i}'] = torch.nn.Parameter(torch.randn((2, 3)))

            def forward(self, x):
                if False:
                    return 10
                a = x.sum()
                for i in range(2):
                    a = a + self.param_list[i].sum()
                    a = a + self.param_dict[f'key_{i}'].sum()
                return a
        export_module = torch.export.export(Mod(), (torch.randn((2, 3)),))
        with self.assertRaisesRegex(RuntimeError, 'Input arg4_1.shape'):
            export_module(torch.randn(6, 6))
        unflattened = export_module.module(flat=False)
        with self.assertRaisesRegex(RuntimeError, 'Input arg4_1.shape'):
            unflattened(torch.randn(6, 6))
if __name__ == '__main__':
    run_tests()