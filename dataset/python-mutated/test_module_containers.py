import os
import sys
from typing import Any, List, Tuple
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.testing._internal.jit_utils import JitTestCase
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestModuleContainers(JitTestCase):

    def test_sequential_intermediary_types(self):
        if False:
            print('Hello World!')

        class A(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x + 3

        class B(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return {'1': x}

        class C(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.foo = torch.nn.Sequential(A(), B())

            def forward(self, x):
                if False:
                    return 10
                return self.foo(x)
        self.checkModule(C(), (torch.tensor(1),))

    def test_moduledict(self):
        if False:
            i = 10
            return i + 15

        class Inner(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return x + 10

        class Inner2(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x * 2

        class Inner3(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return (x - 4) * 3

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                modules = OrderedDict([('one', Inner()), ('two', Inner2()), ('three', Inner3())])
                self.moduledict = nn.ModuleDict(modules)

            def forward(self, x, skip_name):
                if False:
                    return 10
                names = torch.jit.annotate(List[str], [])
                values = []
                for name in self.moduledict:
                    names.append(name)
                for (name, mod) in self.moduledict.items():
                    if name != skip_name:
                        names.append(name)
                        x = mod(x)
                        values.append(x)
                for mod in self.moduledict.values():
                    x = mod(x)
                    values.append(x)
                for key in self.moduledict.keys():
                    names.append(key)
                return (x, names)

        class M2(M):

            def forward(self, x, skip_name):
                if False:
                    return 10
                names = torch.jit.annotate(List[str], [])
                values = []
                x2 = x
                iter = 0
                for name in self.moduledict:
                    names.append(name)
                for (i, (name, mod)) in enumerate(self.moduledict.items()):
                    iter += i
                    if name != skip_name:
                        names.append(name)
                        x = mod(x)
                        values.append(x)
                for (i, mod) in enumerate(self.moduledict.values()):
                    iter += i
                    x = mod(x)
                    values.append(x)
                for (i, key) in enumerate(self.moduledict.keys()):
                    iter += i
                    names.append(key)
                for (mod, mod) in zip(self.moduledict.values(), self.moduledict.values()):
                    iter += i
                    x2 = mod(mod(x2))
                return (x, x2, names, iter)
        for name in ['', 'one', 'two', 'three']:
            inp = torch.tensor(1)
            self.checkModule(M(), (inp, name))
            self.checkModule(M2(), (inp, name))

    def test_custom_container_forward(self):
        if False:
            for i in range(10):
                print('nop')

        class Inner(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return x + 10

        class CustomSequential(nn.Sequential):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__(nn.ReLU(), Inner())

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = x + 3
                for mod in self:
                    x = mod(x)
                return x - 5
        self.checkModule(CustomSequential(), (torch.tensor(0.5),))

        class CustomModuleList(nn.ModuleList):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__([nn.ReLU(), Inner()])

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = x + 3
                for mod in self:
                    x = mod(x)
                return x - 5
        self.checkModule(CustomModuleList(), (torch.tensor(0.5),))

        class CustomModuleDict(nn.ModuleDict):

            def __init__(self):
                if False:
                    return 10
                super().__init__(OrderedDict([('one', Inner()), ('two', nn.ReLU()), ('three', Inner())]))

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = x + 3
                names = torch.jit.annotate(List[str], [])
                for (name, mod) in self.items():
                    x = mod(x)
                    names.append(name)
                return (names, x - 5)
        self.checkModule(CustomModuleDict(), (torch.tensor(0.5),))

    def test_script_module_list_sequential(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.jit.ScriptModule):

            def __init__(self, mod_list):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.mods = mod_list

            @torch.jit.script_method
            def forward(self, v):
                if False:
                    i = 10
                    return i + 15
                for m in self.mods:
                    v = m(v)
                return v
        with torch.jit.optimized_execution(False):
            m = M(nn.Sequential(nn.ReLU()))
            self.assertExportImportModule(m, (torch.randn(2, 2),))

    def test_script_modulelist_index(self):
        if False:
            print('Hello World!')

        class Sub(torch.nn.Module):

            def __init__(self, i):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.i = i

            def forward(self, thing):
                if False:
                    while True:
                        i = 10
                return thing - self.i

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.mods = nn.ModuleList([Sub(i) for i in range(10)])

            def forward(self, v):
                if False:
                    print('Hello World!')
                v = self.mods[4].forward(v)
                v = self.mods[-1].forward(v)
                v = self.mods[-9].forward(v)
                return v
        x = torch.tensor(1)
        self.checkModule(M(), (x,))

        class MForward(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.mods = nn.ModuleList([Sub(i) for i in range(10)])

            def forward(self, v):
                if False:
                    for i in range(10):
                        print('nop')
                v = self.mods[4](v)
                v = self.mods[-1](v)
                v = self.mods[-9](v)
                return v
        self.checkModule(MForward(), (torch.tensor(1),))

        class M2(M):

            def forward(self, v):
                if False:
                    return 10
                return self.mods[-11].forward(v)
        with self.assertRaisesRegexWithHighlight(Exception, 'Index -11 out of range', 'self.mods[-11]'):
            torch.jit.script(M2())

        class M3(M):

            def forward(self, v):
                if False:
                    i = 10
                    return i + 15
                i = 3
                return self.mods[i].forward(v)
        with self.assertRaisesRegexWithHighlight(Exception, 'Enumeration is supported', 'self.mods[i]'):
            torch.jit.script(M3())

        class M4(M):

            def forward(self, v):
                if False:
                    print('Hello World!')
                i = 3
                return self.mods[i].forward(v)
        with self.assertRaisesRegex(Exception, 'will fail because i is not a literal'):
            torch.jit.script(M4())

    def test_module_interface_special_methods(self):
        if False:
            print('Hello World!')

        class CustomModuleInterface(torch.nn.Module):
            pass

        class CustomModuleList(CustomModuleInterface, torch.nn.ModuleList):

            def __init__(self, modules=None):
                if False:
                    return 10
                CustomModuleInterface.__init__(self)
                torch.nn.ModuleList.__init__(self, modules)

        class CustomSequential(CustomModuleInterface, torch.nn.Sequential):

            def __init__(self, modules=None):
                if False:
                    while True:
                        i = 10
                CustomModuleInterface.__init__(self)
                torch.nn.Sequential.__init__(self, modules)

        class CustomModuleDict(CustomModuleInterface, torch.nn.ModuleDict):

            def __init__(self, modules=None):
                if False:
                    while True:
                        i = 10
                CustomModuleInterface.__init__(self)
                torch.nn.ModuleDict.__init__(self, modules)

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.submod = torch.jit.script(torch.nn.ReLU())
                self.modulelist = CustomModuleList([self.submod])
                self.sequential = CustomSequential(self.submod)
                self.moduledict = CustomModuleDict({'submod': self.submod})

            def forward(self, inputs):
                if False:
                    while True:
                        i = 10
                assert self.modulelist[0] is self.submod, '__getitem__ failing for ModuleList'
                assert len(self.modulelist) == 1, '__len__ failing for ModuleList'
                for module in self.modulelist:
                    assert module is self.submod, '__iter__ failing for ModuleList'
                assert self.sequential[0] is self.submod, '__getitem__ failing for Sequential'
                assert len(self.sequential) == 1, '__len__ failing for Sequential'
                for module in self.sequential:
                    assert module is self.submod, '__iter__ failing for Sequential'
                assert self.moduledict['submod'] is self.submod, '__getitem__ failing for ModuleDict'
                assert len(self.moduledict) == 1, '__len__ failing for ModuleDict'
                i = 0
                for key in self.moduledict:
                    i += 1
                assert i == len(self.moduledict), 'iteration failing for ModuleDict'
                assert 'submod' in self.moduledict, '__contains__ fails for ModuleDict'
                for key in self.moduledict.keys():
                    assert key == 'submod', 'keys() fails for ModuleDict'
                for item in self.moduledict.items():
                    assert item[0] == 'submod', 'items() fails for ModuleDict'
                    assert item[1] is self.submod, 'items() fails for ModuleDict'
                for value in self.moduledict.values():
                    assert value is self.submod, 'values() fails for ModuleDict'
                return inputs
        m = MyModule()
        self.checkModule(m, [torch.randn(2, 2)])

    def test_special_method_with_override(self):
        if False:
            while True:
                i = 10

        class CustomModuleInterface(torch.nn.Module):
            pass

        class CustomModuleList(CustomModuleInterface, torch.nn.ModuleList):

            def __init__(self, modules=None):
                if False:
                    for i in range(10):
                        print('nop')
                CustomModuleInterface.__init__(self)
                torch.nn.ModuleList.__init__(self, modules)

            def __len__(self):
                if False:
                    i = 10
                    return i + 15
                return 2

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.submod = torch.jit.script(torch.nn.ReLU())
                self.modulelist = CustomModuleList([self.submod])

            def forward(self, inputs):
                if False:
                    print('Hello World!')
                assert len(self.modulelist) == 2, '__len__ failing for ModuleList'
                return inputs
        m = MyModule()
        self.checkModule(m, [torch.randn(2, 2)])
        mm = torch.jit.script(m)

    def test_moduledict_getitem(self):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.relu = torch.jit.script(torch.nn.ReLU())
                self.tanh = torch.jit.script(torch.nn.Tanh())
                self.moduledict = torch.nn.ModuleDict({'relu': self.relu, 'tanh': self.tanh})

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                assert self.moduledict['relu'] is self.relu
                assert self.moduledict['tanh'] is self.tanh
                return input
        m = MyModule()
        self.checkModule(m, [torch.randn(2, 2)])

    def test_moduledict_keyerror(self):
        if False:
            print('Hello World!')

        class BadModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.moduledict = torch.nn.ModuleDict({'foo': None, 'bar': None})

            def forward(self, input):
                if False:
                    return 10
                assert self.moduledict['blah'] == 'blah', 'this is a keyerror'
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Key Error, blah', "self.moduledict['blah'"):
            b = BadModule()
            torch.jit.script(b)

        class AnotherBadModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.moduledict = torch.nn.ModuleDict({'foo': None, 'bar': None})

            def forward(self, input):
                if False:
                    return 10
                idx = 'blah'
                assert self.moduledict[idx] == 'blah', 'this is a string literal error'
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Unable to extract string literal index. ModuleDict indexing is only supported with string literals. For example, \'i = "a"; self.layers\\[i\\]\\(x\\)\' will fail because i is not a literal.', 'self.moduledict[idx]'):
            b = AnotherBadModule()
            torch.jit.script(b)

    def test_normal_list_attribute_with_modules_error(self):
        if False:
            return 10
        '\n        Test that an attempt to script a module with a regular list attribute\n        containing other modules fails with a relevant error message.\n        '

        class Mod(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.a = [torch.nn.ReLU(), torch.nn.ReLU()]

            def forward(self):
                if False:
                    return 10
                return len(self.a)
        error_msg = 'Could not infer type of list element: Cannot infer concrete type of torch.nn.Module'
        with self.assertRaisesRegexWithHighlight(RuntimeError, error_msg, 'self.a'):
            torch.jit.script(Mod())

    def test_empty_dict_override_contains(self):
        if False:
            return 10

        class CustomModuleInterface(torch.nn.Module):
            pass

        class CustomModuleDict(CustomModuleInterface, torch.nn.ModuleDict):

            def __init__(self, modules=None):
                if False:
                    i = 10
                    return i + 15
                CustomModuleInterface.__init__(self)
                torch.nn.ModuleDict.__init__(self, modules)

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.submod = torch.jit.script(torch.nn.ReLU())
                self.moduledict = CustomModuleDict()

            def forward(self, inputs):
                if False:
                    return 10
                assert 'submod' not in self.moduledict, '__contains__ fails for ModuleDict'
                return inputs
        m = MyModule()
        self.checkModule(m, [torch.randn(2, 2)])

    def test_typed_module_dict(self):
        if False:
            while True:
                i = 10
        '\n        Test that a type annotation can be provided for a ModuleDict that allows\n        non-static indexing.\n        '

        @torch.jit.interface
        class ModuleInterface(torch.nn.Module):

            def forward(self, inp: Any) -> Any:
                if False:
                    i = 10
                    return i + 15
                pass

        class ImplementsInterface(torch.nn.Module):

            def forward(self, inp: Any) -> Any:
                if False:
                    print('Hello World!')
                if isinstance(inp, torch.Tensor):
                    return torch.max(inp, dim=0)
                return inp

        class DoesNotImplementInterface(torch.nn.Module):

            def forward(self, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                if False:
                    while True:
                        i = 10
                return torch.max(inp, dim=0)

        class Mod(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.d = torch.nn.ModuleDict({'module': ImplementsInterface()})

            def forward(self, x: torch.Tensor, key: str) -> Any:
                if False:
                    while True:
                        i = 10
                value: ModuleInterface = self.d[key]
                return value.forward(x)
        m = Mod()
        self.checkModule(m, (torch.randn(2, 2), 'module'))

        class ModDict(torch.nn.ModuleDict):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__({'module': ImplementsInterface()})

            def forward(self, x: torch.Tensor, key: str) -> Any:
                if False:
                    while True:
                        i = 10
                submodule: ModuleInterface = self[key]
                return submodule.forward(x)
        m = ModDict()
        self.checkModule(m, (torch.randn(2, 2), 'module'))

        class ModWithWrongAnnotation(torch.nn.ModuleDict):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.d = torch.nn.ModuleDict({'module': DoesNotImplementInterface()})

            def forward(self, x: torch.Tensor, key: str) -> Any:
                if False:
                    i = 10
                    return i + 15
                submodule: ModuleInterface = self.d[key]
                return submodule.forward(x)
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Attribute module is not of annotated type', 'self.d[key]'):
            torch.jit.script(ModWithWrongAnnotation())

    def test_typed_module_list(self):
        if False:
            while True:
                i = 10
        '\n        Test that a type annotation can be provided for a ModuleList that allows\n        non-static indexing.\n        '

        @torch.jit.interface
        class ModuleInterface(torch.nn.Module):

            def forward(self, inp: Any) -> Any:
                if False:
                    for i in range(10):
                        print('nop')
                pass

        class ImplementsInterface(torch.nn.Module):

            def forward(self, inp: Any) -> Any:
                if False:
                    print('Hello World!')
                if isinstance(inp, torch.Tensor):
                    return torch.max(inp, dim=0)
                return inp

        class DoesNotImplementInterface(torch.nn.Module):

            def forward(self, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                if False:
                    while True:
                        i = 10
                return torch.max(inp, dim=0)

        class Mod(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.l = torch.nn.ModuleList([ImplementsInterface()])

            def forward(self, x: torch.Tensor, idx: int) -> Any:
                if False:
                    i = 10
                    return i + 15
                value: ModuleInterface = self.l[idx]
                return value.forward(x)
        m = Mod()
        self.checkModule(m, (torch.randn(2, 2), 0))

        class ModList(torch.nn.ModuleList):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__([ImplementsInterface()])

            def forward(self, x: torch.Tensor, idx: int) -> Any:
                if False:
                    return 10
                submodule: ModuleInterface = self[idx]
                return submodule.forward(x)
        m = ModList()
        self.checkModule(m, (torch.randn(2, 2), 0))

        class ModWithWrongAnnotation(torch.nn.ModuleList):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.l = torch.nn.ModuleList([DoesNotImplementInterface()])

            def forward(self, x: torch.Tensor, idx: int) -> Any:
                if False:
                    i = 10
                    return i + 15
                submodule: ModuleInterface = self.l[idx]
                return submodule.forward(x)
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Attribute 0 is not of annotated type', 'self.l[idx]'):
            torch.jit.script(ModWithWrongAnnotation())

    def test_module_properties(self):
        if False:
            return 10

        class ModuleWithProperties(torch.nn.Module):
            __jit_unused_properties__ = ['ignored_attr']

            def __init__(self, a: int):
                if False:
                    print('Hello World!')
                super().__init__()
                self.a = a

            def forward(self, a: int, b: int):
                if False:
                    while True:
                        i = 10
                self.attr = a + b
                return self.attr

            @property
            def attr(self):
                if False:
                    i = 10
                    return i + 15
                return self.a

            @property
            def ignored_attr(self):
                if False:
                    return 10
                return sum([self.a])

            @torch.jit.unused
            @property
            def ignored_attr_2(self):
                if False:
                    for i in range(10):
                        print('nop')
                return sum([self.a])

            @ignored_attr_2.setter
            def ignored_attr_2(self, value):
                if False:
                    print('Hello World!')
                self.a = sum([self.a])

            @attr.setter
            def attr(self, a: int):
                if False:
                    return 10
                if a > 0:
                    self.a = a
                else:
                    self.a = 0

        class ModuleWithNoSetter(torch.nn.Module):

            def __init__(self, a: int):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.a = a

            def forward(self, a: int, b: int):
                if False:
                    return 10
                self.attr + a + b

            @property
            def attr(self):
                if False:
                    i = 10
                    return i + 15
                return self.a + 1
        self.checkModule(ModuleWithProperties(5), (5, 6))
        self.checkModule(ModuleWithProperties(5), (-5, -6))
        self.checkModule(ModuleWithNoSetter(5), (5, 6))
        self.checkModule(ModuleWithNoSetter(5), (-5, -6))
        mod = ModuleWithProperties(3)
        scripted_mod = torch.jit.script(mod)
        with self.assertRaisesRegex(AttributeError, 'has no attribute'):
            scripted_mod.ignored_attr

    def test_module_inplace_construct(self):
        if False:
            print('Hello World!')

        class M(nn.Module):

            def __init__(self, start: int):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.linear = nn.Linear(3, 3)
                self.attribute = start
                self.parameter = nn.Parameter(torch.tensor(3, dtype=torch.float))

            def method(self) -> int:
                if False:
                    while True:
                        i = 10
                return self.attribute

            @torch.jit.unused
            def unused_method(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.attribute + self.attribute

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.linear(self.linear(x))

        class N(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.linear = nn.Linear(4, 4)

            @torch.jit.ignore
            def ignored_method(self, x):
                if False:
                    return 10
                return x

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.linear(x)
        m = torch.jit.script(M(3))
        n = torch.jit.script(N())
        n._reconstruct(m._c)
        inp = torch.rand(3)
        with torch.no_grad():
            m_out = m(inp)
            n_out = n(inp)
            self.assertEqual(m_out, n_out)
        self.assertEqual(inp, n.ignored_method(inp))

    def test_parameterlist_script_getitem(self):
        if False:
            return 10

        class MyModule(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.module_list = nn.ModuleList([nn.Linear(1, 1) for _ in range(10)])
                self.parameter_list = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(10)])

            def forward(self, x):
                if False:
                    return 10
                self.module_list[0]
                self.parameter_list[0]
                return x
        self.checkModule(MyModule(), torch.zeros(1))

    def test_parameterlist_script_iter(self):
        if False:
            i = 10
            return i + 15

        class MyModule(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.module_list = nn.ModuleList([nn.Linear(1, 1) for _ in range(10)])
                self.parameter_list = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(10)])

            def forward(self, x):
                if False:
                    print('Hello World!')
                r = x
                for (i, p) in enumerate(self.parameter_list):
                    r = r + p + i
                return r
        self.checkModule(MyModule(), (torch.zeros(1),))

    def test_parameterdict_script_getitem(self):
        if False:
            return 10

        class MyModule(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.parameter_dict = nn.ParameterDict({k: nn.Parameter(torch.zeros(1)) for k in ['a', 'b', 'c']})

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.parameter_dict['a'] * x + self.parameter_dict['b'] * self.parameter_dict['c']
        self.checkModule(MyModule(), (torch.ones(1),))