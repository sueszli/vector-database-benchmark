from typing import List, Any
import torch
import torch.nn as nn
import os
import sys
from torch import Tensor
from torch.testing._internal.jit_utils import JitTestCase, make_global
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class OrigModule(nn.Module):

    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        return inp1 + inp2 + 1

    def two(self, input: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        return input + 2

    def forward(self, input: Tensor) -> Tensor:
        if False:
            return 10
        return input + self.one(input, input) + 1

class NewModule(nn.Module):

    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        return inp1 * inp2 + 1

    def forward(self, input: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        return self.one(input, input + 1)

class TestModuleInterface(JitTestCase):

    def test_not_submodule_interface_call(self):
        if False:
            print('Hello World!')

        @torch.jit.interface
        class ModuleInterface(nn.Module):

            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                if False:
                    print('Hello World!')
                pass

        class TestNotModuleInterfaceCall(nn.Module):
            proxy_mod: ModuleInterface

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                return self.proxy_mod.two(input)
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'object has no attribute or method', 'self.proxy_mod.two'):
            torch.jit.script(TestNotModuleInterfaceCall())

    def test_module_interface(self):
        if False:
            return 10

        @torch.jit.interface
        class OneTwoModule(nn.Module):

            def one(self, x: Tensor, y: Tensor) -> Tensor:
                if False:
                    print('Hello World!')
                pass

            def two(self, x: Tensor) -> Tensor:
                if False:
                    while True:
                        i = 10
                pass

            def forward(self, x: Tensor) -> Tensor:
                if False:
                    i = 10
                    return i + 15
                pass

        @torch.jit.interface
        class OneTwoClass:

            def one(self, x: Tensor, y: Tensor) -> Tensor:
                if False:
                    i = 10
                    return i + 15
                pass

            def two(self, x: Tensor) -> Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                pass

        class FooMod(nn.Module):

            def one(self, x: Tensor, y: Tensor) -> Tensor:
                if False:
                    return 10
                return x + y

            def two(self, x: Tensor) -> Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                return 2 * x

            def forward(self, x: Tensor) -> Tensor:
                if False:
                    while True:
                        i = 10
                return self.one(self.two(x), x)

        class BarMod(nn.Module):

            def one(self, x: Tensor, y: Tensor) -> Tensor:
                if False:
                    i = 10
                    return i + 15
                return x * y

            def two(self, x: Tensor) -> Tensor:
                if False:
                    while True:
                        i = 10
                return 2 / x

            def forward(self, x: Tensor) -> Tensor:
                if False:
                    while True:
                        i = 10
                return self.two(self.one(x, x))

            @torch.jit.export
            def forward2(self, x: Tensor) -> Tensor:
                if False:
                    while True:
                        i = 10
                return self.two(self.one(x, x)) + 1
        make_global(OneTwoModule, OneTwoClass)

        def use_module_interface(mod_list: List[OneTwoModule], x: torch.Tensor):
            if False:
                return 10
            return mod_list[0].forward(x) + mod_list[1].forward(x)

        def use_class_interface(mod_list: List[OneTwoClass], x: Tensor) -> Tensor:
            if False:
                for i in range(10):
                    print('nop')
            return mod_list[0].two(x) + mod_list[1].one(x, x)
        scripted_foo_mod = torch.jit.script(FooMod())
        scripted_bar_mod = torch.jit.script(BarMod())
        self.checkScript(use_module_interface, ([scripted_foo_mod, scripted_bar_mod], torch.rand(3, 4)))
        self.checkScript(use_class_interface, ([scripted_foo_mod, scripted_bar_mod], torch.rand(3, 4)))

        def call_module_interface_on_other_method(mod_interface: OneTwoModule, x: Tensor) -> Tensor:
            if False:
                for i in range(10):
                    print('nop')
            return mod_interface.forward2(x)
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'object has no attribute or method', 'mod_interface.forward2'):
            self.checkScript(call_module_interface_on_other_method, (scripted_bar_mod, torch.rand(3, 4)))

    def test_module_doc_string(self):
        if False:
            return 10

        @torch.jit.interface
        class TestInterface(nn.Module):

            def one(self, inp1, inp2):
                if False:
                    i = 10
                    return i + 15
                pass

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                'stuff 1'
                'stuff 2'
                pass
                'stuff 3'

        class TestModule(nn.Module):
            proxy_mod: TestInterface

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return self.proxy_mod.forward(input)
        input = torch.randn(3, 4)
        self.checkModule(TestModule(), (input,))

    def test_module_interface_subtype(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.interface
        class OneTwoModule(nn.Module):

            def one(self, x: Tensor, y: Tensor) -> Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def two(self, x: Tensor) -> Tensor:
                if False:
                    i = 10
                    return i + 15
                pass

            def forward(self, x: Tensor) -> Tensor:
                if False:
                    return 10
                pass
        make_global(OneTwoModule)

        @torch.jit.script
        def as_module_interface(x: OneTwoModule) -> OneTwoModule:
            if False:
                while True:
                    i = 10
            return x

        @torch.jit.script
        class Foo:

            def one(self, x: Tensor, y: Tensor) -> Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                return x + y

            def two(self, x: Tensor) -> Tensor:
                if False:
                    while True:
                        i = 10
                return 2 * x

            def forward(self, x: Tensor) -> Tensor:
                if False:
                    print('Hello World!')
                return self.one(self.two(x), x)
        with self.assertRaisesRegex(RuntimeError, 'ScriptModule class can be subtype of module interface'):
            as_module_interface(Foo())

        class WrongMod(nn.Module):

            def two(self, x: int) -> int:
                if False:
                    print('Hello World!')
                return 2 * x

            def forward(self, x: Tensor) -> Tensor:
                if False:
                    i = 10
                    return i + 15
                return x + torch.randn(3, self.two(3))
        scripted_wrong_mod = torch.jit.script(WrongMod())
        with self.assertRaisesRegex(RuntimeError, 'is not compatible with interface'):
            as_module_interface(scripted_wrong_mod)

        @torch.jit.interface
        class TensorToAny(nn.Module):

            def forward(self, input: torch.Tensor) -> Any:
                if False:
                    while True:
                        i = 10
                pass
        make_global(TensorToAny)

        @torch.jit.script
        def as_tensor_to_any(x: TensorToAny) -> TensorToAny:
            if False:
                print('Hello World!')
            return x

        @torch.jit.interface
        class AnyToAny(nn.Module):

            def forward(self, input: Any) -> Any:
                if False:
                    i = 10
                    return i + 15
                pass
        make_global(AnyToAny)

        @torch.jit.script
        def as_any_to_any(x: AnyToAny) -> AnyToAny:
            if False:
                for i in range(10):
                    print('nop')
            return x

        class TensorToAnyImplA(nn.Module):

            def forward(self, input: Any) -> Any:
                if False:
                    print('Hello World!')
                return input

        class TensorToAnyImplB(nn.Module):

            def forward(self, input: Any) -> torch.Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                return torch.tensor([1])

        class AnyToAnyImpl(nn.Module):

            def forward(self, input: Any) -> torch.Tensor:
                if False:
                    print('Hello World!')
                return torch.tensor([1])
        as_tensor_to_any(torch.jit.script(TensorToAnyImplA()))
        as_tensor_to_any(torch.jit.script(TensorToAnyImplB()))
        as_any_to_any(torch.jit.script(AnyToAnyImpl()))

    def test_module_interface_inheritance(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'does not support inheritance yet. Please directly'):

            @torch.jit.interface
            class InheritMod(nn.ReLU):

                def three(self, x: Tensor) -> Tensor:
                    if False:
                        i = 10
                        return i + 15
                    return 3 * x

    def test_module_swap(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.interface
        class ModuleInterface(nn.Module):

            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                if False:
                    print('Hello World!')
                pass

            def forward(self, input: Tensor) -> Tensor:
                if False:
                    while True:
                        i = 10
                pass

        class TestModule(nn.Module):
            proxy_mod: ModuleInterface

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                if False:
                    return 10
                return self.proxy_mod.forward(input)
        scripted_mod = torch.jit.script(TestModule())
        input = torch.randn(3, 4)
        self.assertEqual(scripted_mod(input), 3 * input + 2)
        scripted_mod.proxy_mod = torch.jit.script(NewModule())
        self.assertEqual(scripted_mod(input), input * (input + 1) + 1)
        with self.assertRaisesRegex(RuntimeError, 'a ScriptModule with non-scripted module'):
            scripted_mod.proxy_mod = NewModule()

    def test_module_swap_wrong_module(self):
        if False:
            while True:
                i = 10

        @torch.jit.interface
        class ModuleInterface(nn.Module):

            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def forward(self, input: Tensor) -> Tensor:
                if False:
                    while True:
                        i = 10
                pass

        class NewModuleWrong(nn.Module):

            def forward(self, input: int) -> int:
                if False:
                    print('Hello World!')
                return input + 1

        class TestModule(nn.Module):
            proxy_mod: ModuleInterface

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                if False:
                    return 10
                return self.proxy_mod.forward(input)
        scripted_mod = torch.jit.script(TestModule())
        with self.assertRaisesRegex(RuntimeError, 'is not compatible with interface'):
            scripted_mod.proxy_mod = torch.jit.script(NewModuleWrong())

    def test_module_swap_no_lazy_compile(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.interface
        class ModuleInterface(nn.Module):

            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                if False:
                    while True:
                        i = 10
                pass

            def forward(self, input: Tensor) -> Tensor:
                if False:
                    i = 10
                    return i + 15
                pass

        class TestModule(nn.Module):
            proxy_mod: ModuleInterface

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                if False:
                    i = 10
                    return i + 15
                return self.proxy_mod.forward(input)

        class NewModuleMethodNotLazyCompile(nn.Module):

            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                if False:
                    i = 10
                    return i + 15
                return inp1 * inp2 + 1

            def forward(self, input: Tensor) -> Tensor:
                if False:
                    return 10
                return input + 1
        scripted_mod = torch.jit.script(TestModule())
        with self.assertRaisesRegex(RuntimeError, 'is not compatible with interface'):
            scripted_mod.proxy_mod = torch.jit.script(NewModuleMethodNotLazyCompile())

        class NewModuleMethodManualExport(nn.Module):

            @torch.jit.export
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                if False:
                    while True:
                        i = 10
                return inp1 * inp2 + 1

            def forward(self, input: Tensor) -> Tensor:
                if False:
                    while True:
                        i = 10
                return input + 1
        scripted_mod.proxy_mod = torch.jit.script(NewModuleMethodManualExport())
        input = torch.randn(3, 4)
        self.assertEqual(scripted_mod(input), input + 1)

    def test_module_swap_no_module_interface(self):
        if False:
            i = 10
            return i + 15

        class TestNoModuleInterface(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                return self.proxy_mod(input)
        scripted_no_module_interface = torch.jit.script(TestNoModuleInterface())
        scripted_no_module_interface.proxy_mod = torch.jit.script(OrigModule())
        with self.assertRaisesRegex(RuntimeError, "Expected a value of type '__torch__.jit.test_module_interface.OrigModule \\(.*\\)' " + "for field 'proxy_mod', but found '__torch__.jit.test_module_interface.NewModule \\(.*\\)'"):
            scripted_no_module_interface.proxy_mod = torch.jit.script(NewModule())

    def test_script_module_as_interface_swap(self):
        if False:
            print('Hello World!')

        @torch.jit.interface
        class ModuleInterface(nn.Module):

            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                if False:
                    i = 10
                    return i + 15
                pass

            def forward(self, input: Tensor) -> Tensor:
                if False:
                    return 10
                pass

        class OrigScriptModule(torch.jit.ScriptModule):

            @torch.jit.script_method
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                return inp1 + inp2 + 1

            @torch.jit.script_method
            def forward(self, input: Tensor) -> Tensor:
                if False:
                    return 10
                return input + self.one(input, input) + 1

        class NewScriptModule(torch.jit.ScriptModule):

            @torch.jit.script_method
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                if False:
                    print('Hello World!')
                return inp1 * inp2 + 1

            @torch.jit.script_method
            def forward(self, input: Tensor) -> Tensor:
                if False:
                    i = 10
                    return i + 15
                return self.one(input, input + 1)

        class TestNNModuleWithScriptModule(nn.Module):
            proxy_mod: ModuleInterface

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.proxy_mod = OrigScriptModule()

            def forward(self, input: Tensor) -> Tensor:
                if False:
                    i = 10
                    return i + 15
                return self.proxy_mod.forward(input)
        input = torch.randn(3, 4)
        scripted_mod = torch.jit.script(TestNNModuleWithScriptModule())
        self.assertEqual(scripted_mod(input), 3 * input + 2)
        scripted_mod.proxy_mod = NewScriptModule()
        self.assertEqual(scripted_mod(input), input * (input + 1) + 1)

    def test_freeze_module_with_interface(self):
        if False:
            for i in range(10):
                print('nop')

        class SubModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.b = 20

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.b

        class OrigMod(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.a = 0

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):

            def forward(self, x: Tensor) -> int:
                if False:
                    for i in range(10):
                        print('nop')
                pass

        class TestModule(torch.nn.Module):
            proxy_mod: ModInterface

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.proxy_mod(x) + self.sub(x)
        m = torch.jit.script(TestModule())
        m.eval()
        mf = torch._C._freeze_module(m._c)
        mf = torch._C._freeze_module(m._c, freezeInterfaces=True)
        input = torch.tensor([1])
        out_s = m.forward(input)
        out_f = mf.forward(input)
        self.assertEqual(out_s, out_f)

    def test_freeze_module_with_setattr_in_interface(self):
        if False:
            print('Hello World!')

        class SubModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.b = 20

            def forward(self, x):
                if False:
                    return 10
                self.b += 2
                return self.b

            @torch.jit.export
            def getb(self, x):
                if False:
                    print('Hello World!')
                return self.b

        class OrigMod(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.a = 0

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):

            def forward(self, x: Tensor) -> int:
                if False:
                    for i in range(10):
                        print('nop')
                pass

        class TestModule(torch.nn.Module):
            proxy_mod: ModInterface

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.proxy_mod(x) + self.sub.getb(x)
        m = torch.jit.script(TestModule())
        m.proxy_mod = m.sub
        m.eval()
        mf = torch._C._freeze_module(m._c, freezeInterfaces=True)

    def test_freeze_module_with_inplace_mutation_in_interface(self):
        if False:
            return 10

        class SubModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.b = torch.tensor([1.5])

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                self.b[0] += 2
                return self.b

            @torch.jit.export
            def getb(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.b

        class OrigMod(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.a = torch.tensor([0.5])

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):

            def forward(self, x: Tensor) -> Tensor:
                if False:
                    return 10
                pass

        class TestModule(torch.nn.Module):
            proxy_mod: ModInterface

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                y = self.proxy_mod(x)
                z = self.sub.getb(x)
                return y[0] + z[0]
        m = torch.jit.script(TestModule())
        m.proxy_mod = m.sub
        m.sub.b = m.proxy_mod.b
        m.eval()
        mf = torch._C._freeze_module(m._c, freezeInterfaces=True)

    def test_freeze_module_with_mutated_interface(self):
        if False:
            return 10

        class SubModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.b = torch.tensor([1.5])

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.b

            @torch.jit.export
            def getb(self, x):
                if False:
                    print('Hello World!')
                return self.b

        class OrigMod(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.a = torch.tensor([0.5])

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):

            def forward(self, x: Tensor) -> Tensor:
                if False:
                    return 10
                pass

        class TestModule(torch.nn.Module):
            proxy_mod: ModInterface

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()

            def forward(self, x):
                if False:
                    return 10
                self.proxy_mod = self.sub
                y = self.proxy_mod(x)
                z = self.sub.getb(x)
                return y[0] + z[0]
        m = torch.jit.script(TestModule())
        m.eval()
        with self.assertRaisesRegex(RuntimeError, 'Freezing does not support SetAttr on an interface type.'):
            mf = torch._C._freeze_module(m._c, freezeInterfaces=True)

    def test_freeze_module_with_interface_and_fork(self):
        if False:
            for i in range(10):
                print('nop')

        class SubModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.b = torch.tensor([1.5])

            def forward(self, x):
                if False:
                    print('Hello World!')
                self.b[0] += 3.2
                return self.b

        class OrigMod(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.a = torch.tensor([0.5])

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):

            def forward(self, x: Tensor) -> Tensor:
                if False:
                    i = 10
                    return i + 15
                pass

        class TestModule(torch.nn.Module):
            proxy_mod: ModInterface

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                y = self.proxy_mod(x)
                z = self.sub(x)
                return y + z

        class MainModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.test = TestModule()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                fut = torch.jit._fork(self.test.forward, x)
                y = self.test(x)
                z = torch.jit._wait(fut)
                return y + z
        m = torch.jit.script(MainModule())
        m.eval()
        mf = torch._C._freeze_module(m._c, freezeInterfaces=True)

    def test_module_apis_interface(self):
        if False:
            while True:
                i = 10

        @torch.jit.interface
        class ModuleInterface(nn.Module):

            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                if False:
                    while True:
                        i = 10
                pass

        class TestModule(nn.Module):
            proxy_mod: ModuleInterface

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return input * 2

            @torch.jit.export
            def method(self, input):
                if False:
                    i = 10
                    return i + 15
                for module in self.modules():
                    input = module(input)
                return input
        with self.assertRaisesRegex(Exception, 'Could not compile'):
            scripted_mod = torch.jit.script(TestModule())