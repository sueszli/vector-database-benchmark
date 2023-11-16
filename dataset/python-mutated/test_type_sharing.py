import os
import sys
import io
import torch
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_utils import suppress_warnings
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestTypeSharing(JitTestCase):

    def assertSameType(self, m1, m2):
        if False:
            i = 10
            return i + 15
        if not isinstance(m1, torch.jit.ScriptModule):
            m1 = torch.jit.script(m1)
        if not isinstance(m2, torch.jit.ScriptModule):
            m2 = torch.jit.script(m2)
        self.assertEqual(m1._c._type(), m2._c._type())

    def assertDifferentType(self, m1, m2):
        if False:
            while True:
                i = 10
        if not isinstance(m1, torch.jit.ScriptModule):
            m1 = torch.jit.script(m1)
        if not isinstance(m2, torch.jit.ScriptModule):
            m2 = torch.jit.script(m2)
        self.assertNotEqual(m1._c._type(), m2._c._type())

    def test_basic(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self, a, b, c):
                if False:
                    print('Hello World!')
                super().__init__()
                self.a = a
                self.b = b
                self.c = c

            def forward(self, x):
                if False:
                    return 10
                return x
        a = torch.rand(2, 3)
        b = torch.rand(2, 3)
        c = torch.rand(2, 3)
        m1 = M(a, b, c)
        m2 = M(a, b, c)
        self.assertSameType(m1, m2)

    def test_diff_attr_values(self):
        if False:
            print('Hello World!')
        '\n        Types should be shared even if attribute values differ\n        '

        class M(torch.nn.Module):

            def __init__(self, a, b, c):
                if False:
                    print('Hello World!')
                super().__init__()
                self.a = a
                self.b = b
                self.c = c

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x
        a = torch.rand(2, 3)
        b = torch.rand(2, 3)
        c = torch.rand(2, 3)
        m1 = M(a, b, c)
        m2 = M(a * 2, b * 3, c * 4)
        self.assertSameType(m1, m2)

    def test_constants(self):
        if False:
            return 10
        '\n        Types should be shared for identical constant values, and different for different constant values\n        '

        class M(torch.nn.Module):
            __constants__ = ['const']

            def __init__(self, attr, const):
                if False:
                    return 10
                super().__init__()
                self.attr = attr
                self.const = const

            def forward(self):
                if False:
                    i = 10
                    return i + 15
                return self.const
        attr = torch.rand(2, 3)
        m1 = M(attr, 1)
        m2 = M(attr, 1)
        self.assertSameType(m1, m2)
        m3 = M(attr, 2)
        self.assertDifferentType(m1, m3)

    def test_linear(self):
        if False:
            i = 10
            return i + 15
        '\n        Simple example with a real nn Module\n        '
        a = torch.nn.Linear(5, 5)
        b = torch.nn.Linear(5, 5)
        c = torch.nn.Linear(10, 10)
        a = torch.jit.script(a)
        b = torch.jit.script(b)
        c = torch.jit.script(c)
        self.assertSameType(a, b)
        self.assertDifferentType(a, c)

    def test_submodules(self):
        if False:
            print('Hello World!')
        '\n        If submodules differ, the types should differ.\n        '

        class M(torch.nn.Module):

            def __init__(self, in1, out1, in2, out2):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.submod1(x)
                x = self.submod2(x)
                return x
        a = M(1, 1, 2, 2)
        b = M(1, 1, 2, 2)
        self.assertSameType(a, b)
        self.assertSameType(a.submod1, b.submod1)
        c = M(2, 2, 2, 2)
        self.assertDifferentType(a, c)
        self.assertSameType(b.submod2, c.submod1)
        self.assertDifferentType(a.submod1, b.submod2)

    def test_param_vs_attribute(self):
        if False:
            return 10
        "\n        The same module with an `foo` as a parameter vs. attribute shouldn't\n        share types\n        "

        class M(torch.nn.Module):

            def __init__(self, foo):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.foo = foo

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x + self.foo
        as_param = torch.nn.Parameter(torch.ones(2, 2))
        as_attr = torch.ones(2, 2)
        param_mod = M(as_param)
        attr_mod = M(as_attr)
        self.assertDifferentType(attr_mod, param_mod)

    def test_same_but_different_classes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Even if everything about the module is the same, different originating\n        classes should prevent type sharing.\n        '

        class A(torch.nn.Module):
            __constants__ = ['const']

            def __init__(self, in1, out1, in2, out2):
                if False:
                    return 10
                super().__init__()
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)
                self.const = 5

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.submod1(x)
                x = self.submod2(x)
                return x * self.const

        class B(torch.nn.Module):
            __constants__ = ['const']

            def __init__(self, in1, out1, in2, out2):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)
                self.const = 5

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.submod1(x)
                x = self.submod2(x)
                return x * self.const
        a = A(1, 1, 2, 2)
        b = B(1, 1, 2, 2)
        self.assertDifferentType(a, b)

    def test_mutate_attr_value(self):
        if False:
            i = 10
            return i + 15
        '\n        Mutating the value of an attribute should not change type sharing\n        '

        class M(torch.nn.Module):

            def __init__(self, in1, out1, in2, out2):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)
                self.foo = torch.ones(in1, in1)

            def forward(self, x):
                if False:
                    return 10
                x = self.submod1(x)
                x = self.submod2(x)
                return x + self.foo
        a = M(1, 1, 2, 2)
        b = M(1, 1, 2, 2)
        a.foo = torch.ones(2, 2)
        b.foo = torch.rand(2, 2)
        self.assertSameType(a, b)

    def test_assign_python_attr(self):
        if False:
            print('Hello World!')
        '\n        Assigning a new (python-only) attribute should not change type sharing\n        '

        class M(torch.nn.Module):

            def __init__(self, in1, out1, in2, out2):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)
                self.foo = torch.ones(in1, in1)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.submod1(x)
                x = self.submod2(x)
                return x + self.foo
        a = torch.jit.script(M(1, 1, 2, 2))
        b = torch.jit.script(M(1, 1, 2, 2))
        a.new_attr = 'foo bar baz'
        self.assertSameType(a, b)
        a = M(1, 1, 2, 2)
        b = M(1, 1, 2, 2)
        a.new_attr = 'foo bar baz'
        self.assertDifferentType(a, b)

    def test_failed_attribute_compilation(self):
        if False:
            return 10
        '\n        Attributes whose type cannot be inferred should fail cleanly with nice hints\n        '

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.foo = object

            def forward(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.foo
        m = M()
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'failed to convert Python type', 'self.foo'):
            torch.jit.script(m)

    def test_script_function_attribute_different(self):
        if False:
            while True:
                i = 10
        '\n        Different functions passed in should lead to different types\n        '

        @torch.jit.script
        def fn1(x):
            if False:
                print('Hello World!')
            return x + x

        @torch.jit.script
        def fn2(x):
            if False:
                i = 10
                return i + 15
            return x - x

        class M(torch.nn.Module):

            def __init__(self, fn):
                if False:
                    print('Hello World!')
                super().__init__()
                self.fn = fn

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.fn(x)
        fn1_mod = M(fn1)
        fn2_mod = M(fn2)
        self.assertDifferentType(fn1_mod, fn2_mod)

    def test_builtin_function_same(self):
        if False:
            i = 10
            return i + 15

        class Caller(torch.nn.Module):

            def __init__(self, fn):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.fn = fn

            def forward(self, input):
                if False:
                    print('Hello World!')
                return self.fn(input, input)
        c1 = Caller(torch.add)
        c2 = Caller(torch.add)
        self.assertSameType(c1, c2)

    def test_builtin_function_different(self):
        if False:
            i = 10
            return i + 15

        class Caller(torch.nn.Module):

            def __init__(self, fn):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.fn = fn

            def forward(self, input):
                if False:
                    return 10
                return self.fn(input, input)
        c1 = Caller(torch.add)
        c2 = Caller(torch.sub)
        self.assertDifferentType(c1, c2)

    def test_script_function_attribute_same(self):
        if False:
            print('Hello World!')
        '\n        Same functions passed in should lead to same types\n        '

        @torch.jit.script
        def fn(x):
            if False:
                print('Hello World!')
            return x + x

        class M(torch.nn.Module):

            def __init__(self, fn):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.fn = fn

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.fn(x)
        fn1_mod = M(fn)
        fn2_mod = M(fn)
        self.assertSameType(fn1_mod, fn2_mod)

    def test_python_function_attribute_different(self):
        if False:
            print('Hello World!')
        '\n        Different functions passed in should lead to different types\n        '

        def fn1(x):
            if False:
                while True:
                    i = 10
            return x + x

        def fn2(x):
            if False:
                i = 10
                return i + 15
            return x - x

        class M(torch.nn.Module):

            def __init__(self, fn):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.fn = fn

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.fn(x)
        fn1_mod = M(fn1)
        fn2_mod = M(fn2)
        self.assertDifferentType(fn1_mod, fn2_mod)

    def test_python_function_attribute_same(self):
        if False:
            print('Hello World!')
        '\n        Same functions passed in should lead to same types\n        '

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + x

        class M(torch.nn.Module):

            def __init__(self, fn):
                if False:
                    print('Hello World!')
                super().__init__()
                self.fn = fn

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.fn(x)
        fn1_mod = M(fn)
        fn2_mod = M(fn)
        self.assertSameType(fn1_mod, fn2_mod)

    @suppress_warnings
    def test_tracing_gives_different_types(self):
        if False:
            while True:
                i = 10
        "\n        Since we can't guarantee that methods are the same between different\n        trace runs, tracing must always generate a unique type.\n        "

        class M(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                if x.sum() > y.sum():
                    return x
                else:
                    return y
        a = torch.jit.trace(M(), (torch.zeros(1, 1), torch.ones(1, 1)))
        b = torch.jit.trace(M(), (torch.ones(1, 1), torch.zeros(1, 1)))
        self.assertDifferentType(a, b)

    def test_ignored_fns(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self, foo):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.foo = foo

            @torch.jit.ignore
            def ignored(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.foo

            def forward(self):
                if False:
                    i = 10
                    return i + 15
                return self.ignored()
        a = torch.jit.script(M(torch.ones(1)))
        b = torch.jit.script(M(torch.ones(2)))
        self.assertSameType(a, b)
        self.assertNotEqual(a(), b())

    @suppress_warnings
    def test_script_module_containing_traced_module(self):
        if False:
            while True:
                i = 10

        class Traced(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                if x.sum() > 0:
                    return x
                else:
                    return x + x

        class M(torch.nn.Module):

            def __init__(self, input):
                if False:
                    print('Hello World!')
                super().__init__()
                self.traced = torch.jit.trace(Traced(), input)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.traced(x)
        a = M((torch.ones(1),))
        b = M((torch.zeros(1),))
        self.assertDifferentType(a, b)

    def test_loaded_modules_work(self):
        if False:
            i = 10
            return i + 15

        class AB(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.a = 1
                self.b = 1

            def forward(self):
                if False:
                    i = 10
                    return i + 15
                return self.a + self.b

        class A(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.a = 1

            def forward(self):
                if False:
                    return 10
                return self.a

        class Wrapper(torch.nn.Module):

            def __init__(self, sub):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.sub = sub

            def forward(self):
                if False:
                    while True:
                        i = 10
                return self.sub()

        def package(x):
            if False:
                while True:
                    i = 10
            buffer = io.BytesIO()
            torch.jit.save(torch.jit.script(x), buffer)
            buffer.seek(0)
            return torch.jit.script(Wrapper(torch.jit.load(buffer)))
        a = package(AB())
        a()
        b = package(A())
        b()

    def test_module_dict_same_type_different_name(self):
        if False:
            print('Hello World!')
        '\n        We should be able to differentiate between two ModuleDict instances\n        that have different keys but the same value types.\n        '

        class A(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x

        class Foo(torch.nn.Module):

            def __init__(self, s):
                if False:
                    print('Hello World!')
                super().__init__()
                self.dict = torch.nn.ModuleDict(s)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x
        a = Foo({'foo': A()})
        b = Foo({'bar': A()})
        c = Foo({'bar': A()})
        self.assertDifferentType(a, b)
        self.assertSameType(b, c)

    def test_type_sharing_define_in_init(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that types between instances of a ScriptModule\n        subclass that defines methods in its __init__ are not\n        shared.\n        '

        class A(torch.jit.ScriptModule):

            def __init__(self, val):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.define(f'\n                def forward(self) -> int:\n                    return {val}\n                ')
        one = A(1)
        two = A(2)
        self.assertEqual(one(), 1)
        self.assertEqual(two(), 2)

    def test_type_sharing_disabled(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that type sharing can be disabled.\n        '

        class A(torch.nn.Module):

            def __init__(self, sub):
                if False:
                    return 10
                super().__init__()
                self.sub = sub

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x

        class B(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x
        top1 = A(A(B()))
        top2 = A(A(B()))
        top1_s = torch.jit._recursive.create_script_module(top1, torch.jit._recursive.infer_methods_to_compile, share_types=False)
        top2_s = torch.jit._recursive.create_script_module(top2, torch.jit._recursive.infer_methods_to_compile, share_types=False)
        self.assertDifferentType(top1_s, top2_s)
        self.assertDifferentType(top1_s, top1_s.sub)
        self.assertDifferentType(top1_s, top2_s.sub)
        self.assertDifferentType(top2_s, top2_s.sub)
        self.assertDifferentType(top2_s, top1_s.sub)

    def test_type_shared_ignored_attributes(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that types are shared if the exclusion of their\n        ignored attributes makes them equal.\n        '

        class A(torch.nn.Module):
            __jit_ignored_attributes__ = ['a']

            def __init__(self, a, b):
                if False:
                    print('Hello World!')
                super().__init__()
                self.a = a
                self.b = b

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x
        a_with_linear = A(torch.nn.Linear(5, 5), 5)
        a_with_string = A('string', 10)
        self.assertSameType(a_with_linear, a_with_string)

    def test_type_not_shared_ignored_attributes(self):
        if False:
            while True:
                i = 10
        '\n        Test that types are not shared if the exclusion of their\n        ignored attributes makes them not equal.\n        '

        class A(torch.nn.Module):
            __jit_ignored_attributes__ = ['a']

            def __init__(self, a, b, c):
                if False:
                    return 10
                super().__init__()
                self.a = a
                self.b = b
                self.c = c

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x
        mod = A(torch.nn.Linear(5, 5), 5, 'string')
        s1 = torch.jit.script(mod)
        A.__jit_ignored_attributes__ = ['a', 'b']
        s2 = torch.jit.script(mod)
        self.assertDifferentType(s1, s2)