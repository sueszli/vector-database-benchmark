import io
import os
import sys
import unittest
import torch
import torch.nn as nn
from torch.testing import FileCheck
from typing import Any
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, make_global
import torch.testing._internal.jit_utils
from torch.testing._internal.common_utils import IS_SANDCASTLE, skipIfTorchDynamo
from typing import List, Tuple, Iterable, Optional, Dict
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TESTNAME\n\ninstead.')

class TestClassType(JitTestCase):

    def test_reference_semantics(self):
        if False:
            print('Hello World!')
        '\n        Test that modifications made to a class instance in TorchScript\n        are visible in eager.\n        '

        class Foo:

            def __init__(self, a: int):
                if False:
                    i = 10
                    return i + 15
                self.a = a

            def set_a(self, value: int):
                if False:
                    i = 10
                    return i + 15
                self.a = value

            def get_a(self) -> int:
                if False:
                    return 10
                return self.a

            @property
            def attr(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.a
        make_global(Foo)

        def test_fn(obj: Foo):
            if False:
                return 10
            obj.set_a(2)
        scripted_fn = torch.jit.script(test_fn)
        obj = torch.jit.script(Foo(1))
        self.assertEqual(obj.get_a(), 1)
        self.assertEqual(obj.attr, 1)
        scripted_fn(obj)
        self.assertEqual(obj.get_a(), 2)
        self.assertEqual(obj.attr, 2)

    def test_get_with_method(self):
        if False:
            for i in range(10):
                print('nop')

        class FooTest:

            def __init__(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                self.foo = x

            def getFooTest(self):
                if False:
                    return 10
                return self.foo

        def fn(x):
            if False:
                return 10
            foo = FooTest(x)
            return foo.getFooTest()
        input = torch.ones(2, 3)
        self.assertEqual(fn(input), input)

    def test_get_attr(self):
        if False:
            for i in range(10):
                print('nop')

        class FooTest:

            def __init__(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                self.foo = x

        @torch.jit.script
        def fn(x):
            if False:
                while True:
                    i = 10
            foo = FooTest(x)
            return foo.foo
        input = torch.ones(2, 3)
        self.assertEqual(fn(input), input)

    def test_in(self):
        if False:
            while True:
                i = 10

        class FooTest:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                pass

            def __contains__(self, key: str) -> bool:
                if False:
                    i = 10
                    return i + 15
                return key == 'hi'

        @torch.jit.script
        def fn():
            if False:
                return 10
            foo = FooTest()
            return ('hi' in foo, 'no' in foo)
        self.assertEqual(fn(), (True, False))

    def test_set_attr_in_method(self):
        if False:
            for i in range(10):
                print('nop')

        class FooTest:

            def __init__(self, x: int) -> None:
                if False:
                    print('Hello World!')
                self.foo = x

            def incFooTest(self, y: int) -> None:
                if False:
                    i = 10
                    return i + 15
                self.foo = self.foo + y

        @torch.jit.script
        def fn(x: int) -> int:
            if False:
                while True:
                    i = 10
            foo = FooTest(x)
            foo.incFooTest(2)
            return foo.foo
        self.assertEqual(fn(1), 3)

    def test_set_attr_type_mismatch(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Wrong type for attribute assignment', 'self.foo = 10'):

            @torch.jit.script
            class FooTest:

                def __init__(self, x):
                    if False:
                        return 10
                    self.foo = x
                    self.foo = 10

    def test_get_attr_not_initialized(self):
        if False:
            return 10
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'object has no attribute or method', 'self.asdf'):

            @torch.jit.script
            class FooTest:

                def __init__(self, x):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.foo = x

                def get_non_initialized(self):
                    if False:
                        return 10
                    return self.asdf

    def test_set_attr_non_initialized(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Tried to set nonexistent attribute', 'self.bar = y'):

            @torch.jit.script
            class FooTest:

                def __init__(self, x):
                    if False:
                        i = 10
                        return i + 15
                    self.foo = x

                def set_non_initialized(self, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.bar = y

    def test_schema_human_readable(self):
        if False:
            while True:
                i = 10
        '\n        Make sure that the schema is human readable, ie the mode parameter should read "nearest" instead of being displayed in octal\n        aten::__interpolate(Tensor input, int? size=None, float[]? scale_factor=None,\n        str mode=\'nearest\', bool? align_corners=None) -> (Tensor):\n        Expected a value of type \'Optional[int]\' for argument \'size\' but instead found type \'Tensor\'.\n        '
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'nearest', ''):

            @torch.jit.script
            def FooTest(x):
                if False:
                    print('Hello World!')
                return torch.nn.functional.interpolate(x, 'bad')

    def test_type_annotations(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegexWithHighlight(RuntimeError, "Expected a value of type 'bool", ''):

            @torch.jit.script
            class FooTest:

                def __init__(self, x: bool) -> None:
                    if False:
                        print('Hello World!')
                    self.foo = x

            @torch.jit.script
            def fn(x):
                if False:
                    print('Hello World!')
                FooTest(x)
            fn(2)

    def test_conditional_set_attr(self):
        if False:
            return 10
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'assignment cannot be in a control-flow block', ''):

            @torch.jit.script
            class FooTest:

                def __init__(self, x):
                    if False:
                        for i in range(10):
                            print('nop')
                    if 1 == 1:
                        self.attr = x

    def test_class_type_as_param(self):
        if False:
            print('Hello World!')

        class FooTest:

            def __init__(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                self.attr = x
        make_global(FooTest)

        @torch.jit.script
        def fn(foo: FooTest) -> torch.Tensor:
            if False:
                return 10
            return foo.attr

        @torch.jit.script
        def fn2(x):
            if False:
                while True:
                    i = 10
            foo = FooTest(x)
            return fn(foo)
        input = torch.ones(1)
        self.assertEqual(fn2(input), input)

    def test_out_of_order_methods(self):
        if False:
            while True:
                i = 10

        class FooTest:

            def __init__(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                self.x = x
                self.x = self.get_stuff(x)

            def get_stuff(self, y):
                if False:
                    return 10
                return self.x + y

        @torch.jit.script
        def fn(x):
            if False:
                while True:
                    i = 10
            f = FooTest(x)
            return f.x
        input = torch.ones(1)
        self.assertEqual(fn(input), input + input)

    def test_save_load_with_classes(self):
        if False:
            for i in range(10):
                print('nop')

        class FooTest:

            def __init__(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                self.x = x

            def get_x(self):
                if False:
                    return 10
                return self.x

        class MyMod(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, a):
                if False:
                    while True:
                        i = 10
                foo = FooTest(a)
                return foo.get_x()
        m = MyMod()
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)
        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(input, output)

    def test_save_load_with_classes_returned(self):
        if False:
            return 10

        class FooTest:

            def __init__(self, x):
                if False:
                    print('Hello World!')
                self.x = x

            def clone(self):
                if False:
                    print('Hello World!')
                clone = FooTest(self.x)
                return clone

        class MyMod(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, a):
                if False:
                    for i in range(10):
                        print('nop')
                foo = FooTest(a)
                foo_clone = foo.clone()
                return foo_clone.x
        m = MyMod()
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        torch.testing._internal.jit_utils.clear_class_registry()
        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)
        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(input, output)

    def test_save_load_with_classes_nested(self):
        if False:
            while True:
                i = 10

        class FooNestedTest:

            def __init__(self, y):
                if False:
                    i = 10
                    return i + 15
                self.y = y

        class FooNestedTest2:

            def __init__(self, y):
                if False:
                    while True:
                        i = 10
                self.y = y
                self.nested = FooNestedTest(y)

        class FooTest:

            def __init__(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                self.class_attr = FooNestedTest(x)
                self.class_attr2 = FooNestedTest2(x)
                self.x = self.class_attr.y + self.class_attr2.y

        class MyMod(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, a):
                if False:
                    for i in range(10):
                        print('nop')
                foo = FooTest(a)
                return foo.x
        m = MyMod()
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        torch.testing._internal.jit_utils.clear_class_registry()
        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)
        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(2 * input, output)

    def test_python_interop(self):
        if False:
            while True:
                i = 10

        class Foo:

            def __init__(self, x, y):
                if False:
                    print('Hello World!')
                self.x = x
                self.y = y
        make_global(Foo)

        @torch.jit.script
        def use_foo(foo: Foo) -> Foo:
            if False:
                print('Hello World!')
            return foo
        x = torch.ones(2, 3)
        y = torch.zeros(2, 3)
        f = Foo(x, y)
        self.assertEqual(x, f.x)
        self.assertEqual(y, f.y)
        f2 = use_foo(f)
        self.assertEqual(x, f2.x)
        self.assertEqual(y, f2.y)

    def test_class_specialization(self):
        if False:
            return 10

        class Foo:

            def __init__(self, x, y):
                if False:
                    while True:
                        i = 10
                self.x = x
                self.y = y
        make_global(Foo)

        def use_foo(foo: Foo, foo2: Foo, tup: Tuple[Foo, Foo]) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            (a, b) = tup
            return foo.x + foo2.y + a.x + b.y
        x = torch.ones(2, 3)
        y = torch.zeros(2, 3)
        f = Foo(x, y)
        f2 = Foo(x * 2, y * 3)
        f3 = Foo(x * 4, y * 4)
        input = (f, f2, (f, f3))
        sfoo = self.checkScript(use_foo, input)
        graphstr = str(sfoo.graph_for(*input))
        FileCheck().check_count('prim::GetAttr', 4).run(graphstr)

    def test_class_sorting(self):
        if False:
            i = 10
            return i + 15

        class Foo:

            def __init__(self, x: int) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                self.x = x

            def __lt__(self, other) -> bool:
                if False:
                    for i in range(10):
                        print('nop')
                return self.x < other.x

            def getVal(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.x
        make_global(Foo)

        def test(li: List[Foo], reverse: bool=False) -> Tuple[List[int], List[int]]:
            if False:
                return 10
            li_sorted = sorted(li)
            ret_sorted = torch.jit.annotate(List[int], [])
            for foo in li_sorted:
                ret_sorted.append(foo.getVal())
            li.sort(reverse=reverse)
            ret_sort = torch.jit.annotate(List[int], [])
            for foo in li:
                ret_sort.append(foo.getVal())
            return (ret_sorted, ret_sort)
        self.checkScript(test, ([Foo(2), Foo(1), Foo(3)],))
        self.checkScript(test, ([Foo(2), Foo(1), Foo(3)], True))
        self.checkScript(test, ([Foo(2)],))
        self.checkScript(test, ([],))

        @torch.jit.script
        def test_list_no_reverse():
            if False:
                print('Hello World!')
            li = [Foo(3), Foo(1)]
            li.sort()
            return li[0].getVal()
        self.assertEqual(test_list_no_reverse(), 1)

        @torch.jit.script
        def test_sorted_copies():
            if False:
                while True:
                    i = 10
            li = [Foo(3), Foo(1)]
            li_sorted = sorted(li)
            return (li[0].getVal(), li_sorted[0].getVal())
        self.assertEqual(test_sorted_copies(), (3, 1))

        @torch.jit.script
        def test_nested_inside_tuple():
            if False:
                for i in range(10):
                    print('nop')
            li = [(1, Foo(12)), (1, Foo(11))]
            li.sort()
            return [(li[0][0], li[0][1].getVal()), (li[1][0], li[1][1].getVal())]
        self.assertEqual(test_nested_inside_tuple(), [(1, 11), (1, 12)])
        with self.assertRaisesRegexWithHighlight(RuntimeError, "bool' for argument 'reverse", ''):

            @torch.jit.script
            def test():
                if False:
                    i = 10
                    return i + 15
                li = [Foo(1)]
                li.sort(li)
                return li
            test()
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'must define a __lt__', ''):

            @torch.jit.script
            class NoMethod:

                def __init__(self):
                    if False:
                        i = 10
                        return i + 15
                    pass

            @torch.jit.script
            def test():
                if False:
                    return 10
                li = [NoMethod(), NoMethod()]
                li.sort()
                return li
            test()

        @torch.jit.script
        class WrongLt:

            def __init__(self):
                if False:
                    return 10
                pass

            def __lt__(self, other):
                if False:
                    return 10
                pass
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'must define a __lt__', ''):

            @torch.jit.script
            def test():
                if False:
                    return 10
                li = [WrongLt(), WrongLt()]
                li.sort()
                return li
            test()

    def test_class_inheritance(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        class Base:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.b = 2

            def two(self, x):
                if False:
                    while True:
                        i = 10
                return x + self.b
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'does not support inheritance', ''):

            @torch.jit.script
            class Derived(Base):

                def two(self, x):
                    if False:
                        while True:
                            i = 10
                    return x + self.b + 2

    def test_class_inheritance_implicit(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that inheritance is detected in\n        implicit scripting codepaths (e.g. try_ann_to_type).\n        '

        class A:

            def __init__(self, t):
                if False:
                    i = 10
                    return i + 15
                self.t = t

            @staticmethod
            def f(a: torch.Tensor):
                if False:
                    return 10
                return A(a + 1)

        class B(A):

            def __init__(self, t):
                if False:
                    print('Hello World!')
                self.t = t + 10

            @staticmethod
            def f(a: torch.Tensor):
                if False:
                    for i in range(10):
                        print('nop')
                return A(a + 1)
        x = A(torch.tensor([3]))

        def fun(x: Any):
            if False:
                return 10
            if isinstance(x, A):
                return A.f(x.t)
            else:
                return B.f(x.t)
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'object has no attribute or method', ''):
            sc = torch.jit.script(fun)

    @skipIfTorchDynamo('Test does not work with TorchDynamo')
    @unittest.skipIf(IS_SANDCASTLE, "Importing like this doesn't work in fbcode")
    def test_imported_classes(self):
        if False:
            return 10
        import jit._imported_class_test.foo
        import jit._imported_class_test.bar
        import jit._imported_class_test.very.very.nested

        class MyMod(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, a):
                if False:
                    for i in range(10):
                        print('nop')
                foo = jit._imported_class_test.foo.FooSameName(a)
                bar = jit._imported_class_test.bar.FooSameName(a)
                three = jit._imported_class_test.very.very.nested.FooUniqueName(a)
                return foo.x + bar.y + three.y
        m = MyMod()
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        torch.testing._internal.jit_utils.clear_class_registry()
        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)
        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(3 * input, output)

    def test_interface(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        class Foo:

            def __init__(self):
                if False:
                    return 10
                pass

            def one(self, x, y):
                if False:
                    print('Hello World!')
                return x + y

            def two(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return 2 * x

        @torch.jit.script
        class Bar:

            def __init__(self):
                if False:
                    return 10
                pass

            def one(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return x * y

            def two(self, x):
                if False:
                    print('Hello World!')
                return 2 / x

        @torch.jit.interface
        class OneTwo:

            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if False:
                    i = 10
                    return i + 15
                pass

            def two(self, x: torch.Tensor) -> torch.Tensor:
                if False:
                    i = 10
                    return i + 15
                pass

        @torch.jit.interface
        class OneTwoThree:

            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if False:
                    return 10
                pass

            def two(self, x: torch.Tensor) -> torch.Tensor:
                if False:
                    i = 10
                    return i + 15
                pass

            def three(self, x: torch.Tensor) -> torch.Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                pass

        @torch.jit.interface
        class OneTwoWrong:

            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if False:
                    return 10
                pass

            def two(self, x: int) -> int:
                if False:
                    i = 10
                    return i + 15
                pass

        @torch.jit.script
        class NotMember:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                pass

            def one(self, x, y):
                if False:
                    print('Hello World!')
                return x + y

        @torch.jit.script
        class NotMember2:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                pass

            def one(self, x, y):
                if False:
                    while True:
                        i = 10
                return x + y

            def two(self, x: int) -> int:
                if False:
                    for i in range(10):
                        print('nop')
                return 3
        make_global(Foo, Bar, OneTwo, OneTwoThree, OneTwoWrong, NotMember, NotMember2)

        def use_them(x):
            if False:
                return 10
            a = Foo()
            b = Bar()
            c = torch.jit.annotate(List[OneTwo], [a, b])
            for i in range(len(c)):
                x = c[i].one(x, x)
                x = c[i].two(x)
            return x
        self.checkScript(use_them, (torch.rand(3, 4),))

        @torch.jit.script
        def as_interface(x: OneTwo) -> OneTwo:
            if False:
                return 10
            return x

        @torch.jit.script
        def inherit(x: OneTwoThree) -> OneTwo:
            if False:
                i = 10
                return i + 15
            return as_interface(x)
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'does not have method', ''):

            @torch.jit.script
            def wrong1():
                if False:
                    while True:
                        i = 10
                return as_interface(NotMember())
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'is not compatible with interface', ''):

            @torch.jit.script
            def wrong2():
                if False:
                    for i in range(10):
                        print('nop')
                return as_interface(NotMember2())
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'does not have method', ''):

            @torch.jit.script
            def wrong3():
                if False:
                    while True:
                        i = 10
                return inherit(as_interface(Foo()))
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'is not compatible with interface', ''):

            @torch.jit.script
            def wrong4(x: OneTwoWrong) -> int:
                if False:
                    i = 10
                    return i + 15
                return as_interface(x)

        class TestPyAssign(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.proxy_mod = Foo()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.proxy_mod.two(x)
        TestPyAssign.__annotations__ = {'proxy_mod': OneTwo}
        input = torch.rand(3, 4)
        scripted_pyassign_mod = torch.jit.script(TestPyAssign())
        imported_mod = self.getExportImportCopy(scripted_pyassign_mod)
        self.assertEqual(scripted_pyassign_mod(input), imported_mod(input))

        class TestPyAssignError(nn.Module):

            def __init__(self, obj):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.proxy_mod = obj

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.proxy_mod.two(x)
        TestPyAssignError.__annotations__ = {'proxy_mod': OneTwoThree}
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'is not compatible with interface __torch__', ''):
            torch.jit.script(TestPyAssignError(Foo()))

        class PyClass:

            def __init__(self):
                if False:
                    print('Hello World!')
                pass
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'the value is not a TorchScript compatible type', ''):
            torch.jit.script(TestPyAssignError(PyClass()))

    def test_overloaded_fn(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        class Foo:

            def __init__(self, x):
                if False:
                    return 10
                self.x = x

            def __len__(self) -> int:
                if False:
                    while True:
                        i = 10
                return len(self.x)

            def __neg__(self):
                if False:
                    return 10
                self.x = -self.x
                return self

            def __mul__(self, other: torch.Tensor) -> torch.Tensor:
                if False:
                    return 10
                return self.x * other

        def test_overload():
            if False:
                print('Hello World!')
            a = Foo(torch.ones([3, 3]))
            return (len(a), -a * torch.zeros([3, 3]))
        make_global(Foo)
        self.checkScript(test_overload, ())

        @torch.jit.script
        class MyClass:

            def __init__(self, x: int) -> None:
                if False:
                    while True:
                        i = 10
                self.x = x

            def __add__(self, other: int) -> int:
                if False:
                    i = 10
                    return i + 15
                return self.x + other

            def __sub__(self, other: int) -> int:
                if False:
                    return 10
                return self.x - other

            def __mul__(self, other: int) -> int:
                if False:
                    for i in range(10):
                        print('nop')
                return self.x * other

            def __pow__(self, other: int) -> int:
                if False:
                    return 10
                return int(self.x ** other)

            def __truediv__(self, other: int) -> float:
                if False:
                    i = 10
                    return i + 15
                return self.x / other

            def __mod__(self, other: int) -> int:
                if False:
                    print('Hello World!')
                return self.x % other

            def __ne__(self, other: int) -> bool:
                if False:
                    while True:
                        i = 10
                return self.x != other

            def __eq__(self, other: int) -> bool:
                if False:
                    i = 10
                    return i + 15
                return self.x == other

            def __lt__(self, other: int) -> bool:
                if False:
                    while True:
                        i = 10
                return self.x < other

            def __gt__(self, other: int) -> bool:
                if False:
                    for i in range(10):
                        print('nop')
                return self.x > other

            def __le__(self, other: int) -> bool:
                if False:
                    print('Hello World!')
                return self.x <= other

            def __ge__(self, other: int) -> bool:
                if False:
                    print('Hello World!')
                return self.x >= other

            def __and__(self, other: int) -> int:
                if False:
                    while True:
                        i = 10
                return self.x & other

            def __or__(self, other: int) -> int:
                if False:
                    i = 10
                    return i + 15
                return self.x | other

            def __xor__(self, other: int) -> int:
                if False:
                    while True:
                        i = 10
                return self.x ^ other

            def __getitem__(self, other: int) -> int:
                if False:
                    return 10
                return other + 1

            def __setitem__(self, idx: int, val: int) -> None:
                if False:
                    i = 10
                    return i + 15
                self.x = val * idx

            def __call__(self, val: int) -> int:
                if False:
                    return 10
                return self.x * val * 3
        make_global(Foo)

        def add():
            if False:
                return 10
            return MyClass(4) + 3

        def sub():
            if False:
                for i in range(10):
                    print('nop')
            return MyClass(4) - 3

        def mul():
            if False:
                return 10
            return MyClass(4) * 3

        def pow():
            if False:
                i = 10
                return i + 15
            return MyClass(4) ** 3

        def truediv():
            if False:
                while True:
                    i = 10
            return MyClass(4) / 3

        def ne():
            if False:
                for i in range(10):
                    print('nop')
            return MyClass(4) != 3

        def eq():
            if False:
                i = 10
                return i + 15
            return MyClass(4) == 3

        def lt():
            if False:
                i = 10
                return i + 15
            return MyClass(4) < 3

        def gt():
            if False:
                while True:
                    i = 10
            return MyClass(4) > 3

        def le():
            if False:
                while True:
                    i = 10
            return MyClass(4) <= 3

        def ge():
            if False:
                print('Hello World!')
            return MyClass(4) >= 3

        def _and():
            if False:
                return 10
            return MyClass(4) & 3

        def _or():
            if False:
                i = 10
                return i + 15
            return MyClass(4) | 3

        def _xor():
            if False:
                i = 10
                return i + 15
            return MyClass(4) ^ 3

        def getitem():
            if False:
                while True:
                    i = 10
            return MyClass(4)[1]

        def setitem():
            if False:
                i = 10
                return i + 15
            a = MyClass(4)
            a[1] = 5
            return a.x

        def call():
            if False:
                while True:
                    i = 10
            a = MyClass(5)
            return a(2)
        ops = [add, sub, mul, pow, ne, eq, lt, gt, le, ge, _and, _or, _xor, getitem, setitem, call]
        ops.append(truediv)
        for func in ops:
            self.checkScript(func, ())
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'object has no attribute or method', ''):

            @torch.jit.script
            def test():
                if False:
                    i = 10
                    return i + 15
                return Foo(torch.tensor(1)) + Foo(torch.tensor(1))

    def test_cast_overloads(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        class Foo:

            def __init__(self, val: float) -> None:
                if False:
                    i = 10
                    return i + 15
                self.val = val

            def __int__(self):
                if False:
                    return 10
                return int(self.val)

            def __float__(self):
                if False:
                    print('Hello World!')
                return self.val

            def __bool__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return bool(self.val)

            def __str__(self):
                if False:
                    print('Hello World!')
                return str(self.val)
        make_global(Foo)

        def test(foo: Foo) -> Tuple[int, float, bool]:
            if False:
                while True:
                    i = 10
            if foo:
                pass
            return (int(foo), float(foo), bool(foo))
        fn = torch.jit.script(test)
        self.assertEqual(fn(Foo(0.5)), test(0.5))
        self.assertEqual(fn(Foo(0.0)), test(0.0))
        self.assertTrue('0.5' in str(Foo(0.5)))
        self.assertTrue('0.' in str(Foo(0.0)))

        @torch.jit.script
        class BadBool:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                pass

            def __bool__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return (1, 2)
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'expected a bool expression for condition', ''):

            @torch.jit.script
            def test():
                if False:
                    return 10
                if BadBool():
                    print(1)
                    pass

    def test_init_compiled_first(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        class Foo:

            def __before_init__(self):
                if False:
                    print('Hello World!')
                return self.x

            def __init__(self, x, y):
                if False:
                    while True:
                        i = 10
                self.x = x
                self.y = y

    def test_class_constructs_itself(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        class LSTMStateStack:

            def __init__(self, num_layers: int, hidden_size: int) -> None:
                if False:
                    print('Hello World!')
                self.num_layers = num_layers
                self.hidden_size = hidden_size
                self.last_state = (torch.zeros(num_layers, 1, hidden_size), torch.zeros(num_layers, 1, hidden_size))
                self.stack = [(self.last_state[0][-1], self.last_state[0][-1])]

            def copy(self):
                if False:
                    return 10
                other = LSTMStateStack(self.num_layers, self.hidden_size)
                other.stack = list(self.stack)
                return other

    def test_optional_type_promotion(self):
        if False:
            return 10

        @torch.jit.script
        class Leaf:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.x = 1

        @torch.jit.script
        class Tree:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.child = torch.jit.annotate(Optional[Leaf], None)

            def add_child(self, child: Leaf) -> None:
                if False:
                    print('Hello World!')
                self.child = child

    def test_recursive_class(self):
        if False:
            while True:
                i = 10
        '\n        Recursive class types not yet supported. We should give a good error message.\n        '
        with self.assertRaises(RuntimeError):

            @torch.jit.script
            class Tree:

                def __init__(self):
                    if False:
                        print('Hello World!')
                    self.parent = torch.jit.annotate(Optional[Tree], None)

    def test_class_constant(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):
            __constants__ = ['w']

            def __init__(self, w):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.w = w

            def forward(self, x):
                if False:
                    return 10
                y = self.w
                return (x, y)
        for c in (2, 1.0, None, True, 'str', (2, 3), [5.9, 7.3]):
            m = torch.jit.script(M(c))
            buffer = io.BytesIO()
            torch.jit.save(m, buffer)
            buffer.seek(0)
            m_loaded = torch.jit.load(buffer)
            input = torch.rand(2, 3)
            self.assertEqual(m(input), m_loaded(input))
            self.assertEqual(m.w, m_loaded.w)

    def test_py_class_to_ivalue_missing_attribute(self):
        if False:
            print('Hello World!')

        class Foo:
            i: int
            f: float

            def __init__(self, i: int, f: float):
                if False:
                    for i in range(10):
                        print('nop')
                self.i = i
                self.f = f
        make_global(Foo)

        @torch.jit.script
        def test_fn(x: Foo) -> float:
            if False:
                print('Hello World!')
            return x.i + x.f
        test_fn(Foo(3, 4.0))
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'missing attribute i', ''):
            test_fn(torch.rand(3, 4))

    def test_unused_method(self):
        if False:
            i = 10
            return i + 15
        '\n        Test unused methods on scripted classes.\n        '

        @torch.jit.script
        class Unused:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.count: int = 0
                self.items: List[int] = []

            def used(self):
                if False:
                    print('Hello World!')
                self.count += 1
                return self.count

            @torch.jit.unused
            def unused(self, x: int, y: Iterable[int], **kwargs) -> int:
                if False:
                    return 10
                a = next(self.items)
                return a

            def uses_unused(self) -> int:
                if False:
                    i = 10
                    return i + 15
                return self.unused(y='hi', x=3)

        class ModuleWithUnused(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.obj = Unused()

            def forward(self):
                if False:
                    while True:
                        i = 10
                return self.obj.used()

            @torch.jit.export
            def calls_unused(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.obj.unused(3, 'hi')

            @torch.jit.export
            def calls_unused_indirectly(self):
                if False:
                    while True:
                        i = 10
                return self.obj.uses_unused()
        python_module = ModuleWithUnused()
        script_module = torch.jit.script(ModuleWithUnused())
        self.assertEqual(python_module.forward(), script_module.forward())
        with self.assertRaises(torch.jit.Error):
            script_module.calls_unused()
        with self.assertRaises(torch.jit.Error):
            script_module.calls_unused_indirectly()

    def test_self_referential_method(self):
        if False:
            return 10
        '\n        Test that a scripted class can have a method that refers to the class itself\n        in its type annotations.\n        '

        @torch.jit.script
        class Meta:

            def __init__(self, a: int):
                if False:
                    for i in range(10):
                        print('nop')
                self.a = a

            def method(self, other: List['Meta']) -> 'Meta':
                if False:
                    for i in range(10):
                        print('nop')
                return Meta(len(other))

        class ModuleWithMeta(torch.nn.Module):

            def __init__(self, a: int):
                if False:
                    return 10
                super().__init__()
                self.meta = Meta(a)

            def forward(self):
                if False:
                    print('Hello World!')
                new_obj = self.meta.method([self.meta])
                return new_obj.a
        self.checkModule(ModuleWithMeta(5), ())

    def test_type_annotation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that annotating container attributes with types works correctly\n        '

        @torch.jit.script
        class CompetitiveLinkingTokenReplacementUtils:

            def __init__(self):
                if False:
                    return 10
                self.my_list: List[Tuple[float, int, int]] = []
                self.my_dict: Dict[int, int] = {}

        @torch.jit.script
        def foo():
            if False:
                i = 10
                return i + 15
            y = CompetitiveLinkingTokenReplacementUtils()
            new_dict: Dict[int, int] = {1: 1, 2: 2}
            y.my_dict = new_dict
            new_list: List[Tuple[float, int, int]] = [(1.0, 1, 1)]
            y.my_list = new_list
            return y

    def test_default_args(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that methods on class types can have default arguments.\n        '

        @torch.jit.script
        class ClassWithDefaultArgs:

            def __init__(self, a: int=1, b: Optional[List[int]]=None, c: Tuple[int, int, int]=(1, 2, 3), d: Optional[Dict[int, int]]=None, e: Optional[str]=None):
                if False:
                    i = 10
                    return i + 15
                self.int = a
                self.tup = c
                self.str = e
                self.list = [1, 2, 3]
                if b is not None:
                    self.list = b
                self.dict = {1: 2, 3: 4}
                if d is not None:
                    self.dict = d

            def add(self, b: int, scale: float=1.0) -> float:
                if False:
                    return 10
                return self.int * scale + b

        def all_defaults() -> int:
            if False:
                while True:
                    i = 10
            obj: ClassWithDefaultArgs = ClassWithDefaultArgs()
            return obj.int + obj.list[2] + obj.tup[1]

        def some_defaults() -> int:
            if False:
                i = 10
                return i + 15
            obj: ClassWithDefaultArgs = ClassWithDefaultArgs(b=[5, 6, 7])
            return obj.int + obj.list[2] + obj.dict[1]

        def override_defaults() -> int:
            if False:
                return 10
            obj: ClassWithDefaultArgs = ClassWithDefaultArgs(3, [9, 10, 11], (12, 13, 14), {3: 4}, 'str')
            s: int = obj.int
            for x in obj.list:
                s += x
            for y in obj.tup:
                s += y
            s += obj.dict[3]
            st = obj.str
            if st is not None:
                s += len(st)
            return s

        def method_defaults() -> float:
            if False:
                for i in range(10):
                    print('nop')
            obj: ClassWithDefaultArgs = ClassWithDefaultArgs()
            return obj.add(3) + obj.add(3, 0.25)
        self.checkScript(all_defaults, ())
        self.checkScript(some_defaults, ())
        self.checkScript(override_defaults, ())
        self.checkScript(method_defaults, ())

        class ClassWithSomeDefaultArgs:

            def __init__(self, a: int, b: int=1):
                if False:
                    for i in range(10):
                        print('nop')
                self.a = a
                self.b = b

        def default_b() -> int:
            if False:
                print('Hello World!')
            obj: ClassWithSomeDefaultArgs = ClassWithSomeDefaultArgs(1)
            return obj.a + obj.b

        def set_b() -> int:
            if False:
                print('Hello World!')
            obj: ClassWithSomeDefaultArgs = ClassWithSomeDefaultArgs(1, 4)
            return obj.a + obj.b
        self.checkScript(default_b, ())
        self.checkScript(set_b, ())

        class ClassWithMutableArgs:

            def __init__(self, a: List[int]=[1, 2, 3]):
                if False:
                    print('Hello World!')
                self.a = a

        def should_fail():
            if False:
                while True:
                    i = 10
            obj: ClassWithMutableArgs = ClassWithMutableArgs()
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Mutable default parameters are not supported', ''):
            torch.jit.script(should_fail)

    def test_staticmethod(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test static methods on class types.\n        '

        @torch.jit.script
        class ClassWithStaticMethod:

            def __init__(self, a: int, b: int):
                if False:
                    print('Hello World!')
                self.a: int = a
                self.b: int = b

            def get_a(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.a

            def get_b(self):
                if False:
                    i = 10
                    return i + 15
                return self.b

            def __eq__(self, other: 'ClassWithStaticMethod'):
                if False:
                    return 10
                return self.a == other.a and self.b == other.b

            @staticmethod
            def create(args: List['ClassWithStaticMethod']) -> 'ClassWithStaticMethod':
                if False:
                    while True:
                        i = 10
                return ClassWithStaticMethod(args[0].a, args[0].b)

            @staticmethod
            def create_from(a: int, b: int) -> 'ClassWithStaticMethod':
                if False:
                    while True:
                        i = 10
                a = ClassWithStaticMethod(a, b)
                return ClassWithStaticMethod.create([a])

        def test_function(a: int, b: int) -> 'ClassWithStaticMethod':
            if False:
                print('Hello World!')
            return ClassWithStaticMethod.create_from(a, b)
        make_global(ClassWithStaticMethod)
        self.checkScript(test_function, (1, 2))

    def test_classmethod(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test classmethods on class types.\n        '

        @torch.jit.script
        class ClassWithClassMethod:

            def __init__(self, a: int):
                if False:
                    for i in range(10):
                        print('nop')
                self.a: int = a

            def __eq__(self, other: 'ClassWithClassMethod'):
                if False:
                    for i in range(10):
                        print('nop')
                return self.a == other.a

            @classmethod
            def create(cls, a: int) -> 'ClassWithClassMethod':
                if False:
                    i = 10
                    return i + 15
                return cls(a)
        make_global(ClassWithClassMethod)

        def test_function(a: int) -> 'ClassWithClassMethod':
            if False:
                for i in range(10):
                    print('nop')
            x = ClassWithClassMethod(a)
            return x.create(a)
        self.checkScript(test_function, (1,))

    @skipIfTorchDynamo('Not a suitable test for TorchDynamo')
    def test_properties(self):
        if False:
            return 10
        '\n        Test that a scripted class can make use of the @property decorator.\n        '

        def free_function(x: int) -> int:
            if False:
                while True:
                    i = 10
            return x + 1

        @torch.jit.script
        class Properties:
            __jit_unused_properties__ = ['unsupported']

            def __init__(self, a: int):
                if False:
                    for i in range(10):
                        print('nop')
                self.a = a

            @property
            def attr(self) -> int:
                if False:
                    while True:
                        i = 10
                return self.a - 1

            @property
            def unsupported(self) -> int:
                if False:
                    i = 10
                    return i + 15
                return sum([self.a])

            @torch.jit.unused
            @property
            def unsupported_2(self) -> int:
                if False:
                    print('Hello World!')
                return sum([self.a])

            @unsupported_2.setter
            def unsupported_2(self, value):
                if False:
                    i = 10
                    return i + 15
                self.a = sum([self.a])

            @attr.setter
            def attr(self, value: int):
                if False:
                    return 10
                self.a = value + 3

        @torch.jit.script
        class NoSetter:

            def __init__(self, a: int):
                if False:
                    while True:
                        i = 10
                self.a = a

            @property
            def attr(self) -> int:
                if False:
                    return 10
                return free_function(self.a)

        @torch.jit.script
        class MethodThatUsesProperty:

            def __init__(self, a: int):
                if False:
                    i = 10
                    return i + 15
                self.a = a

            @property
            def attr(self) -> int:
                if False:
                    print('Hello World!')
                return self.a - 2

            @attr.setter
            def attr(self, value: int):
                if False:
                    i = 10
                    return i + 15
                self.a = value + 4

            def forward(self):
                if False:
                    print('Hello World!')
                return self.attr

        class ModuleWithProperties(torch.nn.Module):

            def __init__(self, a: int):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.props = Properties(a)

            def forward(self, a: int, b: int, c: int, d: int):
                if False:
                    print('Hello World!')
                self.props.attr = a
                props = Properties(b)
                no_setter = NoSetter(c)
                method_uses_property = MethodThatUsesProperty(a + b)
                props.attr = c
                method_uses_property.attr = d
                return self.props.attr + no_setter.attr + method_uses_property.forward()
        self.checkModule(ModuleWithProperties(5), (5, 6, 7, 8))

    def test_custom_delete(self):
        if False:
            print('Hello World!')
        '\n        Test that del can be called on an instance of a class that\n        overrides __delitem__.\n        '

        class Example:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self._data: Dict[str, torch.Tensor] = {'1': torch.tensor(1.0)}

            def check(self, key: str) -> bool:
                if False:
                    i = 10
                    return i + 15
                return key in self._data

            def __delitem__(self, key: str):
                if False:
                    print('Hello World!')
                del self._data[key]

        def fn() -> bool:
            if False:
                i = 10
                return i + 15
            example = Example()
            del example['1']
            return example.check('1')
        self.checkScript(fn, ())

        class NoDelItem:

            def __init__(self):
                if False:
                    print('Hello World!')
                self._data: Dict[str, torch.Tensor] = {'1': torch.tensor(1.0)}

            def check(self, key: str) -> bool:
                if False:
                    for i in range(10):
                        print('nop')
                return key in self._data

        def fn() -> bool:
            if False:
                for i in range(10):
                    print('nop')
            example = NoDelItem()
            key = '1'
            del example[key]
            return example.check(key)
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Class does not define __delitem__', 'example[key]'):
            self.checkScript(fn, ())

    def test_recursive_script_builtin_type_resolution(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test resolution of built-in torch types(e.g. torch.Tensor, torch.device) when a class is recursively compiled.\n        '
        tensor_t = torch.Tensor
        device_t = torch.device
        device_ty = torch.device

        class A:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                pass

            def f(self, x: tensor_t, y: torch.device) -> tensor_t:
                if False:
                    for i in range(10):
                        print('nop')
                return x.to(device=y)

            def g(self, x: device_t) -> device_ty:
                if False:
                    while True:
                        i = 10
                return x

            def h(self, a: 'A') -> 'A':
                if False:
                    i = 10
                    return i + 15
                return A()

            def i(self, a: List[int]) -> int:
                if False:
                    return 10
                return a[0]

            def j(self, l: List[device_t]) -> device_ty:
                if False:
                    print('Hello World!')
                return l[0]

        def call_f():
            if False:
                for i in range(10):
                    print('nop')
            a = A()
            return a.f(torch.tensor([1]), torch.device('cpu'))

        def call_g():
            if False:
                return 10
            a = A()
            return a.g(torch.device('cpu'))

        def call_i():
            if False:
                for i in range(10):
                    print('nop')
            a = A()
            return a.i([3])

        def call_j():
            if False:
                while True:
                    i = 10
            a = A()
            return a.j([torch.device('cpu'), torch.device('cpu')])
        for fn in [call_f, call_g, call_i, call_j]:
            self.checkScript(fn, ())
            s = self.getExportImportCopy(torch.jit.script(fn))
            self.assertEqual(s(), fn())

    def test_recursive_script_module_builtin_type_resolution(self):
        if False:
            print('Hello World!')
        '\n        Test resolution of built-in torch types(e.g. torch.Tensor, torch.device) when a class is recursively compiled\n        when compiling a module.\n        '

        class Wrapper:

            def __init__(self, t):
                if False:
                    for i in range(10):
                        print('nop')
                self.t = t

            def to(self, l: List[torch.device], device: Optional[torch.device]=None):
                if False:
                    for i in range(10):
                        print('nop')
                return self.t.to(device=device)

        class A(nn.Module):

            def forward(self):
                if False:
                    return 10
                return Wrapper(torch.rand(4, 4))
        scripted = torch.jit.script(A())
        self.getExportImportCopy(scripted)

    def test_class_attribute_wrong_type(self):
        if False:
            return 10
        '\n        Test that the error message displayed when convering a class type\n        to an IValue that has an attribute of the wrong type.\n        '

        @torch.jit.script
        class ValHolder:

            def __init__(self, val):
                if False:
                    i = 10
                    return i + 15
                self.val = val

        class Mod(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.mod1 = ValHolder('1')
                self.mod2 = ValHolder('2')

            def forward(self, cond: bool):
                if False:
                    return 10
                if cond:
                    mod = self.mod1
                else:
                    mod = self.mod2
                return mod.val
        with self.assertRaisesRegexWithHighlight(RuntimeError, "Could not cast attribute 'val' to type Tensor", ''):
            torch.jit.script(Mod())

    def test_recursive_scripting(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that class types are recursively scripted when an Python instance of one\n        is encountered as a module attribute.\n        '

        class Class:

            def __init__(self, a: int):
                if False:
                    while True:
                        i = 10
                self.a = a

            def get_a(self) -> int:
                if False:
                    print('Hello World!')
                return self.a

        class M(torch.nn.Module):

            def __init__(self, obj):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.obj = obj

            def forward(self) -> int:
                if False:
                    while True:
                        i = 10
                return self.obj.get_a()
        self.checkModule(M(Class(4)), ())

    def test_recursive_scripting_failed(self):
        if False:
            while True:
                i = 10
        '\n        Test that class types module attributes that fail to script\n        are added as failed attributes and do not cause compilation itself\n        to fail unless they are used in scripted code.\n        '

        class UnscriptableClass:

            def __init__(self, a: int):
                if False:
                    while True:
                        i = 10
                self.a = a

            def get_a(self) -> bool:
                if False:
                    print('Hello World!')
                return issubclass(self.a, int)

        class ShouldNotCompile(torch.nn.Module):

            def __init__(self, obj):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.obj = obj

            def forward(self) -> bool:
                if False:
                    while True:
                        i = 10
                return self.obj.get_a()
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'failed to convert Python type', ''):
            torch.jit.script(ShouldNotCompile(UnscriptableClass(4)))

        class ShouldCompile(torch.nn.Module):

            def __init__(self, obj):
                if False:
                    return 10
                super().__init__()
                self.obj = obj

            @torch.jit.ignore
            def ignored_method(self) -> bool:
                if False:
                    for i in range(10):
                        print('nop')
                return self.obj.get_a()

            def forward(self, x: int) -> int:
                if False:
                    return 10
                return x + x
        self.checkModule(ShouldCompile(UnscriptableClass(4)), (4,))

    def test_unresolved_class_attributes(self):
        if False:
            return 10

        class UnresolvedAttrClass:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                pass
            ((attr_a, attr_b), [attr_c, attr_d]) = (('', ''), ['', ''])
            attr_e: int = 0

        def fn_a():
            if False:
                i = 10
                return i + 15
            u = UnresolvedAttrClass()
            return u.attr_a

        def fn_b():
            if False:
                print('Hello World!')
            u = UnresolvedAttrClass()
            return u.attr_b

        def fn_c():
            if False:
                print('Hello World!')
            u = UnresolvedAttrClass()
            return u.attr_c

        def fn_d():
            if False:
                i = 10
                return i + 15
            u = UnresolvedAttrClass()
            return u.attr_d

        def fn_e():
            if False:
                print('Hello World!')
            u = UnresolvedAttrClass()
            return u.attr_e
        error_message_regex = 'object has no attribute or method.*is defined as a class attribute'
        for fn in (fn_a, fn_b, fn_c, fn_d, fn_e):
            with self.assertRaisesRegex(RuntimeError, error_message_regex):
                torch.jit.script(fn)