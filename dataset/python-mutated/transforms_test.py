import textwrap
import testslide
from .. import patch_specs, transforms

class PatchTransformsTest(testslide.TestCase):

    def assert_transform(self, original_code: str, patch: patch_specs.Patch, expected_code: str) -> None:
        if False:
            i = 10
            return i + 15
        actual_output = transforms.apply_patches_in_sequence(code=textwrap.dedent(original_code), patches=[patch])
        try:
            self.assertEqual(actual_output.strip(), textwrap.dedent(expected_code).strip())
        except AssertionError as err:
            print('--- Expected ---')
            print(textwrap.dedent(expected_code))
            print('--- Actual ---')
            print(actual_output)
            raise err

    def test_add_to_module__top(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assert_transform(original_code='\n                b: str\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string(''), action=patch_specs.AddAction(content=textwrap.dedent('\n                        from foo import Bar\n                        a: Bar\n                        '), position=patch_specs.AddPosition.TOP_OF_SCOPE)), expected_code='\n                from foo import Bar\n                a: Bar\n                b: str\n                ')

    def test_add_to_module__bottom(self) -> None:
        if False:
            print('Hello World!')
        self.assert_transform(original_code='\n                b: str\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string(''), action=patch_specs.AddAction(content=textwrap.dedent('\n                        def f(x: int) -> int: ...\n                        y: float\n                        '), position=patch_specs.AddPosition.BOTTOM_OF_SCOPE)), expected_code='\n                b: str\n                def f(x: int) -> int: ...\n                y: float\n                ')

    def test_add_to_class__top(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assert_transform(original_code='\n                class MyClass:\n                    b: int\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string('MyClass'), action=patch_specs.AddAction(content=textwrap.dedent('\n                        a: float\n                        def f(self, x: int) -> int: ...\n                        '), position=patch_specs.AddPosition.TOP_OF_SCOPE)), expected_code='\n                class MyClass:\n                    a: float\n                    def f(self, x: int) -> int: ...\n                    b: int\n                ')

    def test_add_to_class__bottom(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assert_transform(original_code='\n                class MyClass:\n                    b: int\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string('MyClass'), action=patch_specs.AddAction(content=textwrap.dedent('\n                        a: float\n                        def f(self, x: int) -> int: ...\n                        '), position=patch_specs.AddPosition.BOTTOM_OF_SCOPE)), expected_code='\n                class MyClass:\n                    b: int\n                    a: float\n                    def f(self, x: int) -> int: ...\n                ')

    def test_add_to_class__force_indent(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assert_transform(original_code='\n                class MyClass: b: int\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string('MyClass'), action=patch_specs.AddAction(content=textwrap.dedent('\n                        a: float\n                        def f(self, x: int) -> int: ...\n                        '), position=patch_specs.AddPosition.BOTTOM_OF_SCOPE)), expected_code='\n                class MyClass:\n                    b: int\n                    a: float\n                    def f(self, x: int) -> int: ...\n                ')

    def test_add_to_class_nested_classes(self) -> None:
        if False:
            return 10
        self.assert_transform(original_code='\n                class OuterClass0:\n                    pass\n                class OuterClass1:\n                    class InnerClass0:\n                        b: int\n                    class InnerClass1:\n                        b: int\n                    class InnerClass2:\n                        b: int\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string('OuterClass1.InnerClass1'), action=patch_specs.AddAction(content=textwrap.dedent('\n                        def f(self, x: int) -> int: ...\n                        '), position=patch_specs.AddPosition.BOTTOM_OF_SCOPE)), expected_code='\n                class OuterClass0:\n                    pass\n                class OuterClass1:\n                    class InnerClass0:\n                        b: int\n                    class InnerClass1:\n                        b: int\n                        def f(self, x: int) -> int: ...\n                    class InnerClass2:\n                        b: int\n                ')

    def test_add_under_import_header(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_transform(original_code='\n                import foo, bar\n                if condition:\n                    from baz import Baz\n                if condition1:\n                    import qux1 as quux\n                elif condition2:\n                    import qux2 as quux\n                else:\n                    import qux3 as quux\n                x: int\n                import this_is_out_of_order\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string(''), action=patch_specs.AddAction(content=textwrap.dedent("\n                        from typing import TypeVar\n                        T = TypeVar('T')\n                        "), position=patch_specs.AddPosition.TOP_OF_SCOPE)), expected_code="\n                import foo, bar\n                if condition:\n                    from baz import Baz\n                if condition1:\n                    import qux1 as quux\n                elif condition2:\n                    import qux2 as quux\n                else:\n                    import qux3 as quux\n                from typing import TypeVar\n                T = TypeVar('T')\n                x: int\n                import this_is_out_of_order\n                ")

    def test_delete__ann_assign(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assert_transform(original_code='\n                x: int\n                y: str\n                z: float\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string(''), action=patch_specs.DeleteAction(name='y')), expected_code='\n                x: int\n                z: float\n                ')

    def test_delete__if_block(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assert_transform(original_code='\n                y: str\n                if condition0:\n                    def f(x: int) -> int: ...\n                elif condition1:\n                    def f(x: str) -> str: ...\n                else:\n                    def f(x: float) -> float: ...\n                z: float\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string(''), action=patch_specs.DeleteAction(name='f')), expected_code='\n                y: str\n                z: float\n                ')

    def test_delete__typealias(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_transform(original_code='\n                from foo import Foo\n                Baz = Foo\n                x: int\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string(''), action=patch_specs.DeleteAction(name='Baz')), expected_code='\n                from foo import Foo\n                x: int\n                ')

    def test_delete__class(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_transform(original_code='\n                class A: pass\n                @classdecorator\n                class B: pass\n                class C: pass\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string(''), action=patch_specs.DeleteAction(name='B')), expected_code='\n                class A: pass\n                class C: pass\n                ')

    def test_delete__function(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_transform(original_code='\n                def f(x: int) -> int: ...\n                def g(x: int) -> int: ...\n                def h(x: int) -> int: ...\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string(''), action=patch_specs.DeleteAction(name='g')), expected_code='\n                def f(x: int) -> int: ...\n                def h(x: int) -> int: ...\n                ')

    def test_delete__overloads(self) -> None:
        if False:
            return 10
        self.assert_transform(original_code='\n                def f(x: int) -> int: ...\n                @overload\n                def g(x: int) -> int: ...\n                @overload\n                def g(x: int) -> int: ...\n                def g(object) -> object: ...\n                def h(x: int) -> int: ...\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string(''), action=patch_specs.DeleteAction(name='g')), expected_code='\n                def f(x: int) -> int: ...\n                def h(x: int) -> int: ...\n                ')

    def test_delete__in_nested_class(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_transform(original_code='\n                class OuterClass0:\n                    class InnerClass1:\n                        x: int\n                class OuterClass1:\n                    class InnerClass0:\n                        x: int\n                    class InnerClass1:\n                        x: int\n                    class InnerClass2:\n                        x: int\n                class OuterClass2:\n                    class InnerClass1:\n                        x: int\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string('OuterClass1.InnerClass1'), action=patch_specs.DeleteAction(name='x')), expected_code='\n                class OuterClass0:\n                    class InnerClass1:\n                        x: int\n                class OuterClass1:\n                    class InnerClass0:\n                        x: int\n                    class InnerClass1:\n                        pass\n                    class InnerClass2:\n                        x: int\n                class OuterClass2:\n                    class InnerClass1:\n                        x: int\n                ')

    def test_delete__import(self) -> None:
        if False:
            return 10
        self.assert_transform(original_code='\n                import foo\n                if condition0:\n                    import bar\n                else:\n                    from baz import bar\n                from bar import qux\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string(''), action=patch_specs.DeleteAction(name='bar')), expected_code='\n                import foo\n                from bar import qux\n                ')

    def test_delete__import_as(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assert_transform(original_code='\n                import foo\n                import bar_alt as bar\n                from baz import bar_alt as bar\n                from bar import qux\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string(''), action=patch_specs.DeleteAction(name='bar')), expected_code='\n                import foo\n                from bar import qux\n                ')

    def test_replace__ann_assign(self) -> None:
        if False:
            print('Hello World!')
        self.assert_transform(original_code='\n                x: int\n                y: str\n                z: float\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string(''), action=patch_specs.ReplaceAction(name='y', content=textwrap.dedent('\n                        w: str\n                        '))), expected_code='\n                x: int\n                w: str\n                z: float\n                ')

    def test_replace__function_with_overloads(self) -> None:
        if False:
            while True:
                i = 10
        self.assert_transform(original_code='\n                def f(x: int) -> int: ...\n                @overload\n                def g(x: int) -> int: ...\n                @overload\n                def g(x: float) -> float: ...\n                def h(x: int) -> int: ...\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string(''), action=patch_specs.ReplaceAction(name='g', content=textwrap.dedent("\n                        T = TypeVar('T')\n                        def g(x: T) -> T: ...\n                        "))), expected_code="\n                def f(x: int) -> int: ...\n                T = TypeVar('T')\n                def g(x: T) -> T: ...\n                def h(x: int) -> int: ...\n                ")

    def test_replace__nested_class(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assert_transform(original_code='\n                class OuterClass0:\n                    class InnerClass1:\n                        x: int\n                class OuterClass1:\n                    class InnerClass0:\n                        x: int\n                    class InnerClass1:\n                        x: int\n                    class InnerClass2:\n                        x: int\n                class OuterClass2:\n                    class InnerClass1:\n                        x: int\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string('OuterClass1.InnerClass1'), action=patch_specs.ReplaceAction(name='x', content='y: float')), expected_code='\n                class OuterClass0:\n                    class InnerClass1:\n                        x: int\n                class OuterClass1:\n                    class InnerClass0:\n                        x: int\n                    class InnerClass1:\n                        y: float\n                    class InnerClass2:\n                        x: int\n                class OuterClass2:\n                    class InnerClass1:\n                        x: int\n                ')

    def test_replace__import(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assert_transform(original_code='\n                import baz\n                from foo import Bar\n                x: Bar\n                ', patch=patch_specs.Patch(parent=patch_specs.QualifiedName.from_string(''), action=patch_specs.ReplaceAction(name='Bar', content='Bar = baz.Bar')), expected_code='\n                import baz\n                Bar = baz.Bar\n                x: Bar\n                ')