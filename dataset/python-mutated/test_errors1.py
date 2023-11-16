"""Tests for displaying errors."""
from pytype.tests import test_base
from pytype.tests import test_utils

class ErrorTest(test_base.BaseTest):
    """Tests for errors."""

    def test_deduplicate(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      def f(x):\n        y = 42\n        y.foobar  # attribute-error[e]\n      f(3)\n      f(4)\n    ')
        self.assertErrorRegexes(errors, {'e': "'foobar' on int$"})

    def test_unknown_global(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      def f():\n        return foobar()  # name-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'foobar'})

    def test_invalid_attribute(self):
        if False:
            return 10
        (ty, errors) = self.InferWithErrors('\n      class A:\n        pass\n      def f():\n        (3).parrot  # attribute-error[e]\n        return "foo"\n    ')
        self.assertTypesMatchPytd(ty, '\n      class A:\n        pass\n\n      def f() -> str: ...\n    ')
        self.assertErrorRegexes(errors, {'e': 'parrot.*int'})

    def test_import_error(self):
        if False:
            i = 10
            return i + 15
        self.InferWithErrors('\n      import rumplestiltskin  # import-error\n    ')

    def test_import_from_error(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      from sys import foobar  # import-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'sys\\.foobar'})

    def test_name_error(self):
        if False:
            return 10
        self.InferWithErrors('\n      foobar  # name-error\n    ')

    def test_wrong_arg_count(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      hex(1, 2, 3, 4)  # wrong-arg-count[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'expects 1.*got 4'})

    def test_wrong_arg_types(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      hex(3j)  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'int.*complex'})

    def test_interpreter_function_name_in_msg(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      class A(list): pass\n      A.append(3)  # missing-parameter[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'function list\\.append'})

    def test_pytd_function_name_in_msg(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', 'class A(list): pass')
            errors = self.CheckWithErrors('\n        import foo\n        foo.A.append(3)  # missing-parameter[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'function list\\.append'})

    def test_builtin_function_name_in_msg(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      x = list\n      x += (1,2)  # missing-parameter[e]\n      ')
        self.assertErrorRegexes(errors, {'e': 'function list\\.__iadd__'})

    def test_rewrite_builtin_function_name(self):
        if False:
            while True:
                i = 10
        'Should rewrite `function builtins.len` to `built-in function len`.'
        errors = self.CheckWithErrors('x = len(None)  # wrong-arg-types[e]')
        self.assertErrorRegexes(errors, {'e': 'Built-in function len'})

    def test_bound_method_name_in_msg(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      "".join(1)  # wrong-arg-types[e]\n      ')
        self.assertErrorRegexes(errors, {'e': 'Function str\\.join'})

    def test_nested_class_method_name_is_msg(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      class A:\n        class B:\n          def f(self):\n            pass\n      A.B().f("oops")  # wrong-arg-count[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Function B.f'})

    def test_pretty_print_wrong_args(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(a: int, b: int, c: int, d: int, e: int): ...\n      ')
            errors = self.CheckWithErrors('\n        import foo\n        foo.f(1, 2, 3, "four", 5)  # wrong-arg-types[e]\n      ', pythonpath=[d.path])
        self.assertErrorSequences(errors, {'e': ['a, b, c, d: int, ...', 'a, b, c, d: str, ...']})

    def test_invalid_base_class(self):
        if False:
            while True:
                i = 10
        self.InferWithErrors('\n      class Foo(3):  # base-class-error\n        pass\n    ')

    def test_invalid_iterator_from_import(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('mod.pyi', '\n        class Codec:\n            def __init__(self) -> None: ...\n      ')
            errors = self.CheckWithErrors('\n        import mod\n        def f():\n          for row in mod.Codec():  # attribute-error[e]\n            pass\n      ', pythonpath=[d.path])
            self.assertErrorSequences(errors, {'e': ['No attribute', '__iter__', 'on mod.Codec']})

    def test_invalid_iterator_from_class(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      class A:\n        pass\n      def f():\n        for row in A():  # attribute-error[e]\n          pass\n    ')
        self.assertErrorRegexes(errors, {'e': '__iter__.*A'})

    def test_iter_on_module(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      import sys\n      for _ in sys:  # module-attr[e]\n        pass\n    ')
        self.assertErrorRegexes(errors, {'e': "__iter__.*module 'sys'"})

    def test_inherit_from_generic(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('mod.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        class Foo(Generic[T]): ...\n        class Bar(Foo[int]): ...\n      ')
            errors = self.CheckWithErrors('\n        import mod\n        chr(mod.Bar())  # wrong-arg-types[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'int.*mod\\.Bar'})

    def test_wrong_keyword_arg(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('mycgi.pyi', '\n        from typing import Union\n        def escape(x: Union[str, int]) -> Union[str, int]: ...\n      ')
            errors = self.CheckWithErrors('\n        import mycgi\n        def foo(s):\n          return mycgi.escape(s, quote=1)  # wrong-keyword-args[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'quote.*mycgi\\.escape'})

    def test_missing_parameter(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def bar(xray, yankee, zulu) -> str: ...\n      ')
            errors = self.CheckWithErrors('\n        import foo\n        foo.bar(1, 2)  # missing-parameter[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'zulu.*foo\\.bar'})

    def test_bad_inheritance(self):
        if False:
            print('Hello World!')
        self.InferWithErrors('\n      class X:\n          pass\n      class Bar(X):\n          pass\n      class Baz(X, Bar):  # mro-error\n          pass\n    ')

    def test_bad_call(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('other.pyi', '\n        def foo(x: int, y: str) -> str: ...\n      ')
            errors = self.CheckWithErrors('\n        import other\n        other.foo(1.2, [])  # wrong-arg-types[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': '\\(x: int'})

    def test_call_uncallable(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      0()  # not-callable[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'int'})

    def test_super_error(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      class A:\n        def __init__(self):\n          super(A, self, "foo").__init__()  # wrong-arg-count[e]\n    ')
        self.assertErrorRegexes(errors, {'e': '2.*3'})

    def test_attribute_error(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('modfoo.pyi', '')
            errors = self.CheckWithErrors('\n        class Foo:\n          def __getattr__(self, name):\n            return "attr"\n        def f():\n          return Foo.foo  # attribute-error[e1]\n        def g(x):\n          if x:\n            y = None\n          else:\n            y = 1\n          return y.bar  # attribute-error[e2]  # attribute-error[e3]\n        def h():\n          return Foo().foo  # No error\n        import modfoo\n        modfoo.baz  # module-attr[e4]\n      ', pythonpath=[d.path])
            if self.python_version == (3, 10):
                e2_msg = "No attribute 'bar' on None"
                e3_msg = "No attribute 'bar' on int"
            else:
                e2_msg = "No attribute 'bar' on int\nIn Optional[int]"
                e3_msg = "No attribute 'bar' on None\nIn Optional[int]"
            self.assertErrorSequences(errors, {'e1': ["No attribute 'foo' on Type[Foo]"], 'e2': [e2_msg], 'e3': [e3_msg], 'e4': ["No attribute 'baz' on module 'modfoo'"]})

    def test_attribute_error_getattribute(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      class Foo:\n        def __getattribute__(self, name):\n          return "attr"\n      def f():\n        return Foo().x  # There should be no error on this line.\n      def g():\n        return Foo.x  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'x'})

    def test_none_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      None.foo  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'foo'})

    def test_pyi_type(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x: list[int]) -> int: ...\n      ')
            errors = self.CheckWithErrors('\n        import foo\n        foo.f([""])  # wrong-arg-types[e]\n      ', deep=True, pythonpath=[d.path])
            self.assertErrorSequences(errors, {'e': ['List[int]', 'List[str]']})

    def test_too_many_args(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      def f():\n        pass\n      f(3)  # wrong-arg-count[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': '0.*1'})

    def test_too_few_args(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      def f(x):\n        pass\n      f()  # missing-parameter[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'x.*f'})

    def test_duplicate_keyword(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      def f(x, y):\n        pass\n      f(3, x=3)  # duplicate-keyword-argument[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'f.*x'})

    def test_bad_import(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        def f() -> int: ...\n        class f: ...\n      ')
            self.InferWithErrors('\n        import a  # pyi-error\n      ', pythonpath=[d.path])

    def test_bad_import_dependency(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from b import X\n        class Y(X): ...\n      ')
            self.InferWithErrors('\n        import a  # pyi-error\n      ', pythonpath=[d.path])

    def test_bad_import_from(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo/a.pyi', '\n        def f() -> int: ...\n        class f: ...\n      ')
            d.create_file('foo/__init__.pyi', '')
            errors = self.CheckWithErrors('\n        from foo import a  # pyi-error[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'foo\\.a'})

    def test_bad_import_from_dependency(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo/a.pyi', '\n          from a import X\n          class Y(X): ...\n      ')
            d.create_file('foo/__init__.pyi', '')
            errors = self.CheckWithErrors('\n        from foo import a  # pyi-error[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'foo\\.a'})

    def test_bad_container(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import SupportsInt\n        class A(SupportsInt[int]): pass\n      ')
            errors = self.CheckWithErrors('\n        import a  # pyi-error[e]\n      ', deep=True, pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'SupportsInt is not a container'})

    def test_bad_type_parameter_order(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, TypeVar\n        K = TypeVar("K")\n        V = TypeVar("V")\n        class A(Generic[K, V]): pass\n        class B(Generic[K, V]): pass\n        class C(A[K, V], B[V, K]): pass\n      ')
            errors = self.CheckWithErrors('\n        import a  # pyi-error[e]\n      ', deep=True, pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'Illegal.*order.*a\\.C'})

    def test_duplicate_type_parameter(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        class A(Generic[T, T]): pass\n      ')
            errors = self.CheckWithErrors('\n        import a  # pyi-error[e]\n      ', deep=True, pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'T'})

    def test_duplicate_generic_base_class(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        V = TypeVar("V")\n        class A(Generic[T], Generic[V]): pass\n      ')
            errors = self.CheckWithErrors('\n        import a  # pyi-error[e]\n      ', deep=True, pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'inherit.*Generic'})

    def test_type_parameter_in_module_constant(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import TypeVar\n        T = TypeVar("T")\n        x = ...  # type: T\n      ')
            errors = self.CheckWithErrors('\n        import a  # pyi-error[e]\n      ', deep=True, pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'a.*T.*a\\.x'})

    def test_type_parameter_in_class_attribute(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        class A(Generic[T]):\n          x = ...  # type: T\n      ')
            errors = self.CheckWithErrors('\n        import a\n        def f():\n          return a.A.x  # unbound-type-param[e]\n      ', deep=True, pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'x.*A.*T'})

    def test_unbound_type_parameter_in_instance_attribute(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import TypeVar\n        T = TypeVar("T")\n        class A:\n          x = ...  # type: T\n      ')
            errors = self.CheckWithErrors('\n        import a  # pyi-error[e]\n      ', deep=True, pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'a.*T.*a\\.A\\.x'})

    def test_print_union_arg(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Union\n        def f(x: Union[int, str]) -> None: ...\n      ')
            errors = self.CheckWithErrors('\n        import a\n        x = a.f(4.2)  # wrong-arg-types[e]\n      ', deep=True, pythonpath=[d.path])
            pattern = ['Expected', 'Union[int, str]', 'Actually passed']
            self.assertErrorSequences(errors, {'e': pattern})

    def test_print_type_arg(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      hex(int)  # wrong-arg-types[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'Actually passed.*Type\\[int\\]'})

    def test_delete_from_set(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      s = {1}\n      del s[1]  # unsupported-operands[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'item deletion'})

    def test_bad_reference(self):
        if False:
            i = 10
            return i + 15
        (ty, errors) = self.InferWithErrors('\n      def main():\n        x = foo  # name-error[e]\n        for foo in []:\n          pass\n        return x\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'foo'})
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def main() -> Any: ...\n    ')

    def test_set_int_attribute(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      x = 42\n      x.y = 42  # not-writable[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'y.*int'})

    def test_invalid_parameters_on_method(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        class A:\n          def __init__(self, x: int) -> None: ...\n      ')
            errors = self.CheckWithErrors('\n        import a\n        x = a.A("")  # wrong-arg-types[e1]\n        x = a.A("", 42)  # wrong-arg-count[e2]\n        x = a.A(42, y="")  # wrong-keyword-args[e3]\n        x = a.A(42, x=42)  # duplicate-keyword-argument[e4]\n        x = a.A()  # missing-parameter[e5]\n      ', pythonpath=[d.path])
            a = 'A\\.__init__'
            self.assertErrorRegexes(errors, {'e1': a, 'e2': a, 'e3': a, 'e4': a, 'e5': a})

    def test_duplicate_keywords(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x, *args, y) -> None: ...\n      ')
            self.InferWithErrors('\n        import foo\n        foo.f(1, y=2)\n        foo.f(1, 2, y=3)\n        foo.f(1, x=1)  # duplicate-keyword-argument\n        # foo.f(y=1, y=2)  # caught by compiler\n      ', deep=True, pythonpath=[d.path])

    def test_invalid_parameters_details(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      float(list())  # wrong-arg-types[e1]\n      float(1, list(), foobar=str)  # wrong-arg-count[e2]\n      float(1, foobar=list())  # wrong-keyword-args[e3]\n      float(1, x="")  # duplicate-keyword-argument[e4]\n      hex()  # missing-parameter[e5]\n    ')
        self.assertErrorSequences(errors, {'e1': ['Actually passed:', 'self, x: List[nothing]'], 'e2': ['_, foobar'], 'e3': ['Actually passed:', 'self, x, foobar'], 'e4': ['Actually passed:', 'self, x, x'], 'e5': ['Actually passed: ()']})

    def test_bad_superclass(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      class A:\n        def f(self):\n          return "foo"\n\n      class B(A):\n        def f(self):\n          return super(self, B).f()  # should be super(B, self)  # wrong-arg-types[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'cls: type.*cls: B'})

    @test_base.skip('Need to type-check second argument to super')
    def test_bad_super_instance(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      class A:\n        pass\n      class B(A):\n        def __init__(self):\n          super(B, A).__init__()  # A cannot be the second argument to super  # wrong-arg-types[e]\n    ', deep=True)
        self.assertErrorSequences(errors, {'e': ['Type[B]', 'Type[A]']})

    def test_bad_name_import(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        import typing\n        x = ...  # type: typing.Rumpelstiltskin\n      ')
            errors = self.CheckWithErrors('\n        import a  # pyi-error[e]\n        x = a.x\n      ', pythonpath=[d.path], deep=True)
            self.assertErrorRegexes(errors, {'e': 'Rumpelstiltskin'})

    def test_bad_name_import_from(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Rumpelstiltskin\n        x = ...  # type: Rumpelstiltskin\n      ')
            errors = self.CheckWithErrors('\n        import a  # pyi-error[e]\n        x = a.x\n      ', pythonpath=[d.path], deep=True)
            self.assertErrorRegexes(errors, {'e': 'Rumpelstiltskin'})

    def test_match_type(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Type\n        class A: ...\n        class B(A): ...\n        class C: ...\n        def f(x: Type[A]) -> bool: ...\n      ')
            (ty, errors) = self.InferWithErrors('\n        import a\n        x = a.f(a.A)\n        y = a.f(a.B)\n        z = a.f(a.C)  # wrong-arg-types[e]\n      ', pythonpath=[d.path], deep=True)
            error = ['Expected', 'Type[a.A]', 'Actual', 'Type[a.C]']
            self.assertErrorSequences(errors, {'e': error})
            self.assertTypesMatchPytd(ty, '\n        import a\n        from typing import Any\n        x = ...  # type: bool\n        y = ...  # type: bool\n        z = ...  # type: Any\n      ')

    def test_match_parameterized_type(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, Type, TypeVar\n        T = TypeVar("T")\n        class A(Generic[T]): ...\n        class B(A[str]): ...\n        def f(x: Type[A[int]]): ...\n      ')
            errors = self.CheckWithErrors('\n        import a\n        x = a.f(a.B)  # wrong-arg-types[e]\n      ', pythonpath=[d.path], deep=True)
            expected_error = ['Expected', 'Type[a.A[int]]', 'Actual', 'Type[a.B]']
            self.assertErrorSequences(errors, {'e': expected_error})

    def test_mro_error(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        class A: ...\n        class B: ...\n        class C(A, B): ...\n        class D(B, A): ...\n        class E(C, D): ...\n      ')
            errors = self.CheckWithErrors('\n        import a\n        x = a.E()  # mro-error[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'E'})

    def test_bad_mro(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        class A(BaseException, ValueError): ...\n      ')
            errors = self.CheckWithErrors('\n        import a\n        class B(a.A): pass  # mro-error[e]\n        raise a.A()\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'A'})

    def test_unsolvable_as_metaclass(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Any\n        def __getattr__(name) -> Any: ...\n      ')
            d.create_file('b.pyi', '\n        from a import A\n        class B(metaclass=A): ...\n      ')
            errors = self.CheckWithErrors("\n        import b\n        class C(b.B):\n          def __init__(self):\n            f = open(self.x, 'r')  # attribute-error[e]\n      ", pythonpath=[d.path], deep=True)
            self.assertErrorRegexes(errors, {'e': 'x.*C'})

    def test_dont_timeout_on_complex(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      if __random__:\n        x = [1]\n      else:\n        x = [1j]\n      x = x + x\n      x = x + x\n      x = x + x\n      x = x + x\n      x = x + x\n      x = x + x\n      x = x + x\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      x = ...  # type: Any\n    ')

    def test_failed_function_call(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        def f(x: str, y: int) -> bool: ...\n        def f(x: str) -> bool: ...\n      ')
            self.InferWithErrors('\n        import a\n        x = a.f(0, "")  # wrong-arg-types\n      ', pythonpath=[d.path])

    def test_noncomputable_method(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        T = TypeVar("T")\n        def copy(x: T) -> T: ...\n      ')
            errors = self.CheckWithErrors('\n        import a\n        class A:\n          def __getattribute__(self, name):\n            return a.copy(self)\n        x = A()()  # not-callable[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'A'})

    def test_bad_type_name(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      X = type(3, (int, object), {"a": 1})  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Actual.*int'})

    def test_bad_type_bases(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      X = type("X", (42,), {"a": 1})  # wrong-arg-types[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['Actual', 'Tuple[int]']})

    def test_half_bad_type_bases(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      X = type("X", (42, object), {"a": 1})  # wrong-arg-types[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['Actual', 'Tuple[int, Type[object]]']})

    def test_bad_type_members(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      X = type("X", (int, object), {0: 1})  # wrong-arg-types[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['Actual', 'Dict[int, int]']})

    def test_recursion(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        class A(B): ...\n        class B(A): ...\n      ')
            (ty, errors) = self.InferWithErrors('\n        import a\n        v = a.A()  # recursion-error[e]\n        x = v.x  # No error because there is an Unsolvable in the MRO of a.A\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        from typing import Any\n        v = ...  # type: a.A\n        x = ...  # type: Any\n      ')
            self.assertErrorRegexes(errors, {'e': 'a\\.A'})

    def test_empty_union_or_optional(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('f1.pyi', '\n        def f(x: Union): ...\n      ')
            d.create_file('f2.pyi', '\n        def f(x: Optional): ...\n      ')
            errors = self.CheckWithErrors('\n        import f1  # pyi-error[e1]\n        import f2  # pyi-error[e2]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e1': 'f1.*Union', 'e2': 'f2.*Optional'})

    def test_bad_dict_attribute(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      x = {"a": 1}\n      y = x.a  # attribute-error[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['a', 'Dict[str, int]']})

    def test_bad_pyi_dict(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Dict\n        x = ...  # type: Dict[str, int, float]\n      ')
            errors = self.CheckWithErrors('\n        import a  # pyi-error[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': '2.*3'})

    def test_call_none(self):
        if False:
            while True:
                i = 10
        self.InferWithErrors('\n      None()  # not-callable\n    ')

    def test_in_none(self):
        if False:
            for i in range(10):
                print('nop')
        self.InferWithErrors('\n      3 in None  # unsupported-operands\n    ')

    def test_no_attr_error(self):
        if False:
            print('Hello World!')
        self.InferWithErrors('\n      if __random__:\n        y = 42\n      else:\n        y = "foo"\n      y.upper  # attribute-error\n    ')

    def test_attr_error(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      if __random__:\n        y = 42\n      else:\n        y = "foo"\n      y.upper  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'upper.*int'})

    def test_print_callable_instance(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      from typing import Callable\n      v = None  # type: Callable[[int], str]\n      hex(v)  # wrong-arg-types[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['Actual', 'Callable[[int], str]']})

    def test_same_name_and_line(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      def f(x):\n        return x + 42  # unsupported-operands[e1]  # unsupported-operands[e2]\n      f("hello")\n      f([])\n    ')
        self.assertErrorRegexes(errors, {'e1': 'str.*int', 'e2': 'List.*int'})

    def test_kwarg_order(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(*args, y, x, z: int): ...\n        def g(x): ...\n      ')
            errors = self.CheckWithErrors('\n        import foo\n        foo.f(x=1, y=2, z="3")  # wrong-arg-types[e1]\n        foo.g(42, v4="the", v3="quick", v2="brown", v1="fox")  # wrong-keyword-args[e2]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e1': 'x, y, z.*x, y, z', 'e2': 'v1, v2, v3, v4'})

    def test_bad_base_class(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      class Foo(None): pass  # base-class-error[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['Invalid base class: None']})

    @test_utils.skipIfPy((3, 10), reason='non-3.10: log one error for all bad options')
    def test_bad_ambiguous_base_class(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      class Bar(None if __random__ else 42): pass  # base-class-error[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['Optional[<instance of int>]']})

    @test_utils.skipUnlessPy((3, 10), reason='3.10: log one error per bad option')
    def test_bad_ambiguous_base_class_310(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      class Bar(None if __random__ else 42): pass  # base-class-error[e1]  # base-class-error[e2]\n    ')
        self.assertErrorSequences(errors, {'e1': ['Invalid base class: None'], 'e2': ['Invalid base class: <instance of int>']})

    def test_callable_in_unsupported_operands(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      def f(x, y=None): pass\n      f in f  # unsupported-operands[e]\n    ')
        typ = 'Callable[[Any, Any], Any]'
        self.assertErrorSequences(errors, {'e': [typ, typ]})

    def test_clean_pyi_namedtuple_names(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import NamedTuple\n        X = NamedTuple("X", [])\n        def f(x: int): ...\n      ')
            errors = self.CheckWithErrors('\n        import foo\n        foo.f(foo.X())  # wrong-arg-types[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'foo.X'})

    def test_bad_annotation(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      list[0]  # not-indexable[e1]\n      dict[1, 2]  # invalid-annotation[e2]  # invalid-annotation[e3]\n      class A: pass\n      A[3]  # not-indexable[e4]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'class list', 'e2': '1.*Not a type', 'e3': '2.*Not a type', 'e4': 'class A'})

    def test_not_protocol(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      a = []\n      a.append(1)\n      a = "".join(a)  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': '\\(.*List\\[int\\]\\)$'})

    def test_protocol_signatures(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      from typing import Sequence\n\n      class Foo:\n        def __len__(self):\n          return 0\n        def __getitem__(self, x: int) -> int:\n          return 0\n\n      def f(x: Sequence[int]):\n        pass\n\n      foo = Foo()\n      f(foo)  # wrong-arg-types[e]\n    ')
        expected = [['Method __getitem__', 'protocol Sequence[int]', 'signature in Foo'], ['def __getitem__(self: Sequence'], ['def __getitem__(self, x: int)']]
        for pattern in expected:
            self.assertErrorSequences(errors, {'e': pattern})

    def test_hidden_error(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      use_option = False\n      def f():\n        if use_option:\n          name_error  # name-error\n    ')

    def test_unknown_in_error(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      def f(x):\n        y = x if __random__ else None\n        return y.groups()  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Optional\\[Any\\]'})

class OperationsTest(test_base.BaseTest):
    """Test operations."""

    def test_binary(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors("\n      def f(): return 3 ** 'foo'  # unsupported-operands[e]\n    ")
        self.assertErrorSequences(errors, {'e': ['**', 'int', 'str', '__pow__ on', 'int']})

    def test_unary(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      def f(): return ~None  # unsupported-operands[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['~', 'None', "'__invert__' on None"]})

    def test_op_and_right_op(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors("\n      def f(): return 'foo' ^ 3  # unsupported-operands[e]\n    ")
        self.assertErrorSequences(errors, {'e': ['^', 'str', 'int', "'__xor__' on", 'str', "'__rxor__' on", 'int']})

    def test_var_name_and_pyval(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors("\n      def f(): return 'foo' ^ 3  # unsupported-operands[e]\n    ")
        self.assertErrorSequences(errors, {'e': ['^', "'foo': str", '3: int']})

class RevealTypeTest(test_base.BaseTest):
    """Tests for pseudo-builtin reveal_type()."""

    def test_reveal_type(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      class Foo:\n        pass\n      reveal_type(Foo)  # reveal-type[e1]\n      reveal_type(Foo())  # reveal-type[e2]\n      reveal_type([1,2,3])  # reveal-type[e3]\n    ')
        self.assertErrorSequences(errors, {'e1': ['Type[Foo]'], 'e2': ['Foo'], 'e3': ['List[int]']})

    def test_reveal_type_expression(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      x = 42\n      y = "foo"\n      reveal_type(x or y)  # reveal-type[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['Union[int, str]']})

    def test_combine_containers(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      from typing import Set, Union\n      x: Set[Union[int, str]]\n      y: Set[Union[str, bytes]]\n      reveal_type(x | y)  # reveal-type[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['Set[Union[bytes, int, str]]']})

class InPlaceOperationsTest(test_base.BaseTest):
    """Test in-place operations."""

    def test_iadd(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      def f(): v = []; v += 3  # unsupported-operands[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['+=', 'List', 'int', '__iadd__ on List', 'Iterable']})

class NoSymbolOperationsTest(test_base.BaseTest):
    """Test operations with no native symbol."""

    def test_getitem(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors("\n      def f(): v = []; return v['foo']  # unsupported-operands[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'item retrieval.*List.*str.*__getitem__ on List.*SupportsIndex'})

    def test_delitem(self):
        if False:
            return 10
        errors = self.CheckWithErrors("\n      def f(): v = {'foo': 3}; del v[3]  # unsupported-operands[e]\n    ")
        d = 'Dict[str, int]'
        self.assertErrorSequences(errors, {'e': ['item deletion', d, 'int', f'__delitem__ on {d}', 'str']})

    def test_setitem(self):
        if False:
            return 10
        errors = self.CheckWithErrors("\n      def f(): v = []; v['foo'] = 3  # unsupported-operands[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'item assignment.*List.*str.*__setitem__ on List.*SupportsIndex'})

    def test_contains(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors("\n      def f(): return 'foo' in 3  # unsupported-operands[e]\n    ")
        self.assertErrorRegexes(errors, {'e': "'in'.*int.*str.*'__contains__' on.*int"})

    def test_recursion(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      def f():\n        if __random__:\n          f()\n          name_error  # name-error\n    ')
if __name__ == '__main__':
    test_base.main()