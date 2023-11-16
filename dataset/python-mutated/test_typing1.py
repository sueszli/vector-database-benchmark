"""Tests for typing.py."""
from pytype.tests import test_base
from pytype.tests import test_utils

class TypingTest(test_base.BaseTest):
    """Tests for typing.py."""

    def test_all(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import typing\n      x = typing.__all__\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      import typing\n      from typing import List\n      x = ...  # type: List[str]\n    ')

    def test_cast1(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import typing\n      def f():\n        return typing.cast(typing.List[int], [])\n    ')
        self.assertTypesMatchPytd(ty, '\n      import typing\n      from typing import Any, List\n      def f() -> List[int]: ...\n    ')

    def test_cast2(self):
        if False:
            return 10
        self.Check('\n      import typing\n      foo = typing.cast(typing.Dict, {})\n    ')

    def test_process_annotation_for_cast(self):
        if False:
            print('Hello World!')
        (ty, _) = self.InferWithErrors('\n      import typing\n      v1 = typing.cast(None, __any_object__)\n      v2 = typing.cast(typing.Union, __any_object__)  # invalid-annotation\n      v3 = typing.cast("A", __any_object__)\n      class A:\n        pass\n    ')
        self.assertTypesMatchPytd(ty, '\n      import typing\n      v1: None\n      v2: typing.Any\n      v3: A\n      class A: ...\n    ')

    def test_no_typevars_for_cast(self):
        if False:
            i = 10
            return i + 15
        self.InferWithErrors('\n        from typing import cast, AnyStr, Type, TypeVar, _T, Union\n        def f(x):\n          return cast(AnyStr, x)  # invalid-annotation\n        f("hello")\n        def g(x):\n          return cast(Union[AnyStr, _T], x)  # invalid-annotation\n        g("quack")\n        ')

    def test_cast_args(self):
        if False:
            print('Hello World!')
        self.assertNoCrash(self.Check, '\n      import typing\n      typing.cast(typing.AnyStr)\n      typing.cast("str")\n      typing.cast()\n      typing.cast(typ=typing.AnyStr, val=__any_object__)\n      typing.cast(typ=str, val=__any_object__)\n      typing.cast(typ="str", val=__any_object__)\n      typing.cast(val=__any_object__)\n      typing.cast(typing.List[typing.AnyStr], [])\n      ')

    def test_generate_type_alias(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import List\n      MyType = List[str]\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      MyType = List[str]\n    ')

    def test_protocol(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing_extensions import Protocol\n      class Foo(Protocol): pass\n    ')

    def test_recursive_tuple(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Tuple\n        class Foo(Tuple[Foo]): ...\n      ')
            ty = self.Infer('\n        import foo\n        x = foo.Foo()[0]\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        x: foo.Foo\n      ')

    def test_base_class(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing import Iterable\n      class Foo(Iterable):\n        pass\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Iterable\n      class Foo(Iterable): ...\n    ')

    def test_type_checking(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import typing\n      if typing.TYPE_CHECKING:\n          pass\n      else:\n          name_error\n    ')

    def test_not_type_checking(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import typing\n      if not typing.TYPE_CHECKING:\n          name_error\n      else:\n          pass\n    ')

    def test_new_type_arg_error(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors("\n      from typing import NewType\n      MyInt = NewType(int, 'MyInt')  # wrong-arg-types[e1]\n      MyStr = NewType(tp='str', name='MyStr')  # wrong-arg-types[e2]\n      MyFunnyNameType = NewType(name=123 if __random__ else 'Abc', tp=int)  # wrong-arg-types[e3]\n      MyFunnyType = NewType(name='Abc', tp=int if __random__ else 'int')  # wrong-arg-types[e4]\n    ")
        self.assertErrorRegexes(errors, {'e1': '.*Expected:.*str.*\\nActually passed:.*Type\\[int\\].*', 'e2': '.*Expected:.*type.*\\nActually passed:.*str.*', 'e3': '.*Expected:.*str.*\\nActually passed:.*Union.*', 'e4': '.*Expected:.*type.*\\nActually passed:.*Union.*'})

    def test_classvar(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import ClassVar\n      class A:\n        x = 5  # type: ClassVar[int]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import ClassVar\n      class A:\n        x: ClassVar[int]\n    ')

    def test_pyi_classvar(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import ClassVar\n        class X:\n          v: ClassVar[int]\n      ')
            self.Check('\n        import foo\n        foo.X.v + 42\n      ', pythonpath=[d.path])

    def test_pyi_classvar_argcount(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import ClassVar\n        class X:\n          v: ClassVar[int, int]\n      ')
            errors = self.CheckWithErrors('\n        import foo  # pyi-error[e]\n      ', pythonpath=[d.path])
        self.assertErrorRegexes(errors, {'e': 'ClassVar.*1.*2'})

    def test_reuse_name(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      from typing import Sequence as Sequence_\n      Sequence = Sequence_[int]\n    ')
        self.assertTypesMatchPytd(ty, '\n      import typing\n      from typing import Any\n      Sequence = typing.Sequence[int]\n      Sequence_: type\n    ')

    def test_type_checking_local(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import TYPE_CHECKING\n      def f():\n        if not TYPE_CHECKING:\n          name_error  # should be ignored\n    ')

class LiteralTest(test_base.BaseTest):
    """Tests for typing.Literal."""

    def test_pyi_parameter(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Literal\n        def f(x: Literal[True]) -> int: ...\n        def f(x: Literal[False]) -> float: ...\n        def f(x: bool) -> complex: ...\n      ')
            ty = self.Infer('\n        import foo\n        x = None  # type: bool\n        v1 = foo.f(True)\n        v2 = foo.f(False)\n        v3 = foo.f(x)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        x: bool\n        v1: int\n        v2: float\n        v3: complex\n      ')

    def test_pyi_return(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Literal\n        def okay() -> Literal[True]: ...\n      ')
            ty = self.Infer('\n        import foo\n        if not foo.okay():\n          x = "oh no"\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, 'import foo')

    def test_pyi_variable(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Literal\n        OKAY: Literal[True]\n      ')
            ty = self.Infer('\n        import foo\n        if not foo.OKAY:\n          x = "oh no"\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, 'import foo')

    def test_pyi_typing_extensions(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing_extensions import Literal\n        OKAY: Literal[True]\n      ')
            ty = self.Infer('\n        import foo\n        if not foo.OKAY:\n          x = "oh no"\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, 'import foo')

    def test_pyi_value(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', "\n        import enum\n        from typing import Literal\n\n        class Color(enum.Enum):\n          RED: str\n\n        def f1(x: Literal[True]) -> None: ...\n        def f2(x: Literal[2]) -> None: ...\n        def f3(x: Literal[None]) -> None: ...\n        def f4(x: Literal['hello']) -> None: ...\n        def f5(x: Literal[b'hello']) -> None: ...\n        def f6(x: Literal[u'hello']) -> None: ...\n        def f7(x: Literal[Color.RED]) -> None: ...\n      ")
            self.Check("\n        import foo\n        foo.f1(True)\n        foo.f2(2)\n        foo.f3(None)\n        foo.f4('hello')\n        foo.f5(b'hello')\n        foo.f6(u'hello')\n        foo.f7(foo.Color.RED)\n      ", pythonpath=[d.path])

    def test_pyi_multiple(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Literal\n        def f(x: Literal[False, None]) -> int: ...\n        def f(x) -> str: ...\n      ')
            ty = self.Infer('\n        import foo\n        v1 = foo.f(False)\n        v2 = foo.f(None)\n        v3 = foo.f(True)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        v1: int\n        v2: int\n        v3: str\n      ')

    def test_reexport(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Literal\n        x: Literal[True]\n        y: Literal[None]\n      ')
            ty = self.Infer('\n        import foo\n        x = foo.x\n        y = foo.y\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        x: bool\n        y: None\n      ')

    def test_string(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import IO, Literal\n        def open(f: str, mode: Literal["r", "rt"]) -> str: ...\n        def open(f: str, mode: Literal["rb"]) -> int: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f1(f):\n          return foo.open(f, mode="r")\n        def f2(f):\n          return foo.open(f, mode="rt")\n        def f3(f):\n          return foo.open(f, mode="rb")\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        def f1(f) -> str: ...\n        def f2(f) -> str: ...\n        def f3(f) -> int: ...\n      ')

    def test_unknown(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Literal\n        def f(x: Literal[True]) -> int: ...\n        def f(x: Literal[False]) -> str: ...\n      ')
            ty = self.Infer('\n        import foo\n        v = foo.f(__any_object__)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any\n        v: Any\n      ')

    def test_literal_constant(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Literal, overload\n        x: Literal["x"]\n        y: Literal["y"]\n        @overload\n        def f(arg: Literal["x"]) -> int: ...\n        @overload\n        def f(arg: Literal["y"]) -> str: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f1():\n          return foo.f(foo.x)\n        def f2():\n          return foo.f(foo.y)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        def f1() -> int: ...\n        def f2() -> str: ...\n      ')

    def test_illegal_literal_class(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Literal\n        class NotEnum:\n          A: int\n        x: Literal[NotEnum.A]\n      ')
            self.CheckWithErrors('\n        import foo  # pyi-error\n      ', pythonpath=[d.path])

    def test_illegal_literal_class_indirect(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class NotEnum:\n          A: int\n      ')
            d.create_file('bar.pyi', '\n        from typing import Literal\n        import foo\n        y: Literal[foo.NotEnum.A]\n      ')
            self.CheckWithErrors('\n        import bar  # pyi-error\n      ', pythonpath=[d.path])

    def test_missing_enum_member(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        import enum\n        from typing import Literal\n        class M(enum.Enum):\n          A: int\n        x: Literal[M.B]\n      ')
            self.CheckWithErrors('\n        import foo  # pyi-error\n      ', pythonpath=[d.path])

    def test_illegal_literal_typevar(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', "\n        from typing import Literal, TypeVar\n        T = TypeVar('T')\n        x: Literal[T]\n      ")
            self.CheckWithErrors('\n        import foo  # pyi-error\n      ', pythonpath=[d.path])

class NotSupportedYetTest(test_base.BaseTest):
    """Tests for importing typing constructs only present in some Python versions.

  We want pytype to behave as follows:

  Is the construct supported by pytype?
  |
  -> No: Log a plain [not supported-yet] error.
  |
  -> Yes: Is the construct being imported from typing_extensions or typing?
     |
     -> typing_extensions: Do not log any errors.
     |
     -> typing: Is the construct present in the runtime typing module?
        |
        -> No: Log [not-supported-yet] with a hint to use typing_extensions.
        |
        -> Yes: Do not log any errors.

  These tests currently use TypeVarTuple (added in 3.11) as the unsupported
  construct and Final (added in 3.8) as the supported construct. Replace them as
  needed as pytype's supported features and runtime versions change.
  """

    def test_unsupported_extension(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      from typing_extensions import TypeVarTuple  # not-supported-yet[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'typing_extensions.TypeVarTuple not supported yet$'})

    def test_unsupported_construct(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      from typing import TypeVarTuple  # not-supported-yet[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'typing.TypeVarTuple not supported yet$'})

    def test_supported_extension(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing_extensions import Final\n    ')

    def test_supported_construct_in_supported_version(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import Final\n    ')
if __name__ == '__main__':
    test_base.main()