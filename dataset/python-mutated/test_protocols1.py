"""Tests for matching against protocols.

Based on PEP 544 https://www.python.org/dev/peps/pep-0544/.
"""
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class ProtocolTest(test_base.BaseTest):
    """Tests for protocol implementation."""

    def test_use_iterable(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class A:\n        def __iter__(self):\n          return iter(__any_object__)\n      v = list(A())\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class A:\n        def __iter__(self) -> Any: ...\n      v = ...  # type: list\n    ')

    def test_generic(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Generic, Protocol, TypeVar\n        T = TypeVar("T")\n        class Foo(Protocol[T]): ...\n      ')
            self.Check('\n        import foo\n      ', pythonpath=[d.path])

    def test_generic_py(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import Protocol, TypeVar\n      T = TypeVar("T")\n      class Foo(Protocol[T]):\n        pass\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Generic, Protocol, TypeVar\n      T = TypeVar("T")\n      class Foo(Generic[T], Protocol): ...\n    ')

    def test_generic_alias(self):
        if False:
            for i in range(10):
                print('nop')
        foo_ty = self.Infer('\n      from typing import Protocol, TypeVar\n      T = TypeVar("T")\n      Foo = Protocol[T]\n\n      class Bar(Foo[T]):\n        pass\n    ')
        self.assertTypesMatchPytd(foo_ty, '\n      from typing import Generic, Protocol, TypeVar\n      T = TypeVar("T")\n      Foo = Protocol[T]\n      class Bar(Generic[T], Protocol): ...\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            ty = self.Infer("\n        import foo\n        from typing import TypeVar\n        T = TypeVar('T')\n        class Baz(foo.Foo[T]):\n          pass\n      ", pythonpath=[d.path])
        self.assertTypesMatchPytd(ty, "\n      import foo\n      from typing import Generic, Protocol, TypeVar\n      T = TypeVar('T')\n      class Baz(Generic[T], Protocol): ...\n    ")

    def test_self_referential_protocol(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Generic, TypeVar\n        _TElem = TypeVar("_TElem")\n        _TIter = TypeVar("_TIter", bound=Iter)\n        class Iter(Generic[_TElem]):\n          def __init__(self): ...\n          def next(self) -> _TElem: ...\n          def __next__(self) -> _TElem: ...\n          def __iter__(self) -> _TIter: ...\n      ')
            self.Check('\n        import foo\n        i = foo.Iter[int]()\n        next(i)\n      ', pythonpath=[d.path])

    def test_attribute(self):
        if False:
            return 10
        self.CheckWithErrors("\n      from typing import Protocol\n      class Foo(Protocol):\n        x = 0\n      class Bar:\n        x = 1\n      class Baz:\n        x = '2'\n      def f(foo):\n        # type: (Foo) -> None\n        pass\n      f(Bar())\n      f(Baz())  # wrong-arg-types\n    ")

    def test_pyi_protocol_in_typevar(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', "\n        from typing import Generic, TypeVar\n        from typing_extensions import Protocol\n\n        T = TypeVar('T', bound=SupportsClose)\n\n        class SupportsClose(Protocol):\n          def close(self) -> object: ...\n\n        class Foo(Generic[T]):\n          def __init__(self, x: T) -> None: ...\n      ")
            self.Check('\n        import foo\n        class Bar:\n          def close(self) -> None:\n            pass\n        foo.Foo(Bar())\n      ', pythonpath=[d.path])
if __name__ == '__main__':
    test_base.main()