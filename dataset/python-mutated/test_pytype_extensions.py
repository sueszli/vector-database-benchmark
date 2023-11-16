"""Tests for pytype_extensions."""
import os
from pytype import errors
from pytype.pytd import pytd_utils
from pytype.tests import test_base

def InitContents():
    if False:
        while True:
            i = 10
    with open(os.path.join(os.path.dirname(__file__), '__init__.py')) as f:
        lines = f.readlines()
    return ''.join(lines)

def _Wrap(method):
    if False:
        i = 10
        return i + 15

    def Wrapper(self, code: str) -> errors.ErrorLog:
        if False:
            for i in range(10):
                print('nop')
        extensions_pyi = pytd_utils.Print(super(CodeTest, self).Infer(InitContents()))
        with self.DepTree([('pytype_extensions.pyi', extensions_pyi)]):
            return method(self, code)
    return Wrapper

class CodeTest(test_base.BaseTest):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super().setUpClass()
        cls.Check = _Wrap(cls.Check)
        cls.CheckWithErrors = _Wrap(cls.CheckWithErrors)
        cls.Infer = _Wrap(cls.Infer)
        cls.InferWithErrors = _Wrap(cls.InferWithErrors)

class DecoratorTest(CodeTest):
    """Tests for pytype_extensions.Decorator."""

    def test_plain_decorator(self):
        if False:
            return 10
        errorlog = self.CheckWithErrors('\n        import pytype_extensions\n\n        @pytype_extensions.Decorator\n        def MyDecorator(f):\n          def wrapper(*a, **kw):\n            return f(*a, **kw)\n          return wrapper\n\n\n        class MyClz(object):\n\n          @MyDecorator\n          def DecoratedMethod(self, i: int) -> float:\n            reveal_type(self)  # reveal-type[e1]\n            return i / 2\n\n          def PytypeTesting(self):\n            reveal_type(self.DecoratedMethod)  # reveal-type[e2]\n            reveal_type(self.DecoratedMethod(1))  # reveal-type[e3]\n\n\n        reveal_type(MyClz.DecoratedMethod)  # reveal-type[e4]\n    ')
        self.assertErrorRegexes(errorlog, {'e1': 'MyClz', 'e2': '.*Callable\\[\\[int\\], float\\].*', 'e3': 'float', 'e4': 'Callable\\[\\[Any, int\\], float\\]'})

    def test_decorator_factory(self):
        if False:
            for i in range(10):
                print('nop')
        errorlog = self.CheckWithErrors("\n        import pytype_extensions\n\n\n        def MyDecoratorFactory(level: int):\n          @pytype_extensions.Decorator\n          def decorator(f):\n            def wrapper(*a, **kw):\n              return f(*a, **kw)\n            return wrapper\n          return decorator\n\n\n        class MyClz(object):\n\n          @MyDecoratorFactory('should be int')  # wrong-arg-types[e1]\n          def MisDecoratedMethod(self) -> int:\n            return 'bad-return-type'  # bad-return-type[e2]\n\n          @MyDecoratorFactory(123)\n          def FactoryDecoratedMethod(self, i: int) -> float:\n            reveal_type(self)  # reveal-type[e3]\n            return i / 2\n\n          def PytypeTesting(self):\n            reveal_type(self.FactoryDecoratedMethod)  # reveal-type[e4]\n            reveal_type(self.FactoryDecoratedMethod(1))  # reveal-type[e5]\n\n\n        reveal_type(MyClz.FactoryDecoratedMethod)  # reveal-type[e6]\n    ")
        self.assertErrorRegexes(errorlog, {'e1': 'Expected.*int.*Actual.*str', 'e2': 'Expected.*int.*Actual.*str', 'e3': 'MyClz', 'e4': '.*Callable\\[\\[int\\], float\\].*', 'e5': 'float', 'e6': 'Callable\\[\\[Any, int\\], float\\]'})

class DataclassTest(CodeTest):
    """Tests for pytype_extensions.Dataclass."""

    def test_basic(self):
        if False:
            return 10
        self.CheckWithErrors("\n      import dataclasses\n      import pytype_extensions\n\n      @dataclasses.dataclass\n      class Foo:\n        x: str\n        y: str\n\n      @dataclasses.dataclass\n      class Bar:\n        x: str\n        y: int\n\n      class Baz:\n        x: str\n        y: int\n\n      def f(x: pytype_extensions.Dataclass[str]):\n        pass\n\n      f(Foo(x='yes', y='1'))  # ok\n      f(Bar(x='no', y=1))  # wrong-arg-types\n      f(Baz())  # wrong-arg-types\n    ")

    def test_fields(self):
        if False:
            print('Hello World!')
        self.Check('\n      import dataclasses\n      import pytype_extensions\n      def f(x: pytype_extensions.Dataclass):\n        return dataclasses.fields(x)\n    ')

class AttrsTest(CodeTest):
    """Test pytype_extensions.Attrs."""

    def test_attr_namespace(self):
        if False:
            return 10
        self.CheckWithErrors("\n      import attr\n      import pytype_extensions\n\n      @attr.s\n      class Foo:\n        x: int = attr.ib()\n        y: int = attr.ib()\n\n      @attr.s\n      class Bar:\n        x: int = attr.ib()\n        y: str = attr.ib()\n\n      class Baz:\n        x: int\n        y: str\n\n      def f(x: pytype_extensions.Attrs[int]):\n        pass\n\n      f(Foo(x=0, y=1))  # ok\n      f(Bar(x=0, y='no'))  # wrong-arg-types\n      f(Baz())  # wrong-arg-types\n    ")

    def test_attrs_namespace(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors("\n      import attrs\n      import pytype_extensions\n\n      @attrs.define\n      class Foo:\n        x: int\n        y: int\n\n      @attrs.define\n      class Bar:\n        x: int\n        y: str\n\n      class Baz:\n        x: int\n        y: str\n\n      def f(x: pytype_extensions.Attrs[int]):\n        pass\n\n      f(Foo(x=0, y=1))  # ok\n      f(Bar(x=0, y='no'))  # wrong-arg-types\n      f(Baz())  # wrong-arg-types\n    ")

def _WrapWithDeps(method, deps):
    if False:
        for i in range(10):
            print('nop')

    def Wrapper(self, code: str) -> errors.ErrorLog:
        if False:
            for i in range(10):
                print('nop')
        extensions_pyi = pytd_utils.Print(super(PyiCodeTest, self).Infer(InitContents()))
        with self.DepTree([('pytype_extensions.pyi', extensions_pyi)] + deps):
            return method(self, code)
    return Wrapper

class PyiCodeTest(test_base.BaseTest):
    _PYI_DEP = None

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        deps = [('foo.pyi', cls._PYI_DEP)]
        cls.Check = _WrapWithDeps(cls.Check, deps)
        cls.CheckWithErrors = _WrapWithDeps(cls.CheckWithErrors, deps)
        cls.Infer = _WrapWithDeps(cls.Infer, deps)
        cls.InferWithErrors = _WrapWithDeps(cls.InferWithErrors, deps)
_ATTRS_PYI = '\n  import attrs\n\n  @attrs.define\n  class Foo:\n    x: int\n    y: int\n'

class AttrsPyiTest(PyiCodeTest):
    _PYI_DEP = _ATTRS_PYI

    def test_basic(self):
        if False:
            print('Hello World!')
        self.Check('\n      import pytype_extensions\n      import foo\n\n      def f(x: pytype_extensions.Attrs[int]):\n        pass\n      f(foo.Foo(1, 2))\n    ')
_DATACLASS_PYI = '\n  import dataclasses\n\n  @dataclasses.dataclass\n  class Foo:\n    x: int\n    y: int\n'

class DataclassPyiTest(PyiCodeTest):
    _PYI_DEP = _DATACLASS_PYI

    def test_basic(self):
        if False:
            return 10
        self.Check('\n      import pytype_extensions\n      import foo\n\n      def f(x: pytype_extensions.Dataclass[int]):\n        pass\n      f(foo.Foo(1, 2))\n    ')
if __name__ == '__main__':
    test_base.main()