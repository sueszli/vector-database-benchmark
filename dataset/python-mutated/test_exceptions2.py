"""Test exceptions."""
from pytype.tests import test_base
from pytype.tests import test_utils

class TestExceptionsPy3(test_base.BaseTest):
    """Exception tests."""

    def test_reraise(self):
        if False:
            i = 10
            return i + 15
        self.assertNoCrash(self.Check, '\n      raise\n    ')

    def test_raise_exception_from(self):
        if False:
            return 10
        self.Check('raise ValueError from NameError')

    def test_exception_message(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('ValueError().message  # attribute-error')

    def test_suppress_context(self):
        if False:
            i = 10
            return i + 15
        self.Check('ValueError().__suppress_context__')

    def test_return_or_call_to_raise(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      from typing import NoReturn\n      def e() -> NoReturn:\n        raise ValueError('this is an error')\n      def f():\n        if __random__:\n          return 16\n        else:\n          e()\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Never\n\n      def e() -> Never: ...\n      def f() -> int: ...\n    ')

    def test_union(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Type, Union\n      class Foo:\n        @property\n        def exception_types(self) -> Type[Union[ValueError, IndexError]]:\n          return ValueError\n      def f(x: Foo):\n        try:\n          pass\n        except x.exception_types as e:\n          return e\n    ')

    def test_bad_union(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      from typing import Type, Optional\n      class Foo:\n        @property\n        def exception_types(self) -> Type[Optional[ValueError]]:\n          return ValueError\n      def f(x: Foo):\n        try:\n          print(x)\n        except x.exception_types as e:  # mro-error[e]\n          return e\n    ')
        self.assertErrorRegexes(errors, {'e': 'NoneType does not inherit from BaseException'})

    @test_utils.skipIfPy((3, 8), reason='failing, not worth fixing since this works again in 3.9')
    def test_no_return_in_finally(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import array\n      import os\n      def f(fd) -> int:\n        try:\n          buf = array.array("l", [0])\n          return buf[0]\n        except (IOError, OSError):\n          return 0\n        finally:\n          os.close(fd)\n    ')

    def test_contextmanager(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      class Foo:\n        def __enter__(self):\n          return self\n        def __exit__(self, exc_type, exc_value, tb):\n          reveal_type(exc_type)  # reveal-type[e]\n          return False\n      with Foo():\n        print(0)\n    ')
        self.assertErrorSequences(errors, {'e': ['Optional[Type[BaseException]]']})

    def test_yield_from(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      def f():\n        yield from g()\n      def g():\n        try:\n          __any_object__()\n        except Exception as e:\n          print(any(s in str(e) for s in 'abcde'))\n        yield None\n    ")

    def test_raise_exc_info(self):
        if False:
            return 10
        self.Check('\n      import sys\n      exception = sys.exc_info()\n      exception_type = exception[0]\n      if exception_type:\n        raise exception_type()\n    ')
if __name__ == '__main__':
    test_base.main()