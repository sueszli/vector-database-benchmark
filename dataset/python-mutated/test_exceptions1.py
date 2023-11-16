"""Test exceptions."""
from pytype.tests import test_base
from pytype.tests import test_utils

class TestExceptions(test_base.BaseTest):
    """Exception tests."""

    def test_exceptions(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f():\n        try:\n          raise ValueError()  # exercise byte_RAISE_VARARGS\n        except ValueError as e:\n          x = "s"\n        finally:  # exercise byte_POP_EXCEPT\n          x = 3\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f() -> int: ...\n    ')

    def test_catching_exceptions(self):
        if False:
            i = 10
            return i + 15
        self.assertNoCrash(self.Check, '\n      try:\n        x[1]\n        print("Shouldn\'t be here...")\n      except NameError:\n        print("caught it!")\n      ')
        self.assertNoCrash(self.Check, '\n      try:\n        x[1]\n        print("Shouldn\'t be here...")\n      except Exception:\n        print("caught it!")\n      ')
        self.assertNoCrash(self.Check, '\n      try:\n        x[1]\n        print("Shouldn\'t be here...")\n      except:\n        print("caught it!")\n      ')

    def test_raise_exception(self):
        if False:
            print('Hello World!')
        self.Check("raise Exception('oops')")

    def test_raise_exception_class(self):
        if False:
            return 10
        self.Check('raise ValueError')

    def test_raise_and_catch_exception(self):
        if False:
            while True:
                i = 10
        self.Check('\n      try:\n        raise ValueError("oops")\n      except ValueError as e:\n        print("Caught: %s" % e)\n      print("All done")\n      ')

    def test_raise_and_catch_exception_in_function(self):
        if False:
            return 10
        self.Check('\n      def fn():\n        raise ValueError("oops")\n\n      try:\n        fn()\n      except ValueError as e:\n        print("Caught: %s" % e)\n      print("done")\n      ')

    def test_global_name_error(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('fooey  # name-error')
        self.assertNoCrash(self.Check, '\n      try:\n        fooey\n        print("Yes fooey?")\n      except NameError:\n        print("No fooey")\n    ')

    def test_local_name_error(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      def fn():\n        fooey  # name-error\n      fn()\n    ')

    def test_catch_local_name_error(self):
        if False:
            print('Hello World!')
        self.assertNoCrash(self.Check, '\n      def fn():\n        try:\n          fooey\n          print("Yes fooey?")\n        except NameError:\n          print("No fooey")\n      fn()\n      ')

    def test_reraise(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      def fn():\n        try:\n          fooey  # name-error\n          print("Yes fooey?")\n        except NameError:\n          print("No fooey")\n          raise\n      fn()\n    ')

    def test_reraise_explicit_exception(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      def fn():\n        try:\n          raise ValueError("ouch")\n        except ValueError as e:\n          print("Caught %s" % e)\n          raise\n      fn()\n    ')

    def test_reraise_in_function_call(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def raise_error(e):\n        raise(e)\n\n      def f():\n        try:\n          return "hello"\n        except Exception as e:\n          raise_error(e)\n\n      f().lower()  # f() should be str, not str|None\n    ')

    def test_finally_while_throwing(self):
        if False:
            print('Hello World!')
        self.Check('\n      def fn():\n        try:\n          print("About to..")\n          raise ValueError("ouch")\n        finally:\n          print("Finally")\n      fn()\n      print("Done")\n    ')

    def test_coverage_issue_92(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      l = []\n      for i in range(3):\n        try:\n          l.append(i)\n        finally:\n          l.append('f')\n        l.append('e')\n      l.append('r')\n      print(l)\n      assert l == [0, 'f', 'e', 1, 'f', 'e', 2, 'f', 'e', 'r']\n      ")

    def test_continue_in_except(self):
        if False:
            return 10
        self.Check("\n      for i in range(3):\n        try:\n          pass\n        except:\n          print(i)\n          continue\n        print('e')\n      ")

    def test_loop_finally_except(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      def f():\n        for s in (1, 2):\n          try:\n            try:\n              break\n            except:\n              continue\n          finally:\n            pass\n      ')

    def test_inherit_from_exception(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo(Exception):\n        pass\n\n      def bar(x):\n        return Foo(x)\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo(Exception):\n        pass\n\n      def bar(x) -> Foo: ...\n    ')

    def test_match_exception_type(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('warnings.pyi', '\n        from typing import Optional, Type, Union\n        def warn(message: Union[str, Warning],\n                 category: Optional[Type[Warning]] = ...,\n                 stacklevel: int = ...) -> None: ...\n      ')
            ty = self.Infer('\n        import warnings\n        def warn():\n          warnings.warn(\n            "set_prefix() is deprecated; use the prefix property",\n            DeprecationWarning, stacklevel=2)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import warnings\n        def warn() -> None: ...\n      ')

    def test_end_finally(self):
        if False:
            return 10
        ty = self.Infer('\n      def foo():\n        try:\n          assert True\n          return 42\n        except Exception:\n          return 42\n    ')
        self.assertTypesMatchPytd(ty, '\n      def foo() -> int: ...\n    ')

    @test_utils.skipFromPy((3, 11), reason='Code gets eliminated very early, not worth fixing')
    def test_dont_eliminate_except_block(self):
        if False:
            return 10
        ty = self.Infer('\n      def foo():\n        try:\n          return 42\n        except Exception:\n          return 1+3j\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def foo() -> Union[int, complex]: ...\n    ')

    @test_utils.skipBeforePy((3, 11), reason='New behaviour in 3.11')
    def test_eliminate_except_block(self):
        if False:
            return 10
        ty = self.Infer('\n      def foo():\n        try:\n          return 42\n        except Exception:\n          return 1+3j\n    ')
        self.assertTypesMatchPytd(ty, '\n      def foo() -> int: ...\n    ')

    def test_assert(self):
        if False:
            return 10
        ty = self.Infer('\n      def foo():\n        try:\n          assert True\n          return 42\n        except:\n          return 1+3j\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n\n      def foo() -> Union[complex, int]: ...\n    ')

    def test_never(self):
        if False:
            return 10
        ty = self.Infer('\n      def f():\n        raise ValueError()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Never\n      def f() -> Never: ...\n    ')

    def test_never_chain(self):
        if False:
            return 10
        ty = self.Infer('\n      def f():\n        raise ValueError()\n      def g():\n        f()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Never\n      def f() -> Never: ...\n      def g() -> Never: ...\n    ')

    def test_try_except_never(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f():\n        try:\n          raise ValueError()\n        except ValueError as e:\n          raise ValueError(str(e))\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Never\n      def f() -> Never: ...\n    ')

    def test_callable_noreturn(self):
        if False:
            return 10
        self.Check('\n      from typing import Callable, NoReturn\n      def f(x: Callable[[], NoReturn]) -> NoReturn:\n        x()\n    ')

    def test_callable_noreturn_branch(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Callable, NoReturn\n      def f(x: Callable[[], NoReturn], y: int) -> int:\n        if y % 2:\n          return y\n        else:\n          x()\n    ')

    def test_return_or_raise(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f():\n        if __random__:\n          return 42\n        else:\n          raise ValueError()\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f() -> int: ...\n    ')

    def test_return_or_raise_set_attribute(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors('\n      def f():\n        raise ValueError()\n      def g():\n        return ""\n      def h():\n        func = f if __random__ else g\n        v = func()\n        v.attr = None  # not-writable\n    ')

    def test_bad_type_self(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      class Foo:\n        def __init__(self):\n          type(42, self)  # wrong-arg-count[e]\n    ')
        self.assertErrorRegexes(errors, {'e': '2.*3'})

    def test_value(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f():\n        try:\n          raise KeyError()\n        except KeyError as e:\n          return e\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f() -> KeyError: ...\n    ')

    def test_value_from_tuple(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import Tuple, Type\n      def f():\n        try:\n          raise KeyError()\n        except (KeyError, ValueError) as e:\n          return e\n      def g():\n        try:\n          raise KeyError()\n        except ((KeyError,),) as e:\n          return e\n      def h():\n        tup = None  # type: Tuple[Type[KeyError], ...]\n        try:\n          raise KeyError()\n        except tup as e:\n          return e\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def f() -> Union[KeyError, ValueError]: ...\n      def g() -> KeyError: ...\n      def h() -> KeyError: ...\n    ')

    def test_bad_type(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      try:\n        x = 1\n      except None:  # mro-error[e1]\n        pass\n      try:\n        x = 2\n      except type(None):  # mro-error[e2]\n        pass\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Not a class', 'e2': 'None.*BaseException'})

    def test_unknown_type(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      try:\n        pass\n      except __any_object__:\n        pass\n    ')

    def test_attribute(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class MyException(BaseException):\n        def __init__(self):\n          self.x = ""\n      def f():\n        try:\n          raise MyException()\n        except MyException as e:\n          return e.x\n    ')
        self.assertTypesMatchPytd(ty, '\n      class MyException(BaseException):\n        x: str\n        def __init__(self) -> None: ...\n      def f() -> str: ...\n    ')

    def test_reuse_name(self):
        if False:
            print('Hello World!')
        self.Check('\n      def f():\n        try:\n          pass\n        except __any_object__ as e:\n          return e\n        try:\n          pass\n        except __any_object__ as e:\n          return e\n    ')

    def test_unknown_base(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class MyException(__any_object__):\n        pass\n      try:\n        pass\n      except MyException:\n        pass\n    ')

    def test_contextmanager(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      _temporaries = {}\n      def f(name):\n        with __any_object__:\n          filename = _temporaries.get(name)\n          (filename, data) = __any_object__\n          if not filename:\n            assert data is not None\n        return filename\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Dict\n      _temporaries: Dict[nothing, nothing]\n      def f(name) -> Any: ...\n    ')

    def test_no_except(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f():\n        try:\n          if __random__:\n            raise ValueError()\n        finally:\n          __any_object__()\n        return 0\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f() -> int: ...\n    ')

    def test_traceback(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      class Foo(Exception):\n        pass\n      x = Foo().with_traceback(None)\n      assert_type(x, Foo)\n    ')
if __name__ == '__main__':
    test_base.main()