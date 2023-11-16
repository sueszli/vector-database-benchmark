"""Tests for disabling errors."""
from pytype.tests import test_base

class DisableTest(test_base.BaseTest):
    """Test error disabling."""

    def test_invalid_directive(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      x = 1  # pytype: this is not a valid pytype directive.  # invalid-directive\n    ')
        self.assertFalse(errors.has_error())

    def test_invalid_disable_error_name(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      x = 1  # pytype: disable=not-an-error.  # invalid-directive[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Invalid error name.*not-an-error'})
        self.assertFalse(errors.has_error())

    def test_disable_error(self):
        if False:
            for i in range(10):
                print('nop')
        self.InferWithErrors('\n      x = a  # name-error\n      x = b  # pytype: disable=name-error\n      x = c  # name-error\n    ')

    def test_open_ended_directive(self):
        if False:
            i = 10
            return i + 15
        "Test that disables in the middle of the file can't be left open-ended."
        (_, errors) = self.InferWithErrors("\n      '''This is a docstring.\n      def f(x):\n        pass\n      class A:\n        pass\n      The above definitions should be ignored.\n      '''\n      # pytype: disable=attribute-error  # ok (before first class/function def)\n      CONSTANT = 42\n      # pytype: disable=not-callable  # ok (before first class/function def)\n      def f(x):\n        # type: ignore  # late-directive[e1]\n        pass\n      def g(): pass\n      x = y  # pytype: disable=name-error  # ok (single line)\n      # pytype: disable=attribute-error  # ok (re-enabled)\n      # pytype: disable=wrong-arg-types  # late-directive[e2]\n      # pytype: enable=attribute-error\n    ")
        self.assertErrorRegexes(errors, {'e1': 'Type checking', 'e2': 'wrong-arg-types'})
        self.assertFalse(errors.has_error())

    def test_skip_file(self):
        if False:
            print('Hello World!')
        self.Check('\n      # pytype: skip-file\n      name_error\n    ')

    def test_implicit_return(self):
        if False:
            while True:
                i = 10
        'Test that the return is attached to the last line of the function.'
        self.Check('\n      class A:\n        def f(self) -> str:\n          if __random__:\n            if __random__:\n              return "a"  # pytype: disable=bad-return-type\n\n      def g() -> str:\n        pass  # pytype: disable=bad-return-type\n\n      def h() -> str:\n        return ([1,\n                 2,\n                 3])  # pytype: disable=bad-return-type\n    ')

    def test_implicit_return_empty_function(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      def f():\n        pass\n\n      def j() -> str:\n        '''docstring'''  # pytype: disable=bad-return-type\n    ")

    def test_implicit_return_not_at_end(self):
        if False:
            while True:
                i = 10
        self.Check("\n      import logging\n      def f() -> str:\n        try:\n          return ''\n        except KeyError:\n          logging.exception(  # pytype: disable=bad-return-type\n              'oops')\n    ")

    def test_implicit_return_annotated_nested_function(self):
        if False:
            return 10
        self.Check("\n      import logging\n      def f():\n        def g() -> str:\n          try:\n            return ''\n          except:\n            logging.exception('oops')  # pytype: disable=bad-return-type\n        return g\n    ")

    def test_implicit_return_annotated_outer_function(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      def f() -> str:\n        def g():\n          pass\n        pass  # pytype: disable=bad-return-type\n    ')

    def test_silence_variable_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      x = [\n          0,\n      ]  # type: None  # pytype: disable=annotation-type-mismatch\n    ')

    def test_disable_location(self):
        if False:
            return 10
        self.Check("\n      import re\n      re.sub(\n        '', object(), '')  # pytype: disable=wrong-arg-types\n    ")

    def test_skip_file_with_comment(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      # pytype: skip-file  # extra comment here\n      import nonsense\n    ')

    def test_missing_parameter_disable(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        def __iter__(self, x, y):\n          pass\n      def f(x):\n        pass\n      f(\n        x=[x for x in Foo],  # pytype: disable=missing-parameter\n      )\n    ')

    def test_silence_parameter_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      def f(\n        x: int = 0.0,\n        y: str = '',\n        **kwargs,\n      ):  # pytype: disable=annotation-type-mismatch\n        pass\n    ")

    def test_do_not_silence_parameter_mismatch(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors("\n      def f(  # annotation-type-mismatch\n        x: int = 0.0,\n        y: str = '',\n        **kwargs,\n      ):\n        pass  # pytype: disable=annotation-type-mismatch\n    ")

    def test_container_disable(self):
        if False:
            print('Hello World!')
        self.Check("\n      x: list[int] = []\n      x.append(\n          ''\n      )  # pytype: disable=container-type-mismatch\n    ")

    def test_multiple_directives(self):
        if False:
            return 10
        'We should support multiple directives on one line.'
        self.Check('\n      a = list() # type: list[int, str]  # pytype: disable=invalid-annotation\n      b = list() # pytype: disable=invalid-annotation  # type: list[int, str]\n      def foo(x): pass\n      c = foo(a, b.i) # pytype: disable=attribute-error  # pytype: disable=wrong-arg-count\n    ')

    def test_bare_annotation(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import AnyStr\n      def f():\n        x: AnyStr  # pytype: disable=invalid-annotation\n    ')

class AttributeErrorDisableTest(test_base.BaseTest):
    """Test attribute-error disabling."""

    def test_disable(self):
        if False:
            return 10
        self.Check("\n      x = [None]\n      y = ''.join(z.oops\n                  for z in x)  # pytype: disable=attribute-error\n    ")

    def test_method_disable(self):
        if False:
            print('Hello World!')
        self.Check("\n      x = [None]\n      y = ''.join(z.oops()\n                  for z in x)  # pytype: disable=attribute-error\n    ")

    def test_iter_disable(self):
        if False:
            print('Hello World!')
        self.Check('\n      x = [y for y in None\n          ]  # pytype: disable=attribute-error\n    ')

    def test_unpack_disable(self):
        if False:
            while True:
                i = 10
        self.Check('\n      x, y, z = (\n        None)  # pytype: disable=attribute-error\n    ')

    def test_contextmanager_disable(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def f():\n        return None\n      with f(\n          ):  # pytype: disable=attribute-error\n        pass\n    ')

    def test_regular_disable(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        pass\n      def f(a):\n        pass\n      f(\n          Foo.nonexistent)  # pytype: disable=attribute-error\n    ')
if __name__ == '__main__':
    test_base.main()