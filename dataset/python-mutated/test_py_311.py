"""Tests for 3.11 support.

These tests are separated into their own file because the main test suite
doesn't pass under python 3.11 yet; this lets us tests individual features as we
get them working.
"""
from pytype.tests import test_base
from pytype.tests import test_utils

@test_utils.skipBeforePy((3, 11), 'Tests specifically for 3.11 support')
class TestPy311(test_base.BaseTest):
    """Tests for python 3.11 support."""

    def test_binop(self):
        if False:
            return 10
        self.Check('\n      def f(x: int | str):\n        pass\n\n      def g(x: int, y: int) -> int:\n        return x & y\n\n      def h(x: int, y: int):\n        x ^= y\n    ')

    def test_method_call(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class A:\n        def f(self):\n          return 42\n\n      x = A().f()\n      assert_type(x, int)\n    ')

    def test_global_call(self):
        if False:
            print('Hello World!')
        self.Check('\n      def f(x):\n        return any(x)\n    ')

    def test_deref1(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def f(*args):\n        def rmdirs(\n            unlink,\n            dirname,\n            removedirs,\n            enoent_error,\n            directory,\n            files,\n        ):\n          for path in [dirname(f) for f in files]:\n            removedirs(path, directory)\n        rmdirs(*args)\n    ')

    def test_deref2(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      def f(x):\n        y = x\n        x = lambda: y\n\n        def g():\n          return x\n    ')

    def test_super(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class A:\n        def __init__(self):\n          super(A, self).__init__()\n    ')

    def test_call_function_ex(self):
        if False:
            print('Hello World!')
        self.Check('\n      import datetime\n      def f(*args):\n        return g(datetime.datetime(*args), 10)\n      def g(x, y):\n        return (x, y)\n    ')

    def test_callable_parameter_in_function(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import collections\n      class C:\n        def __init__(self):\n          self.x = collections.defaultdict(\n              lambda key: key)  # pytype: disable=wrong-arg-types\n    ')

    def test_async_for(self):
        if False:
            return 10
        self.Check('\n      class Client:\n        async def get_or_create_tensorboard(self):\n          response = await __any_object__\n          async for page in response.pages:\n            if page.tensorboards:\n              return response.tensorboards[0].name\n    ')

    def test_yield_from(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      def f():\n        yield 1\n        return 'a', 'b'\n      def g():\n        a, b = yield from f()\n        assert_type(a, str)\n        assert_type(b, str)\n      for x in g():\n        assert_type(x, int)\n    ")

    def test_splat(self):
        if False:
            return 10
        self.Check('\n      def f(value, g):\n        converted = []\n        if isinstance(value, (dict, *tuple({}))):\n          converted.append(value)\n        return g(*converted)\n    ')
if __name__ == '__main__':
    test_base.main()