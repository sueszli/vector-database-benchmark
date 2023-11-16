"""Test functools overlay."""
from pytype.tests import test_base

class TestCachedProperty(test_base.BaseTest):
    """Tests for @cached.property."""

    def test_basic(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import functools\n\n      class A:\n        @functools.cached_property\n        def f(self):\n          return 42\n\n      a = A()\n\n      x = a.f\n      assert_type(x, int)\n\n      a.f = 43\n      x = a.f\n      assert_type(x, int)\n\n      del a.f\n      x = a.f\n      assert_type(x, int)\n    ')

    def test_reingest(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.py', '\n            import functools\n\n            class A:\n              @functools.cached_property\n              def f(self):\n                return 42\n         ')]):
            self.Check('\n        import foo\n\n        a = foo.A()\n\n        x = a.f\n        assert_type(x, int)\n\n        a.f = 43\n        x = a.f\n        assert_type(x, int)\n\n        del a.f\n        x = a.f\n        assert_type(x, int)\n      ')

    @test_base.skip('Not supported yet')
    def test_pyi(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', '\n            import functools\n\n            class A:\n              @functools.cached_property\n              def f(self) -> int: ...\n         ')]):
            self.Check('\n        import foo\n\n        a = A()\n\n        x = a.f\n        assert_type(x, int)\n\n        a.f = 43\n        x = a.f\n        assert_type(x, int)\n\n        del a.f\n        x = a.f\n        assert_type(x, int)\n      ')

    def test_infer(self):
        if False:
            return 10
        ty = self.Infer('\n      from functools import cached_property\n    ')
        self.assertTypesMatchPytd(ty, '\n      import functools\n      cached_property: type[functools.cached_property]\n    ')
if __name__ == '__main__':
    test_base.main()