"""Tests for calling other functions, and the corresponding checks."""
from pytype.tests import test_base
from pytype.tests import test_utils

class CallsTest(test_base.BaseTest):
    """Tests for checking function calls."""

    def test_optional(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('mod.pyi', '\n        def foo(x: int, y: int = ..., z: int = ...) -> int: ...\n      ')
            self.Check('\n        import mod\n        mod.foo(1)\n        mod.foo(1, 2)\n        mod.foo(1, 2, 3)\n      ', pythonpath=[d.path])

    def test_missing(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('mod.pyi', '\n        def foo(x, y) -> int: ...\n      ')
            self.InferWithErrors('\n        import mod\n        mod.foo(1)  # missing-parameter\n      ', pythonpath=[d.path])

    def test_extraneous(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('mod.pyi', '\n        def foo(x, y) -> int: ...\n      ')
            self.InferWithErrors('\n        import mod\n        mod.foo(1, 2, 3)  # wrong-arg-count\n      ', pythonpath=[d.path])

    def test_missing_kwonly(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('mod.pyi', '\n        def foo(x, y, *, z) -> int: ...\n      ')
            (_, errors) = self.InferWithErrors('\n        import mod\n        mod.foo(1, 2)  # missing-parameter[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': '\\bz\\b'})

    def test_extra_keyword(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('mod.pyi', '\n        def foo(x, y) -> int: ...\n      ')
            self.InferWithErrors('\n        import mod\n        mod.foo(1, 2, z=3)  # wrong-keyword-args\n      ', pythonpath=[d.path])

    def test_varargs_with_kwonly(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('mod.pyi', '\n        def foo(*args: int, z: int) -> int: ...\n      ')
            self.Check('\n        import mod\n        mod.foo(1, 2, z=3)\n      ', pythonpath=[d.path])

    def test_varargs_with_missing_kwonly(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('mod.pyi', '\n        def foo(*args: int, z: int) -> int: ...\n      ')
            (_, errors) = self.InferWithErrors('\n        import mod\n        mod.foo(1, 2, 3)  # missing-parameter[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': '\\bz\\b'})
if __name__ == '__main__':
    test_base.main()