"""Tests for typing.AnyStr."""
from pytype.tests import test_base
from pytype.tests import test_utils

class AnyStrTest(test_base.BaseTest):
    """Tests for issues related to AnyStr."""

    def test_type_parameters(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import AnyStr\n        def f(x: AnyStr) -> AnyStr: ...\n      ')
            ty = self.Infer('\n        import a\n        if a.f(""):\n          x = 3\n        if a.f("hello"):\n          y = 3\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        x = ...  # type: int\n        y = ...  # type: int\n      ')

    def test_format(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import AnyStr\n        def f(x: AnyStr) -> AnyStr: ...\n      ')
            self.Check('\n        import foo\n        foo.f("" % __any_object__)\n      ', pythonpath=[d.path])
if __name__ == '__main__':
    test_base.main()