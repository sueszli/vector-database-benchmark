"""Tests for displaying tracebacks in error messages."""
from pytype.tests import test_base

class TracebackTest(test_base.BaseTest):
    """Tests for tracebacks in error messages."""

    def test_build_class(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      class Foo:\n        def f(self, x: Bar):  # name-error[e]\n          pass\n    ')
        self.assertErrorRegexes(errors, {'e': 'Bar.*not defined$'})
if __name__ == '__main__':
    test_base.main()