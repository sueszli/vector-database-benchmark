"""Tests for python3 function call features."""
from pytype.tests import test_base

class TestCalls(test_base.BaseTest):
    """Tests for checking function calls."""

    def test_starstarargs_with_kwonly(self):
        if False:
            i = 10
            return i + 15
        'Args defined as kwonly should be removed from **kwargs.'
        self.Check('\n      def f(a):\n        return a\n      def g(*args, kw=False, **kwargs):\n        # When called from h, **kwargs should not include `kw=True`\n        return f(*args, **kwargs)\n      def h():\n        return g(1, kw=True)\n    ')
if __name__ == '__main__':
    test_base.main()