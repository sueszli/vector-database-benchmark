"""Tests for displaying tracebacks in error messages."""
from pytype.tests import test_base

class TracebackTest(test_base.BaseTest):
    """Tests for tracebacks in error messages."""

    def test_no_traceback(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      def f(x):\n        "hello" + 42  # unsupported-operands[e]\n      f("world")\n    ')
        self.assertErrorRegexes(errors, {'e': 'expects str$'})

    def test_same_traceback(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      def f(x, _):\n        x + 42  # unsupported-operands[e]\n      def g(x):\n        f("hello", x)\n      g("world")\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'Called from.*:\\n  line 4, in g'})

    def test_different_tracebacks(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      def f(x):\n        x + 42  # unsupported-operands[e1]  # unsupported-operands[e2]\n      f("hello")\n      f("world")\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Called from.*:\\n  line 3, in current file', 'e2': 'Called from.*:\\n  line 4, in current file'})

    def test_comprehension(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      def f():\n        return {x.upper() for x in range(10)}  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'upper.*int$'})
        (error,) = errors.errorlog
        self.assertEqual(error.methodname, 'f')

    def test_comprehension_in_traceback(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      def f(x):\n        return x.upper()  # attribute-error[e]\n      def g():\n        return {f(x) for x in range(10)}\n    ')
        self.assertErrorRegexes(errors, {'e': 'Called from.*:\\n  line 4, in g$'})

    def test_no_argument_function(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      def f():\n        return None.attr  # attribute-error[e]\n      f()\n    ')
        self.assertErrorRegexes(errors, {'e': 'attr.*None$'})

    def test_max_callsites(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      def f(s):\n        return "hello, " + s  # unsupported-operands[e1]  # unsupported-operands[e2]  # unsupported-operands[e3]\n      f(0)\n      f(1)\n      f(2)\n      f(3)\n    ')
        self.assertErrorRegexes(errors, {'e1': 'line 3', 'e2': 'line 4', 'e3': 'line 5'})
if __name__ == '__main__':
    test_base.main()