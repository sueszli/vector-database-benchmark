"""Tests for functions module."""
from tensorflow.python.autograph.converters import functions
from tensorflow.python.autograph.converters import return_statements
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.impl import api
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test

class FunctionTransformer(converter_testing.TestCase):

    def test_basic(self):
        if False:
            while True:
                i = 10

        def f(l):
            if False:
                i = 10
                return i + 15
            'Docstring.'
            a = 1
            l += a
            return l
        tr = self.transform(f, functions)
        result_op = tr(constant_op.constant(1))
        self.assertIn('f/', result_op.op.name)
        self.assertEqual('Docstring.', tr.__doc__)

    def test_multiline_docstring(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                i = 10
                return i + 15
            'First sentence.\n\n      Second sentence.\n\n      Returns:\n        Something.\n      '
            return constant_op.constant(1)
        tr = self.transform(f, functions)
        result_op = tr()
        self.assertIn('f/', result_op.op.name)
        self.assertIn('First sentence.', tr.__doc__)
        self.assertIn('Second sentence.', tr.__doc__)

    def test_nested_functions(self):
        if False:
            return 10

        def f(l):
            if False:
                return 10

            def inner_fn(i):
                if False:
                    print('Hello World!')
                return i + 1
            l += 1
            return (l, inner_fn(l))
        tr = self.transform(f, (functions, return_statements))
        (first, second) = tr(constant_op.constant(1))
        self.assertIn('f/', first.op.name)
        self.assertNotIn('inner_fn', first.op.name)
        self.assertIn('f/inner_fn/', second.op.inputs[0].name)

    def test_conversion_context_preserves_in_inner_functions(self):
        if False:
            return 10

        def inner_fn_callee():
            if False:
                while True:
                    i = 10
            self.assertEqual(ag_ctx.control_status_ctx().status, ag_ctx.Status.DISABLED)

        def f():
            if False:
                i = 10
                return i + 15

            def inner_fn():
                if False:
                    return 10
                inner_fn_callee()
            with ag_ctx.ControlStatusCtx(ag_ctx.Status.DISABLED, converter.ConversionOptions(recursive=True)):
                inner_fn()
        tr = self.transform(f, functions)
        tr()

    def test_method(self):
        if False:
            for i in range(10):
                print('nop')

        class TestClass(object):

            def f(self, l):
                if False:
                    for i in range(10):
                        print('nop')

                def inner_fn(i):
                    if False:
                        i = 10
                        return i + 15
                    return i + 1
                l += 1
                return (l, inner_fn(l))
        tr = self.transform(TestClass.f, (functions, return_statements))
        (first, second) = tr(TestClass(), constant_op.constant(1))
        self.assertIn('f/', first.op.name)
        self.assertNotIn('inner_fn', first.op.name)
        self.assertIn('f/inner_fn/', second.op.inputs[0].name)

    def test_lambda_in_return_value(self):
        if False:
            for i in range(10):
                print('nop')

        def f():
            if False:
                while True:
                    i = 10
            return lambda x: x + 1
        tr = self.transform(f, functions)
        result_l = tr()
        self.assertTrue(api.is_autograph_artifact(result_l))
if __name__ == '__main__':
    test.main()