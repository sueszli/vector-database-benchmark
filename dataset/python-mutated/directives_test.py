"""Tests for directives module."""
from tensorflow.python.autograph.converters import directives as directives_converter
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.platform import test

class DirectivesTest(converter_testing.TestCase):

    def test_local_target(self):
        if False:
            i = 10
            return i + 15

        def f():
            if False:
                print('Hello World!')
            l = []
            string_var = 0
            directives.set_element_type(l, 'a', string_var)
        (_, node, _) = self.transform(f, directives_converter, include_ast=True)
        (def_,) = anno.getanno(node.body[0].targets[0], anno.Static.DEFINITIONS)
        d = def_.directives[directives.set_element_type]
        self.assertEqual(d['dtype'].value, 'a')
        self.assertEqual(d['shape'].id, 'string_var')

    def test_argument_target(self):
        if False:
            return 10

        def f(a):
            if False:
                while True:
                    i = 10
            directives.set_element_type(a, 1, shape=2)
            pass
        (_, node, _) = self.transform(f, directives_converter, include_ast=True)
        (def_,) = anno.getanno(node.args.args[0], anno.Static.DEFINITIONS)
        d = def_.directives[directives.set_element_type]
        self.assertEqual(d['dtype'].value, 1)
        self.assertEqual(d['shape'].value, 2)

    def test_loop_target(self):
        if False:
            while True:
                i = 10

        def f():
            if False:
                i = 10
                return i + 15
            a = True
            while True:
                directives.set_loop_options(parallel_iterations=10, back_prop=a)
                pass
        (_, node, _) = self.transform(f, directives_converter, include_ast=True)
        d = anno.getanno(node.body[1], anno.Basic.DIRECTIVES)
        d = d[directives.set_loop_options]
        self.assertEqual(d['parallel_iterations'].value, 10)
        self.assertEqual(d['back_prop'].id, 'a')
        self.assertNotIn('swap_memory', d)

    def test_loop_target_no_loop(self):
        if False:
            return 10

        def f():
            if False:
                print('Hello World!')
            directives.set_loop_options()
            pass
        with self.assertRaisesRegex(ValueError, 'must be used inside a statement'):
            self.transform(f, directives_converter, include_ast=True)

    def test_loop_target_not_first(self):
        if False:
            for i in range(10):
                print('nop')

        def f():
            if False:
                print('Hello World!')
            a = 1
            while True:
                a = 2
                directives.set_loop_options(parallel_iterations=10, back_prop=a)
        with self.assertRaisesRegex(ValueError, 'must be the first statement'):
            self.transform(f, directives_converter, include_ast=True)

    def test_value_verification_does_not_trigger_properties(self):
        if False:
            for i in range(10):
                print('nop')
        self_test = self

        class TestClass(object):

            @property
            def b(self):
                if False:
                    i = 10
                    return i + 15
                self_test.fail('This should never be evaluated')
        tc = TestClass()

        def f():
            if False:
                while True:
                    i = 10
            return tc.b + 1
        (_, node, _) = self.transform(f, directives_converter, include_ast=True)
        self.assertIsNotNone(node)

    def test_value_verification_does_not_trigger_getattr(self):
        if False:
            for i in range(10):
                print('nop')

        class TestClass(object):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.getattr_called = False

            def __getattr__(self, _):
                if False:
                    for i in range(10):
                        print('nop')
                self.getattr_called = True
        tc = TestClass()

        def f():
            if False:
                for i in range(10):
                    print('nop')
            return tc.b + 1
        (_, node, _) = self.transform(f, directives_converter, include_ast=True)
        self.assertIsNotNone(node)
        self.assertFalse(tc.getattr_called)
if __name__ == '__main__':
    test.main()