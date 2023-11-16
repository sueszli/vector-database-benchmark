"""Tests for our test framework."""
import os
from pytype import errors
from pytype.tests import test_base
from pytype.tests import test_utils

class ErrorLogTest(test_base.BaseTest):

    def test_error_comments(self):
        if False:
            return 10
        err = self.CheckWithErrors('\n      a = 10  # a random comment\n      b = "hello" + 3  # unsupported-operands[.mark]\n      c = (10).foo  # attribute-error\n      d = int(int)  # wrong-arg-types[.another_mark]\n    ')
        self.assertEqual({mark: (e.lineno, e.name) for (mark, e) in err.marks.items()}, {'.mark': (2, 'unsupported-operands'), '.another_mark': (4, 'wrong-arg-types')})
        self.assertEqual(err.expected, {2: [('unsupported-operands', '.mark')], 3: [('attribute-error', None)], 4: [('wrong-arg-types', '.another_mark')]})

    def test_multiple_errors_one_line(self):
        if False:
            for i in range(10):
                print('nop')
        err = self.CheckWithErrors('\n      x = (10).foo, "hello".foo  # attribute-error[e1]  # attribute-error[e2]\n    ')
        line = 1
        self.assertEqual(err.expected, {line: [('attribute-error', 'e1'), ('attribute-error', 'e2')]})
        self.assertCountEqual(err.marks, ['e1', 'e2'])
        self.assertIn('on int', err.marks['e1'].message)
        self.assertIn('on str', err.marks['e2'].message)

    def test_different_order_of_errors_one_line(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      x = a.foo, "hello".foo  # name-error[e1]  # attribute-error[e2]\n    ')
        self.CheckWithErrors('\n      x = a.foo, "hello".foo  # attribute-error[e2]  # name-error[e1]\n    ')

    def test_populate_marks(self):
        if False:
            for i in range(10):
                print('nop')
        matcher = test_utils.ErrorMatcher('x = 0')
        self.assertIsNone(matcher.marks)
        matcher.assert_errors_match_expected(errors.ErrorLog())
        self.assertErrorRegexes(matcher, {})
        self.assertIsNotNone(matcher.marks)

    def test_duplicate_mark(self):
        if False:
            print('Hello World!')
        with self.assertRaises(AssertionError) as ctx:
            self.CheckWithErrors('x = 0  # attribute-error[e]  # attribute-error[e]')
        self.assertEqual(str(ctx.exception), 'Mark e already used')

    def test_error_regex_matching(self):
        if False:
            i = 10
            return i + 15
        err = self.CheckWithErrors('\n      a = 10\n      b = "hello"\n      c = a + b  # unsupported-operands\n      d = a.foo()  # attribute-error[.mark]\n    ')
        self.assertErrorRegexes(err, {'.mark': '.*foo.*'})

    def test_error_sequence_matching(self):
        if False:
            i = 10
            return i + 15
        err = self.CheckWithErrors('\n      a = 10\n      b = a < "hello"  # unsupported-operands[.mark]\n      c = a.foo()  # attribute-error\n    ')
        self.assertErrorSequences(err, {'.mark': ['<', 'a: int', "'hello': str"]})

    def test_mismatched_error(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(AssertionError) as ctx:
            self.CheckWithErrors('(10).foo  # wrong-arg-types')
        self.assertIn('Error does not match', str(ctx.exception))

    def test_unexpected_error(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(AssertionError) as ctx:
            self.CheckWithErrors('\n        (10).foo  # attribute-error\n        "hello".foo\n      ')
        self.assertIn('Unexpected error', str(ctx.exception))

    def test_leftover_error(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(AssertionError) as ctx:
            self.CheckWithErrors('x = 0  # attribute-error')
        self.assertIn('Errors not found', str(ctx.exception))

    def test_misspelled_leftover_error(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(AssertionError) as ctx:
            self.CheckWithErrors('x = 0  # misspelled-error')
        self.assertIn('Errors not found', str(ctx.exception))

    def test_mismatched_regex(self):
        if False:
            print('Hello World!')
        err = self.CheckWithErrors('(10).foo  # attribute-error[e]')
        with self.assertRaises(AssertionError) as ctx:
            self.assertErrorRegexes(err, {'e': 'does not match error message'})
        self.assertIn('Bad error message', str(ctx.exception))

    def test_missing_regex(self):
        if False:
            i = 10
            return i + 15
        err = self.CheckWithErrors('(10).foo  # attribute-error[e]')
        with self.assertRaises(AssertionError) as ctx:
            self.assertErrorRegexes(err, {})
        self.assertEqual(str(ctx.exception), 'No matcher for mark e')

    def test_leftover_regex(self):
        if False:
            return 10
        err = self.CheckWithErrors('x = 0')
        with self.assertRaises(AssertionError) as ctx:
            self.assertErrorRegexes(err, {'e': ''})
        self.assertEqual(str(ctx.exception), 'Marks not found in code: e')

    def test_mismatched_sequence(self):
        if False:
            i = 10
            return i + 15
        err = self.CheckWithErrors('(10).foo  # attribute-error[e]')
        with self.assertRaises(AssertionError) as ctx:
            self.assertErrorSequences(err, {'e': ['int', 'foo', 'attribute']})
        self.assertIn('Bad error message', str(ctx.exception))

    def test_bad_check(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(AssertionError) as ctx:
            self.Check('name_error  # name-error')
        self.assertIn('Cannot assert errors', str(ctx.exception))

    def test_bad_infer(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(AssertionError) as ctx:
            self.Infer('name_error  # name-error')
        self.assertIn('Cannot assert errors', str(ctx.exception))

    def test_bad_infer_from_file(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('some_file.py', 'name_error  # name-error')
            with self.assertRaises(AssertionError) as ctx:
                self.InferFromFile(filename=d['some_file.py'], pythonpath=[])
        self.assertIn('Cannot assert errors', str(ctx.exception))

class SkipTest(test_base.BaseTest):

    @test_utils.skipUnlessPy((3, 10), reason='testing skipUnlessPy')
    def test_skip_unless_py(self):
        if False:
            return 10
        self.Check('\n      import sys\n      if sys.version_info.minor != 10:\n        name_error\n    ')

    @test_utils.skipIfPy((3, 10), reason='testing skipIfPy')
    def test_skip_if_py(self):
        if False:
            print('Hello World!')
        self.Check('\n      import sys\n      if sys.version_info.minor == 10:\n        name_error\n    ')

    @test_utils.skipBeforePy((3, 10), reason='testing skipBeforePy')
    def test_skip_before_py(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import sys\n      if sys.version_info.minor < 10:\n        name_error\n    ')

    @test_utils.skipFromPy((3, 10), reason='testing skipFromPy')
    def test_skip_from_py(self):
        if False:
            print('Hello World!')
        self.Check('\n      import sys\n      if sys.version_info.minor >= 10:\n        name_error\n    ')

class DepTreeTest(test_base.BaseTest):

    def test_dep_tree(self):
        if False:
            while True:
                i = 10
        foo_pyi = '\n      class A: pass\n    '
        bar_py = '\n      import foo\n      x = foo.A()\n    '
        deps = [('foo.pyi', foo_pyi), ('bar.py', bar_py)]
        with self.DepTree(deps) as d:
            self.Check('\n        import foo\n        import bar\n        assert_type(bar.x, foo.A)\n      ')
            self.assertCountEqual(os.listdir(d.path), ['foo.pyi', 'bar.pyi'])
if __name__ == '__main__':
    test_base.main()