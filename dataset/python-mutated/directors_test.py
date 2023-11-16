"""Tests for directors.py."""
import sys
import textwrap
from pytype import errors
from pytype.directors import directors
import unittest
_TEST_FILENAME = 'my_file.py'

class LineSetTest(unittest.TestCase):

    def test_no_ranges(self):
        if False:
            for i in range(10):
                print('nop')
        lines = directors._LineSet()
        lines.set_line(2, True)
        self.assertNotIn(0, lines)
        self.assertNotIn(1, lines)
        self.assertIn(2, lines)
        self.assertNotIn(3, lines)

    def test_closed_range(self):
        if False:
            for i in range(10):
                print('nop')
        lines = directors._LineSet()
        lines.start_range(2, True)
        lines.start_range(4, False)
        self.assertNotIn(1, lines)
        self.assertIn(2, lines)
        self.assertIn(3, lines)
        self.assertNotIn(4, lines)
        self.assertNotIn(1000, lines)

    def test_open_range(self):
        if False:
            print('Hello World!')
        lines = directors._LineSet()
        lines.start_range(2, True)
        lines.start_range(4, False)
        lines.start_range(7, True)
        self.assertNotIn(1, lines)
        self.assertIn(2, lines)
        self.assertIn(3, lines)
        self.assertNotIn(4, lines)
        self.assertNotIn(5, lines)
        self.assertNotIn(6, lines)
        self.assertIn(7, lines)
        self.assertIn(1000, lines)

    def test_range_at_zero(self):
        if False:
            for i in range(10):
                print('nop')
        lines = directors._LineSet()
        lines.start_range(0, True)
        lines.start_range(3, False)
        self.assertNotIn(-1, lines)
        self.assertIn(0, lines)
        self.assertIn(1, lines)
        self.assertIn(2, lines)
        self.assertNotIn(3, lines)

    def test_line_overrides_range(self):
        if False:
            i = 10
            return i + 15
        lines = directors._LineSet()
        lines.start_range(2, True)
        lines.start_range(5, False)
        lines.set_line(3, False)
        self.assertIn(2, lines)
        self.assertNotIn(3, lines)
        self.assertIn(4, lines)

    def test_redundant_range(self):
        if False:
            for i in range(10):
                print('nop')
        lines = directors._LineSet()
        lines.start_range(2, True)
        lines.start_range(3, True)
        lines.start_range(5, False)
        lines.start_range(9, False)
        self.assertNotIn(1, lines)
        self.assertIn(2, lines)
        self.assertIn(3, lines)
        self.assertIn(4, lines)
        self.assertNotIn(5, lines)
        self.assertNotIn(9, lines)
        self.assertNotIn(1000, lines)

    def test_enable_disable_on_same_line(self):
        if False:
            return 10
        lines = directors._LineSet()
        lines.start_range(2, True)
        lines.start_range(2, False)
        lines.start_range(3, True)
        lines.start_range(5, False)
        lines.start_range(5, True)
        self.assertNotIn(2, lines)
        self.assertIn(3, lines)
        self.assertIn(4, lines)
        self.assertIn(5, lines)
        self.assertIn(1000, lines)

    def test_decreasing_lines_not_allowed(self):
        if False:
            for i in range(10):
                print('nop')
        lines = directors._LineSet()
        self.assertRaises(ValueError, lines.start_range, -100, True)
        lines.start_range(2, True)
        self.assertRaises(ValueError, lines.start_range, 1, True)

class DirectorTestCase(unittest.TestCase):
    python_version = sys.version_info[:2]

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        super().setUpClass()
        for name in ['test-error', 'test-other-error']:
            errors._error_name(name)

    def _create(self, src, disable=()):
        if False:
            while True:
                i = 10
        self.num_lines = len(src.rstrip().splitlines())
        src = textwrap.dedent(src)
        src_tree = directors.parse_src(src, self.python_version)
        self._errorlog = errors.ErrorLog()
        self._director = directors.Director(src_tree, self._errorlog, _TEST_FILENAME, disable)

    def _should_report(self, expected, lineno, error_name='test-error', filename=_TEST_FILENAME):
        if False:
            for i in range(10):
                print('nop')
        error = errors.Error.for_test(errors.SEVERITY_ERROR, 'message', error_name, filename=filename, lineno=lineno)
        self.assertEqual(expected, self._director.filter_error(error))

class DirectorTest(DirectorTestCase):

    def test_ignore_globally(self):
        if False:
            print('Hello World!')
        self._create('', ['my-error'])
        self._should_report(False, 42, error_name='my-error')

    def test_ignore_one_line(self):
        if False:
            i = 10
            return i + 15
        self._create('\n    # line 2\n    x = 123  # type: ignore\n    # line 4\n    ')
        self._should_report(True, 2)
        self._should_report(False, 3)
        self._should_report(True, 4)

    def test_ignore_one_line_mypy_style(self):
        if False:
            for i in range(10):
                print('nop')
        self._create('\n    # line 2\n    x = 123  # type: ignore[arg-type]\n    # line 4\n    ')
        self._should_report(True, 2)
        self._should_report(False, 3)
        self._should_report(True, 4)

    def test_utf8(self):
        if False:
            while True:
                i = 10
        self._create('\n    x = u"abc□def\\n"\n    ')

    def test_ignore_extra_characters(self):
        if False:
            i = 10
            return i + 15
        self._create('\n    # line 2\n    x = 123  # # type: ignore\n    # line 4\n    ')
        self._should_report(True, 2)
        self._should_report(False, 3)
        self._should_report(True, 4)

    def test_ignore_until_end(self):
        if False:
            while True:
                i = 10
        self._create('\n    # line 2\n    # type: ignore\n    # line 4\n    ')
        self._should_report(True, 2)
        self._should_report(False, 3)
        self._should_report(False, 4)

    def test_out_of_scope(self):
        if False:
            for i in range(10):
                print('nop')
        self._create('\n    # type: ignore\n    ')
        self._should_report(False, 2)
        self._should_report(True, 2, filename=None)
        self._should_report(True, 2, filename='some_other_file.py')
        self._should_report(False, None)
        self._should_report(False, 0)

    def test_disable(self):
        if False:
            return 10
        self._create('\n    # line 2\n    x = 123  # pytype: disable=test-error\n    # line 4\n    ')
        self._should_report(True, 2)
        self._should_report(False, 3)
        self._should_report(True, 4)

    def test_disable_extra_characters(self):
        if False:
            return 10
        self._create('\n    # line 2\n    x = 123  # # pytype: disable=test-error\n    # line 4\n    ')
        self._should_report(True, 2)
        self._should_report(False, 3)
        self._should_report(True, 4)

    def test_disable_until_end(self):
        if False:
            print('Hello World!')
        self._create('\n    # line 2\n    # pytype: disable=test-error\n    # line 4\n    ')
        self._should_report(True, 2)
        self._should_report(False, 3)
        self._should_report(False, 4)

    def test_enable_after_disable(self):
        if False:
            while True:
                i = 10
        self._create('\n    # line 2\n    # pytype: disable=test-error\n    # line 4\n    # pytype: enable=test-error\n    ')
        self._should_report(True, 2)
        self._should_report(False, 3)
        self._should_report(False, 4)
        self._should_report(True, 5)
        self._should_report(True, 100)

    def test_enable_one_line(self):
        if False:
            i = 10
            return i + 15
        self._create('\n    # line 2\n    # pytype: disable=test-error\n    # line 4\n    x = 123 # pytype: enable=test-error\n    ')
        self._should_report(True, 2)
        self._should_report(False, 3)
        self._should_report(False, 4)
        self._should_report(True, 5)
        self._should_report(False, 6)
        self._should_report(False, 100)

    def test_disable_other_error(self):
        if False:
            print('Hello World!')
        self._create('\n    # line 2\n    x = 123  # pytype: disable=test-other-error\n    # line 4\n    ')
        self._should_report(True, 2)
        self._should_report(True, 3)
        self._should_report(False, 3, error_name='test-other-error')
        self._should_report(True, 4)

    def test_disable_multiple_error(self):
        if False:
            while True:
                i = 10
        self._create('\n    # line 2\n    x = 123  # pytype: disable=test-error,test-other-error\n    # line 4\n    ')
        self._should_report(True, 2)
        self._should_report(False, 3)
        self._should_report(False, 3, error_name='test-other-error')
        self._should_report(True, 4)

    def test_disable_all(self):
        if False:
            i = 10
            return i + 15
        self._create('\n    # line 2\n    x = 123  # pytype: disable=*\n    # line 4\n    ')
        self._should_report(True, 2)
        self._should_report(False, 3)
        self._should_report(True, 4)

    def test_multiple_directives(self):
        if False:
            print('Hello World!')
        self._create('\n    x = 123  # sometool: directive=whatever # pytype: disable=test-error\n    ')
        self._should_report(False, 2)

    def test_error_at_line_0(self):
        if False:
            while True:
                i = 10
        self._create('\n    x = "foo"\n    # pytype: disable=attribute-error\n    ')
        self._should_report(False, 0, error_name='attribute-error')

    def test_disable_without_space(self):
        if False:
            for i in range(10):
                print('nop')
        self._create('\n    # line 2\n    x = 123  # pytype:disable=test-error\n    # line 4\n    ')
        self._should_report(True, 2)
        self._should_report(False, 3)
        self._should_report(True, 4)

    def test_invalid_disable(self):
        if False:
            return 10

        def check_warning(message_regex, text):
            if False:
                print('Hello World!')
            self._create(text)
            self.assertLessEqual(1, len(self._errorlog))
            error = list(self._errorlog)[0]
            self.assertEqual(_TEST_FILENAME, error._filename)
            self.assertEqual(1, error.lineno)
            self.assertRegex(str(error), message_regex)
        check_warning('Unknown pytype directive.*disalbe.*', '# pytype: disalbe=test-error')
        check_warning('Invalid error name.*bad-error-name.*', '# pytype: disable=bad-error-name')
        check_warning('Invalid directive syntax', '# pytype: disable')
        check_warning('Invalid directive syntax', '# pytype: ')
        check_warning('Unknown pytype directive.*foo.*', '# pytype: disable=test-error foo=bar')
        check_warning('Invalid directive syntax', '# pytype: disable=test-error ,test-other-error')
        check_warning('Invalid error name', '# pytype: disable=test-error, test-other-error')

    def test_type_comments(self):
        if False:
            print('Hello World!')
        self._create('\n    x = None  # type: int\n    y = None  # allow extra comments # type: str\n    z = None  # type: int  # and extra comments after, too\n    a = None  # type:int  # without a space\n    # type: (int, float) -> str\n    # comment with embedded # type: should-be-discarded\n    ')
        self.assertEqual({2: 'int', 3: 'str', 4: 'int', 5: 'int', 6: '(int, float) -> str'}, self._director.type_comments)

    def test_strings_that_look_like_directives(self):
        if False:
            for i in range(10):
                print('nop')
        self._create('\n    s = "# type: int"\n    x = None  # type: float\n    y = "# type: int"  # type: str\n    ')
        self.assertEqual({3: 'float', 4: 'str'}, self._director.type_comments)

    def test_huge_string(self):
        if False:
            return 10
        src = ['x = (']
        for i in range(2000):
            src.append(f"    'string{i}'")
        src.append(')')
        self._create('\n'.join(src))

    def test_try(self):
        if False:
            print('Hello World!')
        self._create('\n      try:\n        x = None  # type: int\n      except Exception:\n        x = None  # type: str\n      else:\n        x = None  # type: float\n    ')
        self.assertEqual({3: 'int', 5: 'str', 7: 'float'}, self._director.type_comments)

class VariableAnnotationsTest(DirectorTestCase):

    def assertAnnotations(self, expected):
        if False:
            return 10
        actual = {k: (v.name, v.annotation) for (k, v) in self._director.annotations.items()}
        self.assertEqual(expected, actual)

    def test_annotations(self):
        if False:
            return 10
        self._create("\n      v1: int = 0\n      def f():\n        v2: str = ''\n    ")
        self.assertAnnotations({2: ('v1', 'int'), 4: ('v2', 'str')})

    def test_precedence(self):
        if False:
            for i in range(10):
                print('nop')
        self._create('v: int = 0  # type: str')
        self.assertAnnotations({1: ('v', 'int')})

    def test_parameter_annotation(self):
        if False:
            for i in range(10):
                print('nop')
        self._create('\n      def f(\n          x: int = 0):\n        pass\n    ')
        self.assertFalse(self._director.annotations)

    def test_multistatement_line(self):
        if False:
            while True:
                i = 10
        self._create("\n      if __random__: v1: int = 0\n      else: v2: str = ''\n    ")
        self.assertAnnotations({2: ('v1', 'int'), 3: ('v2', 'str')})

    def test_multistatement_line_no_annotation(self):
        if False:
            for i in range(10):
                print('nop')
        self._create('\n      if __random__: v = 0\n      else: v = 1\n    ')
        self.assertFalse(self._director.annotations)

    def test_comment_is_not_an_annotation(self):
        if False:
            print('Hello World!')
        self._create('# FOMO(b/xxx): pylint: disable=invalid-name')
        self.assertFalse(self._director.annotations)

    def test_string_is_not_an_annotation(self):
        if False:
            return 10
        self._create("\n      logging.info('%s: completed: response=%s',  s1, s2)\n      f(':memory:', bar=baz)\n    ")
        self.assertFalse(self._director.annotations)

    def test_multiline_annotation(self):
        if False:
            print('Hello World!')
        self._create('\n      v: Callable[  # a very important comment\n          [], int] = None\n    ')
        self.assertAnnotations({2: ('v', 'Callable[[], int]')})

    def test_multiline_assignment(self):
        if False:
            return 10
        self._create('\n      v: List[int] = [\n          0,\n          1,\n      ]\n    ')
        self.assertAnnotations({2: ('v', 'List[int]')})

    def test_complicated_annotation(self):
        if False:
            return 10
        self._create('v: int if __random__ else str = None')
        self.assertAnnotations({1: ('v', 'int if __random__ else str')})

    def test_colon_in_value(self):
        if False:
            i = 10
            return i + 15
        self._create('v: Dict[str, int] = {x: y}')
        self.assertAnnotations({1: ('v', 'Dict[str, int]')})

    def test_equals_sign_in_value(self):
        if False:
            i = 10
            return i + 15
        self._create('v = {x: f(y=0)}')
        self.assertFalse(self._director.annotations)

    def test_annotation_after_comment(self):
        if False:
            i = 10
            return i + 15
        self._create('\n      # comment\n      v: int = 0\n    ')
        self.assertAnnotations({3: ('v', 'int')})

class LineNumbersTest(DirectorTestCase):

    def test_type_comment_on_multiline_value(self):
        if False:
            return 10
        self._create('\n      v = [\n        ("hello",\n         "world",  # type: should_be_ignored\n\n        )\n      ]  # type: dict\n    ')
        self.assertEqual({2: 'dict'}, self._director.type_comments)

    def test_type_comment_with_trailing_comma(self):
        if False:
            for i in range(10):
                print('nop')
        self._create('\n      v = [\n        ("hello",\n         "world"\n        ),\n      ]  # type: dict\n      w = [\n        ["hello",\n         "world"\n        ],  # some comment\n      ]  # type: dict\n    ')
        self.assertEqual({2: 'dict', 7: 'dict'}, self._director.type_comments)

    def test_decorators(self):
        if False:
            i = 10
            return i + 15
        self._create("\n      class A:\n        '''\n        @decorator in a docstring\n        '''\n        @real_decorator\n        def f(x):\n          x = foo @ bar @ baz\n\n        @decorator(\n            x, y\n        )\n\n        def bar():\n          pass\n    ")
        self.assertEqual(self._director.decorators, {7: ['real_decorator'], 14: ['decorator']})
        self.assertEqual(self._director.decorated_functions, {6: 7, 10: 14})

    def test_stacked_decorators(self):
        if False:
            while True:
                i = 10
        self._create('\n      @decorator(\n          x, y\n      )\n\n      @foo\n\n      class A:\n          pass\n    ')
        self.assertEqual(self._director.decorators, {8: ['decorator', 'foo']})
        self.assertEqual(self._director.decorated_functions, {2: 8, 6: 8})

    def test_overload(self):
        if False:
            i = 10
            return i + 15
        self._create('\n      from typing import overload\n\n      @overload\n      def f() -> int: ...\n\n      @overload\n      def f(x: str) -> str: ...\n\n      def f(x=None):\n        return 0 if x is None else x\n    ')
        self.assertEqual(self._director.decorators, {5: ['overload'], 8: ['overload']})
        self.assertEqual(self._director.decorated_functions, {4: 5, 7: 8})

class DisableDirectivesTest(DirectorTestCase):

    def assertDisables(self, *disable_lines, error_class=None, disables=None):
        if False:
            while True:
                i = 10
        assert not (error_class and disables)
        error_class = error_class or 'wrong-arg-types'
        disables = disables or self._director._disables[error_class]
        for i in range(self.num_lines):
            lineno = i + 1
            if lineno in disable_lines:
                self.assertIn(lineno, disables)
            else:
                self.assertNotIn(lineno, disables)

    def test_basic(self):
        if False:
            print('Hello World!')
        self._create('\n      toplevel(\n          a, b, c, d)  # pytype: disable=wrong-arg-types\n    ')
        self.assertDisables(2, 3)

    def test_nested(self):
        if False:
            return 10
        self._create('\n      toplevel(\n          nested())  # pytype: disable=wrong-arg-types\n    ')
        self.assertDisables(2, 3)

    def test_multiple_nested(self):
        if False:
            for i in range(10):
                print('nop')
        self._create('\n      toplevel(\n        nested1(),\n        nested2(),  # pytype: disable=wrong-arg-types\n        nested3())\n    ')
        self.assertDisables(2, 4)

    def test_multiple_toplevel(self):
        if False:
            return 10
        self._create('\n      toplevel1()\n      toplevel2()  # pytype: disable=wrong-arg-types\n      toplevel3()\n    ')
        self.assertDisables(3)

    def test_deeply_nested(self):
        if False:
            print('Hello World!')
        self._create('\n      toplevel(\n        nested1(),\n        nested2(\n          deeply_nested1(),  # pytype: disable=wrong-arg-types\n          deeply_nested2()),\n        nested3())\n    ')
        self.assertDisables(2, 4, 5)

    def test_non_toplevel(self):
        if False:
            print('Hello World!')
        self._create('\n      x = [\n        f("oops")  # pytype: disable=wrong-arg-types\n      ]\n    ')
        self.assertDisables(2, 3)

    def test_non_toplevel_bad_annotation(self):
        if False:
            i = 10
            return i + 15
        self._create('\n      x: list[int] = [\n        f(\n            "oops")]  # pytype: disable=annotation-type-mismatch\n    ')
        self.assertDisables(2, 4, error_class='annotation-type-mismatch')

    def test_trailing_parenthesis(self):
        if False:
            i = 10
            return i + 15
        self._create('\n      toplevel(\n          a, b, c, d,\n      )  # pytype: disable=wrong-arg-types\n    ')
        self.assertDisables(2, 4)

    def test_multiple_bytecode_blocks(self):
        if False:
            while True:
                i = 10
        self._create('\n      def f():\n        call(a, b, c, d)\n      def g():\n        call(a, b, c, d)  # pytype: disable=wrong-arg-types\n    ')
        self.assertDisables(5)

    def test_compare(self):
        if False:
            while True:
                i = 10
        self._create('\n      import datetime\n      def f(right: datetime.date):\n        left = datetime.datetime(1, 1, 1, 1)\n        return left < right  # pytype: disable=wrong-arg-types\n    ')
        self.assertDisables(5)

    def test_nested_compare(self):
        if False:
            print('Hello World!')
        self._create('\n      f(\n        a,\n        b,\n        (c <\n         d)  # pytype: disable=wrong-arg-types\n      )\n    ')
        self.assertDisables(2, 5, 6)

    def test_iterate(self):
        if False:
            print('Hello World!')
        self._create('\n      class Foo:\n        def __iter__(self, too, many, args):\n          pass\n      foo = Foo()\n      for x in foo:  # pytype: disable=missing-parameter\n        print(x)\n    ')
        self.assertDisables(6, error_class='missing-parameter')

    def test_subscript(self):
        if False:
            while True:
                i = 10
        self._create("\n      class Foo:\n        def __getitem__(self, too, many, args):\n          pass\n      x = Foo()\n      x['X']  # pytype: disable=missing-parameter\n    ")
        self.assertDisables(6, error_class='missing-parameter')

    def test_attrs(self):
        if False:
            while True:
                i = 10
        self._create('\n      import attr\n      def converter(x):\n        return []\n      @attr.s\n      class Foo:\n        x = attr.ib(\n          converter=converter, factory=list, type=dict[str, str]\n        )  # pytype: disable=annotation-type-mismatch\n    ')
        self.assertDisables(7, 9, error_class='annotation-type-mismatch')

    def test_return(self):
        if False:
            print('Hello World!')
        self._create('\n       def f(x):\n         return x\n       def g() -> int:\n         return f(\n             "oops")  # pytype: disable=bad-return-type\n    ')
        self.assertDisables(5, 6, error_class='bad-return-type')

    def test_if(self):
        if False:
            print('Hello World!')
        self._create('\n      if (__random__ and\n          name_error and  # pytype: disable=name-error\n          __random__):\n        pass\n    ')
        self.assertDisables(3, error_class='name-error')

    def test_unsupported(self):
        if False:
            for i in range(10):
                print('nop')
        self._create('\n      x = [\n        "something_unsupported"\n      ]  # pytype: disable=not-supported-yet\n    ')
        self.assertDisables(2, 4, error_class='not-supported-yet')

    def test_range(self):
        if False:
            for i in range(10):
                print('nop')
        self._create('\n      f(\n        # pytype: disable=attribute-error\n        a.nonsense,\n        b.nonsense,\n        # pytype: enable=attribute-error\n      )\n    ')
        self.assertDisables(3, 4, 5, error_class='attribute-error')

    def test_ignore(self):
        if False:
            return 10
        self._create('\n      x = [\n        some_bad_function(\n            "some bad arg")]  # type: ignore\n    ')
        self.assertDisables(2, 3, 4, disables=self._director.ignore)

    def test_ignore_range(self):
        if False:
            for i in range(10):
                print('nop')
        self._create('\n      x = [\n        # type: ignore\n        "oops"\n      ]\n    ')
        self.assertDisables(3, 4, 5, disables=self._director.ignore)

    def test_with_and_backslash_continuation(self):
        if False:
            for i in range(10):
                print('nop')
        self._create('\n      with foo("a",\n               "b"), \\\n           bar("c",\n               "d"),  \\\n           baz("e"):  # pytype: disable=wrong-arg-types\n        pass\n    ')
        self.assertDisables(2, 6)

    def test_not_instantiable(self):
        if False:
            print('Hello World!')
        self._create('\n      x = [\n        A(\n      )]  # pytype: disable=not-instantiable\n    ')
        self.assertDisables(2, 3, 4, error_class='not-instantiable')

    def test_unsupported_operands_in_call(self):
        if False:
            i = 10
            return i + 15
        self._create('\n      some_func(\n        x < y)  # pytype: disable=unsupported-operands\n    ')
        self.assertDisables(2, 3, error_class='unsupported-operands')

    def test_unsupported_operands_in_assignment(self):
        if False:
            print('Hello World!')
        self._create('\n      x["wrong key type"] = (\n        some_call(),\n        "oops")  # pytype: disable=unsupported-operands\n    ')
        self.assertDisables(2, 4, error_class='unsupported-operands')

    def test_header(self):
        if False:
            return 10
        self._create('\n      if (x == 0 and\n          (0).nonsense and  # pytype: disable=attribute-error\n          y == 0):\n        pass\n    ')
        self.assertDisables(2, 3, error_class='attribute-error')

    def test_try(self):
        if False:
            i = 10
            return i + 15
        self._create('\n      try:\n        pass\n      except NonsenseError:  # pytype: disable=name-error\n        pass\n    ')
        self.assertDisables(4, error_class='name-error')

    def test_classdef(self):
        if False:
            for i in range(10):
                print('nop')
        self._create('\n      import abc\n      class Foo:  # pytype: disable=ignored-abstractmethod\n        @abc.abstractmethod\n        def f(self): ...\n    ')
        self.assertDisables(3, error_class='ignored-abstractmethod')

    def test_class_attribute(self):
        if False:
            i = 10
            return i + 15
        self._create('\n      class Foo:\n        x: 0  # pytype: disable=invalid-annotation\n    ')
        self.assertDisables(3, error_class='invalid-annotation')

    def test_nested_call_in_function_decorator(self):
        if False:
            i = 10
            return i + 15
        self._create('\n      @decorate(\n        dict(\n          k1=v(\n            a, b, c),  # pytype: disable=wrong-arg-types\n          k2=v2))\n      def f():\n        pass\n    ')
        self.assertDisables(2, 3, 4, 5)

    def test_nested_call_in_class_decorator(self):
        if False:
            return 10
        self._create('\n      @decorate(\n        dict(\n          k1=v(\n            a, b, c),  # pytype: disable=wrong-arg-types\n          k2=v2))\n      class C:\n        pass\n    ')
        self.assertDisables(2, 3, 4, 5)

class GlobalDirectivesTest(DirectorTestCase):
    """Test global directives."""

    def test_skip_file(self):
        if False:
            print('Hello World!')
        self.assertRaises(directors.SkipFileError, self._create, '\n          # pytype: skip-file\n        ')

    def test_features(self):
        if False:
            while True:
                i = 10
        self._create('\n      # pytype: features=no-return-any\n    ')
        self.assertEqual(self._director.features, {'no-return-any'})

    def test_invalid_features(self):
        if False:
            for i in range(10):
                print('nop')
        self._create('\n      # pytype: features=foo,no-return-any\n    ')
        err = self._errorlog.unique_sorted_errors()[0]
        self.assertEqual(err.name, 'invalid-directive')
        self.assertRegex(err.message, 'Unknown pytype features')
        self.assertRegex(err.message, '.*foo')
if __name__ == '__main__':
    unittest.main()