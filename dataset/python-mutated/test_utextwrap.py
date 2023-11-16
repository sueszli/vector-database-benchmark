"""Tests of the bzrlib.utextwrap."""
from bzrlib import tests, utextwrap
from bzrlib.tests import features
_str_D = u'おはよう'
_str_S = u'hello'
_str_SD = _str_S + _str_D
_str_DS = _str_D + _str_S

class TestUTextWrap(tests.TestCase):

    def check_width(self, text, expected_width):
        if False:
            i = 10
            return i + 15
        w = utextwrap.UTextWrapper()
        self.assertEqual(w._width(text), expected_width, 'Width of %r should be %d' % (text, expected_width))

    def test_width(self):
        if False:
            return 10
        self.check_width(_str_D, 8)
        self.check_width(_str_SD, 13)

    def check_cut(self, text, width, pos):
        if False:
            i = 10
            return i + 15
        w = utextwrap.UTextWrapper()
        self.assertEqual((text[:pos], text[pos:]), w._cut(text, width))

    def test_cut(self):
        if False:
            print('Hello World!')
        s = _str_SD
        self.check_cut(s, 0, 0)
        self.check_cut(s, 1, 1)
        self.check_cut(s, 5, 5)
        self.check_cut(s, 6, 5)
        self.check_cut(s, 7, 6)
        self.check_cut(s, 12, 8)
        self.check_cut(s, 13, 9)
        self.check_cut(s, 14, 9)
        self.check_cut(u'A' * 5, 3, 3)

    def test_split(self):
        if False:
            while True:
                i = 10
        w = utextwrap.UTextWrapper()
        self.assertEqual(list(_str_D), w._split(_str_D))
        self.assertEqual([_str_S] + list(_str_D), w._split(_str_SD))
        self.assertEqual(list(_str_D) + [_str_S], w._split(_str_DS))

    def test_wrap(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(list(_str_D), utextwrap.wrap(_str_D, 1))
        self.assertEqual(list(_str_D), utextwrap.wrap(_str_D, 2))
        self.assertEqual(list(_str_D), utextwrap.wrap(_str_D, 3))
        self.assertEqual(list(_str_D), utextwrap.wrap(_str_D, 3, break_long_words=False))

class TestUTextFill(tests.TestCase):

    def test_fill_simple(self):
        if False:
            while True:
                i = 10
        self.assertEqual('%s\n%s' % (_str_D[:2], _str_D[2:]), utextwrap.fill(_str_D, 4))

    def test_fill_with_breaks(self):
        if False:
            for i in range(10):
                print('nop')
        text = u'spam ham egg spamhamegg' + _str_D + u' spam' + _str_D * 2
        self.assertEqual(u'\n'.join(['spam ham', 'egg spam', 'hamegg' + _str_D[0], _str_D[1:], 'spam' + _str_D[:2], _str_D[2:] + _str_D[:2], _str_D[2:]]), utextwrap.fill(text, 8))

    def test_fill_without_breaks(self):
        if False:
            for i in range(10):
                print('nop')
        text = u'spam ham egg spamhamegg' + _str_D + u' spam' + _str_D * 2
        self.assertEqual(u'\n'.join(['spam ham', 'egg', 'spamhamegg', _str_D, 'spam' + _str_D[:2], _str_D[2:] + _str_D[:2], _str_D[2:]]), utextwrap.fill(text, 8, break_long_words=False))

    def test_fill_indent_with_breaks(self):
        if False:
            while True:
                i = 10
        w = utextwrap.UTextWrapper(8, initial_indent=' ' * 4, subsequent_indent=' ' * 4)
        self.assertEqual(u'\n'.join(['    hell', '    o' + _str_D[0], '    ' + _str_D[1:3], '    ' + _str_D[3]]), w.fill(_str_SD))

    def test_fill_indent_without_breaks(self):
        if False:
            return 10
        w = utextwrap.UTextWrapper(8, initial_indent=' ' * 4, subsequent_indent=' ' * 4)
        w.break_long_words = False
        self.assertEqual(u'\n'.join(['    hello', '    ' + _str_D[:2], '    ' + _str_D[2:]]), w.fill(_str_SD))

    def test_fill_indent_without_breaks_with_fixed_width(self):
        if False:
            i = 10
            return i + 15
        w = utextwrap.UTextWrapper(8, initial_indent=' ' * 4, subsequent_indent=' ' * 4)
        w.break_long_words = False
        w.width = 3
        self.assertEqual(u'\n'.join(['    hello', '    ' + _str_D[0], '    ' + _str_D[1], '    ' + _str_D[2], '    ' + _str_D[3]]), w.fill(_str_SD))

class TestUTextWrapAmbiWidth(tests.TestCase):
    _cyrill_char = u'А'

    def test_ambiwidth1(self):
        if False:
            i = 10
            return i + 15
        w = utextwrap.UTextWrapper(4, ambiguous_width=1)
        s = self._cyrill_char * 8
        self.assertEqual([self._cyrill_char * 4] * 2, w.wrap(s))

    def test_ambiwidth2(self):
        if False:
            for i in range(10):
                print('nop')
        w = utextwrap.UTextWrapper(4, ambiguous_width=2)
        s = self._cyrill_char * 8
        self.assertEqual([self._cyrill_char * 2] * 4, w.wrap(s))
try:
    from test import test_textwrap

    def override_textwrap_symbols(testcase):
        if False:
            i = 10
            return i + 15
        testcase.overrideAttr(test_textwrap, 'TextWrapper', utextwrap.UTextWrapper)
        testcase.overrideAttr(test_textwrap, 'wrap', utextwrap.wrap)
        testcase.overrideAttr(test_textwrap, 'fill', utextwrap.fill)

    def setup_both(testcase, base_class, reused_class):
        if False:
            i = 10
            return i + 15
        super(base_class, testcase).setUp()
        override_textwrap_symbols(testcase)
        reused_class.setUp(testcase)

    class TestWrap(tests.TestCase, test_textwrap.WrapTestCase):

        def setUp(self):
            if False:
                while True:
                    i = 10
            setup_both(self, TestWrap, test_textwrap.WrapTestCase)

    class TestLongWord(tests.TestCase, test_textwrap.LongWordTestCase):

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            setup_both(self, TestLongWord, test_textwrap.LongWordTestCase)

    class TestIndent(tests.TestCase, test_textwrap.IndentTestCases):

        def setUp(self):
            if False:
                print('Hello World!')
            setup_both(self, TestIndent, test_textwrap.IndentTestCases)
except ImportError:

    class TestWrap(tests.TestCase):

        def test_wrap(self):
            if False:
                for i in range(10):
                    print('nop')
            raise tests.TestSkipped('test.test_textwrap is not available.')

    class TestLongWord(tests.TestCase):

        def test_longword(self):
            if False:
                for i in range(10):
                    print('nop')
            raise tests.TestSkipped('test.test_textwrap is not available.')

    class TestIndent(tests.TestCase):

        def test_indent(self):
            if False:
                for i in range(10):
                    print('nop')
            raise tests.TestSkipped('test.test_textwrap is not available.')