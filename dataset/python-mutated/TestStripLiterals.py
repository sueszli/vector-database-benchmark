from Cython.Build.Dependencies import strip_string_literals
from Cython.TestUtils import CythonTest

class TestStripLiterals(CythonTest):

    def t(self, before, expected):
        if False:
            return 10
        (actual, literals) = strip_string_literals(before, prefix='_L')
        self.assertEqual(expected, actual)
        for (key, value) in literals.items():
            actual = actual.replace(key, value)
        self.assertEqual(before, actual)

    def test_empty(self):
        if False:
            print('Hello World!')
        self.t('', '')

    def test_single_quote(self):
        if False:
            for i in range(10):
                print('nop')
        self.t("'x'", "'_L1_'")

    def test_double_quote(self):
        if False:
            for i in range(10):
                print('nop')
        self.t('"x"', '"_L1_"')

    def test_nested_quotes(self):
        if False:
            print('Hello World!')
        self.t(' \'"\' "\'" ', ' \'_L1_\' "_L2_" ')

    def test_triple_quote(self):
        if False:
            for i in range(10):
                print('nop')
        self.t(" '''a\n''' ", " '''_L1_''' ")

    def test_backslash(self):
        if False:
            print('Hello World!')
        self.t("'a\\'b'", "'_L1_'")
        self.t("'a\\\\'", "'_L1_'")
        self.t("'a\\\\\\'b'", "'_L1_'")

    def test_unicode(self):
        if False:
            return 10
        self.t("u'abc'", "u'_L1_'")

    def test_raw(self):
        if False:
            for i in range(10):
                print('nop')
        self.t("r'abc\\\\'", "r'_L1_'")

    def test_raw_unicode(self):
        if False:
            i = 10
            return i + 15
        self.t("ru'abc\\\\'", "ru'_L1_'")

    def test_comment(self):
        if False:
            while True:
                i = 10
        self.t('abc # foo', 'abc #_L1_')

    def test_comment_and_quote(self):
        if False:
            while True:
                i = 10
        self.t("abc # 'x'", 'abc #_L1_')
        self.t("'abc#'", "'_L1_'")

    def test_include(self):
        if False:
            for i in range(10):
                print('nop')
        self.t("include 'a.pxi' # something here", "include '_L1_' #_L2_")

    def test_extern(self):
        if False:
            i = 10
            return i + 15
        self.t("cdef extern from 'a.h': # comment", "cdef extern from '_L1_': #_L2_")