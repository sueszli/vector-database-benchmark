import difflib
from test.support import findfile
import unittest
import doctest
import sys

class TestWithAscii(unittest.TestCase):

    def test_one_insert(self):
        if False:
            for i in range(10):
                print('nop')
        sm = difflib.SequenceMatcher(None, 'b' * 100, 'a' + 'b' * 100)
        self.assertAlmostEqual(sm.ratio(), 0.995, places=3)
        self.assertEqual(list(sm.get_opcodes()), [('insert', 0, 0, 0, 1), ('equal', 0, 100, 1, 101)])
        self.assertEqual(sm.bpopular, set())
        sm = difflib.SequenceMatcher(None, 'b' * 100, 'b' * 50 + 'a' + 'b' * 50)
        self.assertAlmostEqual(sm.ratio(), 0.995, places=3)
        self.assertEqual(list(sm.get_opcodes()), [('equal', 0, 50, 0, 50), ('insert', 50, 50, 50, 51), ('equal', 50, 100, 51, 101)])
        self.assertEqual(sm.bpopular, set())

    def test_one_delete(self):
        if False:
            while True:
                i = 10
        sm = difflib.SequenceMatcher(None, 'a' * 40 + 'c' + 'b' * 40, 'a' * 40 + 'b' * 40)
        self.assertAlmostEqual(sm.ratio(), 0.994, places=3)
        self.assertEqual(list(sm.get_opcodes()), [('equal', 0, 40, 0, 40), ('delete', 40, 41, 40, 40), ('equal', 41, 81, 40, 80)])

    def test_bjunk(self):
        if False:
            while True:
                i = 10
        sm = difflib.SequenceMatcher(isjunk=lambda x: x == ' ', a='a' * 40 + 'b' * 40, b='a' * 44 + 'b' * 40)
        self.assertEqual(sm.bjunk, set())
        sm = difflib.SequenceMatcher(isjunk=lambda x: x == ' ', a='a' * 40 + 'b' * 40, b='a' * 44 + 'b' * 40 + ' ' * 20)
        self.assertEqual(sm.bjunk, {' '})
        sm = difflib.SequenceMatcher(isjunk=lambda x: x in [' ', 'b'], a='a' * 40 + 'b' * 40, b='a' * 44 + 'b' * 40 + ' ' * 20)
        self.assertEqual(sm.bjunk, {' ', 'b'})

class TestAutojunk(unittest.TestCase):
    """Tests for the autojunk parameter added in 2.7"""

    def test_one_insert_homogenous_sequence(self):
        if False:
            for i in range(10):
                print('nop')
        seq1 = 'b' * 200
        seq2 = 'a' + 'b' * 200
        sm = difflib.SequenceMatcher(None, seq1, seq2)
        self.assertAlmostEqual(sm.ratio(), 0, places=3)
        self.assertEqual(sm.bpopular, {'b'})
        sm = difflib.SequenceMatcher(None, seq1, seq2, autojunk=False)
        self.assertAlmostEqual(sm.ratio(), 0.9975, places=3)
        self.assertEqual(sm.bpopular, set())

class TestSFbugs(unittest.TestCase):

    def test_ratio_for_null_seqn(self):
        if False:
            print('Hello World!')
        s = difflib.SequenceMatcher(None, [], [])
        self.assertEqual(s.ratio(), 1)
        self.assertEqual(s.quick_ratio(), 1)
        self.assertEqual(s.real_quick_ratio(), 1)

    def test_comparing_empty_lists(self):
        if False:
            for i in range(10):
                print('nop')
        group_gen = difflib.SequenceMatcher(None, [], []).get_grouped_opcodes()
        self.assertRaises(StopIteration, next, group_gen)
        diff_gen = difflib.unified_diff([], [])
        self.assertRaises(StopIteration, next, diff_gen)

    def test_matching_blocks_cache(self):
        if False:
            i = 10
            return i + 15
        s = difflib.SequenceMatcher(None, 'abxcd', 'abcd')
        first = s.get_matching_blocks()
        second = s.get_matching_blocks()
        self.assertEqual(second[0].size, 2)
        self.assertEqual(second[1].size, 2)
        self.assertEqual(second[2].size, 0)

    def test_added_tab_hint(self):
        if False:
            i = 10
            return i + 15
        diff = list(difflib.Differ().compare(['\tI am a buggy'], ['\t\tI am a bug']))
        self.assertEqual('- \tI am a buggy', diff[0])
        self.assertEqual('? \t          --\n', diff[1])
        self.assertEqual('+ \t\tI am a bug', diff[2])
        self.assertEqual('? +\n', diff[3])

    def test_hint_indented_properly_with_tabs(self):
        if False:
            i = 10
            return i + 15
        diff = list(difflib.Differ().compare(['\t \t \t^'], ['\t \t \t^\n']))
        self.assertEqual('- \t \t \t^', diff[0])
        self.assertEqual('+ \t \t \t^\n', diff[1])
        self.assertEqual('? \t \t \t +\n', diff[2])

    def test_mdiff_catch_stop_iteration(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(list(difflib._mdiff(['2'], ['3'], 1)), [((1, '\x00-2\x01'), (1, '\x00+3\x01'), True)])
patch914575_from1 = '\n   1. Beautiful is beTTer than ugly.\n   2. Explicit is better than implicit.\n   3. Simple is better than complex.\n   4. Complex is better than complicated.\n'
patch914575_to1 = '\n   1. Beautiful is better than ugly.\n   3.   Simple is better than complex.\n   4. Complicated is better than complex.\n   5. Flat is better than nested.\n'
patch914575_nonascii_from1 = '\n   1. Beautiful is beTTer than ugly.\n   2. Explicit is better than ımplıcıt.\n   3. Simple is better than complex.\n   4. Complex is better than complicated.\n'
patch914575_nonascii_to1 = '\n   1. Beautiful is better than ügly.\n   3.   Sımple is better than complex.\n   4. Complicated is better than cömplex.\n   5. Flat is better than nested.\n'
patch914575_from2 = '\n\t\tLine 1: preceded by from:[tt] to:[ssss]\n  \t\tLine 2: preceded by from:[sstt] to:[sssst]\n  \t \tLine 3: preceded by from:[sstst] to:[ssssss]\nLine 4:  \thas from:[sst] to:[sss] after :\nLine 5: has from:[t] to:[ss] at end\t\n'
patch914575_to2 = '\n    Line 1: preceded by from:[tt] to:[ssss]\n    \tLine 2: preceded by from:[sstt] to:[sssst]\n      Line 3: preceded by from:[sstst] to:[ssssss]\nLine 4:   has from:[sst] to:[sss] after :\nLine 5: has from:[t] to:[ss] at end\n'
patch914575_from3 = 'line 0\n1234567890123456789012345689012345\nline 1\nline 2\nline 3\nline 4   changed\nline 5   changed\nline 6   changed\nline 7\nline 8  subtracted\nline 9\n1234567890123456789012345689012345\nshort line\njust fits in!!\njust fits in two lines yup!!\nthe end'
patch914575_to3 = 'line 0\n1234567890123456789012345689012345\nline 1\nline 2    added\nline 3\nline 4   chanGEd\nline 5a  chanGed\nline 6a  changEd\nline 7\nline 8\nline 9\n1234567890\nanother long line that needs to be wrapped\njust fitS in!!\njust fits in two lineS yup!!\nthe end'

class TestSFpatches(unittest.TestCase):

    def test_html_diff(self):
        if False:
            for i in range(10):
                print('nop')
        f1a = (patch914575_from1 + '123\n' * 10) * 3
        t1a = (patch914575_to1 + '123\n' * 10) * 3
        f1b = '456\n' * 10 + f1a
        t1b = '456\n' * 10 + t1a
        f1a = f1a.splitlines()
        t1a = t1a.splitlines()
        f1b = f1b.splitlines()
        t1b = t1b.splitlines()
        f2 = patch914575_from2.splitlines()
        t2 = patch914575_to2.splitlines()
        f3 = patch914575_from3
        t3 = patch914575_to3
        i = difflib.HtmlDiff()
        j = difflib.HtmlDiff(tabsize=2)
        k = difflib.HtmlDiff(wrapcolumn=14)
        full = i.make_file(f1a, t1a, 'from', 'to', context=False, numlines=5)
        tables = '\n'.join(['<h2>Context (first diff within numlines=5(default))</h2>', i.make_table(f1a, t1a, 'from', 'to', context=True), '<h2>Context (first diff after numlines=5(default))</h2>', i.make_table(f1b, t1b, 'from', 'to', context=True), '<h2>Context (numlines=6)</h2>', i.make_table(f1a, t1a, 'from', 'to', context=True, numlines=6), '<h2>Context (numlines=0)</h2>', i.make_table(f1a, t1a, 'from', 'to', context=True, numlines=0), '<h2>Same Context</h2>', i.make_table(f1a, f1a, 'from', 'to', context=True), '<h2>Same Full</h2>', i.make_table(f1a, f1a, 'from', 'to', context=False), '<h2>Empty Context</h2>', i.make_table([], [], 'from', 'to', context=True), '<h2>Empty Full</h2>', i.make_table([], [], 'from', 'to', context=False), '<h2>tabsize=2</h2>', j.make_table(f2, t2), '<h2>tabsize=default</h2>', i.make_table(f2, t2), '<h2>Context (wrapcolumn=14,numlines=0)</h2>', k.make_table(f3.splitlines(), t3.splitlines(), context=True, numlines=0), '<h2>wrapcolumn=14,splitlines()</h2>', k.make_table(f3.splitlines(), t3.splitlines()), '<h2>wrapcolumn=14,splitlines(True)</h2>', k.make_table(f3.splitlines(True), t3.splitlines(True))])
        actual = full.replace('</body>', '\n%s\n</body>' % tables)
        with open(findfile('test_difflib_expect.html'), encoding='utf-8') as fp:
            self.assertEqual(actual, fp.read())

    def test_recursion_limit(self):
        if False:
            for i in range(10):
                print('nop')
        limit = sys.getrecursionlimit()
        old = [(i % 2 and 'K:%d' or 'V:A:%d') % i for i in range(limit * 2)]
        new = [(i % 2 and 'K:%d' or 'V:B:%d') % i for i in range(limit * 2)]
        difflib.SequenceMatcher(None, old, new).get_opcodes()

    def test_make_file_default_charset(self):
        if False:
            i = 10
            return i + 15
        html_diff = difflib.HtmlDiff()
        output = html_diff.make_file(patch914575_from1.splitlines(), patch914575_to1.splitlines())
        self.assertIn('content="text/html; charset=utf-8"', output)

    def test_make_file_iso88591_charset(self):
        if False:
            return 10
        html_diff = difflib.HtmlDiff()
        output = html_diff.make_file(patch914575_from1.splitlines(), patch914575_to1.splitlines(), charset='iso-8859-1')
        self.assertIn('content="text/html; charset=iso-8859-1"', output)

    def test_make_file_usascii_charset_with_nonascii_input(self):
        if False:
            print('Hello World!')
        html_diff = difflib.HtmlDiff()
        output = html_diff.make_file(patch914575_nonascii_from1.splitlines(), patch914575_nonascii_to1.splitlines(), charset='us-ascii')
        self.assertIn('content="text/html; charset=us-ascii"', output)
        self.assertIn('&#305;mpl&#305;c&#305;t', output)

class TestOutputFormat(unittest.TestCase):

    def test_tab_delimiter(self):
        if False:
            i = 10
            return i + 15
        args = ['one', 'two', 'Original', 'Current', '2005-01-26 23:30:50', '2010-04-02 10:20:52']
        ud = difflib.unified_diff(*args, lineterm='')
        self.assertEqual(list(ud)[0:2], ['--- Original\t2005-01-26 23:30:50', '+++ Current\t2010-04-02 10:20:52'])
        cd = difflib.context_diff(*args, lineterm='')
        self.assertEqual(list(cd)[0:2], ['*** Original\t2005-01-26 23:30:50', '--- Current\t2010-04-02 10:20:52'])

    def test_no_trailing_tab_on_empty_filedate(self):
        if False:
            return 10
        args = ['one', 'two', 'Original', 'Current']
        ud = difflib.unified_diff(*args, lineterm='')
        self.assertEqual(list(ud)[0:2], ['--- Original', '+++ Current'])
        cd = difflib.context_diff(*args, lineterm='')
        self.assertEqual(list(cd)[0:2], ['*** Original', '--- Current'])

    def test_range_format_unified(self):
        if False:
            while True:
                i = 10
        spec = '           Each <range> field shall be of the form:\n             %1d", <beginning line number>  if the range contains exactly one line,\n           and:\n            "%1d,%1d", <beginning line number>, <number of lines> otherwise.\n           If a range is empty, its beginning line number shall be the number of\n           the line just before the range, or 0 if the empty range starts the file.\n        '
        fmt = difflib._format_range_unified
        self.assertEqual(fmt(3, 3), '3,0')
        self.assertEqual(fmt(3, 4), '4')
        self.assertEqual(fmt(3, 5), '4,2')
        self.assertEqual(fmt(3, 6), '4,3')
        self.assertEqual(fmt(0, 0), '0,0')

    def test_range_format_context(self):
        if False:
            print('Hello World!')
        spec = '           The range of lines in file1 shall be written in the following format\n           if the range contains two or more lines:\n               "*** %d,%d ****\n", <beginning line number>, <ending line number>\n           and the following format otherwise:\n               "*** %d ****\n", <ending line number>\n           The ending line number of an empty range shall be the number of the preceding line,\n           or 0 if the range is at the start of the file.\n\n           Next, the range of lines in file2 shall be written in the following format\n           if the range contains two or more lines:\n               "--- %d,%d ----\n", <beginning line number>, <ending line number>\n           and the following format otherwise:\n               "--- %d ----\n", <ending line number>\n        '
        fmt = difflib._format_range_context
        self.assertEqual(fmt(3, 3), '3')
        self.assertEqual(fmt(3, 4), '4')
        self.assertEqual(fmt(3, 5), '4,5')
        self.assertEqual(fmt(3, 6), '4,6')
        self.assertEqual(fmt(0, 0), '0')

class TestBytes(unittest.TestCase):

    def check(self, diff):
        if False:
            for i in range(10):
                print('nop')
        diff = list(diff)
        for line in diff:
            self.assertIsInstance(line, bytes, 'all lines of diff should be bytes, but got: %r' % line)

    def test_byte_content(self):
        if False:
            while True:
                i = 10
        a = [b'hello', b'andr\xe9']
        b = [b'hello', b'andr\xc3\xa9']
        unified = difflib.unified_diff
        context = difflib.context_diff
        check = self.check
        check(difflib.diff_bytes(unified, a, a))
        check(difflib.diff_bytes(unified, a, b))
        check(difflib.diff_bytes(unified, a, a, b'a', b'a'))
        check(difflib.diff_bytes(unified, a, b, b'a', b'b'))
        check(difflib.diff_bytes(unified, a, a, b'a', b'a', b'2005', b'2013'))
        check(difflib.diff_bytes(unified, a, b, b'a', b'b', b'2005', b'2013'))
        check(difflib.diff_bytes(context, a, a))
        check(difflib.diff_bytes(context, a, b))
        check(difflib.diff_bytes(context, a, a, b'a', b'a'))
        check(difflib.diff_bytes(context, a, b, b'a', b'b'))
        check(difflib.diff_bytes(context, a, a, b'a', b'a', b'2005', b'2013'))
        check(difflib.diff_bytes(context, a, b, b'a', b'b', b'2005', b'2013'))

    def test_byte_filenames(self):
        if False:
            i = 10
            return i + 15
        fna = b'\xb3odz.txt'
        fnb = b'\xc5\x82odz.txt'
        a = [b'\xa3odz is a city in Poland.']
        b = [b'\xc5\x81odz is a city in Poland.']
        check = self.check
        unified = difflib.unified_diff
        context = difflib.context_diff
        check(difflib.diff_bytes(unified, a, b, fna, fnb))
        check(difflib.diff_bytes(context, a, b, fna, fnb))

        def assertDiff(expect, actual):
            if False:
                for i in range(10):
                    print('nop')
            actual = list(actual)
            self.assertEqual(len(expect), len(actual))
            for (e, a) in zip(expect, actual):
                self.assertEqual(e, a)
        expect = [b'--- \xb3odz.txt', b'+++ \xc5\x82odz.txt', b'@@ -1 +1 @@', b'-\xa3odz is a city in Poland.', b'+\xc5\x81odz is a city in Poland.']
        actual = difflib.diff_bytes(unified, a, b, fna, fnb, lineterm=b'')
        assertDiff(expect, actual)
        datea = b'2005-03-18'
        dateb = b'2005-03-19'
        check(difflib.diff_bytes(unified, a, b, fna, fnb, datea, dateb))
        check(difflib.diff_bytes(context, a, b, fna, fnb, datea, dateb))
        expect = [b'--- \xb3odz.txt\t2005-03-18', b'+++ \xc5\x82odz.txt\t2005-03-19', b'@@ -1 +1 @@', b'-\xa3odz is a city in Poland.', b'+\xc5\x81odz is a city in Poland.']
        actual = difflib.diff_bytes(unified, a, b, fna, fnb, datea, dateb, lineterm=b'')
        assertDiff(expect, actual)

    def test_mixed_types_content(self):
        if False:
            print('Hello World!')
        a = [b'hello']
        b = ['hello']
        unified = difflib.unified_diff
        context = difflib.context_diff
        expect = "lines to compare must be str, not bytes (b'hello')"
        self._assert_type_error(expect, unified, a, b)
        self._assert_type_error(expect, unified, b, a)
        self._assert_type_error(expect, context, a, b)
        self._assert_type_error(expect, context, b, a)
        expect = "all arguments must be bytes, not str ('hello')"
        self._assert_type_error(expect, difflib.diff_bytes, unified, a, b)
        self._assert_type_error(expect, difflib.diff_bytes, unified, b, a)
        self._assert_type_error(expect, difflib.diff_bytes, context, a, b)
        self._assert_type_error(expect, difflib.diff_bytes, context, b, a)

    def test_mixed_types_filenames(self):
        if False:
            return 10
        a = ['hello\n']
        b = ['ohell\n']
        fna = b'ol\xe9.txt'
        fnb = b'ol\xc3a9.txt'
        self._assert_type_error("all arguments must be str, not: b'ol\\xe9.txt'", difflib.unified_diff, a, b, fna, fnb)

    def test_mixed_types_dates(self):
        if False:
            return 10
        a = [b'foo\n']
        b = [b'bar\n']
        datea = '1 fév'
        dateb = '3 fév'
        self._assert_type_error("all arguments must be bytes, not str ('1 fév')", difflib.diff_bytes, difflib.unified_diff, a, b, b'a', b'b', datea, dateb)
        a = ['foo\n']
        b = ['bar\n']
        list(difflib.unified_diff(a, b, 'a', 'b', datea, dateb))

    def _assert_type_error(self, msg, generator, *args):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError) as ctx:
            list(generator(*args))
        self.assertEqual(msg, str(ctx.exception))

class TestJunkAPIs(unittest.TestCase):

    def test_is_line_junk_true(self):
        if False:
            i = 10
            return i + 15
        for line in ['#', '  ', ' #', '# ', ' # ', '']:
            self.assertTrue(difflib.IS_LINE_JUNK(line), repr(line))

    def test_is_line_junk_false(self):
        if False:
            for i in range(10):
                print('nop')
        for line in ['##', ' ##', '## ', 'abc ', 'abc #', 'Mr. Moose is up!']:
            self.assertFalse(difflib.IS_LINE_JUNK(line), repr(line))

    def test_is_line_junk_REDOS(self):
        if False:
            while True:
                i = 10
        evil_input = '\t' * 1000000 + '##'
        self.assertFalse(difflib.IS_LINE_JUNK(evil_input))

    def test_is_character_junk_true(self):
        if False:
            return 10
        for char in [' ', '\t']:
            self.assertTrue(difflib.IS_CHARACTER_JUNK(char), repr(char))

    def test_is_character_junk_false(self):
        if False:
            for i in range(10):
                print('nop')
        for char in ['a', '#', '\n', '\x0c', '\r', '\x0b']:
            self.assertFalse(difflib.IS_CHARACTER_JUNK(char), repr(char))

class TestFindLongest(unittest.TestCase):

    def longer_match_exists(self, a, b, n):
        if False:
            while True:
                i = 10
        return any((b_part in a for b_part in [b[i:i + n + 1] for i in range(0, len(b) - n - 1)]))

    def test_default_args(self):
        if False:
            while True:
                i = 10
        a = 'foo bar'
        b = 'foo baz bar'
        sm = difflib.SequenceMatcher(a=a, b=b)
        match = sm.find_longest_match()
        self.assertEqual(match.a, 0)
        self.assertEqual(match.b, 0)
        self.assertEqual(match.size, 6)
        self.assertEqual(a[match.a:match.a + match.size], b[match.b:match.b + match.size])
        self.assertFalse(self.longer_match_exists(a, b, match.size))
        match = sm.find_longest_match(alo=2, blo=4)
        self.assertEqual(match.a, 3)
        self.assertEqual(match.b, 7)
        self.assertEqual(match.size, 4)
        self.assertEqual(a[match.a:match.a + match.size], b[match.b:match.b + match.size])
        self.assertFalse(self.longer_match_exists(a[2:], b[4:], match.size))
        match = sm.find_longest_match(bhi=5, blo=1)
        self.assertEqual(match.a, 1)
        self.assertEqual(match.b, 1)
        self.assertEqual(match.size, 4)
        self.assertEqual(a[match.a:match.a + match.size], b[match.b:match.b + match.size])
        self.assertFalse(self.longer_match_exists(a, b[1:5], match.size))

    def test_longest_match_with_popular_chars(self):
        if False:
            print('Hello World!')
        a = 'dabcd'
        b = 'd' * 100 + 'abc' + 'd' * 100
        sm = difflib.SequenceMatcher(a=a, b=b)
        match = sm.find_longest_match(0, len(a), 0, len(b))
        self.assertEqual(match.a, 0)
        self.assertEqual(match.b, 99)
        self.assertEqual(match.size, 5)
        self.assertEqual(a[match.a:match.a + match.size], b[match.b:match.b + match.size])
        self.assertFalse(self.longer_match_exists(a, b, match.size))

def setUpModule():
    if False:
        return 10
    difflib.HtmlDiff._default_prefix = 0

def load_tests(loader, tests, pattern):
    if False:
        while True:
            i = 10
    tests.addTest(doctest.DocTestSuite(difflib))
    return tests
if __name__ == '__main__':
    unittest.main()