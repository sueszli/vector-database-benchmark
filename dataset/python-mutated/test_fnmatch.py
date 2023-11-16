"""Test cases for the fnmatch module."""
import unittest
import os
import string
import warnings
from fnmatch import fnmatch, fnmatchcase, translate, filter

class FnmatchTestCase(unittest.TestCase):

    def check_match(self, filename, pattern, should_match=True, fn=fnmatch):
        if False:
            while True:
                i = 10
        if should_match:
            self.assertTrue(fn(filename, pattern), 'expected %r to match pattern %r' % (filename, pattern))
        else:
            self.assertFalse(fn(filename, pattern), 'expected %r not to match pattern %r' % (filename, pattern))

    def test_fnmatch(self):
        if False:
            for i in range(10):
                print('nop')
        check = self.check_match
        check('abc', 'abc')
        check('abc', '?*?')
        check('abc', '???*')
        check('abc', '*???')
        check('abc', '???')
        check('abc', '*')
        check('abc', 'ab[cd]')
        check('abc', 'ab[!de]')
        check('abc', 'ab[de]', False)
        check('a', '??', False)
        check('a', 'b', False)
        check('\\', '[\\]')
        check('a', '[!\\]')
        check('\\', '[!\\]', False)
        check('foo\nbar', 'foo*')
        check('foo\nbar\n', 'foo*')
        check('\nfoo', 'foo*', False)
        check('\n', '*')

    def test_slow_fnmatch(self):
        if False:
            i = 10
            return i + 15
        check = self.check_match
        check('a' * 50, '*a*a*a*a*a*a*a*a*a*a')
        check('a' * 50 + 'b', '*a*a*a*a*a*a*a*a*a*a', False)

    def test_mix_bytes_str(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, fnmatch, 'test', b'*')
        self.assertRaises(TypeError, fnmatch, b'test', '*')
        self.assertRaises(TypeError, fnmatchcase, 'test', b'*')
        self.assertRaises(TypeError, fnmatchcase, b'test', '*')

    def test_fnmatchcase(self):
        if False:
            print('Hello World!')
        check = self.check_match
        check('abc', 'abc', True, fnmatchcase)
        check('AbC', 'abc', False, fnmatchcase)
        check('abc', 'AbC', False, fnmatchcase)
        check('AbC', 'AbC', True, fnmatchcase)
        check('usr/bin', 'usr/bin', True, fnmatchcase)
        check('usr\\bin', 'usr/bin', False, fnmatchcase)
        check('usr/bin', 'usr\\bin', False, fnmatchcase)
        check('usr\\bin', 'usr\\bin', True, fnmatchcase)

    def test_bytes(self):
        if False:
            while True:
                i = 10
        self.check_match(b'test', b'te*')
        self.check_match(b'test\xff', b'te*\xff')
        self.check_match(b'foo\nbar', b'foo*')

    def test_case(self):
        if False:
            for i in range(10):
                print('nop')
        ignorecase = os.path.normcase('ABC') == os.path.normcase('abc')
        check = self.check_match
        check('abc', 'abc')
        check('AbC', 'abc', ignorecase)
        check('abc', 'AbC', ignorecase)
        check('AbC', 'AbC')

    def test_sep(self):
        if False:
            for i in range(10):
                print('nop')
        normsep = os.path.normcase('\\') == os.path.normcase('/')
        check = self.check_match
        check('usr/bin', 'usr/bin')
        check('usr\\bin', 'usr/bin', normsep)
        check('usr/bin', 'usr\\bin', normsep)
        check('usr\\bin', 'usr\\bin')

    def test_char_set(self):
        if False:
            print('Hello World!')
        ignorecase = os.path.normcase('ABC') == os.path.normcase('abc')
        check = self.check_match
        tescases = string.ascii_lowercase + string.digits + string.punctuation
        for c in tescases:
            check(c, '[az]', c in 'az')
            check(c, '[!az]', c not in 'az')
        for c in tescases:
            check(c, '[AZ]', c in 'az' and ignorecase)
            check(c, '[!AZ]', c not in 'az' or not ignorecase)
        for c in string.ascii_uppercase:
            check(c, '[az]', c in 'AZ' and ignorecase)
            check(c, '[!az]', c not in 'AZ' or not ignorecase)
        for c in tescases:
            check(c, '[aa]', c == 'a')
        for c in tescases:
            check(c, '[^az]', c in '^az')
            check(c, '[[az]', c in '[az')
            check(c, '[!]]', c != ']')
        check('[', '[')
        check('[]', '[]')
        check('[!', '[!')
        check('[!]', '[!]')

    def test_range(self):
        if False:
            i = 10
            return i + 15
        ignorecase = os.path.normcase('ABC') == os.path.normcase('abc')
        normsep = os.path.normcase('\\') == os.path.normcase('/')
        check = self.check_match
        tescases = string.ascii_lowercase + string.digits + string.punctuation
        for c in tescases:
            check(c, '[b-d]', c in 'bcd')
            check(c, '[!b-d]', c not in 'bcd')
            check(c, '[b-dx-z]', c in 'bcdxyz')
            check(c, '[!b-dx-z]', c not in 'bcdxyz')
        for c in tescases:
            check(c, '[B-D]', c in 'bcd' and ignorecase)
            check(c, '[!B-D]', c not in 'bcd' or not ignorecase)
        for c in string.ascii_uppercase:
            check(c, '[b-d]', c in 'BCD' and ignorecase)
            check(c, '[!b-d]', c not in 'BCD' or not ignorecase)
        for c in tescases:
            check(c, '[b-b]', c == 'b')
        for c in tescases:
            check(c, '[!-#]', c not in '-#')
            check(c, '[!--.]', c not in '-.')
            check(c, '[^-`]', c in '^_`')
            if not (normsep and c == '/'):
                check(c, '[[-^]', c in '[\\]^')
                check(c, '[\\-^]', c in '\\]^')
            check(c, '[b-]', c in '-b')
            check(c, '[!b-]', c not in '-b')
            check(c, '[-b]', c in '-b')
            check(c, '[!-b]', c not in '-b')
            check(c, '[-]', c in '-')
            check(c, '[!-]', c not in '-')
        for c in tescases:
            check(c, '[d-b]', False)
            check(c, '[!d-b]', True)
            check(c, '[d-bx-z]', c in 'xyz')
            check(c, '[!d-bx-z]', c not in 'xyz')
            check(c, '[d-b^-`]', c in '^_`')
            if not (normsep and c == '/'):
                check(c, '[d-b[-^]', c in '[\\]^')

    def test_sep_in_char_set(self):
        if False:
            i = 10
            return i + 15
        normsep = os.path.normcase('\\') == os.path.normcase('/')
        check = self.check_match
        check('/', '[/]')
        check('\\', '[\\]')
        check('/', '[\\]', normsep)
        check('\\', '[/]', normsep)
        check('[/]', '[/]', False)
        check('[\\\\]', '[/]', False)
        check('\\', '[\\t]')
        check('/', '[\\t]', normsep)
        check('t', '[\\t]')
        check('\t', '[\\t]', False)

    def test_sep_in_range(self):
        if False:
            i = 10
            return i + 15
        normsep = os.path.normcase('\\') == os.path.normcase('/')
        check = self.check_match
        check('a/b', 'a[.-0]b', not normsep)
        check('a\\b', 'a[.-0]b', False)
        check('a\\b', 'a[Z-^]b', not normsep)
        check('a/b', 'a[Z-^]b', False)
        check('a/b', 'a[/-0]b', not normsep)
        check('a\\b', 'a[/-0]b', False)
        check('a[/-0]b', 'a[/-0]b', False)
        check('a[\\-0]b', 'a[/-0]b', False)
        check('a/b', 'a[.-/]b')
        check('a\\b', 'a[.-/]b', normsep)
        check('a[.-/]b', 'a[.-/]b', False)
        check('a[.-\\]b', 'a[.-/]b', False)
        check('a\\b', 'a[\\-^]b')
        check('a/b', 'a[\\-^]b', normsep)
        check('a[\\-^]b', 'a[\\-^]b', False)
        check('a[/-^]b', 'a[\\-^]b', False)
        check('a\\b', 'a[Z-\\]b', not normsep)
        check('a/b', 'a[Z-\\]b', False)
        check('a[Z-\\]b', 'a[Z-\\]b', False)
        check('a[Z-/]b', 'a[Z-\\]b', False)

    def test_warnings(self):
        if False:
            print('Hello World!')
        with warnings.catch_warnings():
            warnings.simplefilter('error', Warning)
            check = self.check_match
            check('[', '[[]')
            check('&', '[a&&b]')
            check('|', '[a||b]')
            check('~', '[a~~b]')
            check(',', '[a-z+--A-Z]')
            check('.', '[a-z--/A-Z]')

class TranslateTestCase(unittest.TestCase):

    def test_translate(self):
        if False:
            print('Hello World!')
        import re
        self.assertEqual(translate('*'), '(?s:.*)\\Z')
        self.assertEqual(translate('?'), '(?s:.)\\Z')
        self.assertEqual(translate('a?b*'), '(?s:a.b.*)\\Z')
        self.assertEqual(translate('[abc]'), '(?s:[abc])\\Z')
        self.assertEqual(translate('[]]'), '(?s:[]])\\Z')
        self.assertEqual(translate('[!x]'), '(?s:[^x])\\Z')
        self.assertEqual(translate('[^x]'), '(?s:[\\^x])\\Z')
        self.assertEqual(translate('[x'), '(?s:\\[x)\\Z')
        self.assertEqual(translate('*.txt'), '(?s:.*\\.txt)\\Z')
        self.assertEqual(translate('*********'), '(?s:.*)\\Z')
        self.assertEqual(translate('A*********'), '(?s:A.*)\\Z')
        self.assertEqual(translate('*********A'), '(?s:.*A)\\Z')
        self.assertEqual(translate('A*********?[?]?'), '(?s:A.*.[?].)\\Z')
        t = translate('**a*a****a')
        digits = re.findall('\\d+', t)
        self.assertEqual(len(digits), 4)
        self.assertEqual(digits[0], digits[1])
        self.assertEqual(digits[2], digits[3])
        g1 = f'g{digits[0]}'
        g2 = f'g{digits[2]}'
        self.assertEqual(t, f'(?s:(?=(?P<{g1}>.*?a))(?P={g1})(?=(?P<{g2}>.*?a))(?P={g2}).*a)\\Z')
        r1 = translate('**a**a**a*')
        r2 = translate('**b**b**b*')
        r3 = translate('*c*c*c*')
        fatre = '|'.join([r1, r2, r3])
        self.assertTrue(re.match(fatre, 'abaccad'))
        self.assertTrue(re.match(fatre, 'abxbcab'))
        self.assertTrue(re.match(fatre, 'cbabcaxc'))
        self.assertFalse(re.match(fatre, 'dabccbad'))

class FilterTestCase(unittest.TestCase):

    def test_filter(self):
        if False:
            while True:
                i = 10
        self.assertEqual(filter(['Python', 'Ruby', 'Perl', 'Tcl'], 'P*'), ['Python', 'Perl'])
        self.assertEqual(filter([b'Python', b'Ruby', b'Perl', b'Tcl'], b'P*'), [b'Python', b'Perl'])

    def test_mix_bytes_str(self):
        if False:
            return 10
        self.assertRaises(TypeError, filter, ['test'], b'*')
        self.assertRaises(TypeError, filter, [b'test'], '*')

    def test_case(self):
        if False:
            while True:
                i = 10
        ignorecase = os.path.normcase('P') == os.path.normcase('p')
        self.assertEqual(filter(['Test.py', 'Test.rb', 'Test.PL'], '*.p*'), ['Test.py', 'Test.PL'] if ignorecase else ['Test.py'])
        self.assertEqual(filter(['Test.py', 'Test.rb', 'Test.PL'], '*.P*'), ['Test.py', 'Test.PL'] if ignorecase else ['Test.PL'])

    def test_sep(self):
        if False:
            return 10
        normsep = os.path.normcase('\\') == os.path.normcase('/')
        self.assertEqual(filter(['usr/bin', 'usr', 'usr\\lib'], 'usr/*'), ['usr/bin', 'usr\\lib'] if normsep else ['usr/bin'])
        self.assertEqual(filter(['usr/bin', 'usr', 'usr\\lib'], 'usr\\*'), ['usr/bin', 'usr\\lib'] if normsep else ['usr\\lib'])
if __name__ == '__main__':
    unittest.main()