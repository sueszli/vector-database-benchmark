"""Tests for _rio_*."""
from bzrlib import rio, tests

def load_tests(standard_tests, module, loader):
    if False:
        return 10
    (suite, _) = tests.permute_tests_for_extension(standard_tests, loader, 'bzrlib._rio_py', 'bzrlib._rio_pyx')
    return suite

class TestValidTag(tests.TestCase):
    module = None

    def test_ok(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.module._valid_tag('foo'))

    def test_no_spaces(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(self.module._valid_tag('foo bla'))

    def test_numeric(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.module._valid_tag('3foo423'))

    def test_no_colon(self):
        if False:
            while True:
                i = 10
        self.assertFalse(self.module._valid_tag('foo:bla'))

    def test_type_error(self):
        if False:
            return 10
        self.assertRaises(TypeError, self.module._valid_tag, 423)

    def test_empty(self):
        if False:
            print('Hello World!')
        self.assertFalse(self.module._valid_tag(''))

    def test_unicode(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, self.module._valid_tag, u'foo')

    def test_non_ascii_char(self):
        if False:
            print('Hello World!')
        self.assertFalse(self.module._valid_tag('µ'))

class TestReadUTF8Stanza(tests.TestCase):
    module = None

    def assertReadStanza(self, result, line_iter):
        if False:
            while True:
                i = 10
        s = self.module._read_stanza_utf8(line_iter)
        self.assertEqual(result, s)
        if s is not None:
            for (tag, value) in s.iter_pairs():
                self.assertIsInstance(tag, str)
                self.assertIsInstance(value, unicode)

    def assertReadStanzaRaises(self, exception, line_iter):
        if False:
            while True:
                i = 10
        self.assertRaises(exception, self.module._read_stanza_utf8, line_iter)

    def test_no_string(self):
        if False:
            while True:
                i = 10
        self.assertReadStanzaRaises(TypeError, [21323])

    def test_empty(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertReadStanza(None, [])

    def test_none(self):
        if False:
            print('Hello World!')
        self.assertReadStanza(None, [''])

    def test_simple(self):
        if False:
            return 10
        self.assertReadStanza(rio.Stanza(foo='bar'), ['foo: bar\n', ''])

    def test_multi_line(self):
        if False:
            return 10
        self.assertReadStanza(rio.Stanza(foo='bar\nbla'), ['foo: bar\n', '\tbla\n'])

    def test_repeated(self):
        if False:
            for i in range(10):
                print('nop')
        s = rio.Stanza()
        s.add('foo', 'bar')
        s.add('foo', 'foo')
        self.assertReadStanza(s, ['foo: bar\n', 'foo: foo\n'])

    def test_invalid_early_colon(self):
        if False:
            return 10
        self.assertReadStanzaRaises(ValueError, ['f:oo: bar\n'])

    def test_invalid_tag(self):
        if False:
            return 10
        self.assertReadStanzaRaises(ValueError, ['f%oo: bar\n'])

    def test_continuation_too_early(self):
        if False:
            while True:
                i = 10
        self.assertReadStanzaRaises(ValueError, ['\tbar\n'])

    def test_large(self):
        if False:
            while True:
                i = 10
        value = 'bla' * 9000
        self.assertReadStanza(rio.Stanza(foo=value), ['foo: %s\n' % value])

    def test_non_ascii_char(self):
        if False:
            print('Hello World!')
        self.assertReadStanza(rio.Stanza(foo=u'nåme'), [u'foo: nåme\n'.encode('utf-8')])

class TestReadUnicodeStanza(tests.TestCase):
    module = None

    def assertReadStanza(self, result, line_iter):
        if False:
            while True:
                i = 10
        s = self.module._read_stanza_unicode(line_iter)
        self.assertEqual(result, s)
        if s is not None:
            for (tag, value) in s.iter_pairs():
                self.assertIsInstance(tag, str)
                self.assertIsInstance(value, unicode)

    def assertReadStanzaRaises(self, exception, line_iter):
        if False:
            while True:
                i = 10
        self.assertRaises(exception, self.module._read_stanza_unicode, line_iter)

    def test_no_string(self):
        if False:
            i = 10
            return i + 15
        self.assertReadStanzaRaises(TypeError, [21323])

    def test_empty(self):
        if False:
            while True:
                i = 10
        self.assertReadStanza(None, [])

    def test_none(self):
        if False:
            while True:
                i = 10
        self.assertReadStanza(None, [u''])

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        self.assertReadStanza(rio.Stanza(foo='bar'), [u'foo: bar\n', u''])

    def test_multi_line(self):
        if False:
            print('Hello World!')
        self.assertReadStanza(rio.Stanza(foo='bar\nbla'), [u'foo: bar\n', u'\tbla\n'])

    def test_repeated(self):
        if False:
            i = 10
            return i + 15
        s = rio.Stanza()
        s.add('foo', 'bar')
        s.add('foo', 'foo')
        self.assertReadStanza(s, [u'foo: bar\n', u'foo: foo\n'])

    def test_invalid_early_colon(self):
        if False:
            i = 10
            return i + 15
        self.assertReadStanzaRaises(ValueError, [u'f:oo: bar\n'])

    def test_invalid_tag(self):
        if False:
            return 10
        self.assertReadStanzaRaises(ValueError, [u'f%oo: bar\n'])

    def test_continuation_too_early(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertReadStanzaRaises(ValueError, [u'\tbar\n'])

    def test_large(self):
        if False:
            i = 10
            return i + 15
        value = u'bla' * 9000
        self.assertReadStanza(rio.Stanza(foo=value), [u'foo: %s\n' % value])

    def test_non_ascii_char(self):
        if False:
            print('Hello World!')
        self.assertReadStanza(rio.Stanza(foo=u'nåme'), [u'foo: nåme\n'])