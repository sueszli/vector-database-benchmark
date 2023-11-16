"""Tests for rio serialization

A simple, reproducible structured IO format.

rio itself works in Unicode strings.  It is typically encoded to UTF-8,
but this depends on the transport.
"""
import re
from tempfile import TemporaryFile
from bzrlib import rio
from bzrlib.tests import TestCase
from bzrlib.rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file

class TestRio(TestCase):

    def test_stanza(self):
        if False:
            return 10
        'Construct rio stanza in memory'
        s = Stanza(number='42', name='fred')
        self.assertTrue('number' in s)
        self.assertFalse('color' in s)
        self.assertFalse('42' in s)
        self.assertEqual(list(s.iter_pairs()), [('name', 'fred'), ('number', '42')])
        self.assertEqual(s.get('number'), '42')
        self.assertEqual(s.get('name'), 'fred')

    def test_value_checks(self):
        if False:
            i = 10
            return i + 15
        'rio checks types on construction'

    def test_empty_value(self):
        if False:
            for i in range(10):
                print('nop')
        'Serialize stanza with empty field'
        s = Stanza(empty='')
        self.assertEqualDiff(s.to_string(), 'empty: \n')

    def test_to_lines(self):
        if False:
            i = 10
            return i + 15
        'Write simple rio stanza to string'
        s = Stanza(number='42', name='fred')
        self.assertEqual(list(s.to_lines()), ['name: fred\n', 'number: 42\n'])

    def test_as_dict(self):
        if False:
            i = 10
            return i + 15
        'Convert rio Stanza to dictionary'
        s = Stanza(number='42', name='fred')
        sd = s.as_dict()
        self.assertEqual(sd, dict(number='42', name='fred'))

    def test_to_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Write rio to file'
        tmpf = TemporaryFile()
        s = Stanza(a_thing='something with "quotes like \\"this\\""', number='42', name='fred')
        s.write(tmpf)
        tmpf.seek(0)
        self.assertEqualDiff(tmpf.read(), '\na_thing: something with "quotes like \\"this\\""\nname: fred\nnumber: 42\n'[1:])

    def test_multiline_string(self):
        if False:
            return 10
        tmpf = TemporaryFile()
        s = Stanza(motto='war is peace\nfreedom is slavery\nignorance is strength')
        s.write(tmpf)
        tmpf.seek(0)
        self.assertEqualDiff(tmpf.read(), 'motto: war is peace\n\tfreedom is slavery\n\tignorance is strength\n')
        tmpf.seek(0)
        s2 = read_stanza(tmpf)
        self.assertEqual(s, s2)

    def test_read_stanza(self):
        if False:
            while True:
                i = 10
        'Load stanza from string'
        lines = 'revision: mbp@sourcefrog.net-123-abc\ntimestamp: 1130653962\ntimezone: 36000\ncommitter: Martin Pool <mbp@test.sourcefrog.net>\n'.splitlines(True)
        s = read_stanza(lines)
        self.assertTrue('revision' in s)
        self.assertEqualDiff(s.get('revision'), 'mbp@sourcefrog.net-123-abc')
        self.assertEqual(list(s.iter_pairs()), [('revision', 'mbp@sourcefrog.net-123-abc'), ('timestamp', '1130653962'), ('timezone', '36000'), ('committer', 'Martin Pool <mbp@test.sourcefrog.net>')])
        self.assertEqual(len(s), 4)

    def test_repeated_field(self):
        if False:
            i = 10
            return i + 15
        'Repeated field in rio'
        s = Stanza()
        for (k, v) in [('a', '10'), ('b', '20'), ('a', '100'), ('b', '200'), ('a', '1000'), ('b', '2000')]:
            s.add(k, v)
        s2 = read_stanza(s.to_lines())
        self.assertEqual(s, s2)
        self.assertEqual(s.get_all('a'), map(str, [10, 100, 1000]))
        self.assertEqual(s.get_all('b'), map(str, [20, 200, 2000]))

    def test_backslash(self):
        if False:
            return 10
        s = Stanza(q='\\')
        t = s.to_string()
        self.assertEqualDiff(t, 'q: \\\n')
        s2 = read_stanza(s.to_lines())
        self.assertEqual(s, s2)

    def test_blank_line(self):
        if False:
            return 10
        s = Stanza(none='', one='\n', two='\n\n')
        self.assertEqualDiff(s.to_string(), 'none: \none: \n\t\ntwo: \n\t\n\t\n')
        s2 = read_stanza(s.to_lines())
        self.assertEqual(s, s2)

    def test_whitespace_value(self):
        if False:
            i = 10
            return i + 15
        s = Stanza(space=' ', tabs='\t\t\t', combo='\n\t\t\n')
        self.assertEqualDiff(s.to_string(), 'combo: \n\t\t\t\n\t\nspace:  \ntabs: \t\t\t\n')
        s2 = read_stanza(s.to_lines())
        self.assertEqual(s, s2)
        self.rio_file_stanzas([s])

    def test_quoted(self):
        if False:
            for i in range(10):
                print('nop')
        'rio quoted string cases'
        s = Stanza(q1='"hello"', q2=' "for', q3='\n\n"for"\n', q4='for\n"\nfor', q5='\n', q6='"', q7='""', q8='\\', q9='\\"\\"')
        s2 = read_stanza(s.to_lines())
        self.assertEqual(s, s2)

    def test_read_empty(self):
        if False:
            i = 10
            return i + 15
        'Detect end of rio file'
        s = read_stanza([])
        self.assertEqual(s, None)
        self.assertTrue(s is None)

    def test_read_nul_byte(self):
        if False:
            i = 10
            return i + 15
        'File consisting of a nul byte causes an error.'
        self.assertRaises(ValueError, read_stanza, ['\x00'])

    def test_read_nul_bytes(self):
        if False:
            print('Hello World!')
        'File consisting of many nul bytes causes an error.'
        self.assertRaises(ValueError, read_stanza, ['\x00' * 100])

    def test_read_iter(self):
        if False:
            print('Hello World!')
        'Read several stanzas from file'
        tmpf = TemporaryFile()
        tmpf.write('version_header: 1\n\nname: foo\nval: 123\n\nname: bar\nval: 129319\n')
        tmpf.seek(0)
        reader = read_stanzas(tmpf)
        read_iter = iter(reader)
        stuff = list(reader)
        self.assertEqual(stuff, [Stanza(version_header='1'), Stanza(name='foo', val='123'), Stanza(name='bar', val='129319')])

    def test_read_several(self):
        if False:
            i = 10
            return i + 15
        'Read several stanzas from file'
        tmpf = TemporaryFile()
        tmpf.write('version_header: 1\n\nname: foo\nval: 123\n\nname: quoted\naddress:   "Willowglen"\n\t  42 Wallaby Way\n\t  Sydney\n\nname: bar\nval: 129319\n')
        tmpf.seek(0)
        s = read_stanza(tmpf)
        self.assertEqual(s, Stanza(version_header='1'))
        s = read_stanza(tmpf)
        self.assertEqual(s, Stanza(name='foo', val='123'))
        s = read_stanza(tmpf)
        self.assertEqualDiff(s.get('name'), 'quoted')
        self.assertEqualDiff(s.get('address'), '  "Willowglen"\n  42 Wallaby Way\n  Sydney')
        s = read_stanza(tmpf)
        self.assertEqual(s, Stanza(name='bar', val='129319'))
        s = read_stanza(tmpf)
        self.assertEqual(s, None)
        self.check_rio_file(tmpf)

    def check_rio_file(self, real_file):
        if False:
            print('Hello World!')
        real_file.seek(0)
        read_write = rio_file(RioReader(real_file)).read()
        real_file.seek(0)
        self.assertEqual(read_write, real_file.read())

    @staticmethod
    def stanzas_to_str(stanzas):
        if False:
            return 10
        return rio_file(stanzas).read()

    def rio_file_stanzas(self, stanzas):
        if False:
            for i in range(10):
                print('nop')
        new_stanzas = list(RioReader(rio_file(stanzas)))
        self.assertEqual(new_stanzas, stanzas)

    def test_tricky_quoted(self):
        if False:
            i = 10
            return i + 15
        tmpf = TemporaryFile()
        tmpf.write('s: "one"\n\ns: \n\t"one"\n\t\n\ns: "\n\ns: ""\n\ns: """\n\ns: \n\t\n\ns: \\\n\ns: \n\t\\\n\t\\\\\n\t\n\ns: word\\\n\ns: quote"\n\ns: backslashes\\\\\\\n\ns: both\\"\n\n')
        tmpf.seek(0)
        expected_vals = ['"one"', '\n"one"\n', '"', '""', '"""', '\n', '\\', '\n\\\n\\\\\n', 'word\\', 'quote"', 'backslashes\\\\\\', 'both\\"']
        for expected in expected_vals:
            stanza = read_stanza(tmpf)
            self.rio_file_stanzas([stanza])
            self.assertEqual(len(stanza), 1)
            self.assertEqualDiff(stanza.get('s'), expected)

    def test_write_empty_stanza(self):
        if False:
            return 10
        'Write empty stanza'
        l = list(Stanza().to_lines())
        self.assertEqual(l, [])

    def test_rio_raises_type_error(self):
        if False:
            print('Hello World!')
        'TypeError on adding invalid type to Stanza'
        s = Stanza()
        self.assertRaises(TypeError, s.add, 'foo', {})

    def test_rio_raises_type_error_key(self):
        if False:
            while True:
                i = 10
        'TypeError on adding invalid type to Stanza'
        s = Stanza()
        self.assertRaises(TypeError, s.add, 10, {})

    def test_rio_unicode(self):
        if False:
            print('Hello World!')
        uni_data = u'オ'
        s = Stanza(foo=uni_data)
        self.assertEqual(s.get('foo'), uni_data)
        raw_lines = s.to_lines()
        self.assertEqual(raw_lines, ['foo: ' + uni_data.encode('utf-8') + '\n'])
        new_s = read_stanza(raw_lines)
        self.assertEqual(new_s.get('foo'), uni_data)

    def test_rio_to_unicode(self):
        if False:
            for i in range(10):
                print('nop')
        uni_data = u'オ'
        s = Stanza(foo=uni_data)
        unicode_str = s.to_unicode()
        self.assertEqual(u'foo: %s\n' % (uni_data,), unicode_str)
        new_s = rio.read_stanza_unicode(unicode_str.splitlines(True))
        self.assertEqual(uni_data, new_s.get('foo'))

    def test_nested_rio_unicode(self):
        if False:
            print('Hello World!')
        uni_data = u'オ'
        s = Stanza(foo=uni_data)
        parent_stanza = Stanza(child=s.to_unicode())
        raw_lines = parent_stanza.to_lines()
        self.assertEqual(['child: foo: ' + uni_data.encode('utf-8') + '\n', '\t\n'], raw_lines)
        new_parent = read_stanza(raw_lines)
        child_text = new_parent.get('child')
        self.assertEqual(u'foo: %s\n' % uni_data, child_text)
        new_child = rio.read_stanza_unicode(child_text.splitlines(True))
        self.assertEqual(uni_data, new_child.get('foo'))

    def mail_munge(self, lines, dos_nl=True):
        if False:
            print('Hello World!')
        new_lines = []
        for line in lines:
            line = re.sub(' *\n', '\n', line)
            if dos_nl:
                line = re.sub('([^\r])\n', '\\1\r\n', line)
            new_lines.append(line)
        return new_lines

    def test_patch_rio(self):
        if False:
            i = 10
            return i + 15
        stanza = Stanza(data='#\n\r\\r ', space=' ' * 255, hash='#' * 255)
        lines = rio.to_patch_lines(stanza)
        for line in lines:
            self.assertContainsRe(line, '^# ')
            self.assertTrue(72 >= len(line))
        for line in rio.to_patch_lines(stanza, max_width=12):
            self.assertTrue(12 >= len(line))
        new_stanza = rio.read_patch_stanza(self.mail_munge(lines, dos_nl=False))
        lines = self.mail_munge(lines)
        new_stanza = rio.read_patch_stanza(lines)
        self.assertEqual('#\n\r\\r ', new_stanza.get('data'))
        self.assertEqual(' ' * 255, new_stanza.get('space'))
        self.assertEqual('#' * 255, new_stanza.get('hash'))

    def test_patch_rio_linebreaks(self):
        if False:
            while True:
                i = 10
        stanza = Stanza(breaktest='linebreak -/' * 30)
        self.assertContainsRe(rio.to_patch_lines(stanza, 71)[0], 'linebreak\\\\\n')
        stanza = Stanza(breaktest='linebreak-/' * 30)
        self.assertContainsRe(rio.to_patch_lines(stanza, 70)[0], 'linebreak-\\\\\n')
        stanza = Stanza(breaktest='linebreak/' * 30)
        self.assertContainsRe(rio.to_patch_lines(stanza, 70)[0], 'linebreak\\\\\n')