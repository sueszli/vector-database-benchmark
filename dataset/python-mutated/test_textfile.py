from StringIO import StringIO
from bzrlib.errors import BinaryFile
from bzrlib.tests import TestCase, TestCaseInTempDir
from bzrlib.textfile import text_file, check_text_lines, check_text_path

class TextFile(TestCase):

    def test_text_file(self):
        if False:
            i = 10
            return i + 15
        s = StringIO('ab' * 2048)
        self.assertEqual(text_file(s).read(), s.getvalue())
        s = StringIO('a' * 1023 + '\x00')
        self.assertRaises(BinaryFile, text_file, s)
        s = StringIO('a' * 1024 + '\x00')
        self.assertEqual(text_file(s).read(), s.getvalue())

    def test_check_text_lines(self):
        if False:
            for i in range(10):
                print('nop')
        lines = ['ab' * 2048]
        check_text_lines(lines)
        lines = ['a' * 1023 + '\x00']
        self.assertRaises(BinaryFile, check_text_lines, lines)

class TextPath(TestCaseInTempDir):

    def test_text_file(self):
        if False:
            print('Hello World!')
        with file('boo', 'wb') as f:
            f.write('ab' * 2048)
        check_text_path('boo')
        with file('boo', 'wb') as f:
            f.write('a' * 1023 + '\x00')
        self.assertRaises(BinaryFile, check_text_path, 'boo')