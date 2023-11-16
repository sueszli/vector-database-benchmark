"""Test pyshell, coverage 12%."""
from idlelib import pyshell
import unittest
from test.support import requires
from tkinter import Tk

class FunctionTest(unittest.TestCase):

    def test_restart_line_wide(self):
        if False:
            return 10
        eq = self.assertEqual
        for (file, mul, extra) in (('', 22, ''), ('finame', 21, '=')):
            width = 60
            bar = mul * '='
            with self.subTest(file=file, bar=bar):
                file = file or 'Shell'
                line = pyshell.restart_line(width, file)
                eq(len(line), width)
                eq(line, f'{bar + extra} RESTART: {file} {bar}')

    def test_restart_line_narrow(self):
        if False:
            for i in range(10):
                print('nop')
        (expect, taglen) = ('= RESTART: Shell', 16)
        for width in (taglen - 1, taglen, taglen + 1):
            with self.subTest(width=width):
                self.assertEqual(pyshell.restart_line(width, ''), expect)
        self.assertEqual(pyshell.restart_line(taglen + 2, ''), expect + ' =')

class PyShellFileListTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        requires('gui')
        cls.root = Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        cls.root.destroy()
        del cls.root

    def test_init(self):
        if False:
            return 10
        psfl = pyshell.PyShellFileList(self.root)
        self.assertEqual(psfl.EditorWindow, pyshell.PyShellEditorWindow)
        self.assertIsNone(psfl.pyshell)

class PyShellRemoveLastNewlineAndSurroundingWhitespaceTest(unittest.TestCase):
    regexp = pyshell.PyShell._last_newline_re

    def all_removed(self, text):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('', self.regexp.sub('', text))

    def none_removed(self, text):
        if False:
            print('Hello World!')
        self.assertEqual(text, self.regexp.sub('', text))

    def check_result(self, text, expected):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(expected, self.regexp.sub('', text))

    def test_empty(self):
        if False:
            return 10
        self.all_removed('')

    def test_newline(self):
        if False:
            while True:
                i = 10
        self.all_removed('\n')

    def test_whitespace_no_newline(self):
        if False:
            return 10
        self.all_removed(' ')
        self.all_removed('  ')
        self.all_removed('   ')
        self.all_removed(' ' * 20)
        self.all_removed('\t')
        self.all_removed('\t\t')
        self.all_removed('\t\t\t')
        self.all_removed('\t' * 20)
        self.all_removed('\t ')
        self.all_removed(' \t')
        self.all_removed(' \t \t ')
        self.all_removed('\t \t \t')

    def test_newline_with_whitespace(self):
        if False:
            for i in range(10):
                print('nop')
        self.all_removed(' \n')
        self.all_removed('\t\n')
        self.all_removed(' \t\n')
        self.all_removed('\t \n')
        self.all_removed('\n ')
        self.all_removed('\n\t')
        self.all_removed('\n \t')
        self.all_removed('\n\t ')
        self.all_removed(' \n ')
        self.all_removed('\t\n ')
        self.all_removed(' \n\t')
        self.all_removed('\t\n\t')
        self.all_removed('\t \t \t\n')
        self.all_removed(' \t \t \n')
        self.all_removed('\n\t \t \t')
        self.all_removed('\n \t \t ')

    def test_multiple_newlines(self):
        if False:
            i = 10
            return i + 15
        self.check_result('\n\n', '\n')
        self.check_result('\n' * 5, '\n' * 4)
        self.check_result('\n' * 5 + '\t', '\n' * 4)
        self.check_result('\n' * 20, '\n' * 19)
        self.check_result('\n' * 20 + ' ', '\n' * 19)
        self.check_result(' \n \n ', ' \n')
        self.check_result(' \n\n ', ' \n')
        self.check_result(' \n\n', ' \n')
        self.check_result('\t\n\n', '\t\n')
        self.check_result('\n\n ', '\n')
        self.check_result('\n\n\t', '\n')
        self.check_result(' \n \n ', ' \n')
        self.check_result('\t\n\t\n\t', '\t\n')

    def test_non_whitespace(self):
        if False:
            while True:
                i = 10
        self.none_removed('a')
        self.check_result('a\n', 'a')
        self.check_result('a\n ', 'a')
        self.check_result('a \n ', 'a')
        self.check_result('a \n\t', 'a')
        self.none_removed('-')
        self.check_result('-\n', '-')
        self.none_removed('.')
        self.check_result('.\n', '.')

    def test_unsupported_whitespace(self):
        if False:
            print('Hello World!')
        self.none_removed('\x0b')
        self.none_removed('\n\x0b')
        self.check_result('\x0b\n', '\x0b')
        self.none_removed(' \n\x0b')
        self.check_result('\x0b\n ', '\x0b')
if __name__ == '__main__':
    unittest.main(verbosity=2)