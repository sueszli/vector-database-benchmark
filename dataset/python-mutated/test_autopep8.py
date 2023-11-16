"""Test suite for autopep8.

Unit tests go in "UnitTests". System tests go in "SystemTests".

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import re
import sys
import time
import contextlib
import io
import shutil
import stat
from subprocess import Popen, PIPE
from tempfile import mkstemp, mkdtemp
import tokenize
import unittest
import warnings
from io import StringIO
ROOT_DIR = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.insert(0, ROOT_DIR)
import autopep8
from autopep8 import get_module_imports_on_top_of_file
FAKE_CONFIGURATION = os.path.join(ROOT_DIR, 'test', 'fake_configuration')
FAKE_PYCODESTYLE_CONFIGURATION = os.path.join(ROOT_DIR, 'test', 'fake_pycodestyle_configuration')
if 'AUTOPEP8_COVERAGE' in os.environ and int(os.environ['AUTOPEP8_COVERAGE']):
    AUTOPEP8_CMD_TUPLE = (sys.executable, '-Wignore::DeprecationWarning', '-m', 'coverage', 'run', '--branch', '--parallel', '--omit=*/site-packages/*', os.path.join(ROOT_DIR, 'autopep8.py'))
else:
    AUTOPEP8_CMD_TUPLE = (sys.executable, '-Wignore::DeprecationWarning', os.path.join(ROOT_DIR, 'autopep8.py'))

class UnitTests(unittest.TestCase):
    maxDiff = None

    def test_compile_value_error(self):
        if False:
            return 10
        source = '"\\xhh" \\'
        self.assertFalse(autopep8.check_syntax(source))

    def test_find_newline_only_cr(self):
        if False:
            for i in range(10):
                print('nop')
        source = ['print(1)\r', 'print(2)\r', 'print3\r']
        self.assertEqual(autopep8.CR, autopep8.find_newline(source))

    def test_find_newline_only_lf(self):
        if False:
            print('Hello World!')
        source = ['print(1)\n', 'print(2)\n', 'print3\n']
        self.assertEqual(autopep8.LF, autopep8.find_newline(source))

    def test_find_newline_only_crlf(self):
        if False:
            return 10
        source = ['print(1)\r\n', 'print(2)\r\n', 'print3\r\n']
        self.assertEqual(autopep8.CRLF, autopep8.find_newline(source))

    def test_find_newline_cr1_and_lf2(self):
        if False:
            return 10
        source = ['print(1)\n', 'print(2)\r', 'print3\n']
        self.assertEqual(autopep8.LF, autopep8.find_newline(source))

    def test_find_newline_cr1_and_crlf2(self):
        if False:
            return 10
        source = ['print(1)\r\n', 'print(2)\r', 'print3\r\n']
        self.assertEqual(autopep8.CRLF, autopep8.find_newline(source))

    def test_find_newline_should_default_to_lf(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(autopep8.LF, autopep8.find_newline([]))
        self.assertEqual(autopep8.LF, autopep8.find_newline(['', '']))

    def test_detect_encoding(self):
        if False:
            return 10
        self.assertEqual('utf-8', autopep8.detect_encoding(os.path.join(ROOT_DIR, 'test', 'test_autopep8.py')))

    def test_detect_encoding_with_cookie(self):
        if False:
            print('Hello World!')
        self.assertEqual('iso-8859-1', autopep8.detect_encoding(os.path.join(ROOT_DIR, 'test', 'iso_8859_1.py')))

    def test_readlines_from_file_with_bad_encoding(self):
        if False:
            print('Hello World!')
        'Bad encoding should not cause an exception.'
        self.assertEqual(['# -*- coding: zlatin-1 -*-\n'], autopep8.readlines_from_file(os.path.join(ROOT_DIR, 'test', 'bad_encoding.py')))

    def test_readlines_from_file_with_bad_encoding2(self):
        if False:
            print('Hello World!')
        'Bad encoding should not cause an exception.'
        with warnings.catch_warnings(record=True):
            self.assertTrue(autopep8.readlines_from_file(os.path.join(ROOT_DIR, 'test', 'bad_encoding2.py')))

    def test_fix_whitespace(self):
        if False:
            while True:
                i = 10
        self.assertEqual('a b', autopep8.fix_whitespace('a    b', offset=1, replacement=' '))

    def test_fix_whitespace_with_tabs(self):
        if False:
            print('Hello World!')
        self.assertEqual('a b', autopep8.fix_whitespace('a\t  \t  b', offset=1, replacement=' '))

    def test_multiline_string_lines(self):
        if False:
            print('Hello World!')
        self.assertEqual({2}, autopep8.multiline_string_lines("'''\n'''\n"))

    def test_multiline_string_lines_with_many(self):
        if False:
            while True:
                i = 10
        self.assertEqual({2, 7, 10, 11, 12}, autopep8.multiline_string_lines("'''\n'''\n''''''\n''''''\n''''''\n'''\n'''\n\n'''\n\n\n'''\n"))

    def test_multiline_string_should_not_report_single_line(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(set(), autopep8.multiline_string_lines("'''abc'''\n"))

    def test_multiline_string_should_not_report_docstrings(self):
        if False:
            return 10
        self.assertEqual({5}, autopep8.multiline_string_lines("def foo():\n    '''Foo.\n    Bar.'''\n    hello = '''\n'''\n"))

    def test_supported_fixes(self):
        if False:
            while True:
                i = 10
        self.assertIn('E121', [f[0] for f in autopep8.supported_fixes()])

    def test_shorten_comment(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('# ' + '=' * 72 + '\n', autopep8.shorten_comment('# ' + '=' * 100 + '\n', max_line_length=79))

    def test_shorten_comment_should_not_split_numbers(self):
        if False:
            for i in range(10):
                print('nop')
        line = '# ' + '0' * 100 + '\n'
        self.assertEqual(line, autopep8.shorten_comment(line, max_line_length=79))

    def test_shorten_comment_should_not_split_words(self):
        if False:
            for i in range(10):
                print('nop')
        line = '# ' + 'a' * 100 + '\n'
        self.assertEqual(line, autopep8.shorten_comment(line, max_line_length=79))

    def test_shorten_comment_should_not_split_urls(self):
        if False:
            i = 10
            return i + 15
        line = '# http://foo.bar/' + 'abc-' * 100 + '\n'
        self.assertEqual(line, autopep8.shorten_comment(line, max_line_length=79))

    def test_shorten_comment_should_not_modify_special_comments(self):
        if False:
            print('Hello World!')
        line = '#!/bin/blah ' + ' x' * 90 + '\n'
        self.assertEqual(line, autopep8.shorten_comment(line, max_line_length=79))

    def test_format_block_comments(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('# abc', fix_e265_and_e266('#abc'))
        self.assertEqual('# abc', fix_e265_and_e266('####abc'))
        self.assertEqual('# abc', fix_e265_and_e266('##   #   ##abc'))
        self.assertEqual('# abc "# noqa"', fix_e265_and_e266('# abc "# noqa"'))
        self.assertEqual('# *abc', fix_e265_and_e266('#*abc'))

    def test_format_block_comments_should_leave_outline_alone(self):
        if False:
            return 10
        line = '###################################################################\n##   Some people like these crazy things. So leave them alone.   ##\n###################################################################\n'
        self.assertEqual(line, fix_e265_and_e266(line))
        line = '#################################################################\n#   Some people like these crazy things. So leave them alone.   #\n#################################################################\n'
        self.assertEqual(line, fix_e265_and_e266(line))

    def test_format_block_comments_with_multiple_lines(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual("# abc\n  # blah blah\n    # four space indentation\n''' #do not modify strings\n#do not modify strings\n#do not modify strings\n#do not modify strings'''\n#\n", fix_e265_and_e266("# abc\n  #blah blah\n    #four space indentation\n''' #do not modify strings\n#do not modify strings\n#do not modify strings\n#do not modify strings'''\n#\n"))

    def test_format_block_comments_should_not_corrupt_special_comments(self):
        if False:
            while True:
                i = 10
        self.assertEqual('#: abc', fix_e265_and_e266('#: abc'))
        self.assertEqual('#!/bin/bash\n', fix_e265_and_e266('#!/bin/bash\n'))

    def test_format_block_comments_should_only_touch_real_comments(self):
        if False:
            for i in range(10):
                print('nop')
        commented_out_code = '#x = 1'
        self.assertEqual(commented_out_code, fix_e266(commented_out_code))

    def test_fix_file(self):
        if False:
            i = 10
            return i + 15
        self.assertIn('import ', autopep8.fix_file(filename=os.path.join(ROOT_DIR, 'test', 'example.py')))

    def test_fix_file_with_diff(self):
        if False:
            while True:
                i = 10
        filename = os.path.join(ROOT_DIR, 'test', 'example.py')
        self.assertIn('@@', autopep8.fix_file(filename=filename, options=autopep8.parse_args(['--diff', filename])))

    def test_fix_lines(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('print(123)\n', autopep8.fix_lines(['print( 123 )\n'], options=autopep8.parse_args([''])))

    def test_fix_code(self):
        if False:
            while True:
                i = 10
        self.assertEqual('print(123)\n', autopep8.fix_code('print( 123 )\n'))

    def test_fix_code_with_empty_string(self):
        if False:
            print('Hello World!')
        self.assertEqual('', autopep8.fix_code(''))

    def test_fix_code_with_multiple_lines(self):
        if False:
            return 10
        self.assertEqual('print(123)\nx = 4\n', autopep8.fix_code('print( 123 )\nx   =4'))

    def test_fix_code_byte_string(self):
        if False:
            return 10
        'This feature is here for friendliness to Python 2.'
        self.assertEqual('print(123)\n', autopep8.fix_code(b'print( 123 )\n'))

    def test_fix_code_with_options(self):
        if False:
            print('Hello World!')
        self.assertEqual('print(123)\n', autopep8.fix_code('print( 123 )\n', options={'ignore': ['W']}))
        self.assertEqual('print( 123 )\n', autopep8.fix_code('print( 123 )\n', options={'ignore': ['E']}))

    def test_fix_code_with_bad_options(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            autopep8.fix_code('print( 123 )\n', options={'ignor': ['W']})
        with self.assertRaises(ValueError):
            autopep8.fix_code('print( 123 )\n', options={'ignore': 'W'})

    def test_normalize_line_endings(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(['abc\n', 'def\n', '123\n', 'hello\n', 'world\n'], autopep8.normalize_line_endings(['abc\n', 'def\n', '123\n', 'hello\r\n', 'world\r'], '\n'))

    def test_normalize_line_endings_with_crlf(self):
        if False:
            print('Hello World!')
        self.assertEqual(['abc\r\n', 'def\r\n', '123\r\n', 'hello\r\n', 'world\r\n'], autopep8.normalize_line_endings(['abc\n', 'def\r\n', '123\r\n', 'hello\r\n', 'world\r'], '\r\n'))

    def test_normalize_multiline(self):
        if False:
            while True:
                i = 10
        self.assertEqual('def foo(): pass', autopep8.normalize_multiline('def foo():'))
        self.assertEqual('def _(): return 1', autopep8.normalize_multiline('return 1'))
        self.assertEqual('@decorator\ndef _(): pass', autopep8.normalize_multiline('@decorator\n'))
        self.assertEqual('class A: pass', autopep8.normalize_multiline('class A:'))

    def test_code_match(self):
        if False:
            return 10
        self.assertTrue(autopep8.code_match('E2', select=['E2', 'E3'], ignore=[]))
        self.assertTrue(autopep8.code_match('E26', select=['E2', 'E3'], ignore=[]))
        self.assertFalse(autopep8.code_match('E26', select=[], ignore=['E']))
        self.assertFalse(autopep8.code_match('E2', select=['E2', 'E3'], ignore=['E2']))
        self.assertFalse(autopep8.code_match('E26', select=['W'], ignore=['']))
        self.assertFalse(autopep8.code_match('E26', select=['W'], ignore=['E1']))

    def test_split_at_offsets(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual([''], autopep8.split_at_offsets('', [0]))
        self.assertEqual(['1234'], autopep8.split_at_offsets('1234', [0]))
        self.assertEqual(['1', '234'], autopep8.split_at_offsets('1234', [1]))
        self.assertEqual(['12', '34'], autopep8.split_at_offsets('1234', [2]))
        self.assertEqual(['12', '3', '4'], autopep8.split_at_offsets('1234', [2, 3]))

    def test_split_at_offsets_with_out_of_order(self):
        if False:
            while True:
                i = 10
        self.assertEqual(['12', '3', '4'], autopep8.split_at_offsets('1234', [3, 2]))

    def test_fix_2to3(self):
        if False:
            while True:
                i = 10
        self.assertEqual('try: pass\nexcept ValueError as e: pass\n', autopep8.fix_2to3('try: pass\nexcept ValueError, e: pass\n'))
        self.assertEqual('while True: pass\n', autopep8.fix_2to3('while 1: pass\n'))
        self.assertEqual('import sys\nsys.maxsize\n', autopep8.fix_2to3('import sys\nsys.maxint\n'))

    def test_fix_2to3_subset(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'type(res) == type(42)\n'
        fixed = 'isinstance(res, type(42))\n'
        self.assertEqual(fixed, autopep8.fix_2to3(line))
        self.assertEqual(fixed, autopep8.fix_2to3(line, select=['E721']))
        self.assertEqual(fixed, autopep8.fix_2to3(line, select=['E7']))
        self.assertEqual(line, autopep8.fix_2to3(line, select=['W']))
        self.assertEqual(line, autopep8.fix_2to3(line, select=['E999']))
        self.assertEqual(line, autopep8.fix_2to3(line, ignore=['E721']))

    def test_is_python_file(self):
        if False:
            print('Hello World!')
        self.assertTrue(autopep8.is_python_file(os.path.join(ROOT_DIR, 'autopep8.py')))
        with temporary_file_context('#!/usr/bin/env python') as filename:
            self.assertTrue(autopep8.is_python_file(filename))
        with temporary_file_context('#!/usr/bin/python') as filename:
            self.assertTrue(autopep8.is_python_file(filename))
        with temporary_file_context('#!/usr/bin/python3') as filename:
            self.assertTrue(autopep8.is_python_file(filename))
        with temporary_file_context('#!/usr/bin/pythonic') as filename:
            self.assertFalse(autopep8.is_python_file(filename))
        with temporary_file_context('###!/usr/bin/python') as filename:
            self.assertFalse(autopep8.is_python_file(filename))
        self.assertFalse(autopep8.is_python_file(os.devnull))
        self.assertFalse(autopep8.is_python_file('/bin/bash'))

    def test_match_file(self):
        if False:
            for i in range(10):
                print('nop')
        with temporary_file_context('', suffix='.py', prefix='.') as filename:
            self.assertFalse(autopep8.match_file(filename, exclude=[]), msg=filename)
        self.assertFalse(autopep8.match_file(os.devnull, exclude=[]))
        with temporary_file_context('', suffix='.py', prefix='') as filename:
            self.assertTrue(autopep8.match_file(filename, exclude=[]), msg=filename)

    def test_match_file_with_dummy_file(self):
        if False:
            while True:
                i = 10
        filename = 'notexists.dummyfile.dummy'
        self.assertEqual(autopep8.match_file(filename, exclude=[]), False)

    def test_find_files(self):
        if False:
            for i in range(10):
                print('nop')
        temp_directory = mkdtemp()
        try:
            target = os.path.join(temp_directory, 'dir')
            os.mkdir(target)
            with open(os.path.join(target, 'a.py'), 'w'):
                pass
            exclude = os.path.join(target, 'ex')
            os.mkdir(exclude)
            with open(os.path.join(exclude, 'b.py'), 'w'):
                pass
            sub = os.path.join(exclude, 'sub')
            os.mkdir(sub)
            with open(os.path.join(sub, 'c.py'), 'w'):
                pass
            cwd = os.getcwd()
            os.chdir(temp_directory)
            try:
                files = list(autopep8.find_files(['dir'], True, [os.path.join('dir', 'ex')]))
            finally:
                os.chdir(cwd)
            file_names = [os.path.basename(f) for f in files]
            self.assertIn('a.py', file_names)
            self.assertNotIn('b.py', file_names)
            self.assertNotIn('c.py', file_names)
        finally:
            shutil.rmtree(temp_directory)

    def test_line_shortening_rank(self):
        if False:
            i = 10
            return i + 15
        self.assertGreater(autopep8.line_shortening_rank('(1\n+1)\n', indent_word='    ', max_line_length=79), autopep8.line_shortening_rank('(1+\n1)\n', indent_word='    ', max_line_length=79))
        self.assertGreaterEqual(autopep8.line_shortening_rank('(1+\n1)\n', indent_word='    ', max_line_length=79), autopep8.line_shortening_rank('(1+1)\n', indent_word='    ', max_line_length=79))
        autopep8.line_shortening_rank('\n', indent_word='    ', max_line_length=79)
        self.assertGreater(autopep8.line_shortening_rank('[foo(\nx) for x in y]\n', indent_word='    ', max_line_length=79), autopep8.line_shortening_rank('[foo(x)\nfor x in y]\n', indent_word='    ', max_line_length=79))

    def test_extract_code_from_function(self):
        if False:
            for i in range(10):
                print('nop')

        def fix_e123():
            if False:
                i = 10
                return i + 15
            pass
        self.assertEqual('e123', autopep8.extract_code_from_function(fix_e123))

        def foo():
            if False:
                i = 10
                return i + 15
            pass
        self.assertEqual(None, autopep8.extract_code_from_function(foo))

        def fix_foo():
            if False:
                print('Hello World!')
            pass
        self.assertEqual(None, autopep8.extract_code_from_function(fix_foo))

        def e123():
            if False:
                print('Hello World!')
            pass
        self.assertEqual(None, autopep8.extract_code_from_function(e123))

        def fix_():
            if False:
                for i in range(10):
                    print('nop')
            pass
        self.assertEqual(None, autopep8.extract_code_from_function(fix_))

    def test_reindenter(self):
        if False:
            while True:
                i = 10
        reindenter = autopep8.Reindenter('if True:\n  pass\n')
        self.assertEqual('if True:\n    pass\n', reindenter.run())

    def test_reindenter_with_non_standard_indent_size(self):
        if False:
            while True:
                i = 10
        reindenter = autopep8.Reindenter('if True:\n  pass\n')
        self.assertEqual('if True:\n   pass\n', reindenter.run(3))

    def test_reindenter_with_good_input(self):
        if False:
            print('Hello World!')
        lines = 'if True:\n    pass\n'
        reindenter = autopep8.Reindenter(lines)
        self.assertEqual(lines, reindenter.run())

    def test_reindenter_should_leave_stray_comment_alone(self):
        if False:
            i = 10
            return i + 15
        lines = '  #\nif True:\n  pass\n'
        reindenter = autopep8.Reindenter(lines)
        self.assertEqual('  #\nif True:\n    pass\n', reindenter.run())

    @unittest.skipIf('AUTOPEP8_COVERAGE' in os.environ, 'exists form-feed')
    def test_reindenter_not_affect_with_formfeed(self):
        if False:
            print('Hello World!')
        lines = "print('hello')\n\x0c\nprint('python')\n"
        reindenter = autopep8.Reindenter(lines)
        self.assertEqual(lines, reindenter.run())

    def test_fix_e225_avoid_failure(self):
        if False:
            i = 10
            return i + 15
        fix_pep8 = autopep8.FixPEP8(filename='', options=autopep8.parse_args(['']), contents='    1\n')
        self.assertEqual([], fix_pep8.fix_e225({'line': 1, 'column': 5}))

    def test_fix_e271_ignore_redundant(self):
        if False:
            i = 10
            return i + 15
        fix_pep8 = autopep8.FixPEP8(filename='', options=autopep8.parse_args(['']), contents='x = 1\n')
        self.assertEqual([], fix_pep8.fix_e271({'line': 1, 'column': 2}))

    def test_fix_e401_avoid_non_import(self):
        if False:
            return 10
        fix_pep8 = autopep8.FixPEP8(filename='', options=autopep8.parse_args(['']), contents='    1\n')
        self.assertEqual([], fix_pep8.fix_e401({'line': 1, 'column': 5}))

    def test_fix_e711_avoid_failure(self):
        if False:
            return 10
        fix_pep8 = autopep8.FixPEP8(filename='', options=autopep8.parse_args(['']), contents='None == x\n')
        self.assertEqual(None, fix_pep8.fix_e711({'line': 1, 'column': 6}))
        self.assertEqual([], fix_pep8.fix_e711({'line': 1, 'column': 700}))
        fix_pep8 = autopep8.FixPEP8(filename='', options=autopep8.parse_args(['']), contents='x <> None\n')
        self.assertEqual([], fix_pep8.fix_e711({'line': 1, 'column': 3}))

    def test_fix_e712_avoid_failure(self):
        if False:
            return 10
        fix_pep8 = autopep8.FixPEP8(filename='', options=autopep8.parse_args(['']), contents='True == x\n')
        self.assertEqual([], fix_pep8.fix_e712({'line': 1, 'column': 5}))
        self.assertEqual([], fix_pep8.fix_e712({'line': 1, 'column': 700}))
        fix_pep8 = autopep8.FixPEP8(filename='', options=autopep8.parse_args(['']), contents='x != True\n')
        self.assertEqual([], fix_pep8.fix_e712({'line': 1, 'column': 3}))
        fix_pep8 = autopep8.FixPEP8(filename='', options=autopep8.parse_args(['']), contents='x == False\n')
        self.assertEqual([], fix_pep8.fix_e712({'line': 1, 'column': 3}))

    def test_get_diff_text(self):
        if False:
            while True:
                i = 10
        self.assertEqual('-foo\n+bar\n', '\n'.join(autopep8.get_diff_text(['foo\n'], ['bar\n'], '').split('\n')[3:]))

    def test_get_diff_text_without_newline(self):
        if False:
            return 10
        self.assertEqual('-foo\n\\ No newline at end of file\n+foo\n', '\n'.join(autopep8.get_diff_text(['foo'], ['foo\n'], '').split('\n')[3:]))

    def test_count_unbalanced_brackets(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(0, autopep8.count_unbalanced_brackets('()'))
        self.assertEqual(1, autopep8.count_unbalanced_brackets('('))
        self.assertEqual(2, autopep8.count_unbalanced_brackets('(['))
        self.assertEqual(1, autopep8.count_unbalanced_brackets('[])'))
        self.assertEqual(1, autopep8.count_unbalanced_brackets("'','.join(['%s=%s' % (col, col)')"))

    def test_commented_out_code_lines(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual([1, 4], autopep8.commented_out_code_lines('#x = 1\n#Hello\n#Hello world.\n#html_use_index = True\n'))

    def test_standard_deviation(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertAlmostEqual(2, autopep8.standard_deviation([2, 4, 4, 4, 5, 5, 7, 9]))
        self.assertAlmostEqual(0, autopep8.standard_deviation([]))
        self.assertAlmostEqual(0, autopep8.standard_deviation([1]))
        self.assertAlmostEqual(0.5, autopep8.standard_deviation([1, 2]))

    def test_priority_key_with_non_existent_key(self):
        if False:
            print('Hello World!')
        pep8_result = {'id': 'foobar'}
        self.assertGreater(autopep8._priority_key(pep8_result), 1)

    def test_decode_filename(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('foo.py', autopep8.decode_filename(b'foo.py'))

    def test_almost_equal(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(autopep8.code_almost_equal('[1, 2, 3\n    4, 5]\n', '[1, 2, 3\n4, 5]\n'))
        self.assertTrue(autopep8.code_almost_equal('[1,2,3\n    4,5]\n', '[1, 2, 3\n4,5]\n'))
        self.assertFalse(autopep8.code_almost_equal('[1, 2, 3\n    4, 5]\n', '[1, 2, 3, 4,\n    5]\n'))

    def test_token_offsets(self):
        if False:
            for i in range(10):
                print('nop')
        text = '1\n'
        string_io = io.StringIO(text)
        self.assertEqual([(tokenize.NUMBER, '1', 0, 1), (tokenize.NEWLINE, '\n', 1, 2), (tokenize.ENDMARKER, '', 2, 2)], list(autopep8.token_offsets(tokenize.generate_tokens(string_io.readline))))

    def test_token_offsets_with_multiline(self):
        if False:
            while True:
                i = 10
        text = "x = '''\n1\n2\n'''\n"
        string_io = io.StringIO(text)
        self.assertEqual([(tokenize.NAME, 'x', 0, 1), (tokenize.OP, '=', 2, 3), (tokenize.STRING, "'''\n1\n2\n'''", 4, 15), (tokenize.NEWLINE, '\n', 15, 16), (tokenize.ENDMARKER, '', 16, 16)], list(autopep8.token_offsets(tokenize.generate_tokens(string_io.readline))))

    def test_token_offsets_with_escaped_newline(self):
        if False:
            i = 10
            return i + 15
        text = 'True or \\\n    False\n'
        string_io = io.StringIO(text)
        self.assertEqual([(tokenize.NAME, 'True', 0, 4), (tokenize.NAME, 'or', 5, 7), (tokenize.NAME, 'False', 11, 16), (tokenize.NEWLINE, '\n', 16, 17), (tokenize.ENDMARKER, '', 17, 17)], list(autopep8.token_offsets(tokenize.generate_tokens(string_io.readline))))

    def test_shorten_line_candidates_are_valid(self):
        if False:
            print('Hello World!')
        for text in ['[xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx, y] = [1, 2]\n', 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx, y = [1, 2]\n', 'lambda xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: line_shortening_rank(x,\n                                           indent_word,\n                                           max_line_length)\n']:
            indent = autopep8._get_indentation(text)
            source = text[len(indent):]
            assert source.lstrip() == source
            tokens = list(autopep8.generate_tokens(source))
            for candidate in autopep8.shorten_line(tokens, source, indent, indent_word='    ', max_line_length=79, aggressive=10, experimental=True, previous_line=''):
                self.assertEqual(re.sub('\\s', '', text), re.sub('\\s', '', candidate))

    def test_get_fixed_long_line_empty(self):
        if False:
            while True:
                i = 10
        line = ''
        self.assertEqual(line, autopep8.get_fixed_long_line(line, line, line))

class SystemTests(unittest.TestCase):
    maxDiff = None

    def test_e101(self):
        if False:
            i = 10
            return i + 15
        line = 'while True:\n    if True:\n    \t1\n'
        fixed = 'while True:\n    if True:\n        1\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e101_with_indent_size_1(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'while True:\n    if True:\n    \t1\n'
        fixed = 'while True:\n if True:\n  1\n'
        with autopep8_context(line, options=['--indent-size=1']) as result:
            self.assertEqual(fixed, result)

    def test_e101_with_indent_size_2(self):
        if False:
            i = 10
            return i + 15
        line = 'while True:\n    if True:\n    \t1\n'
        fixed = 'while True:\n  if True:\n    1\n'
        with autopep8_context(line, options=['--indent-size=2']) as result:
            self.assertEqual(fixed, result)

    def test_e101_with_indent_size_3(self):
        if False:
            i = 10
            return i + 15
        line = 'while True:\n    if True:\n    \t1\n'
        fixed = 'while True:\n   if True:\n      1\n'
        with autopep8_context(line, options=['--indent-size=3']) as result:
            self.assertEqual(fixed, result)

    def test_e101_should_not_expand_non_indentation_tabs(self):
        if False:
            i = 10
            return i + 15
        line = "while True:\n    if True:\n    \t1 == '\t'\n"
        fixed = "while True:\n    if True:\n        1 == '\t'\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e101_should_ignore_multiline_strings(self):
        if False:
            return 10
        line = "x = '''\nwhile True:\n    if True:\n    \t1\n'''\n"
        fixed = "x = '''\nwhile True:\n    if True:\n    \t1\n'''\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e101_should_fix_docstrings(self):
        if False:
            return 10
        line = "class Bar(object):\n\n    def foo():\n        '''\n\tdocstring\n        '''\n"
        fixed = "class Bar(object):\n\n    def foo():\n        '''\n        docstring\n        '''\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e101_when_pep8_mistakes_first_tab_in_string(self):
        if False:
            print('Hello World!')
        line = "x = '''\n\tHello.\n'''\nif True:\n    123\n"
        fixed = "x = '''\n\tHello.\n'''\nif True:\n    123\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e101_should_ignore_multiline_strings_complex(self):
        if False:
            return 10
        line = "print(3 !=  4, '''\nwhile True:\n    if True:\n    \t1\n\t''', 4 !=  5)\n"
        fixed = "print(3 != 4, '''\nwhile True:\n    if True:\n    \t1\n\t''', 4 != 5)\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e101_with_comments(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'while True:  # My inline comment\n             # with a hanging\n             # comment.\n    # Hello\n    if True:\n    \t# My comment\n    \t1\n    \t# My other comment\n'
        fixed = 'while True:  # My inline comment\n    # with a hanging\n    # comment.\n    # Hello\n    if True:\n        # My comment\n        1\n        # My other comment\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e101_skip_if_bad_indentation(self):
        if False:
            print('Hello World!')
        line = 'try:\n\t    pass\n    except:\n        pass\n'
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_e101_skip_innocuous(self):
        if False:
            for i in range(10):
                print('nop')
        p = Popen(list(AUTOPEP8_CMD_TUPLE) + ['-vvv', '--select=E101', '--diff', '--global-config={}'.format(os.devnull), os.path.join(ROOT_DIR, 'test', 'e101_example.py')], stdout=PIPE, stderr=PIPE)
        output = [x.decode('utf-8') for x in p.communicate()][0]
        setup_cfg_file = os.path.join(ROOT_DIR, 'setup.cfg')
        tox_ini_file = os.path.join(ROOT_DIR, 'tox.ini')
        expected = 'read config path: /dev/null\nread config path: {}\nread config path: {}\n'.format(setup_cfg_file, tox_ini_file)
        self.assertEqual(expected, output)

    def test_e111_short(self):
        if False:
            return 10
        line = 'class Dummy:\n\n  def __init__(self):\n    pass\n'
        fixed = 'class Dummy:\n\n    def __init__(self):\n        pass\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e111_long(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'class Dummy:\n\n     def __init__(self):\n          pass\n'
        fixed = 'class Dummy:\n\n    def __init__(self):\n        pass\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e111_longer(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'while True:\n      if True:\n            1\n      elif True:\n            2\n'
        fixed = 'while True:\n    if True:\n        1\n    elif True:\n        2\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e111_multiple_levels(self):
        if False:
            return 10
        line = "while True:\n    if True:\n       1\n\n# My comment\nprint('abc')\n\n"
        fixed = "while True:\n    if True:\n        1\n\n# My comment\nprint('abc')\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e111_with_dedent(self):
        if False:
            return 10
        line = 'def foo():\n    if True:\n         2\n    1\n'
        fixed = 'def foo():\n    if True:\n        2\n    1\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e111_with_other_errors(self):
        if False:
            return 10
        line = "def foo():\n    if True:\n         (2 , 1)\n    1\n    if True:\n           print('hello')\t\n    2\n"
        fixed = "def foo():\n    if True:\n        (2, 1)\n    1\n    if True:\n        print('hello')\n    2\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e111_should_not_modify_string_contents(self):
        if False:
            print('Hello World!')
        line = "if True:\n x = '''\n 1\n '''\n"
        fixed = "if True:\n    x = '''\n 1\n '''\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e112_should_leave_bad_syntax_alone(self):
        if False:
            i = 10
            return i + 15
        line = 'if True:\npass\n'
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_e113(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'a = 1\n    b = 2\n'
        fixed = 'a = 1\nb = 2\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e113_bad_syntax(self):
        if False:
            for i in range(10):
                print('nop')
        line = '    pass\n'
        fixed = 'pass\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e114(self):
        if False:
            print('Hello World!')
        line = '   # a = 1\n'
        fixed = '# a = 1\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e115(self):
        if False:
            i = 10
            return i + 15
        line = 'if True:\n# A comment.\n    pass\n'
        fixed = 'if True:\n    # A comment.\n    pass\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e116(self):
        if False:
            return 10
        line = 'a = 1\n    # b = 2\n'
        fixed = 'a = 1\n# b = 2\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e117(self):
        if False:
            while True:
                i = 10
        line = "for a in [1, 2, 3]:\n    print('hello world')\n    for b in [1, 2, 3]:\n            print(a, b)\n"
        fixed = "for a in [1, 2, 3]:\n    print('hello world')\n    for b in [1, 2, 3]:\n        print(a, b)\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e12_reindent(self):
        if False:
            i = 10
            return i + 15
        line = 'def foo_bar(baz, frop,\n    fizz, bang):  # E128\n    pass\n\n\nif True:\n    x = {\n         }  # E123\n#: E121\nprint "E121", (\n  "dent")\n#: E122\nprint "E122", (\n"dent")\n#: E124\nprint "E124", ("visual",\n               "indent_two"\n              )\n#: E125\nif (row < 0 or self.moduleCount <= row or\n    col < 0 or self.moduleCount <= col):\n    raise Exception("%s,%s - %s" % (row, col, self.moduleCount))\n#: E126\nprint "E126", (\n            "dent")\n#: E127\nprint "E127", ("over-",\n                  "over-indent")\n#: E128\nprint "E128", ("under-",\n              "under-indent")\n'
        fixed = 'def foo_bar(baz, frop,\n            fizz, bang):  # E128\n    pass\n\n\nif True:\n    x = {\n    }  # E123\n#: E121\nprint "E121", (\n    "dent")\n#: E122\nprint "E122", (\n    "dent")\n#: E124\nprint "E124", ("visual",\n               "indent_two"\n               )\n#: E125\nif (row < 0 or self.moduleCount <= row or\n        col < 0 or self.moduleCount <= col):\n    raise Exception("%s,%s - %s" % (row, col, self.moduleCount))\n#: E126\nprint "E126", (\n    "dent")\n#: E127\nprint "E127", ("over-",\n               "over-indent")\n#: E128\nprint "E128", ("under-",\n               "under-indent")\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e12_reindent_with_multiple_fixes(self):
        if False:
            while True:
                i = 10
        line = "\nsql = 'update %s set %s %s' % (from_table,\n                               ','.join(['%s=%s' % (col, col) for col in cols]),\n        where_clause)\n"
        fixed = "\nsql = 'update %s set %s %s' % (from_table,\n                               ','.join(['%s=%s' % (col, col)\n                                        for col in cols]),\n                               where_clause)\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e12_tricky(self):
        if False:
            i = 10
            return i + 15
        line = '#: E126\nif (\n    x == (\n        3\n    ) or\n    x == (\n    3\n    ) or\n        y == 4):\n    pass\n'
        fixed = '#: E126\nif (\n    x == (\n        3\n    ) or\n    x == (\n        3\n    ) or\n        y == 4):\n    pass\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e12_large(self):
        if False:
            return 10
        line = "class BogusController(controller.CementBaseController):\n\n    class Meta:\n        pass\n\nclass BogusController2(controller.CementBaseController):\n\n    class Meta:\n        pass\n\nclass BogusController3(controller.CementBaseController):\n\n    class Meta:\n        pass\n\nclass BogusController4(controller.CementBaseController):\n\n    class Meta:\n        pass\n\nclass TestBaseController(controller.CementBaseController):\n\n    class Meta:\n        pass\n\nclass TestBaseController2(controller.CementBaseController):\n\n    class Meta:\n        pass\n\nclass TestStackedController(controller.CementBaseController):\n\n    class Meta:\n        arguments = [\n            ]\n\nclass TestDuplicateController(controller.CementBaseController):\n\n    class Meta:\n\n        config_defaults = dict(\n            foo='bar',\n            )\n\n        arguments = [\n            (['-f2', '--foo2'], dict(action='store'))\n            ]\n\n    def my_command(self):\n        pass\n"
        fixed = "class BogusController(controller.CementBaseController):\n\n    class Meta:\n        pass\n\n\nclass BogusController2(controller.CementBaseController):\n\n    class Meta:\n        pass\n\n\nclass BogusController3(controller.CementBaseController):\n\n    class Meta:\n        pass\n\n\nclass BogusController4(controller.CementBaseController):\n\n    class Meta:\n        pass\n\n\nclass TestBaseController(controller.CementBaseController):\n\n    class Meta:\n        pass\n\n\nclass TestBaseController2(controller.CementBaseController):\n\n    class Meta:\n        pass\n\n\nclass TestStackedController(controller.CementBaseController):\n\n    class Meta:\n        arguments = [\n        ]\n\n\nclass TestDuplicateController(controller.CementBaseController):\n\n    class Meta:\n\n        config_defaults = dict(\n            foo='bar',\n        )\n\n        arguments = [\n            (['-f2', '--foo2'], dict(action='store'))\n        ]\n\n    def my_command(self):\n        pass\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e12_with_bad_indentation(self):
        if False:
            print('Hello World!')
        line = '\n\n\ndef bar():\n    foo(1,\n      2)\n\n\ndef baz():\n     pass\n\n    pass\n'
        fixed = '\n\n\ndef bar():\n    foo(1,\n        2)\n\n\ndef baz():\n     pass\n\n    pass\n'
        with autopep8_context(line, options=['--select=E12']) as result:
            self.assertEqual(fixed, result)

    def test_e121_with_multiline_string(self):
        if False:
            return 10
        line = "testing = \\\n'''inputs: d c b a\n'''\n"
        fixed = "testing = \\\n    '''inputs: d c b a\n'''\n"
        with autopep8_context(line, options=['--select=E12']) as result:
            self.assertEqual(fixed, result)

    def test_e122_with_fallback(self):
        if False:
            i = 10
            return i + 15
        line = "foooo('',\n      scripts=[''],\n      classifiers=[\n      'Development Status :: 4 - Beta',\n      'Environment :: Console',\n      'Intended Audience :: Developers',\n      ])\n"
        fixed = "foooo('',\n      scripts=[''],\n      classifiers=[\n          'Development Status :: 4 - Beta',\n          'Environment :: Console',\n          'Intended Audience :: Developers',\n      ])\n"
        with autopep8_context(line, options=[]) as result:
            self.assertEqual(fixed, result)

    def test_e123(self):
        if False:
            print('Hello World!')
        line = 'if True:\n    foo = (\n        )\n'
        fixed = 'if True:\n    foo = (\n    )\n'
        with autopep8_context(line, options=['--select=E12']) as result:
            self.assertEqual(fixed, result)

    def test_e123_with_escaped_newline(self):
        if False:
            return 10
        line = '\nx = \\\n    (\n)\n'
        fixed = '\nx = \\\n    (\n    )\n'
        with autopep8_context(line, options=['--select=E12']) as result:
            self.assertEqual(fixed, result)

    def test_e128_with_aaa_option(self):
        if False:
            i = 10
            return i + 15
        line = 'def extractBlocks(self):\n    addLine = (self.matchMultiple(linesIncludePatterns, line)\n       and not self.matchMultiple(linesExcludePatterns, line)) or emptyLine\n'
        fixed = 'def extractBlocks(self):\n    addLine = (\n        self.matchMultiple(\n            linesIncludePatterns,\n            line) and not self.matchMultiple(\n            linesExcludePatterns,\n            line)) or emptyLine\n'
        with autopep8_context(line, options=['-aaa']) as result:
            self.assertEqual(fixed, result)

    def test_e129(self):
        if False:
            for i in range(10):
                print('nop')
        line = "if (a and\n    b in [\n        'foo',\n    ] or\n    c):\n    pass\n"
        fixed = "if (a and\n    b in [\n        'foo',\n    ] or\n        c):\n    pass\n"
        with autopep8_context(line, options=['--select=E129']) as result:
            self.assertEqual(fixed, result)

    def test_e125_with_multiline_string(self):
        if False:
            return 10
        line = "for foo in '''\n    abc\n    123\n    '''.strip().split():\n    print(foo)\n"
        with autopep8_context(line, options=['--select=E12']) as result:
            self.assertEqual(line, result)

    def test_e125_with_multiline_string_okay(self):
        if False:
            while True:
                i = 10
        line = "def bar(\n    a='''a'''):\n    print(foo)\n"
        fixed = "def bar(\n        a='''a'''):\n    print(foo)\n"
        with autopep8_context(line, options=['--select=E12']) as result:
            self.assertEqual(fixed, result)

    def test_e126(self):
        if False:
            print('Hello World!')
        line = 'if True:\n    posted = models.DateField(\n            default=datetime.date.today,\n            help_text="help"\n    )\n'
        fixed = 'if True:\n    posted = models.DateField(\n        default=datetime.date.today,\n        help_text="help"\n    )\n'
        with autopep8_context(line, options=['--select=E12']) as result:
            self.assertEqual(fixed, result)

    def test_e126_should_not_interfere_with_other_fixes(self):
        if False:
            return 10
        line = "self.assertEqual('bottom 1',\n    SimpleNamedNode.objects.filter(id__gt=1).exclude(\n        name='bottom 3').filter(\n            name__in=['bottom 3', 'bottom 1'])[0].name)\n"
        fixed = "self.assertEqual('bottom 1',\n                 SimpleNamedNode.objects.filter(id__gt=1).exclude(\n                     name='bottom 3').filter(\n                     name__in=['bottom 3', 'bottom 1'])[0].name)\n"
        with autopep8_context(line, options=['--select=E12']) as result:
            self.assertEqual(fixed, result)

    def test_e127(self):
        if False:
            i = 10
            return i + 15
        line = 'if True:\n    if True:\n        chksum = (sum([int(value[i]) for i in xrange(0, 9, 2)]) * 7 -\n                          sum([int(value[i]) for i in xrange(1, 9, 2)])) % 10\n'
        fixed = 'if True:\n    if True:\n        chksum = (sum([int(value[i]) for i in xrange(0, 9, 2)]) * 7 -\n                  sum([int(value[i]) for i in xrange(1, 9, 2)])) % 10\n'
        with autopep8_context(line, options=['--select=E12']) as result:
            self.assertEqual(fixed, result)

    def test_e127_align_visual_indent(self):
        if False:
            print('Hello World!')
        line = 'def draw(self):\n    color = [([0.2, 0.1, 0.3], [0.2, 0.1, 0.3], [0.2, 0.1, 0.3]),\n               ([0.9, 0.3, 0.5], [0.5, 1.0, 0.5], [0.3, 0.3, 0.9])  ][self._p._colored ]\n    self.draw_background(color)\n'
        fixed = 'def draw(self):\n    color = [([0.2, 0.1, 0.3], [0.2, 0.1, 0.3], [0.2, 0.1, 0.3]),\n             ([0.9, 0.3, 0.5], [0.5, 1.0, 0.5], [0.3, 0.3, 0.9])][self._p._colored]\n    self.draw_background(color)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e127_align_visual_indent_okay(self):
        if False:
            return 10
        'This is for code coverage.'
        line = 'want = (have + _leading_space_count(\n        after[jline - 1]) -\n        _leading_space_count(lines[jline]))\n'
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_e127_with_backslash(self):
        if False:
            for i in range(10):
                print('nop')
        line = '\nif True:\n    if True:\n        self.date = meta.session.query(schedule.Appointment)\\\n            .filter(schedule.Appointment.id ==\n                                      appointment_id).one().agenda.endtime\n'
        fixed = '\nif True:\n    if True:\n        self.date = meta.session.query(schedule.Appointment)\\\n            .filter(schedule.Appointment.id ==\n                    appointment_id).one().agenda.endtime\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e127_with_bracket_then_parenthesis(self):
        if False:
            for i in range(10):
                print('nop')
        line = '\nif True:\n    foo = [food(1)\n               for bar in bars]\n'
        fixed = '\nif True:\n    foo = [food(1)\n           for bar in bars]\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e12_with_backslash(self):
        if False:
            return 10
        line = "\nif True:\n    assert reeval == parsed, \\\n            'Repr gives different object:\\n  %r !=\\n  %r' % (parsed, reeval)\n"
        fixed = "\nif True:\n    assert reeval == parsed, \\\n        'Repr gives different object:\\n  %r !=\\n  %r' % (parsed, reeval)\n"
        with autopep8_context(line, options=['--select=E12']) as result:
            self.assertEqual(fixed, result)

    def test_e133(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'if True:\n    e = [\n        1, 2\n    ]\n'
        fixed = 'if True:\n    e = [\n        1, 2\n        ]\n'
        with autopep8_context(line, options=['--hang-closing']) as result:
            self.assertEqual(fixed, result)

    def test_e133_no_indentation_line(self):
        if False:
            while True:
                i = 10
        line = 'e = [\n    1, 2\n]\n'
        fixed = 'e = [\n    1, 2\n    ]\n'
        with autopep8_context(line, options=['--hang-closing']) as result:
            self.assertEqual(fixed, result)

    def test_e133_not_effected(self):
        if False:
            i = 10
            return i + 15
        line = 'if True:\n    e = [\n        1, 2\n        ]\n'
        with autopep8_context(line, options=['--hang-closing']) as result:
            self.assertEqual(line, result)

    def test_w191(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'while True:\n\tif True:\n\t\t1\n'
        fixed = 'while True:\n    if True:\n        1\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_w191_ignore(self):
        if False:
            return 10
        line = 'while True:\n\tif True:\n\t\t1\n'
        with autopep8_context(line, options=['--aggressive', '--ignore=W191']) as result:
            self.assertEqual(line, result)

    def test_e131_with_select_option(self):
        if False:
            i = 10
            return i + 15
        line = 'd = f(\n    a="hello"\n        "world",\n    b=1)\n'
        fixed = 'd = f(\n    a="hello"\n    "world",\n    b=1)\n'
        with autopep8_context(line, options=['--select=E131']) as result:
            self.assertEqual(fixed, result)

    def test_e131_invalid_indent_with_select_option(self):
        if False:
            print('Hello World!')
        line = 'd = (\n    "hello"\n  "world")\n'
        fixed = 'd = (\n    "hello"\n    "world")\n'
        with autopep8_context(line, options=['--select=E131']) as result:
            self.assertEqual(fixed, result)

    def test_e201(self):
        if False:
            print('Hello World!')
        line = '(   1)\n'
        fixed = '(1)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e202(self):
        if False:
            while True:
                i = 10
        line = '(1   )\n[2  ]\n{3  }\n'
        fixed = '(1)\n[2]\n{3}\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e202_multiline(self):
        if False:
            return 10
        line = "\n('''\na\nb\nc\n''' )\n"
        fixed = "\n('''\na\nb\nc\n''')\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e202_skip_multiline_with_escaped_newline(self):
        if False:
            print('Hello World!')
        line = "\n\n('c\\\n' )\n"
        fixed = "\n\n('c\\\n')\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e203_colon(self):
        if False:
            for i in range(10):
                print('nop')
        line = '{4 : 3}\n'
        fixed = '{4: 3}\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e203_comma(self):
        if False:
            while True:
                i = 10
        line = '[1 , 2  , 3]\n'
        fixed = '[1, 2, 3]\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e203_semicolon(self):
        if False:
            while True:
                i = 10
        line = "print(a, end=' ') ; nl = 0\n"
        fixed = "print(a, end=' '); nl = 0\n"
        with autopep8_context(line, options=['--select=E203']) as result:
            self.assertEqual(fixed, result)

    def test_e203_with_newline(self):
        if False:
            print('Hello World!')
        line = "print(a\n, end=' ')\n"
        fixed = "print(a, end=' ')\n"
        with autopep8_context(line, options=['--select=E203']) as result:
            self.assertEqual(fixed, result)

    def test_e211(self):
        if False:
            print('Hello World!')
        line = 'd = [1, 2, 3]\nprint(d  [0])\n'
        fixed = 'd = [1, 2, 3]\nprint(d[0])\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e221(self):
        if False:
            print('Hello World!')
        line = 'a = 1  + 1\n'
        fixed = 'a = 1 + 1\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e221_do_not_skip_multiline(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'def javascript(self):\n    return u"""\n<script type="text/javascript" src="++resource++ptg.shufflegallery/jquery.promptu-menu.js"></script>\n<script type="text/javascript">\n$(function(){\n    $(\'ul.promptu-menu\').promptumenu({width: %(width)i, height: %(height)i, rows: %(rows)i, columns: %(columns)i, direction: \'%(direction)s\', intertia: %(inertia)i, pages: %(pages)i});\n\t$(\'ul.promptu-menu a\').click(function(e) {\n        e.preventDefault();\n    });\n    $(\'ul.promptu-menu a\').dblclick(function(e) {\n        window.location.replace($(this).attr("href"));\n    });\n});\n</script>\n    """  % {\n    }\n'
        fixed = 'def javascript(self):\n    return u"""\n<script type="text/javascript" src="++resource++ptg.shufflegallery/jquery.promptu-menu.js"></script>\n<script type="text/javascript">\n$(function(){\n    $(\'ul.promptu-menu\').promptumenu({width: %(width)i, height: %(height)i, rows: %(rows)i, columns: %(columns)i, direction: \'%(direction)s\', intertia: %(inertia)i, pages: %(pages)i});\n\t$(\'ul.promptu-menu a\').click(function(e) {\n        e.preventDefault();\n    });\n    $(\'ul.promptu-menu a\').dblclick(function(e) {\n        window.location.replace($(this).attr("href"));\n    });\n});\n</script>\n    """ % {\n    }\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e222(self):
        if False:
            while True:
                i = 10
        line = 'a = 1 +  1\n'
        fixed = 'a = 1 + 1\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e222_with_multiline(self):
        if False:
            while True:
                i = 10
        line = 'a =   """bar\nbaz"""\n'
        fixed = 'a = """bar\nbaz"""\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e223(self):
        if False:
            while True:
                i = 10
        line = 'a = 1\t+ 1\n'
        fixed = 'a = 1 + 1\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e223_double(self):
        if False:
            return 10
        line = 'a = 1\t\t+ 1\n'
        fixed = 'a = 1 + 1\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e223_with_tab_indentation(self):
        if False:
            i = 10
            return i + 15
        line = 'class Foo():\n\n\tdef __init__(self):\n\t\tx= 1\t+ 3\n'
        fixed = 'class Foo():\n\n\tdef __init__(self):\n\t\tx = 1 + 3\n'
        with autopep8_context(line, options=['--ignore=E1,W191']) as result:
            self.assertEqual(fixed, result)

    def test_e224(self):
        if False:
            while True:
                i = 10
        line = 'a = 11 +\t1\n'
        fixed = 'a = 11 + 1\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e224_double(self):
        if False:
            return 10
        line = 'a = 11 +\t\t1\n'
        fixed = 'a = 11 + 1\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e224_with_tab_indentation(self):
        if False:
            i = 10
            return i + 15
        line = 'class Foo():\n\n\tdef __init__(self):\n\t\tx= \t3\n'
        fixed = 'class Foo():\n\n\tdef __init__(self):\n\t\tx = 3\n'
        with autopep8_context(line, options=['--ignore=E1,W191']) as result:
            self.assertEqual(fixed, result)

    def test_e225(self):
        if False:
            for i in range(10):
                print('nop')
        line = '1+1\n2 +2\n3+ 3\n'
        fixed = '1 + 1\n2 + 2\n3 + 3\n'
        with autopep8_context(line, options=['--select=E,W']) as result:
            self.assertEqual(fixed, result)

    def test_e225_with_indentation_fix(self):
        if False:
            while True:
                i = 10
        line = "class Foo(object):\n\n  def bar(self):\n    return self.elephant!='test'\n"
        fixed = "class Foo(object):\n\n    def bar(self):\n        return self.elephant != 'test'\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e226(self):
        if False:
            for i in range(10):
                print('nop')
        line = '1*1\n2*2\n3*3\n'
        fixed = '1 * 1\n2 * 2\n3 * 3\n'
        with autopep8_context(line, options=['--select=E22']) as result:
            self.assertEqual(fixed, result)

    def test_e227(self):
        if False:
            return 10
        line = '1&1\n2&2\n3&3\n'
        fixed = '1 & 1\n2 & 2\n3 & 3\n'
        with autopep8_context(line, options=['--select=E22']) as result:
            self.assertEqual(fixed, result)

    def test_e228(self):
        if False:
            i = 10
            return i + 15
        line = '1%1\n2%2\n3%3\n'
        fixed = '1 % 1\n2 % 2\n3 % 3\n'
        with autopep8_context(line, options=['--select=E22']) as result:
            self.assertEqual(fixed, result)

    def test_e231(self):
        if False:
            for i in range(10):
                print('nop')
        line = '[1,2,3]\n'
        fixed = '[1, 2, 3]\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e231_with_many_commas(self):
        if False:
            while True:
                i = 10
        fixed = str(list(range(200))) + '\n'
        line = re.sub(', ', ',', fixed)
        with autopep8_context(line, options=['--select=E231']) as result:
            self.assertEqual(fixed, result)

    def test_e231_with_colon_after_comma(self):
        if False:
            for i in range(10):
                print('nop')
        'ws_comma fixer ignores this case.'
        line = 'a[b1,:]\n'
        fixed = 'a[b1, :]\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e231_should_only_do_ws_comma_once(self):
        if False:
            return 10
        "If we don't check appropriately, we end up doing ws_comma multiple\n        times and skipping all other fixes."
        line = 'print( 1 )\nfoo[0,:]\nbar[zap[0][0]:zig[0][0],:]\n'
        fixed = 'print(1)\nfoo[0, :]\nbar[zap[0][0]:zig[0][0], :]\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e241(self):
        if False:
            while True:
                i = 10
        line = 'l = (1,  2)\n'
        fixed = 'l = (1, 2)\n'
        with autopep8_context(line, options=['--select=E']) as result:
            self.assertEqual(fixed, result)

    def test_e241_should_be_enabled_by_aggressive(self):
        if False:
            print('Hello World!')
        line = 'l = (1,  2)\n'
        fixed = 'l = (1, 2)\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e241_double(self):
        if False:
            while True:
                i = 10
        line = 'l = (1,   2)\n'
        fixed = 'l = (1, 2)\n'
        with autopep8_context(line, options=['--select=E']) as result:
            self.assertEqual(fixed, result)

    def test_e242(self):
        if False:
            print('Hello World!')
        line = 'l = (1,\t2)\n'
        fixed = 'l = (1, 2)\n'
        with autopep8_context(line, options=['--select=E']) as result:
            self.assertEqual(fixed, result)

    def test_e242_double(self):
        if False:
            while True:
                i = 10
        line = 'l = (1,\t\t2)\n'
        fixed = 'l = (1, 2)\n'
        with autopep8_context(line, options=['--select=E']) as result:
            self.assertEqual(fixed, result)

    def test_e251(self):
        if False:
            while True:
                i = 10
        line = 'def a(arg = 1):\n    print(arg)\n'
        fixed = 'def a(arg=1):\n    print(arg)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e251_with_escaped_newline(self):
        if False:
            print('Hello World!')
        line = '1\n\n\ndef a(arg=\\\n1):\n    print(arg)\n'
        fixed = '1\n\n\ndef a(arg=1):\n    print(arg)\n'
        with autopep8_context(line, options=['--select=E251']) as result:
            self.assertEqual(fixed, result)

    def test_e251_with_calling(self):
        if False:
            return 10
        line = 'foo(bar= True)\n'
        fixed = 'foo(bar=True)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e251_with_argument_on_next_line(self):
        if False:
            return 10
        line = 'foo(bar\n=None)\n'
        fixed = 'foo(bar=None)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e252(self):
        if False:
            while True:
                i = 10
        line = 'def a(arg1: int=1, arg2: int =1, arg3: int= 1):\n    print(arg)\n'
        fixed = 'def a(arg1: int = 1, arg2: int = 1, arg3: int = 1):\n    print(arg)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e252_with_argument_on_next_line(self):
        if False:
            i = 10
            return i + 15
        line = 'def a(arg: int\n=1):\n    print(arg)\n'
        fixed = 'def a(arg: int\n= 1):\n    print(arg)\n'
        with autopep8_context(line, options=['--select=E252']) as result:
            self.assertEqual(fixed, result)

    def test_e252_with_escaped_newline(self):
        if False:
            i = 10
            return i + 15
        line = 'def a(arg: int\\\n=1):\n    print(arg)\n'
        fixed = 'def a(arg: int\\\n= 1):\n    print(arg)\n'
        with autopep8_context(line, options=['--select=E252']) as result:
            self.assertEqual(fixed, result)

    def test_e261(self):
        if False:
            i = 10
            return i + 15
        line = "print('a b ')# comment\n"
        fixed = "print('a b ')  # comment\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e261_with_inline_commented_out_code(self):
        if False:
            return 10
        line = '1 # 0 + 0\n'
        fixed = '1  # 0 + 0\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e261_with_dictionary(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'd = {# comment\n1: 2}\n'
        fixed = 'd = {  # comment\n    1: 2}\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e261_with_dictionary_no_space(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'd = {#comment\n1: 2}\n'
        fixed = 'd = {  # comment\n    1: 2}\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e261_with_comma(self):
        if False:
            i = 10
            return i + 15
        line = '{1: 2 # comment\n , }\n'
        fixed = '{1: 2  # comment\n , }\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e262_more_space(self):
        if False:
            return 10
        line = "print('a b ')  #  comment\n"
        fixed = "print('a b ')  # comment\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e262_none_space(self):
        if False:
            return 10
        line = "print('a b ')  #comment\n"
        fixed = "print('a b ')  # comment\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e262_hash_in_string(self):
        if False:
            while True:
                i = 10
        line = "print('a b  #string')  #comment\n"
        fixed = "print('a b  #string')  # comment\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e262_hash_in_string_and_multiple_hashes(self):
        if False:
            return 10
        line = "print('a b  #string')  #comment #comment\n"
        fixed = "print('a b  #string')  # comment #comment\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e262_more_complex(self):
        if False:
            print('Hello World!')
        line = "print('a b ')  #comment\n123\n"
        fixed = "print('a b ')  # comment\n123\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e265(self):
        if False:
            print('Hello World!')
        line = '#A comment\n123\n'
        fixed = '# A comment\n123\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e265_ignores_special_comments(self):
        if False:
            i = 10
            return i + 15
        line = '#!python\n456\n'
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_e265_ignores_special_comments_in_middle_of_file(self):
        if False:
            i = 10
            return i + 15
        line = '123\n\n#!python\n456\n'
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_e265_only(self):
        if False:
            print('Hello World!')
        line = '##A comment\n#B comment\n123\n'
        fixed = '## A comment\n# B comment\n123\n'
        with autopep8_context(line, options=['--select=E265']) as result:
            self.assertEqual(fixed, result)

    def test_e265_issue662(self):
        if False:
            print('Hello World!')
        line = '#print(" ")\n'
        fixed = '# print(" ")\n'
        with autopep8_context(line, options=['--select=E265']) as result:
            self.assertEqual(fixed, result)

    def test_ignore_e265(self):
        if False:
            print('Hello World!')
        line = '## A comment\n#B comment\n123\n'
        fixed = '# A comment\n#B comment\n123\n'
        with autopep8_context(line, options=['--ignore=E265']) as result:
            self.assertEqual(fixed, result)

    def test_e266(self):
        if False:
            while True:
                i = 10
        line = '## comment\n123\n'
        fixed = '# comment\n123\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e266_only(self):
        if False:
            print('Hello World!')
        line = '## A comment\n#B comment\n123\n'
        fixed = '# A comment\n#B comment\n123\n'
        with autopep8_context(line, options=['--select=E266']) as result:
            self.assertEqual(fixed, result)

    def test_e266_issue662(self):
        if False:
            i = 10
            return i + 15
        line = '## comment\n'
        fixed = '# comment\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_ignore_e266(self):
        if False:
            while True:
                i = 10
        line = '##A comment\n#B comment\n123\n'
        fixed = '## A comment\n# B comment\n123\n'
        with autopep8_context(line, options=['--ignore=E266']) as result:
            self.assertEqual(fixed, result)

    def test_e271(self):
        if False:
            return 10
        line = 'True and  False\n'
        fixed = 'True and False\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e271_with_multiline(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'if True and  False \\\n        True:\n    pass\n'
        fixed = 'if True and False \\\n        True:\n    pass\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e272(self):
        if False:
            return 10
        line = 'True  and False\n'
        fixed = 'True and False\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e273(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'True and\tFalse\n'
        fixed = 'True and False\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e274(self):
        if False:
            i = 10
            return i + 15
        line = 'True\tand False\n'
        fixed = 'True and False\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e306(self):
        if False:
            return 10
        line = "\ndef test_descriptors(self):\n\n        class descriptor(object):\n            def __init__(self, fn):\n                self.fn = fn\n            def __get__(self, obj, owner):\n                if obj is not None:\n                    return self.fn(obj, obj)\n                else:\n                    return self\n            def method(self):\n                return 'method'\n"
        fixed = "\ndef test_descriptors(self):\n\n    class descriptor(object):\n        def __init__(self, fn):\n            self.fn = fn\n\n        def __get__(self, obj, owner):\n            if obj is not None:\n                return self.fn(obj, obj)\n            else:\n                return self\n\n        def method(self):\n            return 'method'\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e301(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'class k:\n    s = 0\n    def f():\n        print(1)\n'
        fixed = 'class k:\n    s = 0\n\n    def f():\n        print(1)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e301_extended_with_docstring(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'class Foo(object):\n    """Test."""\n    def foo(self):\n\n\n\n        """Test."""\n        def bar():\n            pass\n'
        fixed = 'class Foo(object):\n    """Test."""\n\n    def foo(self):\n        """Test."""\n        def bar():\n            pass\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_not_e301_extended_with_comment(self):
        if False:
            while True:
                i = 10
        line = 'class Foo(object):\n\n    """Test."""\n\n    # A comment.\n    def foo(self):\n        pass\n'
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_e302(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'def f():\n    print(1)\n\ndef ff():\n    print(2)\n'
        fixed = 'def f():\n    print(1)\n\n\ndef ff():\n    print(2)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e302_bug(self):
        if False:
            print('Hello World!')
        'Avoid creating bad syntax.'
        line = 'def repeatable_expr():      return [bracketed_choice, simple_match, rule_ref],\\\n                                    Optional(repeat_operator)\n# def match():                return [simple_match , mixin_rule_match] TODO\ndef simple_match():         return [str_match, re_match]\n'
        self.assertTrue(autopep8.check_syntax(line))
        with autopep8_context(line) as result:
            self.assertTrue(autopep8.check_syntax(result))

    def test_e303(self):
        if False:
            for i in range(10):
                print('nop')
        line = '\n\n\n# alpha\n\n1\n'
        fixed = '\n\n# alpha\n\n1\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e303_extended(self):
        if False:
            print('Hello World!')
        line = 'def foo():\n\n    """Document."""\n'
        fixed = 'def foo():\n    """Document."""\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e303_with_e305(self):
        if False:
            while True:
                i = 10
        line = 'def foo():\n    pass\n\n\n\n# comment   (E303)\na = 1     # (E305)\n'
        fixed = 'def foo():\n    pass\n\n\n# comment   (E303)\na = 1     # (E305)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e304(self):
        if False:
            print('Hello World!')
        line = '@contextmanager\n\ndef f():\n    print(1)\n'
        fixed = '@contextmanager\ndef f():\n    print(1)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e304_with_comment(self):
        if False:
            while True:
                i = 10
        line = '@contextmanager\n# comment\n\ndef f():\n    print(1)\n'
        fixed = '@contextmanager\n# comment\ndef f():\n    print(1)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e305(self):
        if False:
            while True:
                i = 10
        line = 'def a():\n    pass\na()\n'
        fixed = 'def a():\n    pass\n\n\na()\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e401(self):
        if False:
            print('Hello World!')
        line = 'import os, sys\n'
        fixed = 'import os\nimport sys\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e401_with_indentation(self):
        if False:
            print('Hello World!')
        line = 'def a():\n    import os, sys\n'
        fixed = 'def a():\n    import os\n    import sys\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e401_should_ignore_commented_comma(self):
        if False:
            i = 10
            return i + 15
        line = 'import bdist_egg, egg  # , not a module, neither is this\n'
        fixed = 'import bdist_egg\nimport egg  # , not a module, neither is this\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e401_should_ignore_commented_comma_with_indentation(self):
        if False:
            while True:
                i = 10
        line = 'if True:\n    import bdist_egg, egg  # , not a module, neither is this\n'
        fixed = 'if True:\n    import bdist_egg\n    import egg  # , not a module, neither is this\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e401_should_ignore_false_positive(self):
        if False:
            i = 10
            return i + 15
        line = 'import bdist_egg; bdist_egg.write_safety_flag(cmd.egg_info, safe)\n'
        with autopep8_context(line, options=['--select=E401']) as result:
            self.assertEqual(line, result)

    def test_e401_with_escaped_newline_case(self):
        if False:
            return 10
        line = 'import foo, \\\n    bar\n'
        fixed = 'import foo\nimport \\\n    bar\n'
        with autopep8_context(line, options=['--select=E401']) as result:
            self.assertEqual(fixed, result)

    def test_e402(self):
        if False:
            print('Hello World!')
        line = 'a = 1\nimport os\n'
        fixed = 'import os\na = 1\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e402_duplicate_module(self):
        if False:
            return 10
        line = 'a = 1\nimport os\nprint(os)\nimport os\n'
        fixed = 'import os\na = 1\nprint(os)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e402_with_future_import(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'from __future__ import print_function\na = 1\nimport os\n'
        fixed = 'from __future__ import print_function\nimport os\na = 1\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e401_with_multiline_from_import(self):
        if False:
            i = 10
            return i + 15
        line = 'from os import (\n    chroot\n)\ndef f():\n    pass\nfrom a import b\nfrom b import c\nfrom c import d\n'
        fixed = 'from a import b\nfrom c import d\nfrom b import c\nfrom os import (\n    chroot\n)\n\n\ndef f():\n    pass\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e402_with_multiline_from_future_import(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'from __future__ import (\n    absolute_import,\n    print_function\n)\ndef f():\n    pass\nimport os\n'
        fixed = 'from __future__ import (\n    absolute_import,\n    print_function\n)\nimport os\n\n\ndef f():\n    pass\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e402_with_module_doc(self):
        if False:
            print('Hello World!')
        line1 = '"""\nmodule doc\n"""\na = 1\nimport os\n'
        fixed1 = '"""\nmodule doc\n"""\nimport os\na = 1\n'
        line2 = '# comment\nr"""\nmodule doc\n"""\na = 1\nimport os\n'
        fixed2 = '# comment\nr"""\nmodule doc\n"""\nimport os\na = 1\n'
        line3 = "u'''one line module doc'''\na = 1\nimport os\n"
        fixed3 = "u'''one line module doc'''\nimport os\na = 1\n"
        line4 = '\'\'\'\n"""\ndoc\'\'\'\na = 1\nimport os\n'
        fixed4 = '\'\'\'\n"""\ndoc\'\'\'\nimport os\na = 1\n'
        for (line, fixed) in [(line1, fixed1), (line2, fixed2), (line3, fixed3), (line4, fixed4)]:
            with autopep8_context(line) as result:
                self.assertEqual(fixed, result)

    def test_e402_import_some_modules(self):
        if False:
            return 10
        line = 'a = 1\nfrom csv import (\n    reader,\n    writer,\n)\nimport os\nprint(os, reader, writer)\nimport os\n'
        fixed = 'import os\nfrom csv import (\n    reader,\n    writer,\n)\na = 1\nprint(os, reader, writer)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e402_with_dunder(self):
        if False:
            while True:
                i = 10
        line = '__all__ = ["a", "b"]\ndef f():\n    pass\nimport os\n'
        fixed = 'import os\n__all__ = ["a", "b"]\n\n\ndef f():\n    pass\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e402_with_dunder_lines(self):
        if False:
            print('Hello World!')
        line = '__all__ = [\n    "a",\n    "b",\n]\ndef f():\n    pass\nimport os\n'
        fixed = 'import os\n__all__ = [\n    "a",\n    "b",\n]\n\n\ndef f():\n    pass\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e501_basic(self):
        if False:
            while True:
                i = 10
        line = '\nprint(111, 111, 111, 111, 222, 222, 222, 222, 222, 222, 222, 222, 222, 333, 333, 333, 333)\n'
        fixed = '\nprint(111, 111, 111, 111, 222, 222, 222, 222,\n      222, 222, 222, 222, 222, 333, 333, 333, 333)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_dictionary(self):
        if False:
            while True:
                i = 10
        line = "myDict = { 'kg': 1, 'tonnes': tonne, 't/y': tonne / year, 'Mt/y': 1e6 * tonne / year}\n"
        fixed = "myDict = {\n    'kg': 1,\n    'tonnes': tonne,\n    't/y': tonne / year,\n    'Mt/y': 1e6 * tonne / year}\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_in(self):
        if False:
            print('Hello World!')
        line = "if True:\n    if True:\n        if True:\n            if True:\n                if True:\n                    if True:\n                        if True:\n                            if True:\n                                if k_left in ('any', k_curr) and k_right in ('any', k_curr):\n                                    pass\n"
        fixed = "if True:\n    if True:\n        if True:\n            if True:\n                if True:\n                    if True:\n                        if True:\n                            if True:\n                                if k_left in ('any', k_curr) and k_right in ('any', k_curr):\n                                    pass\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_commas_and_colons(self):
        if False:
            for i in range(10):
                print('nop')
        line = "foobar = {'aaaaaaaaaaaa': 'bbbbbbbbbbbbbbbb', 'dddddd': 'eeeeeeeeeeeeeeee', 'ffffffffffff': 'gggggggg'}\n"
        fixed = "foobar = {'aaaaaaaaaaaa': 'bbbbbbbbbbbbbbbb',\n          'dddddd': 'eeeeeeeeeeeeeeee', 'ffffffffffff': 'gggggggg'}\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_inline_comments(self):
        if False:
            return 10
        line = "'                                                          '  # Long inline comments should be moved above.\nif True:\n    '                                                          '  # Long inline comments should be moved above.\n"
        fixed = "# Long inline comments should be moved above.\n'                                                          '\nif True:\n    # Long inline comments should be moved above.\n    '                                                          '\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_inline_comments_should_skip_multiline(self):
        if False:
            return 10
        line = "'''This should be left alone. -----------------------------------------------------\n\n'''  # foo\n\n'''This should be left alone. -----------------------------------------------------\n\n''' \\\n# foo\n\n'''This should be left alone. -----------------------------------------------------\n\n''' \\\n\\\n# foo\n"
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(line, result)

    def test_e501_with_inline_comments_should_skip_keywords(self):
        if False:
            return 10
        line = "'                                                          '  # noqa Long inline comments should be moved above.\nif True:\n    '                                                          '  # pylint: disable-msgs=E0001\n    '                                                          '  # pragma: no cover\n    '                                                          '  # pragma: no cover\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(line, result)

    def test_e501_with_inline_comments_should_skip_keywords_without_aggressive(self):
        if False:
            print('Hello World!')
        line = "'                                                          '  # noqa Long inline comments should be moved above.\nif True:\n    '                                                          '  # pylint: disable-msgs=E0001\n    '                                                          '  # pragma: no cover\n    '                                                          '  # pragma: no cover\n"
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_e501_with_inline_comments_should_skip_edge_cases(self):
        if False:
            i = 10
            return i + 15
        line = "if True:\n    x = \\\n        '                                                          '  # Long inline comments should be moved above.\n"
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_e501_basic_should_prefer_balanced_brackets(self):
        if False:
            return 10
        line = 'if True:\n    reconstructed = iradon(radon(image), filter="ramp", interpolation="nearest")\n'
        fixed = 'if True:\n    reconstructed = iradon(radon(image), filter="ramp",\n                           interpolation="nearest")\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_very_long_line(self):
        if False:
            return 10
        line = 'x = [3244234243234, 234234234324, 234234324, 23424234, 234234234, 234234, 234243, 234243, 234234234324, 234234324, 23424234, 234234234, 234234, 234243, 234243]\n'
        fixed = 'x = [\n    3244234243234,\n    234234234324,\n    234234324,\n    23424234,\n    234234234,\n    234234,\n    234243,\n    234243,\n    234234234324,\n    234234324,\n    23424234,\n    234234234,\n    234234,\n    234243,\n    234243]\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_lambda(self):
        if False:
            return 10
        line = 'self.mock_group.modify_state.side_effect = lambda *_: defer.fail(NoSuchScalingGroupError(1, 2))\n'
        fixed = 'self.mock_group.modify_state.side_effect = lambda *_: defer.fail(\n    NoSuchScalingGroupError(1, 2))\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e501_shorten_with_backslash(self):
        if False:
            return 10
        line = 'class Bar(object):\n\n    def bar(self, position):\n        if 0 <= position <= self._blocks[-1].position + len(self._blocks[-1].text):\n            pass\n'
        fixed = 'class Bar(object):\n\n    def bar(self, position):\n        if 0 <= position <= self._blocks[-1].position + \\\n                len(self._blocks[-1].text):\n            pass\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_shorten_at_commas_skip(self):
        if False:
            while True:
                i = 10
        line = "parser.add_argument('source_corpus', help='corpus name/path relative to an nltk_data directory')\nparser.add_argument('target_corpus', help='corpus name/path relative to an nltk_data directory')\n"
        fixed = "parser.add_argument(\n    'source_corpus',\n    help='corpus name/path relative to an nltk_data directory')\nparser.add_argument(\n    'target_corpus',\n    help='corpus name/path relative to an nltk_data directory')\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_shorter_length(self):
        if False:
            i = 10
            return i + 15
        line = "foooooooooooooooooo('abcdefghijklmnopqrstuvwxyz')\n"
        fixed = "foooooooooooooooooo(\n    'abcdefghijklmnopqrstuvwxyz')\n"
        with autopep8_context(line, options=['--max-line-length=40']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_indent(self):
        if False:
            return 10
        line = '\ndef d():\n    print(111, 111, 111, 111, 222, 222, 222, 222, 222, 222, 222, 222, 222, 333, 333, 333, 333)\n'
        fixed = '\ndef d():\n    print(111, 111, 111, 111, 222, 222, 222, 222,\n          222, 222, 222, 222, 222, 333, 333, 333, 333)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e501_alone_with_indentation(self):
        if False:
            i = 10
            return i + 15
        line = '\nif True:\n    print(111, 111, 111, 111, 222, 222, 222, 222, 222, 222, 222, 222, 222, 333, 333, 333, 333)\n'
        fixed = '\nif True:\n    print(111, 111, 111, 111, 222, 222, 222, 222,\n          222, 222, 222, 222, 222, 333, 333, 333, 333)\n'
        with autopep8_context(line, options=['--select=E501']) as result:
            self.assertEqual(fixed, result)

    def test_e501_alone_with_tuple(self):
        if False:
            print('Hello World!')
        line = "\nfooooooooooooooooooooooooooooooo000000000000000000000000 = [1,\n                                                            ('TransferTime', 'FLOAT')\n                                                           ]\n"
        fixed = "\nfooooooooooooooooooooooooooooooo000000000000000000000000 = [1,\n                                                            ('TransferTime',\n                                                             'FLOAT')\n                                                           ]\n"
        with autopep8_context(line, options=['--select=E501']) as result:
            self.assertEqual(fixed, result)

    def test_e501_should_not_try_to_break_at_every_paren_in_arithmetic(self):
        if False:
            while True:
                i = 10
        line = "term3 = w6 * c5 * (8.0 * psi4 * (11.0 - 24.0 * t2) - 28 * psi3 * (1 - 6.0 * t2) + psi2 * (1 - 32 * t2) - psi * (2.0 * t2) + t4) / 720.0\nthis_should_be_shortened = ('                                                                 ', '            ')\n"
        fixed = "term3 = w6 * c5 * (8.0 * psi4 * (11.0 - 24.0 * t2) - 28 * psi3 *\n                   (1 - 6.0 * t2) + psi2 * (1 - 32 * t2) - psi * (2.0 * t2) + t4) / 720.0\nthis_should_be_shortened = (\n    '                                                                 ',\n    '            ')\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_arithmetic_operator_with_indent(self):
        if False:
            return 10
        line = 'def d():\n    111 + 111 + 111 + 111 + 111 + 222 + 222 + 222 + 222 + 222 + 222 + 222 + 222 + 222 + 333 + 333 + 333 + 333\n'
        fixed = 'def d():\n    111 + 111 + 111 + 111 + 111 + 222 + 222 + 222 + 222 + \\\n        222 + 222 + 222 + 222 + 222 + 333 + 333 + 333 + 333\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e501_more_complicated(self):
        if False:
            print('Hello World!')
        line = "\nblahblah = os.environ.get('blahblah') or os.environ.get('blahblahblah') or os.environ.get('blahblahblahblah')\n"
        fixed = "\nblahblah = os.environ.get('blahblah') or os.environ.get(\n    'blahblahblah') or os.environ.get('blahblahblahblah')\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e501_skip_even_more_complicated(self):
        if False:
            for i in range(10):
                print('nop')
        line = '\nif True:\n    if True:\n        if True:\n            blah = blah.blah_blah_blah_bla_bl(blahb.blah, blah.blah,\n                                              blah=blah.label, blah_blah=blah_blah,\n                                              blah_blah2=blah_blah)\n'
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_e501_avoid_breaking_at_empty_parentheses_if_possible(self):
        if False:
            i = 10
            return i + 15
        line = 'someverylongindenttionwhatnot().foo().bar().baz("and here is a long string 123456789012345678901234567890")\n'
        fixed = 'someverylongindenttionwhatnot().foo().bar().baz(\n    "and here is a long string 123456789012345678901234567890")\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_logical_fix(self):
        if False:
            print('Hello World!')
        line = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx(aaaaaaaaaaaaaaaaaaaaaaa,\n                             bbbbbbbbbbbbbbbbbbbbbbbbbbbb, cccccccccccccccccccccccccccc, dddddddddddddddddddddddd)\n'
        fixed = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx(\n    aaaaaaaaaaaaaaaaaaaaaaa,\n    bbbbbbbbbbbbbbbbbbbbbbbbbbbb,\n    cccccccccccccccccccccccccccc,\n    dddddddddddddddddddddddd)\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_logical_fix_and_physical_fix(self):
        if False:
            i = 10
            return i + 15
        line = '# ------------------------------------ ------------------------------------------\nxxxxxxxxxxxxxxxxxxxxxxxxxxxx(aaaaaaaaaaaaaaaaaaaaaaa,\n                             bbbbbbbbbbbbbbbbbbbbbbbbbbbb, cccccccccccccccccccccccccccc, dddddddddddddddddddddddd)\n'
        fixed = '# ------------------------------------ -----------------------------------\nxxxxxxxxxxxxxxxxxxxxxxxxxxxx(\n    aaaaaaaaaaaaaaaaaaaaaaa,\n    bbbbbbbbbbbbbbbbbbbbbbbbbbbb,\n    cccccccccccccccccccccccccccc,\n    dddddddddddddddddddddddd)\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_logical_fix_and_adjacent_strings(self):
        if False:
            print('Hello World!')
        line = 'print(\'a-----------------------\' \'b-----------------------\' \'c-----------------------\'\n      \'d-----------------------\'\'e\'"f"r"g")\n'
        fixed = 'print(\n    \'a-----------------------\'\n    \'b-----------------------\'\n    \'c-----------------------\'\n    \'d-----------------------\'\n    \'e\'\n    "f"\n    r"g")\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_multiple_lines(self):
        if False:
            print('Hello World!')
        line = '\nfoo_bar_zap_bing_bang_boom(111, 111, 111, 111, 222, 222, 222, 222, 222, 222, 222, 222, 222, 333, 333,\n                           111, 111, 111, 111, 222, 222, 222, 222, 222, 222, 222, 222, 222, 333, 333)\n'
        fixed = '\nfoo_bar_zap_bing_bang_boom(\n    111,\n    111,\n    111,\n    111,\n    222,\n    222,\n    222,\n    222,\n    222,\n    222,\n    222,\n    222,\n    222,\n    333,\n    333,\n    111,\n    111,\n    111,\n    111,\n    222,\n    222,\n    222,\n    222,\n    222,\n    222,\n    222,\n    222,\n    222,\n    333,\n    333)\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_multiple_lines_and_quotes(self):
        if False:
            return 10
        line = "\nif True:\n    xxxxxxxxxxx = xxxxxxxxxxxxxxxxx(xxxxxxxxxxx, xxxxxxxxxxxxxxxx={'xxxxxxxxxxxx': 'xxxxx',\n                                                                   'xxxxxxxxxxx': xx,\n                                                                   'xxxxxxxx': False,\n                                                                   })\n"
        fixed = "\nif True:\n    xxxxxxxxxxx = xxxxxxxxxxxxxxxxx(\n        xxxxxxxxxxx,\n        xxxxxxxxxxxxxxxx={\n            'xxxxxxxxxxxx': 'xxxxx',\n            'xxxxxxxxxxx': xx,\n            'xxxxxxxx': False,\n        })\n"
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_do_not_break_on_keyword(self):
        if False:
            return 10
        line = "\nif True:\n    long_variable_name = tempfile.mkstemp(prefix='abcdefghijklmnopqrstuvwxyz0123456789')\n"
        fixed = "\nif True:\n    long_variable_name = tempfile.mkstemp(\n        prefix='abcdefghijklmnopqrstuvwxyz0123456789')\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e501_do_not_begin_line_with_comma(self):
        if False:
            for i in range(10):
                print('nop')
        line = '\ndef dummy():\n    if True:\n        if True:\n            if True:\n                object = ModifyAction( [MODIFY70.text, OBJECTBINDING71.text, COLON72.text], MODIFY70.getLine(), MODIFY70.getCharPositionInLine() )\n'
        fixed = '\ndef dummy():\n    if True:\n        if True:\n            if True:\n                object = ModifyAction([MODIFY70.text, OBJECTBINDING71.text, COLON72.text], MODIFY70.getLine(\n                ), MODIFY70.getCharPositionInLine())\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e501_should_not_break_on_dot(self):
        if False:
            print('Hello World!')
        line = 'if True:\n    if True:\n        raise xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(\'xxxxxxxxxxxxxxxxx "{d}" xxxxxxxxxxxxxx\'.format(d=\'xxxxxxxxxxxxxxx\'))\n'
        fixed = 'if True:\n    if True:\n        raise xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(\n            \'xxxxxxxxxxxxxxxxx "{d}" xxxxxxxxxxxxxx\'.format(d=\'xxxxxxxxxxxxxxx\'))\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_comment(self):
        if False:
            print('Hello World!')
        line = '123\nif True:\n    if True:\n        if True:\n            if True:\n                if True:\n                    if True:\n                        # This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n                        pass\n\n# http://foo.bar/abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-\n\n# The following is ugly commented-out code and should not be touched.\n#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx = 1\n'
        fixed = '123\nif True:\n    if True:\n        if True:\n            if True:\n                if True:\n                    if True:\n                        # This is a long comment that should be wrapped. I will\n                        # wrap it using textwrap to be within 72 characters.\n                        pass\n\n# http://foo.bar/abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-\n\n# The following is ugly commented-out code and should not be touched.\n# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx = 1\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_comment_should_not_modify_docstring(self):
        if False:
            i = 10
            return i + 15
        line = 'def foo():\n    """\n                        # This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n    """\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(line, result)

    def test_e501_should_only_modify_last_comment(self):
        if False:
            for i in range(10):
                print('nop')
        line = '123\nif True:\n    if True:\n        if True:\n            if True:\n                if True:\n                    if True:\n                        # This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n                        # 1. This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n                        # 2. This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n                        # 3. This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n'
        fixed = '123\nif True:\n    if True:\n        if True:\n            if True:\n                if True:\n                    if True:\n                        # This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n                        # 1. This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n                        # 2. This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n                        # 3. This is a long comment that should be wrapped. I\n                        # will wrap it using textwrap to be within 72\n                        # characters.\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_should_not_interfere_with_non_comment(self):
        if False:
            i = 10
            return i + 15
        line = '\n\n"""\n# not actually a comment %d. 12345678901234567890, 12345678901234567890, 12345678901234567890.\n""" % (0,)\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(line, result)

    def test_e501_should_cut_comment_pattern(self):
        if False:
            i = 10
            return i + 15
        line = '123\n# -- Useless lines ----------------------------------------------------------------------\n321\n'
        fixed = '123\n# -- Useless lines -------------------------------------------------------\n321\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_function_should_not_break_on_colon(self):
        if False:
            for i in range(10):
                print('nop')
        line = '\nclass Useless(object):\n\n    def _table_field_is_plain_widget(self, widget):\n        if widget.__class__ == Widget or\\\n                (widget.__class__ == WidgetMeta and Widget in widget.__bases__):\n            return True\n\n        return False\n'
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_e501_should_break_before_tuple_start(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'xxxxxxxxxxxxx(aaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbb, cccccccccc, (dddddddddddddddddddddd, eeeeeeeeeeee, fffffffffff, gggggggggg))\n'
        fixed = 'xxxxxxxxxxxxx(aaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbb, cccccccccc,\n              (dddddddddddddddddddddd, eeeeeeeeeeee, fffffffffff, gggggggggg))\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_aggressive(self):
        if False:
            while True:
                i = 10
        line = 'models = {\n    \'auth.group\': {\n        \'Meta\': {\'object_name\': \'Group\'},\n        \'permissions\': (\'django.db.models.fields.related.ManyToManyField\', [], {\'to\': "orm[\'auth.Permission\']", \'symmetrical\': \'False\', \'blank\': \'True\'})\n    },\n    \'auth.permission\': {\n        \'Meta\': {\'ordering\': "(\'content_type__app_label\', \'content_type__model\', \'codename\')", \'unique_together\': "((\'content_type\', \'codename\'),)", \'object_name\': \'Permission\'},\n        \'name\': (\'django.db.models.fields.CharField\', [], {\'max_length\': \'50\'})\n    },\n}\n'
        fixed = 'models = {\n    \'auth.group\': {\n        \'Meta\': {\n            \'object_name\': \'Group\'},\n        \'permissions\': (\n            \'django.db.models.fields.related.ManyToManyField\',\n            [],\n            {\n                \'to\': "orm[\'auth.Permission\']",\n                \'symmetrical\': \'False\',\n                \'blank\': \'True\'})},\n    \'auth.permission\': {\n        \'Meta\': {\n            \'ordering\': "(\'content_type__app_label\', \'content_type__model\', \'codename\')",\n            \'unique_together\': "((\'content_type\', \'codename\'),)",\n            \'object_name\': \'Permission\'},\n        \'name\': (\n            \'django.db.models.fields.CharField\',\n            [],\n            {\n                \'max_length\': \'50\'})},\n}\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_aggressive_and_multiple_logical_lines(self):
        if False:
            print('Hello World!')
        line = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx(aaaaaaaaaaaaaaaaaaaaaaa,\n                             bbbbbbbbbbbbbbbbbbbbbbbbbbbb, cccccccccccccccccccccccccccc, dddddddddddddddddddddddd)\nxxxxxxxxxxxxxxxxxxxxxxxxxxxx(aaaaaaaaaaaaaaaaaaaaaaa,\n                             bbbbbbbbbbbbbbbbbbbbbbbbbbbb, cccccccccccccccccccccccccccc, dddddddddddddddddddddddd)\n'
        fixed = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx(\n    aaaaaaaaaaaaaaaaaaaaaaa,\n    bbbbbbbbbbbbbbbbbbbbbbbbbbbb,\n    cccccccccccccccccccccccccccc,\n    dddddddddddddddddddddddd)\nxxxxxxxxxxxxxxxxxxxxxxxxxxxx(\n    aaaaaaaaaaaaaaaaaaaaaaa,\n    bbbbbbbbbbbbbbbbbbbbbbbbbbbb,\n    cccccccccccccccccccccccccccc,\n    dddddddddddddddddddddddd)\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_aggressive_and_multiple_logical_lines_with_math(self):
        if False:
            while True:
                i = 10
        line = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx([-1 + 5 / 10,\n                                                                            100,\n                                                                            -3 - 4])\n'
        fixed = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(\n    [-1 + 5 / 10, 100, -3 - 4])\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_aggressive_and_import(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'from . import (xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,\n               yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy)\n'
        fixed = 'from . import (\n    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,\n    yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy)\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_aggressive_and_massive_number_of_logical_lines(self):
        if False:
            for i in range(10):
                print('nop')
        "We do not care about results here.\n\n        We just want to know that it doesn't take a ridiculous amount of\n        time. Caching is currently required to avoid repeately trying\n        the same line.\n\n        "
        line = '# encoding: utf-8\nimport datetime\nfrom south.db import db\nfrom south.v2 import SchemaMigration\nfrom django.db import models\n\nfrom provider.compat import user_model_label\n\n\nclass Migration(SchemaMigration):\n\n    def forwards(self, orm):\n\n        # Adding model \'Client\'\n        db.create_table(\'oauth2_client\', (\n            (\'id\', self.gf(\'django.db.models.fields.AutoField\')(primary_key=True)),\n            (\'user\', self.gf(\'django.db.models.fields.related.ForeignKey\')(to=orm[user_model_label])),\n            (\'url\', self.gf(\'django.db.models.fields.URLField\')(max_length=200)),\n            (\'redirect_uri\', self.gf(\'django.db.models.fields.URLField\')(max_length=200)),\n            (\'client_id\', self.gf(\'django.db.models.fields.CharField\')(default=\'37b581bdc702c732aa65\', max_length=255)),\n            (\'client_secret\', self.gf(\'django.db.models.fields.CharField\')(default=\'5cf90561f7566aa81457f8a32187dcb8147c7b73\', max_length=255)),\n            (\'client_type\', self.gf(\'django.db.models.fields.IntegerField\')()),\n        ))\n        db.send_create_signal(\'oauth2\', [\'Client\'])\n\n        # Adding model \'Grant\'\n        db.create_table(\'oauth2_grant\', (\n            (\'id\', self.gf(\'django.db.models.fields.AutoField\')(primary_key=True)),\n            (\'user\', self.gf(\'django.db.models.fields.related.ForeignKey\')(to=orm[user_model_label])),\n            (\'client\', self.gf(\'django.db.models.fields.related.ForeignKey\')(to=orm[\'oauth2.Client\'])),\n            (\'code\', self.gf(\'django.db.models.fields.CharField\')(default=\'f0cda1a5f4ae915431ff93f477c012b38e2429c4\', max_length=255)),\n            (\'expires\', self.gf(\'django.db.models.fields.DateTimeField\')(default=datetime.datetime(2012, 2, 8, 10, 43, 45, 620301))),\n            (\'redirect_uri\', self.gf(\'django.db.models.fields.CharField\')(max_length=255, blank=True)),\n            (\'scope\', self.gf(\'django.db.models.fields.IntegerField\')(default=0)),\n        ))\n        db.send_create_signal(\'oauth2\', [\'Grant\'])\n\n        # Adding model \'AccessToken\'\n        db.create_table(\'oauth2_accesstoken\', (\n            (\'id\', self.gf(\'django.db.models.fields.AutoField\')(primary_key=True)),\n            (\'user\', self.gf(\'django.db.models.fields.related.ForeignKey\')(to=orm[user_model_label])),\n            (\'token\', self.gf(\'django.db.models.fields.CharField\')(default=\'b10b8f721e95117cb13c\', max_length=255)),\n            (\'client\', self.gf(\'django.db.models.fields.related.ForeignKey\')(to=orm[\'oauth2.Client\'])),\n            (\'expires\', self.gf(\'django.db.models.fields.DateTimeField\')(default=datetime.datetime(2013, 2, 7, 10, 33, 45, 618854))),\n            (\'scope\', self.gf(\'django.db.models.fields.IntegerField\')(default=0)),\n        ))\n        db.send_create_signal(\'oauth2\', [\'AccessToken\'])\n\n        # Adding model \'RefreshToken\'\n        db.create_table(\'oauth2_refreshtoken\', (\n            (\'id\', self.gf(\'django.db.models.fields.AutoField\')(primary_key=True)),\n            (\'user\', self.gf(\'django.db.models.fields.related.ForeignKey\')(to=orm[user_model_label])),\n            (\'token\', self.gf(\'django.db.models.fields.CharField\')(default=\'84035a870dab7c820c2c501fb0b10f86fdf7a3fe\', max_length=255)),\n            (\'access_token\', self.gf(\'django.db.models.fields.related.OneToOneField\')(related_name=\'refresh_token\', unique=True, to=orm[\'oauth2.AccessToken\'])),\n            (\'client\', self.gf(\'django.db.models.fields.related.ForeignKey\')(to=orm[\'oauth2.Client\'])),\n            (\'expired\', self.gf(\'django.db.models.fields.BooleanField\')(default=False)),\n        ))\n        db.send_create_signal(\'oauth2\', [\'RefreshToken\'])\n\n\n    def backwards(self, orm):\n\n        # Deleting model \'Client\'\n        db.delete_table(\'oauth2_client\')\n\n        # Deleting model \'Grant\'\n        db.delete_table(\'oauth2_grant\')\n\n        # Deleting model \'AccessToken\'\n        db.delete_table(\'oauth2_accesstoken\')\n\n        # Deleting model \'RefreshToken\'\n        db.delete_table(\'oauth2_refreshtoken\')\n\n\n    models = {\n        \'auth.group\': {\n            \'Meta\': {\'object_name\': \'Group\'},\n            \'id\': (\'django.db.models.fields.AutoField\', [], {\'primary_key\': \'True\'}),\n            \'name\': (\'django.db.models.fields.CharField\', [], {\'unique\': \'True\', \'max_length\': \'80\'}),\n            \'permissions\': (\'django.db.models.fields.related.ManyToManyField\', [], {\'to\': "orm[\'auth.Permission\']", \'symmetrical\': \'False\', \'blank\': \'True\'})\n        },\n        \'auth.permission\': {\n            \'Meta\': {\'ordering\': "(\'content_type__app_label\', \'content_type__model\', \'codename\')", \'unique_together\': "((\'content_type\', \'codename\'),)", \'object_name\': \'Permission\'},\n            \'codename\': (\'django.db.models.fields.CharField\', [], {\'max_length\': \'100\'}),\n            \'content_type\': (\'django.db.models.fields.related.ForeignKey\', [], {\'to\': "orm[\'contenttypes.ContentType\']"}),\n            \'id\': (\'django.db.models.fields.AutoField\', [], {\'primary_key\': \'True\'}),\n            \'name\': (\'django.db.models.fields.CharField\', [], {\'max_length\': \'50\'})\n        },\n        user_model_label: {\n            \'Meta\': {\'object_name\': user_model_label.split(\'.\')[-1]},\n            \'date_joined\': (\'django.db.models.fields.DateTimeField\', [], {\'default\': \'datetime.datetime.now\'}),\n            \'email\': (\'django.db.models.fields.EmailField\', [], {\'max_length\': \'75\', \'blank\': \'True\'}),\n            \'first_name\': (\'django.db.models.fields.CharField\', [], {\'max_length\': \'30\', \'blank\': \'True\'}),\n            \'groups\': (\'django.db.models.fields.related.ManyToManyField\', [], {\'to\': "orm[\'auth.Group\']", \'symmetrical\': \'False\', \'blank\': \'True\'}),\n            \'id\': (\'django.db.models.fields.AutoField\', [], {\'primary_key\': \'True\'}),\n            \'is_active\': (\'django.db.models.fields.BooleanField\', [], {\'default\': \'True\'}),\n            \'is_staff\': (\'django.db.models.fields.BooleanField\', [], {\'default\': \'False\'}),\n            \'is_superuser\': (\'django.db.models.fields.BooleanField\', [], {\'default\': \'False\'}),\n            \'last_login\': (\'django.db.models.fields.DateTimeField\', [], {\'default\': \'datetime.datetime.now\'}),\n            \'last_name\': (\'django.db.models.fields.CharField\', [], {\'max_length\': \'30\', \'blank\': \'True\'}),\n            \'password\': (\'django.db.models.fields.CharField\', [], {\'max_length\': \'128\'}),\n            \'user_permissions\': (\'django.db.models.fields.related.ManyToManyField\', [], {\'to\': "orm[\'auth.Permission\']", \'symmetrical\': \'False\', \'blank\': \'True\'}),\n            \'username\': (\'django.db.models.fields.CharField\', [], {\'unique\': \'True\', \'max_length\': \'30\'})\n        },\n        \'contenttypes.contenttype\': {\n            \'Meta\': {\'ordering\': "(\'name\',)", \'unique_together\': "((\'app_label\', \'model\'),)", \'object_name\': \'ContentType\', \'db_table\': "\'django_content_type\'"},\n            \'app_label\': (\'django.db.models.fields.CharField\', [], {\'max_length\': \'100\'}),\n            \'id\': (\'django.db.models.fields.AutoField\', [], {\'primary_key\': \'True\'}),\n            \'model\': (\'django.db.models.fields.CharField\', [], {\'max_length\': \'100\'}),\n            \'name\': (\'django.db.models.fields.CharField\', [], {\'max_length\': \'100\'})\n        },\n        \'oauth2.accesstoken\': {\n            \'Meta\': {\'object_name\': \'AccessToken\'},\n            \'client\': (\'django.db.models.fields.related.ForeignKey\', [], {\'to\': "orm[\'oauth2.Client\']"}),\n            \'expires\': (\'django.db.models.fields.DateTimeField\', [], {\'default\': \'datetime.datetime(2013, 2, 7, 10, 33, 45, 624553)\'}),\n            \'id\': (\'django.db.models.fields.AutoField\', [], {\'primary_key\': \'True\'}),\n            \'scope\': (\'django.db.models.fields.IntegerField\', [], {\'default\': \'0\'}),\n            \'token\': (\'django.db.models.fields.CharField\', [], {\'default\': "\'d5c1f65020ebdc89f20c\'", \'max_length\': \'255\'}),\n            \'user\': (\'django.db.models.fields.related.ForeignKey\', [], {\'to\': "orm[\'%s\']" % user_model_label})\n        },\n        \'oauth2.client\': {\n            \'Meta\': {\'object_name\': \'Client\'},\n            \'client_id\': (\'django.db.models.fields.CharField\', [], {\'default\': "\'306fb26cbcc87dd33cdb\'", \'max_length\': \'255\'}),\n            \'client_secret\': (\'django.db.models.fields.CharField\', [], {\'default\': "\'7e5785add4898448d53767f15373636b918cf0e3\'", \'max_length\': \'255\'}),\n            \'client_type\': (\'django.db.models.fields.IntegerField\', [], {}),\n            \'id\': (\'django.db.models.fields.AutoField\', [], {\'primary_key\': \'True\'}),\n            \'redirect_uri\': (\'django.db.models.fields.URLField\', [], {\'max_length\': \'200\'}),\n            \'url\': (\'django.db.models.fields.URLField\', [], {\'max_length\': \'200\'}),\n            \'user\': (\'django.db.models.fields.related.ForeignKey\', [], {\'to\': "orm[\'%s\']" % user_model_label})\n        },\n        \'oauth2.grant\': {\n            \'Meta\': {\'object_name\': \'Grant\'},\n            \'client\': (\'django.db.models.fields.related.ForeignKey\', [], {\'to\': "orm[\'oauth2.Client\']"}),\n            \'code\': (\'django.db.models.fields.CharField\', [], {\'default\': "\'310b2c63e27306ecf5307569dd62340cc4994b73\'", \'max_length\': \'255\'}),\n            \'expires\': (\'django.db.models.fields.DateTimeField\', [], {\'default\': \'datetime.datetime(2012, 2, 8, 10, 43, 45, 625956)\'}),\n            \'id\': (\'django.db.models.fields.AutoField\', [], {\'primary_key\': \'True\'}),\n            \'redirect_uri\': (\'django.db.models.fields.CharField\', [], {\'max_length\': \'255\', \'blank\': \'True\'}),\n            \'scope\': (\'django.db.models.fields.IntegerField\', [], {\'default\': \'0\'}),\n            \'user\': (\'django.db.models.fields.related.ForeignKey\', [], {\'to\': "orm[\'%s\']" % user_model_label})\n        },\n        \'oauth2.refreshtoken\': {\n            \'Meta\': {\'object_name\': \'RefreshToken\'},\n            \'access_token\': (\'django.db.models.fields.related.OneToOneField\', [], {\'related_name\': "\'refresh_token\'", \'unique\': \'True\', \'to\': "orm[\'oauth2.AccessToken\']"}),\n            \'client\': (\'django.db.models.fields.related.ForeignKey\', [], {\'to\': "orm[\'oauth2.Client\']"}),\n            \'expired\': (\'django.db.models.fields.BooleanField\', [], {\'default\': \'False\'}),\n            \'id\': (\'django.db.models.fields.AutoField\', [], {\'primary_key\': \'True\'}),\n            \'token\': (\'django.db.models.fields.CharField\', [], {\'default\': "\'ef0ab76037f17769ab2975a816e8f41a1c11d25e\'", \'max_length\': \'255\'}),\n            \'user\': (\'django.db.models.fields.related.ForeignKey\', [], {\'to\': "orm[\'%s\']" % user_model_label})\n        }\n    }\n\n    complete_apps = [\'oauth2\']\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(''.join(line.split()), ''.join(result.split()))

    def test_e501_shorten_comment_with_aggressive(self):
        if False:
            while True:
                i = 10
        line = '# --------- ----------------------------------------------------------------------\n'
        fixed = '# --------- --------------------------------------------------------------\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_shorten_comment_without_aggressive(self):
        if False:
            while True:
                i = 10
        'Do nothing without aggressive.'
        line = 'def foo():\n    pass\n# --------- ----------------------------------------------------------------------\n'
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_e501_with_aggressive_and_escaped_newline(self):
        if False:
            print('Hello World!')
        line = 'if True or \\\n    False:  # test test test test test test test test test test test test test test\n    pass\n'
        fixed = 'if True or \\\n        False:  # test test test test test test test test test test test test test test\n    pass\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_aggressive_and_multiline_string(self):
        if False:
            i = 10
            return i + 15
        line = "print('---------------------------------------------------------------------',\n      ('================================================', '====================='),\n      '''--------------------------------------------------------------------------------\n      ''')\n"
        fixed = "print(\n    '---------------------------------------------------------------------',\n    ('================================================',\n     '====================='),\n    '''--------------------------------------------------------------------------------\n      ''')\n"
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_aggressive_and_multiline_string_with_addition(self):
        if False:
            print('Hello World!')
        line = 'def f():\n    email_text += """<html>This is a really long docstring that goes over the column limit and is multi-line.<br><br>\n<b>Czar: </b>"""+despot["Nicholas"]+"""<br>\n<b>Minion: </b>"""+serf["Dmitri"]+"""<br>\n<b>Residence: </b>"""+palace["Winter"]+"""<br>\n</body>\n</html>"""\n'
        fixed = 'def f():\n    email_text += """<html>This is a really long docstring that goes over the column limit and is multi-line.<br><br>\n<b>Czar: </b>""" + despot["Nicholas"] + """<br>\n<b>Minion: </b>""" + serf["Dmitri"] + """<br>\n<b>Residence: </b>""" + palace["Winter"] + """<br>\n</body>\n</html>"""\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_aggressive_and_multiline_string_in_parens(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'def f():\n    email_text += ("""<html>This is a really long docstring that goes over the column limit and is multi-line.<br><br>\n<b>Czar: </b>"""+despot["Nicholas"]+"""<br>\n<b>Minion: </b>"""+serf["Dmitri"]+"""<br>\n<b>Residence: </b>"""+palace["Winter"]+"""<br>\n</body>\n</html>""")\n'
        fixed = 'def f():\n    email_text += (\n        """<html>This is a really long docstring that goes over the column limit and is multi-line.<br><br>\n<b>Czar: </b>""" +\n        despot["Nicholas"] +\n        """<br>\n<b>Minion: </b>""" +\n        serf["Dmitri"] +\n        """<br>\n<b>Residence: </b>""" +\n        palace["Winter"] +\n        """<br>\n</body>\n</html>""")\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_aggressive_and_indentation(self):
        if False:
            i = 10
            return i + 15
        line = 'if True:\n    # comment here\n    print(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n          bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,cccccccccccccccccccccccccccccccccccccccccc)\n'
        fixed = 'if True:\n    # comment here\n    print(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n          bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,\n          cccccccccccccccccccccccccccccccccccccccccc)\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_multiple_keys_and_aggressive(self):
        if False:
            print('Hello World!')
        line = "one_two_three_four_five_six = {'one two three four five': 12345, 'asdfsdflsdkfjl sdflkjsdkfkjsfjsdlkfj sdlkfjlsfjs': '343',\n                               1: 1}\n"
        fixed = "one_two_three_four_five_six = {\n    'one two three four five': 12345,\n    'asdfsdflsdkfjl sdflkjsdkfkjsfjsdlkfj sdlkfjlsfjs': '343',\n    1: 1}\n"
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_aggressive_and_carriage_returns_only(self):
        if False:
            return 10
        'Make sure _find_logical() does not crash.'
        line = 'if True:\r    from aaaaaaaaaaaaaaaa import bbbbbbbbbbbbbbbbbbb\r    \r    ccccccccccc = None\r'
        fixed = 'if True:\r    from aaaaaaaaaaaaaaaa import bbbbbbbbbbbbbbbbbbb\r\r    ccccccccccc = None\r'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_should_ignore_imports(self):
        if False:
            i = 10
            return i + 15
        line = 'import logging, os, bleach, commonware, urllib2, json, time, requests, urlparse, re\n'
        with autopep8_context(line, options=['--select=E501']) as result:
            self.assertEqual(line, result)

    def test_e501_should_not_do_useless_things(self):
        if False:
            i = 10
            return i + 15
        line = "foo('                                                                            ')\n"
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_e501_aggressive_with_percent(self):
        if False:
            return 10
        line = 'raise MultiProjectException("Ambiguous workspace: %s=%s, %s" % ( varname, varname_path, os.path.abspath(config_filename)))\n'
        fixed = 'raise MultiProjectException(\n    "Ambiguous workspace: %s=%s, %s" %\n    (varname, varname_path, os.path.abspath(config_filename)))\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_aggressive_with_def(self):
        if False:
            print('Hello World!')
        line = 'def foo(sldfkjlsdfsdf, kksdfsdfsf,sdfsdfsdf, sdfsdfkdk, szdfsdfsdf, sdfsdfsdfsdlkfjsdlf, sdfsdfddf,sdfsdfsfd, sdfsdfdsf):\n    pass\n'
        fixed = 'def foo(sldfkjlsdfsdf, kksdfsdfsf, sdfsdfsdf, sdfsdfkdk, szdfsdfsdf,\n        sdfsdfsdfsdlkfjsdlf, sdfsdfddf, sdfsdfsfd, sdfsdfdsf):\n    pass\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_more_aggressive_with_def(self):
        if False:
            print('Hello World!')
        line = 'def foobar(sldfkjlsdfsdf, kksdfsdfsf,sdfsdfsdf, sdfsdfkdk, szdfsdfsdf, sdfsdfsdfsdlkfjsdlf, sdfsdfddf,sdfsdfsfd, sdfsdfdsf):\n    pass\n'
        fixed = 'def foobar(\n        sldfkjlsdfsdf,\n        kksdfsdfsf,\n        sdfsdfsdf,\n        sdfsdfkdk,\n        szdfsdfsdf,\n        sdfsdfsdfsdlkfjsdlf,\n        sdfsdfddf,\n        sdfsdfsfd,\n        sdfsdfdsf):\n    pass\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_aggressive_with_tuple(self):
        if False:
            i = 10
            return i + 15
        line = "def f():\n    man_this_is_a_very_long_function_name(an_extremely_long_variable_name,\n                                          ('a string that is long: %s'%'bork'))\n"
        fixed = "def f():\n    man_this_is_a_very_long_function_name(\n        an_extremely_long_variable_name,\n        ('a string that is long: %s' % 'bork'))\n"
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_aggressive_with_tuple_in_list(self):
        if False:
            print('Hello World!')
        line = "def f(self):\n    self._xxxxxxxx(aaaaaa, bbbbbbbbb, cccccccccccccccccc,\n                   [('mmmmmmmmmm', self.yyyyyyyyyy.zzzzzzz/_DDDDD)], eee, 'ff')\n"
        fixed = "def f(self):\n    self._xxxxxxxx(aaaaaa, bbbbbbbbb, cccccccccccccccccc, [\n                   ('mmmmmmmmmm', self.yyyyyyyyyy.zzzzzzz / _DDDDD)], eee, 'ff')\n"
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_aggressive_decorator(self):
        if False:
            while True:
                i = 10
        line = "@foo(('xxxxxxxxxxxxxxxxxxxxxxxxxx', users.xxxxxxxxxxxxxxxxxxxxxxxxxx), ('yyyyyyyyyyyy', users.yyyyyyyyyyyy), ('zzzzzzzzzzzzzz', users.zzzzzzzzzzzzzz))\n"
        fixed = "@foo(('xxxxxxxxxxxxxxxxxxxxxxxxxx', users.xxxxxxxxxxxxxxxxxxxxxxxxxx),\n     ('yyyyyyyyyyyy', users.yyyyyyyyyyyy), ('zzzzzzzzzzzzzz', users.zzzzzzzzzzzzzz))\n"
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_aggressive_long_class_name(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'class AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA(BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB):\n    pass\n'
        fixed = 'class AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA(\n        BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB):\n    pass\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_aggressive_long_comment_and_long_line(self):
        if False:
            i = 10
            return i + 15
        line = "def foo():\n    # This is not a novel to be tossed aside lightly. It should be throw with great force.\n    self.xxxxxxxxx(_('yyyyyyyyyyyyy yyyyyyyyyyyy yyyyyyyy yyyyyyyy y'), 'zzzzzzzzzzzzzzzzzzz', bork='urgent')\n"
        fixed = "def foo():\n    # This is not a novel to be tossed aside lightly. It should be throw with\n    # great force.\n    self.xxxxxxxxx(\n        _('yyyyyyyyyyyyy yyyyyyyyyyyy yyyyyyyy yyyyyyyy y'),\n        'zzzzzzzzzzzzzzzzzzz',\n        bork='urgent')\n"
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_aggressive_intermingled_comments(self):
        if False:
            print('Hello World!')
        line = "A = [\n    # A comment\n    ['aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'bbbbbbbbbbbbbbbbbbbbbb', 'cccccccccccccccccccccc']\n]\n"
        fixed = "A = [\n    # A comment\n    ['aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',\n     'bbbbbbbbbbbbbbbbbbbbbb',\n     'cccccccccccccccccccccc']\n]\n"
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_if_line_over_limit(self):
        if False:
            while True:
                i = 10
        line = 'if not xxxxxxxxxxxx(aaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbb, cccccccccccccc, dddddddddddddddddddddd):\n    return 1\n'
        fixed = 'if not xxxxxxxxxxxx(\n        aaaaaaaaaaaaaaaaaa,\n        bbbbbbbbbbbbbbbb,\n        cccccccccccccc,\n        dddddddddddddddddddddd):\n    return 1\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_for_line_over_limit(self):
        if False:
            return 10
        line = 'for aaaaaaaaa in xxxxxxxxxxxx(aaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbb, cccccccccccccc, dddddddddddddddddddddd):\n    pass\n'
        fixed = 'for aaaaaaaaa in xxxxxxxxxxxx(\n        aaaaaaaaaaaaaaaaaa,\n        bbbbbbbbbbbbbbbb,\n        cccccccccccccc,\n        dddddddddddddddddddddd):\n    pass\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_while_line_over_limit(self):
        if False:
            while True:
                i = 10
        line = 'while xxxxxxxxxxxx(aaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbb, cccccccccccccc, dddddddddddddddddddddd):\n    pass\n'
        fixed = 'while xxxxxxxxxxxx(\n        aaaaaaaaaaaaaaaaaa,\n        bbbbbbbbbbbbbbbb,\n        cccccccccccccc,\n        dddddddddddddddddddddd):\n    pass\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e501_avoid_breaking_at_opening_slice(self):
        if False:
            while True:
                i = 10
        "Prevents line break on slice notation, dict access in this example:\n\n        GYakymOSMc=GYakymOSMW(GYakymOSMJ,GYakymOSMA,GYakymOSMr,GYakymOSMw[\n                  'abc'],GYakymOSMU,GYakymOSMq,GYakymOSMH,GYakymOSMl,svygreNveyvarf=GYakymOSME)\n\n        "
        line = "GYakymOSMc=GYakymOSMW(GYakymOSMJ,GYakymOSMA,GYakymOSMr,GYakymOSMw['abc'],GYakymOSMU,GYakymOSMq,GYakymOSMH,GYakymOSMl,svygreNveyvarf=GYakymOSME)\n"
        fixed = "GYakymOSMc = GYakymOSMW(GYakymOSMJ, GYakymOSMA, GYakymOSMr,\n                        GYakymOSMw['abc'], GYakymOSMU, GYakymOSMq, GYakymOSMH, GYakymOSMl, svygreNveyvarf=GYakymOSME)\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e501_avoid_breaking_at_multi_level_slice(self):
        if False:
            print('Hello World!')
        "Prevents line break on slice notation, dict access in this example:\n\n        GYakymOSMc=GYakymOSMW(GYakymOSMJ,GYakymOSMA,GYakymOSMr,GYakymOSMw['abc'][\n        'def'],GYakymOSMU,GYakymOSMq,GYakymOSMH,GYakymOSMl,svygreNveyvarf=GYakymOSME)\n\n        "
        line = "GYakymOSMc=GYakymOSMW(GYakymOSMJ,GYakymOSMA,GYakymOSMr,GYakymOSMw['abc']['def'],GYakymOSMU,GYakymOSMq,GYakymOSMH,GYakymOSMl,svygreNveyvarf=GYakymOSME)\n"
        fixed = "GYakymOSMc = GYakymOSMW(GYakymOSMJ, GYakymOSMA, GYakymOSMr,\n                        GYakymOSMw['abc']['def'], GYakymOSMU, GYakymOSMq, GYakymOSMH, GYakymOSMl, svygreNveyvarf=GYakymOSME)\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    @unittest.skipIf(sys.version_info.major >= 3 and sys.version_info.minor < 8 or sys.version_info.major < 3, 'syntax error in Python3.7 and lower version')
    def test_e501_with_pep572_assignment_expressions(self):
        if False:
            return 10
        line = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = 1\nif bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb := aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:\n    print(bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(line, result)

    def test_e502(self):
        if False:
            while True:
                i = 10
        line = "print('abc'\\\n      'def')\n"
        fixed = "print('abc'\n      'def')\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e701(self):
        if False:
            return 10
        line = 'if True: print(True)\n'
        fixed = 'if True:\n    print(True)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e701_with_escaped_newline(self):
        if False:
            i = 10
            return i + 15
        line = 'if True:\\\nprint(True)\n'
        fixed = 'if True:\n    print(True)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    @unittest.skipIf(sys.version_info >= (3, 12), 'not detech in Python3.12+')
    def test_e701_with_escaped_newline_and_spaces(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'if True:    \\   \nprint(True)\n'
        fixed = 'if True:\n    print(True)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e702(self):
        if False:
            i = 10
            return i + 15
        line = 'print(1); print(2)\n'
        fixed = 'print(1)\nprint(2)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e702_after_colon_should_be_untouched(self):
        if False:
            i = 10
            return i + 15
        line = 'def foo(): print(1); print(2)\n'
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_e702_with_semicolon_at_end(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'print(1);\n'
        fixed = 'print(1)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e702_with_semicolon_and_space_at_end(self):
        if False:
            i = 10
            return i + 15
        line = 'print(1); \n'
        fixed = 'print(1)\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e702_with_whitespace(self):
        if False:
            while True:
                i = 10
        line = 'print(1) ; print(2)\n'
        fixed = 'print(1)\nprint(2)\n'
        with autopep8_context(line, options=['--select=E702']) as result:
            self.assertEqual(fixed, result)

    def test_e702_with_non_ascii_file(self):
        if False:
            print('Hello World!')
        line = "# -*- coding: utf-8 -*-\n# French comment with accent é\n# Un commentaire en français avec un accent é\n\nimport time\n\ntime.strftime('%d-%m-%Y');\n"
        fixed = "# -*- coding: utf-8 -*-\n# French comment with accent é\n# Un commentaire en français avec un accent é\n\nimport time\n\ntime.strftime('%d-%m-%Y')\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e702_with_escaped_newline(self):
        if False:
            return 10
        line = '1; \\\n2\n'
        fixed = '1\n2\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e702_with_escaped_newline_with_indentation(self):
        if False:
            while True:
                i = 10
        line = '1; \\\n    2\n'
        fixed = '1\n2\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e702_more_complicated(self):
        if False:
            i = 10
            return i + 15
        line = 'def foo():\n    if bar : bar+=1;  bar=bar*bar   ; return bar\n'
        fixed = 'def foo():\n    if bar:\n        bar += 1\n        bar = bar * bar\n        return bar\n'
        with autopep8_context(line, options=['--select=E,W']) as result:
            self.assertEqual(fixed, result)

    def test_e702_with_semicolon_in_string(self):
        if False:
            print('Hello World!')
        line = 'print(";");\n'
        fixed = 'print(";")\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e702_with_semicolon_in_string_to_the_right(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'x = "x"; y = "y;y"\n'
        fixed = 'x = "x"\ny = "y;y"\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e702_indent_correctly(self):
        if False:
            i = 10
            return i + 15
        line = '\n(\n    1,\n    2,\n    3); 4; 5; 5  # pyflakes\n'
        fixed = '\n(\n    1,\n    2,\n    3)\n4\n5\n5  # pyflakes\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e702_with_triple_quote(self):
        if False:
            print('Hello World!')
        line = '"""\n      hello\n   """; 1\n'
        fixed = '"""\n      hello\n   """\n1\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e702_with_triple_quote_and_indent(self):
        if False:
            print('Hello World!')
        line = 'def f():\n    """\n      hello\n   """; 1\n'
        fixed = 'def f():\n    """\n      hello\n   """\n    1\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e702_with_semicolon_after_string(self):
        if False:
            i = 10
            return i + 15
        line = "raise IOError('abc '\n              'def.');\n"
        fixed = "raise IOError('abc '\n              'def.')\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e702_with_dict_semicolon(self):
        if False:
            for i in range(10):
                print('nop')
        line = "MY_CONST = [\n    {'A': 1},\n    {'B': 2}\n];\n"
        fixed = "MY_CONST = [\n    {'A': 1},\n    {'B': 2}\n]\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e702_with_e701_and_only_select_e702_option(self):
        if False:
            return 10
        line = 'for i in range(3):\n    if i == 1: print(i); continue\n    print(i)\n'
        with autopep8_context(line, options=['--select=E702']) as result:
            self.assertEqual(line, result)

    def test_e703_with_inline_comment(self):
        if False:
            while True:
                i = 10
        line = 'a = 5;    # inline comment\n'
        fixed = 'a = 5    # inline comment\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e703_in_example_of_readme(self):
        if False:
            return 10
        line = "def example2(): return ('' in {'f': 2}) in {'has_key() is deprecated': True};\n"
        fixed = "def example2(): return ('' in {'f': 2}) in {'has_key() is deprecated': True}\n"
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e704(self):
        if False:
            return 10
        line = 'def f(x): return 2*x\n'
        fixed = 'def f(x):\n    return 2 * x\n'
        with autopep8_context(line, options=['-aaa']) as result:
            self.assertEqual(fixed, result)

    def test_e704_not_work_with_aa_option(self):
        if False:
            return 10
        line = 'def f(x): return 2*x\n'
        with autopep8_context(line, options=['-aa', '--select=E704']) as result:
            self.assertEqual(line, result)

    def test_e711(self):
        if False:
            print('Hello World!')
        line = 'foo == None\n'
        fixed = 'foo is None\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)
        line = 'None == foo\n'
        fixed = 'None is foo\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e711_in_conditional(self):
        if False:
            return 10
        line = 'if foo == None and None == foo:\npass\n'
        fixed = 'if foo is None and None is foo:\npass\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e711_in_conditional_with_multiple_instances(self):
        if False:
            return 10
        line = 'if foo == None and bar == None:\npass\n'
        fixed = 'if foo is None and bar is None:\npass\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e711_with_not_equals_none(self):
        if False:
            i = 10
            return i + 15
        line = 'foo != None\n'
        fixed = 'foo is not None\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e712(self):
        if False:
            return 10
        line = 'foo == True\n'
        fixed = 'foo\n'
        with autopep8_context(line, options=['-aa', '--select=E712']) as result:
            self.assertEqual(fixed, result)

    def test_e712_in_conditional_with_multiple_instances(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'if foo == True and bar == True:\npass\n'
        fixed = 'if foo and bar:\npass\n'
        with autopep8_context(line, options=['-aa', '--select=E712']) as result:
            self.assertEqual(fixed, result)

    def test_e712_with_false(self):
        if False:
            i = 10
            return i + 15
        line = 'foo != False\n'
        fixed = 'foo\n'
        with autopep8_context(line, options=['-aa', '--select=E712']) as result:
            self.assertEqual(fixed, result)

    def test_e712_with_special_case_equal_not_true(self):
        if False:
            i = 10
            return i + 15
        line = 'if foo != True:\n    pass\n'
        fixed = 'if not foo:\n    pass\n'
        with autopep8_context(line, options=['-aa', '--select=E712']) as result:
            self.assertEqual(fixed, result)

    def test_e712_with_special_case_equal_false(self):
        if False:
            while True:
                i = 10
        line = 'if foo == False:\n    pass\n'
        fixed = 'if not foo:\n    pass\n'
        with autopep8_context(line, options=['-aa', '--select=E712']) as result:
            self.assertEqual(fixed, result)

    def test_e712_with_dict_value(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'if d["key"] != True:\n    pass\n'
        fixed = 'if not d["key"]:\n    pass\n'
        with autopep8_context(line, options=['-aa', '--select=E712']) as result:
            self.assertEqual(fixed, result)

    def test_e712_only_if_aggressive_level_2(self):
        if False:
            return 10
        line = 'foo == True\n'
        with autopep8_context(line, options=['-a']) as result:
            self.assertEqual(line, result)

    def test_e711_and_e712(self):
        if False:
            while True:
                i = 10
        line = 'if (foo == None and bar == True) or (foo != False and bar != None):\npass\n'
        fixed = 'if (foo is None and bar) or (foo and bar is not None):\npass\n'
        with autopep8_context(line, options=['-aa']) as result:
            self.assertEqual(fixed, result)

    def test_e713(self):
        if False:
            return 10
        line = 'if not x in y:\n    pass\n'
        fixed = 'if x not in y:\n    pass\n'
        with autopep8_context(line, options=['-aa', '--select=E713']) as result:
            self.assertEqual(fixed, result)

    def test_e713_more(self):
        if False:
            i = 10
            return i + 15
        line = 'if not "." in y:\n    pass\n'
        fixed = 'if "." not in y:\n    pass\n'
        with autopep8_context(line, options=['-aa', '--select=E713']) as result:
            self.assertEqual(fixed, result)

    def test_e713_with_in(self):
        if False:
            while True:
                i = 10
        line = 'if not "." in y and "," in y:\n    pass\n'
        fixed = 'if "." not in y and "," in y:\n    pass\n'
        with autopep8_context(line, options=['-aa', '--select=E713']) as result:
            self.assertEqual(fixed, result)

    def test_e713_with_tuple(self):
        if False:
            return 10
        line = '\nif not role in ("domaincontroller_master",\n                "domaincontroller_backup",\n                "domaincontroller_slave",\n                "memberserver",\n                ):\n    pass\n'
        fixed = '\nif role not in ("domaincontroller_master",\n                "domaincontroller_backup",\n                "domaincontroller_slave",\n                "memberserver",\n                ):\n    pass\n'
        with autopep8_context(line, options=['-aa', '--select=E713']) as result:
            self.assertEqual(fixed, result)

    def test_e713_chain(self):
        if False:
            i = 10
            return i + 15
        line = 'if "@" not in x or not "/" in y:\n    pass\n'
        fixed = 'if "@" not in x or "/" not in y:\n    pass\n'
        with autopep8_context(line, options=['-aa', '--select=E713']) as result:
            self.assertEqual(fixed, result)

    def test_e713_chain2(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'if "@" not in x or "[" not in x or not "/" in y:\n    pass\n'
        fixed = 'if "@" not in x or "[" not in x or "/" not in y:\n    pass\n'
        with autopep8_context(line, options=['-aa', '--select=E713']) as result:
            self.assertEqual(fixed, result)

    def test_e713_chain3(self):
        if False:
            i = 10
            return i + 15
        line = 'if not "@" in x or "[" not in x or not "/" in y:\n    pass\n'
        fixed = 'if "@" not in x or "[" not in x or "/" not in y:\n    pass\n'
        with autopep8_context(line, options=['-aa', '--select=E713']) as result:
            self.assertEqual(fixed, result)

    def test_e713_chain4(self):
        if False:
            return 10
        line = 'if not "." in y and not "," in y:\n    pass\n'
        fixed = 'if "." not in y and "," not in y:\n    pass\n'
        with autopep8_context(line, options=['-aa', '--select=E713']) as result:
            self.assertEqual(fixed, result)

    def test_e714(self):
        if False:
            print('Hello World!')
        line = 'if not x is y:\n    pass\n'
        fixed = 'if x is not y:\n    pass\n'
        with autopep8_context(line, options=['-aa', '--select=E714']) as result:
            self.assertEqual(fixed, result)

    def test_e714_with_is(self):
        if False:
            while True:
                i = 10
        line = 'if not x is y or x is z:\n    pass\n'
        fixed = 'if x is not y or x is z:\n    pass\n'
        with autopep8_context(line, options=['-aa', '--select=E714']) as result:
            self.assertEqual(fixed, result)

    def test_e714_chain(self):
        if False:
            print('Hello World!')
        line = 'if not x is y or not x is z:\n    pass\n'
        fixed = 'if x is not y or x is not z:\n    pass\n'
        with autopep8_context(line, options=['-aa', '--select=E714']) as result:
            self.assertEqual(fixed, result)

    def test_e713_and_e714(self):
        if False:
            i = 10
            return i + 15
        line = '\nif not x is y:\n    pass\nif not role in ("domaincontroller_master",\n                "domaincontroller_backup",\n                "domaincontroller_slave",\n                "memberserver",\n                ):\n    pass\n'
        fixed = '\nif x is not y:\n    pass\nif role not in ("domaincontroller_master",\n                "domaincontroller_backup",\n                "domaincontroller_slave",\n                "memberserver",\n                ):\n    pass\n'
        with autopep8_context(line, options=['-aa', '--select=E713,E714']) as result:
            self.assertEqual(fixed, result)

    def test_e713_with_single_quote(self):
        if False:
            return 10
        line = "if not 'DC IP' in info:\n"
        fixed = "if 'DC IP' not in info:\n"
        with autopep8_context(line, options=['-aa', '--select=E713,E714']) as result:
            self.assertEqual(fixed, result)

    def test_e714_with_single_quote(self):
        if False:
            i = 10
            return i + 15
        line = "if not 'DC IP' is info:\n"
        fixed = "if 'DC IP' is not info:\n"
        with autopep8_context(line, options=['-aa', '--select=E713,E714']) as result:
            self.assertEqual(fixed, result)

    def test_e721(self):
        if False:
            return 10
        line = "type('') == type('')\n"
        fixed = "isinstance('', type(''))\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e721_with_str(self):
        if False:
            print('Hello World!')
        line = "str == type('')\n"
        fixed = "isinstance('', str)\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e721_in_conditional(self):
        if False:
            print('Hello World!')
        line = "if str == type(''):\n    pass\n"
        fixed = "if isinstance('', str):\n    pass\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e722(self):
        if False:
            return 10
        line = 'try:\n    print(a)\nexcept:\n    pass\n'
        fixed = 'try:\n    print(a)\nexcept BaseException:\n    pass\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e722_with_if_else_stmt(self):
        if False:
            i = 10
            return i + 15
        line = 'try:\n    print(a)\nexcept:\n    if a==b:\n        print(a)\n    else:\n        print(b)\n'
        fixed = 'try:\n    print(a)\nexcept BaseException:\n    if a == b:\n        print(a)\n    else:\n        print(b)\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e722_non_aggressive(self):
        if False:
            while True:
                i = 10
        line = 'try:\n    print(a)\nexcept:\n    pass\n'
        with autopep8_context(line, options=[]) as result:
            self.assertEqual(line, result)

    def test_e731(self):
        if False:
            while True:
                i = 10
        line = 'a = lambda x: x * 2\n'
        fixed = 'def a(x): return x * 2\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e731_no_arg(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'a = lambda: x * 2\n'
        fixed = 'def a(): return x * 2\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e731_with_tuple_arg(self):
        if False:
            print('Hello World!')
        line = 'a = lambda (x, y), z: x * 2\n'
        fixed = 'def a((x, y), z): return x * 2\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e731_with_args(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'a = lambda x, y: x * 2 + y\n'
        fixed = 'def a(x, y): return x * 2 + y\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_e731_with_select_option(self):
        if False:
            i = 10
            return i + 15
        line = 'a = lambda x: x * 2\n'
        fixed = 'def a(x): return x * 2\n'
        with autopep8_context(line, options=['--select=E731']) as result:
            self.assertEqual(fixed, result)

    def test_e731_with_default_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'a = lambda k, d=None: bar.get("%s/%s" % (prefix, k), d)\n'
        fixed = 'def a(k, d=None): return bar.get("%s/%s" % (prefix, k), d)\n'
        with autopep8_context(line, options=['--select=E731']) as result:
            self.assertEqual(fixed, result)

    @unittest.skipIf(sys.version_info >= (3, 12), 'not detech in Python3.12+')
    def test_e901_should_cause_indentation_screw_up(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'def tmp(g):\n    g(4)))\n\n    if not True:\n        pass\n        pass\n'
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_should_preserve_vertical_tab(self):
        if False:
            while True:
                i = 10
        line = '#Memory Bu\x0bffer Register:\n'
        fixed = '# Memory Bu\x0bffer Register:\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_w191_should_ignore_multiline_strings(self):
        if False:
            while True:
                i = 10
        line = "print(3 !=  4, '''\nwhile True:\n    if True:\n    \t1\n\t''', 4  != 5)\nif True:\n\t123\n"
        fixed = "print(3 != 4, '''\nwhile True:\n    if True:\n    \t1\n\t''', 4 != 5)\nif True:\n    123\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_w191_should_ignore_tabs_in_strings(self):
        if False:
            print('Hello World!')
        line = "if True:\n\tx = '''\n\t\tblah\n\tif True:\n\t1\n\t'''\nif True:\n\t123\nelse:\n\t32\n"
        fixed = "if True:\n    x = '''\n\t\tblah\n\tif True:\n\t1\n\t'''\nif True:\n    123\nelse:\n    32\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_w291(self):
        if False:
            while True:
                i = 10
        line = "print('a b ')\t \n"
        fixed = "print('a b ')\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_w291_with_comment(self):
        if False:
            for i in range(10):
                print('nop')
        line = "print('a b ')  # comment\t \n"
        fixed = "print('a b ')  # comment\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_w292(self):
        if False:
            while True:
                i = 10
        line = '1\n2'
        fixed = '1\n2\n'
        with autopep8_context(line, options=['--aggressive', '--select=W292']) as result:
            self.assertEqual(fixed, result)

    def test_w292_ignore(self):
        if False:
            for i in range(10):
                print('nop')
        line = '1\n2'
        with autopep8_context(line, options=['--aggressive', '--ignore=W292']) as result:
            self.assertEqual(line, result)

    def test_w293(self):
        if False:
            while True:
                i = 10
        line = '1\n \n2\n'
        fixed = '1\n\n2\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_w391(self):
        if False:
            for i in range(10):
                print('nop')
        line = '  \n'
        fixed = ''
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_w391_more_complex(self):
        if False:
            i = 10
            return i + 15
        line = '123\n456\n  \n'
        fixed = '123\n456\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_w503(self):
        if False:
            return 10
        line = '(width == 0\n + height == 0)\n'
        fixed = '(width == 0 +\n height == 0)\n'
        with autopep8_context(line, options=['--select=W503']) as result:
            self.assertEqual(fixed, result)

    def test_w503_with_ignore_w504(self):
        if False:
            while True:
                i = 10
        line = '(width == 0\n + height == 0)\n'
        fixed = '(width == 0 +\n height == 0)\n'
        with autopep8_context(line, options=['--ignore=E,W504']) as result:
            self.assertEqual(fixed, result)

    def test_w504_with_ignore_w503(self):
        if False:
            print('Hello World!')
        line = '(width == 0 +\n height == 0)\n'
        fixed = '(width == 0\n + height == 0)\n'
        with autopep8_context(line, options=['--ignore=E,W503']) as result:
            self.assertEqual(fixed, result)

    def test_w503_w504_none_ignored(self):
        if False:
            for i in range(10):
                print('nop')
        line = '(width == 0 +\n height == 0\n+ depth == 0)\n'
        fixed = '(width == 0 +\n height == 0\n+ depth == 0)\n'
        with autopep8_context(line, options=['--ignore=E']) as result:
            self.assertEqual(fixed, result)

    def test_w503_w504_both_ignored(self):
        if False:
            i = 10
            return i + 15
        line = '(width == 0 +\n height == 0\n+ depth == 0)\n'
        fixed = '(width == 0 +\n height == 0\n+ depth == 0)\n'
        with autopep8_context(line, options=['--ignore=E,W503, W504']) as result:
            self.assertEqual(fixed, result)

    def test_w503_skip_default(self):
        if False:
            i = 10
            return i + 15
        line = '(width == 0\n + height == 0)\n'
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_w503_and_or(self):
        if False:
            for i in range(10):
                print('nop')
        line = '(width == 0\n and height == 0\n or name == "")\n'
        fixed = '(width == 0 and\n height == 0 or\n name == "")\n'
        with autopep8_context(line, options=['--select=W503']) as result:
            self.assertEqual(fixed, result)

    def test_w503_with_comment(self):
        if False:
            i = 10
            return i + 15
        line = '(width == 0  # this is comment\n + height == 0)\n'
        fixed = '(width == 0 +  # this is comment\n height == 0)\n'
        with autopep8_context(line, options=['--select=W503']) as result:
            self.assertEqual(fixed, result)

    def test_w503_with_comment_into_point_out_line(self):
        if False:
            print('Hello World!')
        line = 'def test():\n    return (\n        True not in []\n        and False  # comment required\n    )\n'
        fixed = 'def test():\n    return (\n        True not in [] and\n        False  # comment required\n    )\n'
        with autopep8_context(line, options=['--select=W503']) as result:
            self.assertEqual(fixed, result)

    def test_w503_with_comment_double(self):
        if False:
            i = 10
            return i + 15
        line = '(\n    1111  # C1\n    and 22222222  # C2\n    and 333333333333  # C3\n)\n'
        fixed = '(\n    1111 and  # C1\n    22222222 and  # C2\n    333333333333  # C3\n)\n'
        with autopep8_context(line, options=['--select=W503']) as result:
            self.assertEqual(fixed, result)

    def test_w503_with_comment_with_only_comment_block_charactor(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'if (True  #\n    and True\n    and True):\n    print(1)\n'
        fixed = 'if (True and  #\n    True and\n    True):\n    print(1)\n'
        with autopep8_context(line, options=['--select=W503']) as result:
            self.assertEqual(fixed, result)

    def test_w503_over_5lines(self):
        if False:
            i = 10
            return i + 15
        line = 'X = (\n    1  # 1\n    + 2  # 2\n    + 3  # 3\n    + 4  # 4\n    + 5  # 5\n    + 6  # 6\n    + 7  # 7\n)\n'
        fixed = 'X = (\n    1 +  # 1\n    2 +  # 2\n    3 +  # 3\n    4 +  # 4\n    5 +  # 5\n    6 +  # 6\n    7  # 7\n)\n'
        with autopep8_context(line, options=['--select=W503']) as result:
            self.assertEqual(fixed, result)

    def test_w503_with_line_comment(self):
        if False:
            print('Hello World!')
        line = '(width == 0\n # this is comment\n + height == 0)\n'
        fixed = '(width == 0 +\n # this is comment\n height == 0)\n'
        with autopep8_context(line, options=['--select=W503', '--ignore=E']) as result:
            self.assertEqual(fixed, result)

    def test_w503_with_empty_line(self):
        if False:
            while True:
                i = 10
        line = '\n# this is comment\na = 2\nb = (1 +\n     2 +\n     3) / 2.0\n'
        fixed = '\n# this is comment\na = 2\nb = (1 +\n     2 +\n     3) / 2.0\n'
        with autopep8_context(line, options=['--ignore=E721']) as result:
            self.assertEqual(fixed, result)

    def test_w503_with_line_comments(self):
        if False:
            i = 10
            return i + 15
        line = '(width == 0\n # this is comment\n # comment2\n + height == 0)\n'
        fixed = '(width == 0 +\n # this is comment\n # comment2\n height == 0)\n'
        with autopep8_context(line, options=['--select=W503', '--ignore=E']) as result:
            self.assertEqual(fixed, result)

    def test_ignore_only_w503_with_select_w(self):
        if False:
            print('Hello World!')
        line = 'a = (\n    11 + 22 +\n    33 +\n    44\n    + 55\n)\n'
        fixed = 'a = (\n    11 + 22\n    + 33\n    + 44\n    + 55\n)\n'
        with autopep8_context(line, options=['--select=W', '--ignore=W503']) as result:
            self.assertEqual(fixed, result)
        with autopep8_context(line, options=['--select=W5', '--ignore=W503']) as result:
            self.assertEqual(fixed, result)
        with autopep8_context(line, options=['--select=W50', '--ignore=W503']) as result:
            self.assertEqual(fixed, result)

    def test_ignore_only_w504_with_select_w(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'a = (\n    11 + 22 +\n    33 +\n    44\n    + 55\n)\n'
        fixed = 'a = (\n    11 + 22 +\n    33 +\n    44 +\n    55\n)\n'
        with autopep8_context(line, options=['--select=W', '--ignore=W504']) as result:
            self.assertEqual(fixed, result)
        with autopep8_context(line, options=['--select=W5', '--ignore=W504']) as result:
            self.assertEqual(fixed, result)
        with autopep8_context(line, options=['--select=W50', '--ignore=W504']) as result:
            self.assertEqual(fixed, result)

    def test_ignore_w503_and_w504_with_select_w(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'a = (\n    11 + 22 +\n    33 +\n    44\n    + 55\n)\n'
        with autopep8_context(line, options=['--select=W', '--ignore=W503,W504']) as result:
            self.assertEqual(line, result)
        with autopep8_context(line, options=['--select=W5', '--ignore=W503,W504']) as result:
            self.assertEqual(line, result)
        with autopep8_context(line, options=['--select=W50', '--ignore=W503,W504']) as result:
            self.assertEqual(line, result)

    def test_w504(self):
        if False:
            for i in range(10):
                print('nop')
        line = '(width == 0 +\n height == 0)\n'
        fixed = '(width == 0\n + height == 0)\n'
        with autopep8_context(line, options=['--select=W504', '--ignore=E']) as result:
            self.assertEqual(fixed, result)

    def test_w504_comment_on_first_line(self):
        if False:
            i = 10
            return i + 15
        line = 'x = (1 | # test\n2)\n'
        fixed = 'x = (1 # test\n| 2)\n'
        with autopep8_context(line, options=['--select=W504', '--ignore=E']) as result:
            self.assertEqual(fixed, result)

    def test_w504_comment_on_second_line(self):
        if False:
            i = 10
            return i + 15
        line = 'x = (1 |\n2) # test\n'
        fixed = 'x = (1\n| 2) # test\n'
        with autopep8_context(line, options=['--select=W504', '--ignore=E']) as result:
            self.assertEqual(fixed, result)

    def test_w504_comment_on_each_lines(self):
        if False:
            print('Hello World!')
        line = 'x = (1 |# test\n2 |# test\n3) # test\n'
        fixed = 'x = (1# test\n| 2# test\n| 3) # test\n'
        with autopep8_context(line, options=['--select=W504', '--ignore=E']) as result:
            self.assertEqual(fixed, result)

    def test_w504_with_e265_ignore_option(self):
        if False:
            i = 10
            return i + 15
        line = '(width == 0 +\n height == 0)\n'
        with autopep8_context(line, options=['--ignore=E265']) as result:
            self.assertEqual(line, result)

    def test_w504_with_e265_ignore_option_regression(self):
        if False:
            while True:
                i = 10
        line = 'if True:\n    if True:\n        if (\n                link.is_wheel and\n                isinstance(link.comes_from, HTMLPage) and\n                link.comes_from.url.startswith(index_url)\n        ):\n            _store_wheel_in_cache(file_path, index_url)\n'
        with autopep8_context(line, options=['--ignore=E265']) as result:
            self.assertEqual(line, result)

    def test_w504_with_line_comment(self):
        if False:
            return 10
        line = '(width == 0 +\n # this is comment\n height == 0)\n'
        fixed = '(width == 0\n # this is comment\n + height == 0)\n'
        with autopep8_context(line, options=['--select=W504', '--ignore=E']) as result:
            self.assertEqual(fixed, result)

    def test_w504_not_applied_by_default_when_modifying_with_ignore(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'q = 1\n\n\n\n\ndef x(y, z):\n    if (\n            y and\n            z\n    ):\n        pass\n'
        fixed = line.replace('\n\n\n\n', '\n\n')
        with autopep8_context(line, options=['--ignore=E265']) as result:
            self.assertEqual(fixed, result)

    def test_w503_and_w504_conflict(self):
        if False:
            return 10
        line = "if True:\n    if True:\n        assert_equal(self.nodes[0].getbalance(\n        ), bal + Decimal('50.00000000') + Decimal('2.19000000'))  # block reward + tx\n"
        fixed = "if True:\n    if True:\n        assert_equal(\n            self.nodes[0].getbalance(),\n            bal +\n            Decimal('50.00000000') +\n            Decimal('2.19000000'))  # block reward + tx\n"
        with autopep8_context(line, options=['-aa', '--select=E,W']) as result:
            self.assertEqual(fixed, result)
        with autopep8_context(line, options=['-aa', '--select=E,W5']) as result:
            self.assertEqual(fixed, result)
        with autopep8_context(line, options=['-aa', '--select=E,W50']) as result:
            self.assertEqual(fixed, result)

    def test_w605_simple(self):
        if False:
            return 10
        line = "escape = '\\.jpg'\n"
        fixed = "escape = '\\\\.jpg'\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_w605_identical_token(self):
        if False:
            return 10
        line = "escape = foo('\\.bar', '\\.kilroy')\n"
        fixed = "escape = foo('\\\\.bar', '\\\\.kilroy')\n"
        with autopep8_context(line, options=['--aggressive', '--pep8-passes', '5']) as result:
            self.assertEqual(fixed, result, 'Two tokens get r added')
        line = "escape = foo('\\.bar', '\\\\.kilroy')\n"
        fixed = "escape = foo('\\\\.bar', '\\\\.kilroy')\n"
        with autopep8_context(line, options=['--aggressive', '--pep8-passes', '5']) as result:
            self.assertEqual(fixed, result, 'r not added if already there')
        line = "escape = foo('\\.bar', '\\.bar')\n"
        fixed = "escape = foo('\\\\.bar', '\\\\.bar')\n"
        with autopep8_context(line, options=['--aggressive', '--pep8-passes', '5']) as result:
            self.assertEqual(fixed, result)

    def test_w605_with_invalid_syntax(self):
        if False:
            print('Hello World!')
        line = "escape = rr'\\.jpg'\n"
        fixed = "escape = rr'\\\\.jpg'\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_w605_with_multilines(self):
        if False:
            print('Hello World!')
        line = "regex = '\\d+(\\.\\d+){3}$'\nfoo = validators.RegexValidator(\n    regex='\\d+(\\.\\d+){3}$')\n"
        fixed = "regex = '\\\\d+(\\\\.\\\\d+){3}$'\nfoo = validators.RegexValidator(\n    regex='\\\\d+(\\\\.\\\\d+){3}$')\n"
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_trailing_whitespace_in_multiline_string(self):
        if False:
            i = 10
            return i + 15
        line = 'x = """ \nhello"""    \n'
        fixed = 'x = """ \nhello"""\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_trailing_whitespace_in_multiline_string_aggressive(self):
        if False:
            print('Hello World!')
        line = 'x = """ \nhello"""    \n'
        fixed = 'x = """\nhello"""\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_execfile_in_lambda_should_not_be_modified(self):
        if False:
            for i in range(10):
                print('nop')
        'Modifying this to the exec() form is invalid in Python 2.'
        line = 'lambda: execfile("foo.py")\n'
        with autopep8_context(line, options=['--aggressive']) as result:
            self.assertEqual(line, result)

    def test_range(self):
        if False:
            while True:
                i = 10
        line = 'print( 1 )\nprint( 2 )\n print( 3 )\n'
        fixed = 'print( 1 )\nprint(2)\n print( 3 )\n'
        with autopep8_context(line, options=['--line-range', '2', '2']) as result:
            self.assertEqual(fixed, result)

    def test_range_line_number_changes_from_one_line(self):
        if False:
            while True:
                i = 10
        line = 'a=12\na=1; b=2;c=3\nd=4;\n\ndef f(a = 1):\n    pass\n'
        fixed = 'a=12\na = 1\nb = 2\nc = 3\nd=4;\n\ndef f(a = 1):\n    pass\n'
        with autopep8_context(line, options=['--line-range', '2', '2']) as result:
            self.assertEqual(fixed, result)

    def test_range_indent_changes_small_range(self):
        if False:
            return 10
        line = '\nif True:\n  (1, \n    2,\n3)\nelif False:\n  a = 1\nelse:\n  a = 2\n\nc = 1\nif True:\n  c = 2\n  a = (1,\n2)\n'
        fixed2_5 = '\nif True:\n  (1,\n   2,\n   3)\nelif False:\n  a = 1\nelse:\n  a = 2\n\nc = 1\nif True:\n  c = 2\n  a = (1,\n2)\n'
        with autopep8_context(line, options=['--line-range', '2', '5']) as result:
            self.assertEqual(fixed2_5, result)

    def test_range_indent_deep_if_blocks_first_block(self):
        if False:
            return 10
        line = '\nif a:\n  if a = 1:\n    b = 1\n  else:\n    b = 2\nelif a == 0:\n  b = 3\nelse:\n  b = 4\n'
        with autopep8_context(line, options=['--line-range', '2', '5']) as result:
            self.assertEqual(line, result)

    def test_range_indent_deep_if_blocks_second_block(self):
        if False:
            print('Hello World!')
        line = '\nif a:\n  if a = 1:\n    b = 1\n  else:\n    b = 2\nelif a == 0:\n  b = 3\nelse:\n  b = 4\n'
        with autopep8_context(line, options=['--line-range', '6', '9']) as result:
            self.assertEqual(line, result)

    def test_range_indent_continued_statements_partial(self):
        if False:
            while True:
                i = 10
        line = '\nif a == 1:\n\ttry:\n\t  foo\n\texcept AttributeError:\n\t  pass\n\telse:\n\t  "nooo"\n\tb = 1\n'
        with autopep8_context(line, options=['--line-range', '2', '6']) as result:
            self.assertEqual(line, result)

    def test_range_indent_continued_statements_last_block(self):
        if False:
            return 10
        line = '\nif a == 1:\n\ttry:\n\t  foo\n\texcept AttributeError:\n\t  pass\n\telse:\n\t  "nooo"\n\tb = 1\n'
        with autopep8_context(line, options=['--line-range', '6', '9']) as result:
            self.assertEqual(line, result)

    def test_range_with_broken_syntax(self):
        if False:
            while True:
                i = 10
        line = 'if True:\n   if True:\n      pass\n else:\n    pass\n'
        with autopep8_context(line, options=['--line-range', '1', '1']) as result:
            self.assertEqual(line, result)

    def test_long_import_line(self):
        if False:
            for i in range(10):
                print('nop')
        line = 's\nfrom t import a,     bbbbbbbbbbbbbbbbbbbbbbbbbbbbb, ccccccccccccccccccccccccccccccc, ddddddddddddddddddddddddddddddddddd\n'
        fixed = 'from t import a,     bbbbbbbbbbbbbbbbbbbbbbbbbbbbb, ccccccccccccccccccccccccccccccc, ddddddddddddddddddddddddddddddddddd\ns\n'
        with autopep8_context(line) as result:
            self.assertEqual(fixed, result)

    def test_exchange_multiple_imports_with_def(self):
        if False:
            return 10
        line = 'def f(n):\n    return n\nfrom a import fa\nfrom b import fb\nfrom c import fc\n'
        with autopep8_context(line) as result:
            self.assertEqual(result[:4], 'from')

    @unittest.skipIf(sys.version_info.major >= 3 and sys.version_info.minor < 8 or sys.version_info.major < 3, 'syntax error in Python3.7 and lower version')
    def test_with_walrus_operator(self):
        if False:
            for i in range(10):
                print('nop')
        'check pycodestyle 2.6.0+'
        line = 'sql_stmt = ""\nwith open(filename) as f:\n    while line := f.readline():\n        sql_stmt += line\n'
        with autopep8_context(line) as result:
            self.assertEqual(line, result)

    def test_autopep8_disable(self):
        if False:
            return 10
        test_code = '# autopep8: off\ndef f():\n    aaaaaaaaaaa.bbbbbbb([\n        (\'xxxxxxxxxx\', \'yyyyyy\',\n         \'Heaven hath no wrath like love to hatred turned. Nor hell a fury like a woman scorned.\'),\n        (\'xxxxxxx\', \'yyyyyyyyyyy\', "To the last I grapple with thee. From hell\'s heart I stab at thee. For hate\'s sake I spit my last breath at thee!")])\n# autopep8: on\n'
        expected_output = '# autopep8: off\ndef f():\n    aaaaaaaaaaa.bbbbbbb([\n        (\'xxxxxxxxxx\', \'yyyyyy\',\n         \'Heaven hath no wrath like love to hatred turned. Nor hell a fury like a woman scorned.\'),\n        (\'xxxxxxx\', \'yyyyyyyyyyy\', "To the last I grapple with thee. From hell\'s heart I stab at thee. For hate\'s sake I spit my last breath at thee!")])\n# autopep8: on\n'
        with autopep8_context(test_code) as result:
            self.assertEqual(expected_output, result)

    def test_autopep8_disable_multi(self):
        if False:
            print('Hello World!')
        test_code = 'fix=1\n# autopep8: off\nskip=1\n# autopep8: on\nfix=2\n# autopep8: off\nskip=2\n# autopep8: on\nfix=3\n'
        expected_output = 'fix = 1\n# autopep8: off\nskip=1\n# autopep8: on\nfix = 2\n# autopep8: off\nskip=2\n# autopep8: on\nfix = 3\n'
        with autopep8_context(test_code) as result:
            self.assertEqual(expected_output, result)

    def test_fmt_disable(self):
        if False:
            i = 10
            return i + 15
        test_code = '# fmt: off\ndef f():\n    aaaaaaaaaaa.bbbbbbb([\n        (\'xxxxxxxxxx\', \'yyyyyy\',\n         \'Heaven hath no wrath like love to hatred turned. Nor hell a fury like a woman scorned.\'),\n        (\'xxxxxxx\', \'yyyyyyyyyyy\', "To the last I grapple with thee. From hell\'s heart I stab at thee. For hate\'s sake I spit my last breath at thee!")])\n# fmt: on\n'
        expected_output = '# fmt: off\ndef f():\n    aaaaaaaaaaa.bbbbbbb([\n        (\'xxxxxxxxxx\', \'yyyyyy\',\n         \'Heaven hath no wrath like love to hatred turned. Nor hell a fury like a woman scorned.\'),\n        (\'xxxxxxx\', \'yyyyyyyyyyy\', "To the last I grapple with thee. From hell\'s heart I stab at thee. For hate\'s sake I spit my last breath at thee!")])\n# fmt: on\n'
        with autopep8_context(test_code) as result:
            self.assertEqual(expected_output, result)

    def test_fmt_disable_without_reenable(self):
        if False:
            while True:
                i = 10
        test_code = '# fmt: off\nprint(123)\n'
        expected_output = '# fmt: off\nprint(123)\n'
        with autopep8_context(test_code) as result:
            self.assertEqual(expected_output, result)

    def test_fmt_disable_with_double_reenable(self):
        if False:
            while True:
                i = 10
        test_code = '# fmt: off\nprint( 123 )\n# fmt: on\nprint( 123 )\n# fmt: on\nprint( 123 )\n'
        expected_output = '# fmt: off\nprint( 123 )\n# fmt: on\nprint(123)\n# fmt: on\nprint(123)\n'
        with autopep8_context(test_code) as result:
            self.assertEqual(expected_output, result)

    def test_fmt_double_disable_and_reenable(self):
        if False:
            while True:
                i = 10
        test_code = '# fmt: off\nprint( 123 )\n# fmt: off\nprint( 123 )\n# fmt: on\nprint( 123 )\n'
        expected_output = '# fmt: off\nprint( 123 )\n# fmt: off\nprint( 123 )\n# fmt: on\nprint(123)\n'
        with autopep8_context(test_code) as result:
            self.assertEqual(expected_output, result)

    def test_fmt_multi_disable_and_reenable(self):
        if False:
            print('Hello World!')
        test_code = 'fix=1\n# fmt: off\nskip=1\n# fmt: on\nfix=2\n# fmt: off\nskip=2\n# fmt: on\nfix=3\n'
        expected_output = 'fix = 1\n# fmt: off\nskip=1\n# fmt: on\nfix = 2\n# fmt: off\nskip=2\n# fmt: on\nfix = 3\n'
        with autopep8_context(test_code) as result:
            self.assertEqual(expected_output, result)

    def test_fmt_multi_disable_complex(self):
        if False:
            while True:
                i = 10
        test_code = 'fix=1\n# fmt: off\nskip=1\n# fmt: off\nfix=2\n# fmt: off\nskip=2\n# fmt: on\nfix=3\n'
        expected_output = 'fix = 1\n# fmt: off\nskip=1\n# fmt: off\nfix=2\n# fmt: off\nskip=2\n# fmt: on\nfix = 3\n'
        with autopep8_context(test_code) as result:
            self.assertEqual(expected_output, result)

    def test_fmt_multi_disable_complex_multi(self):
        if False:
            return 10
        test_code = 'fix=1\n# fmt: off\nskip=1\n# fmt: off\nfix=2\n# fmt: on\nfix=22\n# fmt: on\nfix=222\n# fmt: off\nskip=2\n# fmt: on\nfix=3\n'
        expected_output = 'fix = 1\n# fmt: off\nskip=1\n# fmt: off\nfix=2\n# fmt: on\nfix = 22\n# fmt: on\nfix = 222\n# fmt: off\nskip=2\n# fmt: on\nfix = 3\n'
        with autopep8_context(test_code) as result:
            self.assertEqual(expected_output, result)

    def test_general_disable(self):
        if False:
            print('Hello World!')
        test_code = "# fmt: off\n\nimport math, sys;\n\ndef example1():\n    # This is a long comment. This should be wrapped to fit within 72 characters.\n    some_tuple=(   1,2, 3,'a'  );\n    some_variable={'long':'Long code lines should be wrapped within 79 characters.',\n    'other':[math.pi, 100,200,300,9876543210,'This is a long string that goes on'],\n    'more':{'inner':'This whole logical line should be wrapped.',some_tuple:[1,\n    20,300,40000,500000000,60000000000000000]}}\n    return (some_tuple, some_variable)\ndef example2(): return {'has_key() is deprecated':True}.has_key(\n    {'f':2}.has_key(''));\nclass Example3(   object ):\n    def __init__    ( self, bar ):\n    # Comments should have a space after the hash.\n    if bar : bar+=1;  bar=bar* bar   ; return bar\n    else:\n        some_string = '''\n                    Indentation in multiline strings should not be touched.\nOnly actual code should be reindented.\n'''\n        return (sys.path, some_string)\n# fmt: on\n\nimport math, sys;\n\ndef example1():\n    # This is a long comment. This should be wrapped to fit within 72 characters.\n    some_tuple=(   1,2, 3,'a'  );\n    some_variable={'long':'Long code lines should be wrapped within 79 characters.',\n    'other':[math.pi, 100,200,300,9876543210,'This is a long string that goes on'],\n    'more':{'inner':'This whole logical line should be wrapped.',some_tuple:[1,\n    20,300,40000,500000000,60000000000000000]}}\n    return (some_tuple, some_variable)\ndef example2(): return {'has_key() is deprecated':True}.has_key(\n    {'f':2}.has_key(''));\nclass Example3(   object ):\n    def __init__    ( self, bar ):\n    # Comments should have a space after the hash.\n    if bar : bar+=1;  bar=bar* bar   ; return bar\n    else:\n        some_string = '''\n                    Indentation in multiline strings should not be touched.\nOnly actual code should be reindented.\n'''\n        return (sys.path, some_string)\n\n\n"
        expected_output = "# fmt: off\n\nimport sys\nimport math\nimport math, sys;\n\ndef example1():\n    # This is a long comment. This should be wrapped to fit within 72 characters.\n    some_tuple=(   1,2, 3,'a'  );\n    some_variable={'long':'Long code lines should be wrapped within 79 characters.',\n    'other':[math.pi, 100,200,300,9876543210,'This is a long string that goes on'],\n    'more':{'inner':'This whole logical line should be wrapped.',some_tuple:[1,\n    20,300,40000,500000000,60000000000000000]}}\n    return (some_tuple, some_variable)\ndef example2(): return {'has_key() is deprecated':True}.has_key(\n    {'f':2}.has_key(''));\nclass Example3(   object ):\n    def __init__    ( self, bar ):\n    # Comments should have a space after the hash.\n    if bar : bar+=1;  bar=bar* bar   ; return bar\n    else:\n        some_string = '''\n                    Indentation in multiline strings should not be touched.\nOnly actual code should be reindented.\n'''\n        return (sys.path, some_string)\n# fmt: on\n\n\ndef example1():\n    # This is a long comment. This should be wrapped to fit within 72 characters.\n    some_tuple = (1, 2, 3, 'a')\n    some_variable = {'long': 'Long code lines should be wrapped within 79 characters.',\n                     'other': [math.pi, 100, 200, 300, 9876543210, 'This is a long string that goes on'],\n                     'more': {'inner': 'This whole logical line should be wrapped.', some_tuple: [1,\n                                                                                                  20, 300, 40000, 500000000, 60000000000000000]}}\n    return (some_tuple, some_variable)\n\n\ndef example2(): return {'has_key() is deprecated': True}.has_key(\n    {'f': 2}.has_key(''))\n\n\nclass Example3(object):\n    def __init__(self, bar):\n        # Comments should have a space after the hash.\n    if bar:\n        bar += 1\n        bar = bar * bar\n        return bar\n    else:\n        some_string = '''\n                    Indentation in multiline strings should not be touched.\nOnly actual code should be reindented.\n'''\n        return (sys.path, some_string)\n"
        with autopep8_context(test_code) as result:
            self.assertEqual(expected_output, result)

class UtilityFunctionTests(unittest.TestCase):

    def test_get_module_imports(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'import os\nimport sys\n\nif True:\n    print(1)\n'
        target_line_index = 8
        result = get_module_imports_on_top_of_file(line.splitlines(), target_line_index)
        self.assertEqual(result, 0)

    def test_get_module_imports_case_of_autopep8(self):
        if False:
            for i in range(10):
                print('nop')
        line = "#!/usr/bin/python\n\n# comment\n# comment\n\n'''this module ...\n\nthis module ...\n'''\n\nimport os\nimport sys\n\nif True:\n    print(1)\n"
        target_line_index = 11
        result = get_module_imports_on_top_of_file(line.splitlines(), target_line_index)
        self.assertEqual(result, 10)

class CommandLineTests(unittest.TestCase):
    maxDiff = None

    def test_e122_and_e302_with_backslash(self):
        if False:
            while True:
                i = 10
        line = 'import sys\n\\\ndef f():\n    pass\n'
        fixed = 'import sys\n\n\n\\\ndef f():\n    pass\n'
        with autopep8_subprocess(line, [], timeout=3) as (result, retcode):
            self.assertEqual(fixed, result)
            self.assertEqual(retcode, autopep8.EXIT_CODE_OK)

    def test_diff(self):
        if False:
            i = 10
            return i + 15
        line = "'abc'  \n"
        fixed = "-'abc'  \n+'abc'\n"
        with autopep8_subprocess(line, ['--diff']) as (result, retcode):
            self.assertEqual(fixed, '\n'.join(result.split('\n')[3:]))
            self.assertEqual(retcode, autopep8.EXIT_CODE_OK)

    def test_diff_with_exit_code_option(self):
        if False:
            i = 10
            return i + 15
        line = "'abc'  \n"
        fixed = "-'abc'  \n+'abc'\n"
        with autopep8_subprocess(line, ['--diff', '--exit-code']) as (result, retcode):
            self.assertEqual(fixed, '\n'.join(result.split('\n')[3:]))
            self.assertEqual(retcode, autopep8.EXIT_CODE_EXISTS_DIFF)

    def test_non_diff_with_exit_code_option(self):
        if False:
            for i in range(10):
                print('nop')
        line = "'abc'\n"
        with autopep8_subprocess(line, ['--diff', '--exit-code']) as (result, retcode):
            self.assertEqual('', '\n'.join(result.split('\n')[3:]))
            self.assertEqual(retcode, autopep8.EXIT_CODE_OK)

    def test_non_diff_with_exit_code_and_jobs_options(self):
        if False:
            while True:
                i = 10
        line = "'abc'\n"
        with autopep8_subprocess(line, ['-j0', '--diff', '--exit-code']) as (result, retcode):
            self.assertEqual('', '\n'.join(result.split('\n')[3:]))
            self.assertEqual(retcode, autopep8.EXIT_CODE_OK)

    def test_diff_with_empty_file(self):
        if False:
            return 10
        with autopep8_subprocess('', ['--diff']) as (result, retcode):
            self.assertEqual('\n'.join(result.split('\n')[3:]), '')
            self.assertEqual(retcode, autopep8.EXIT_CODE_OK)

    def test_diff_with_nonexistent_file(self):
        if False:
            i = 10
            return i + 15
        p = Popen(list(AUTOPEP8_CMD_TUPLE) + ['--diff', 'non_existent_file'], stdout=PIPE, stderr=PIPE)
        error = p.communicate()[1].decode('utf-8')
        self.assertIn('non_existent_file', error)

    def test_diff_with_standard_in(self):
        if False:
            while True:
                i = 10
        p = Popen(list(AUTOPEP8_CMD_TUPLE) + ['--diff', '-'], stdout=PIPE, stderr=PIPE)
        error = p.communicate()[1].decode('utf-8')
        self.assertIn('cannot', error)

    def test_indent_size_is_zero(self):
        if False:
            print('Hello World!')
        line = "'abc'\n"
        with autopep8_subprocess(line, ['--indent-size=0']) as (result, retcode):
            self.assertEqual(retcode, autopep8.EXIT_CODE_ARGPARSE_ERROR)

    def test_exit_code_with_io_error(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'import sys\ndef a():\n    print(1)\n'
        with readonly_temporary_file_context(line) as filename:
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + ['--in-place', filename], stdout=PIPE, stderr=PIPE)
            p.communicate()
            self.assertEqual(p.returncode, autopep8.EXIT_CODE_ERROR)

    def test_pep8_passes(self):
        if False:
            i = 10
            return i + 15
        line = "'abc'  \n"
        fixed = "'abc'\n"
        with autopep8_subprocess(line, ['--pep8-passes', '0']) as (result, retcode):
            self.assertEqual(fixed, result)
            self.assertEqual(retcode, autopep8.EXIT_CODE_OK)

    def test_pep8_ignore(self):
        if False:
            for i in range(10):
                print('nop')
        line = "'abc'  \n"
        with autopep8_subprocess(line, ['--ignore=E,W']) as (result, retcode):
            self.assertEqual(line, result)
            self.assertEqual(retcode, autopep8.EXIT_CODE_OK)

    def test_pep8_ignore_should_handle_trailing_comma_gracefully(self):
        if False:
            while True:
                i = 10
        line = "'abc'  \n"
        fixed = "'abc'\n"
        with autopep8_subprocess(line, ['--ignore=,']) as (result, retcode):
            self.assertEqual(fixed, result)
            self.assertEqual(retcode, autopep8.EXIT_CODE_OK)

    def test_help(self):
        if False:
            return 10
        p = Popen(list(AUTOPEP8_CMD_TUPLE) + ['-h'], stdout=PIPE)
        self.assertIn('usage:', p.communicate()[0].decode('utf-8').lower())

    @unittest.skipIf(sys.version_info >= (3, 12), 'not detech in Python3.12+')
    def test_verbose(self):
        if False:
            i = 10
            return i + 15
        line = 'bad_syntax)'
        with temporary_file_context(line) as filename:
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename, '-vvv'], stdout=PIPE, stderr=PIPE)
            verbose_error = p.communicate()[1].decode('utf-8')
        self.assertIn("'fix_e901' is not defined", verbose_error)

    def test_verbose_diff(self):
        if False:
            i = 10
            return i + 15
        line = '+'.join(100 * ['323424234234'])
        with temporary_file_context(line) as filename:
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename, '-vvvv', '--diff'], stdout=PIPE, stderr=PIPE)
            verbose_error = p.communicate()[1].decode('utf-8')
        self.assertIn('------------', verbose_error)

    def test_verbose_with_select_e702(self):
        if False:
            print('Hello World!')
        line = 'for i in range(3):\n    if i == 1: print(i); continue\n    print(i)\n'
        with temporary_file_context(line) as filename:
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename, '-vvv', '--select=E702'], stdout=PIPE, stderr=PIPE)
            verbose_error = p.communicate()[1].decode('utf-8')
        self.assertIn(' with other compound statements', verbose_error)

    def test_in_place(self):
        if False:
            print('Hello World!')
        line = "'abc'  \n"
        fixed = "'abc'\n"
        with temporary_file_context(line) as filename:
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename, '--in-place'])
            p.wait()
            with open(filename) as f:
                self.assertEqual(fixed, f.read())
            self.assertEqual(p.returncode, autopep8.EXIT_CODE_OK)

    def test_in_place_no_modifications_no_writes(self):
        if False:
            i = 10
            return i + 15
        with temporary_file_context('import os\n') as filename:
            os.chmod(filename, 292)
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename, '--in-place'], stderr=PIPE)
            (_, err) = p.communicate()
            self.assertEqual(err, b'')
            self.assertEqual(p.returncode, autopep8.EXIT_CODE_OK)

    def test_in_place_no_modifications_no_writes_with_empty_file(self):
        if False:
            print('Hello World!')
        with temporary_file_context('') as filename:
            os.chmod(filename, 292)
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename, '--in-place'], stderr=PIPE)
            (_, err) = p.communicate()
            self.assertEqual(err, b'')
            self.assertEqual(p.returncode, autopep8.EXIT_CODE_OK)

    def test_in_place_with_w292(self):
        if False:
            i = 10
            return i + 15
        line = 'import os'
        fixed = 'import os\n'
        with temporary_file_context(line) as filename:
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename, '--in-place'])
            p.wait()
            with open(filename) as f:
                self.assertEqual(fixed, f.read())

    def test_in_place_with_exit_code_option(self):
        if False:
            for i in range(10):
                print('nop')
        line = "'abc'  \n"
        fixed = "'abc'\n"
        with temporary_file_context(line) as filename:
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename, '--in-place', '--exit-code'])
            p.wait()
            with open(filename) as f:
                self.assertEqual(fixed, f.read())
            self.assertEqual(p.returncode, autopep8.EXIT_CODE_EXISTS_DIFF)

    def test_in_place_with_exit_code_option_with_w391(self):
        if False:
            for i in range(10):
                print('nop')
        line = '\n\n\n'
        fixed = ''
        with temporary_file_context(line) as filename:
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename, '--in-place', '--exit-code'])
            p.wait()
            with open(filename) as f:
                self.assertEqual(fixed, f.read())
            self.assertEqual(p.returncode, autopep8.EXIT_CODE_EXISTS_DIFF)

    def test_parallel_jobs(self):
        if False:
            while True:
                i = 10
        line = "'abc'  \n"
        fixed = "'abc'\n"
        with temporary_file_context(line) as filename_a:
            with temporary_file_context(line) as filename_b:
                p = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename_a, filename_b, '--jobs=3', '--in-place'])
                p.wait()
                with open(filename_a) as f:
                    self.assertEqual(fixed, f.read())
                with open(filename_b) as f:
                    self.assertEqual(fixed, f.read())

    def test_parallel_jobs_with_diff_option(self):
        if False:
            while True:
                i = 10
        line = "'abc'  \n"
        with temporary_file_context(line) as filename_a:
            with temporary_file_context(line) as filename_b:
                files = list(set([filename_a, filename_b]))
                p = Popen(list(AUTOPEP8_CMD_TUPLE) + files + ['--jobs=3', '--diff'], stdout=PIPE)
                p.wait()
                output = p.stdout.read().decode()
                output = output.replace('\r\n', '\n')
                p.stdout.close()
                actual_diffs = []
                for filename in files:
                    actual_diffs.append("--- original/{filename}\n+++ fixed/{filename}\n@@ -1 +1 @@\n-'abc'  {blank}\n+'abc'\n".format(filename=filename, blank=''))
                self.assertEqual(p.returncode, autopep8.EXIT_CODE_OK)
                for actual_diff in actual_diffs:
                    self.assertIn(actual_diff, output)

    def test_parallel_jobs_with_inplace_option_and_io_error(self):
        if False:
            while True:
                i = 10
        temp_directory = mkdtemp(dir='.')
        try:
            file_a = os.path.join(temp_directory, 'a.py')
            with open(file_a, 'w') as output:
                output.write("'abc'  \n")
            os.chmod(file_a, stat.S_IRUSR)
            os.mkdir(os.path.join(temp_directory, 'd'))
            file_b = os.path.join(temp_directory, 'd', 'b.py')
            with open(file_b, 'w') as output:
                output.write('123  \n')
            os.chmod(file_b, stat.S_IRUSR)
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [temp_directory, '--recursive', '--in-place'], stdout=PIPE, stderr=PIPE)
            p.communicate()[0].decode('utf-8')
            self.assertEqual(p.returncode, autopep8.EXIT_CODE_ERROR)
        finally:
            shutil.rmtree(temp_directory)

    def test_parallel_jobs_with_automatic_cpu_count(self):
        if False:
            return 10
        line = "'abc'  \n"
        fixed = "'abc'\n"
        with temporary_file_context(line) as filename_a:
            with temporary_file_context(line) as filename_b:
                p = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename_a, filename_b, '--jobs=0', '--in-place'])
                p.wait()
                with open(filename_a) as f:
                    self.assertEqual(fixed, f.read())
                with open(filename_b) as f:
                    self.assertEqual(fixed, f.read())

    def test_in_place_with_empty_file(self):
        if False:
            i = 10
            return i + 15
        line = ''
        with temporary_file_context(line) as filename:
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename, '--in-place'])
            p.wait()
            self.assertEqual(0, p.returncode)
            with open(filename) as f:
                self.assertEqual(f.read(), line)

    def test_in_place_and_diff(self):
        if False:
            return 10
        line = "'abc'  \n"
        with temporary_file_context(line) as filename:
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename, '--in-place', '--diff'], stderr=PIPE)
            result = p.communicate()[1].decode('utf-8')
        self.assertIn('--in-place and --diff are mutually exclusive', result)

    def test_recursive(self):
        if False:
            for i in range(10):
                print('nop')
        temp_directory = mkdtemp(dir='.')
        try:
            with open(os.path.join(temp_directory, 'a.py'), 'w') as output:
                output.write("'abc'  \n")
            os.mkdir(os.path.join(temp_directory, 'd'))
            with open(os.path.join(temp_directory, 'd', 'b.py'), 'w') as output:
                output.write('123  \n')
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [temp_directory, '--recursive', '--diff'], stdout=PIPE)
            result = p.communicate()[0].decode('utf-8')
            self.assertEqual("-'abc'  \n+'abc'", '\n'.join(result.split('\n')[3:5]))
            self.assertEqual('-123  \n+123', '\n'.join(result.split('\n')[8:10]))
        finally:
            shutil.rmtree(temp_directory)

    def test_recursive_should_not_crash_on_unicode_filename(self):
        if False:
            print('Hello World!')
        temp_directory = mkdtemp(dir='.')
        try:
            for filename in ['x.py', 'é.py', 'é.txt']:
                with open(os.path.join(temp_directory, filename), 'w'):
                    pass
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [temp_directory, '--recursive', '--diff'], stdout=PIPE)
            self.assertFalse(p.communicate()[0])
            self.assertEqual(0, p.returncode)
        finally:
            shutil.rmtree(temp_directory)

    def test_recursive_should_ignore_hidden(self):
        if False:
            return 10
        temp_directory = mkdtemp(dir='.')
        temp_subdirectory = mkdtemp(prefix='.', dir=temp_directory)
        try:
            with open(os.path.join(temp_subdirectory, 'a.py'), 'w') as output:
                output.write("'abc'  \n")
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [temp_directory, '--recursive', '--diff'], stdout=PIPE)
            result = p.communicate()[0].decode('utf-8')
            self.assertEqual(0, p.returncode)
            self.assertEqual('', result)
        finally:
            shutil.rmtree(temp_directory)

    def test_exclude(self):
        if False:
            print('Hello World!')
        temp_directory = mkdtemp(dir='.')
        try:
            with open(os.path.join(temp_directory, 'a.py'), 'w') as output:
                output.write("'abc'  \n")
            os.mkdir(os.path.join(temp_directory, 'd'))
            with open(os.path.join(temp_directory, 'd', 'b.py'), 'w') as output:
                output.write('123  \n')
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [temp_directory, '--recursive', '--exclude=a*', '--diff'], stdout=PIPE)
            result = p.communicate()[0].decode('utf-8')
            self.assertNotIn('abc', result)
            self.assertIn('123', result)
        finally:
            shutil.rmtree(temp_directory)

    def test_exclude_with_directly_file_args(self):
        if False:
            i = 10
            return i + 15
        temp_directory = mkdtemp(dir='.')
        try:
            filepath_a = os.path.join(temp_directory, 'a.py')
            with open(filepath_a, 'w') as output:
                output.write("'abc'  \n")
            os.mkdir(os.path.join(temp_directory, 'd'))
            filepath_b = os.path.join(temp_directory, 'd', 'b.py')
            with open(os.path.join(filepath_b), 'w') as output:
                output.write('123  \n')
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + ['--exclude=*/a.py', '--diff', filepath_a, filepath_b], stdout=PIPE)
            result = p.communicate()[0].decode('utf-8')
            self.assertNotIn('abc', result)
            self.assertIn('123', result)
        finally:
            shutil.rmtree(temp_directory)

    def test_invalid_option_combinations(self):
        if False:
            for i in range(10):
                print('nop')
        line = "'abc'  \n"
        with temporary_file_context(line) as filename:
            for options in [['--recursive', filename], ['--jobs=2', filename], ['--max-line-length=0', filename], [], ['-', '--in-place'], ['-', '--recursive'], ['-', filename], ['--line-range', '0', '2', filename], ['--line-range', '2', '1', filename], ['--line-range', '-1', '-1', filename]]:
                p = Popen(list(AUTOPEP8_CMD_TUPLE) + options, stderr=PIPE)
                result = p.communicate()[1].decode('utf-8')
                self.assertNotEqual(0, p.returncode, msg=str(options))
                self.assertTrue(len(result))

    def test_list_fixes(self):
        if False:
            return 10
        with autopep8_subprocess('', options=['--list-fixes']) as (result, retcode):
            self.assertIn('E121', result)
            self.assertEqual(retcode, autopep8.EXIT_CODE_OK)

    def test_fixpep8_class_constructor(self):
        if False:
            return 10
        line = 'print(1)\nprint(2)\n'
        with temporary_file_context(line) as filename:
            pep8obj = autopep8.FixPEP8(filename, None)
        self.assertEqual(''.join(pep8obj.source), line)

    def test_inplace_with_multi_files(self):
        if False:
            while True:
                i = 10
        exception = None
        with disable_stderr():
            try:
                autopep8.parse_args(['test.py', 'dummy.py'])
            except SystemExit as e:
                exception = e
        self.assertTrue(exception)
        self.assertEqual(exception.code, autopep8.EXIT_CODE_ARGPARSE_ERROR)

    def test_standard_out_should_use_native_line_ending(self):
        if False:
            print('Hello World!')
        line = '1\r\n2\r\n3\r\n'
        with temporary_file_context(line) as filename:
            process = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename], stdout=PIPE)
            self.assertEqual(os.linesep.join(['1', '2', '3', '']), process.communicate()[0].decode('utf-8'))

    def test_standard_out_should_use_native_line_ending_with_cr_input(self):
        if False:
            return 10
        line = '1\r2\r3\r'
        with temporary_file_context(line) as filename:
            process = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename], stdout=PIPE)
            self.assertEqual(os.linesep.join(['1', '2', '3', '']), process.communicate()[0].decode('utf-8'))

    def test_standard_in(self):
        if False:
            print('Hello World!')
        line = 'print( 1 )\n'
        fixed = 'print(1)' + os.linesep
        process = Popen(list(AUTOPEP8_CMD_TUPLE) + ['-'], stdout=PIPE, stdin=PIPE)
        self.assertEqual(fixed, process.communicate(line.encode('utf-8'))[0].decode('utf-8'))

    def test_exit_code_should_be_set_when_standard_in(self):
        if False:
            return 10
        line = 'print( 1 )\n'
        process = Popen(list(AUTOPEP8_CMD_TUPLE) + ['--exit-code', '-'], stdout=PIPE, stdin=PIPE)
        process.communicate(line.encode('utf-8'))[0].decode('utf-8')
        self.assertEqual(process.returncode, autopep8.EXIT_CODE_EXISTS_DIFF)

class ConfigurationTests(unittest.TestCase):

    def test_local_config(self):
        if False:
            for i in range(10):
                print('nop')
        args = autopep8.parse_args([os.path.join(FAKE_CONFIGURATION, 'foo.py'), '--global-config={}'.format(os.devnull), '-vvv'], apply_config=True)
        self.assertEqual(args.indent_size, 2)

    def test_config_override(self):
        if False:
            i = 10
            return i + 15
        args = autopep8.parse_args([os.path.join(FAKE_CONFIGURATION, 'foo.py'), '--indent-size=7'], apply_config=True)
        self.assertEqual(args.indent_size, 7)

    def test_config_false_with_local(self):
        if False:
            print('Hello World!')
        args = autopep8.parse_args([os.path.join(FAKE_CONFIGURATION, 'foo.py'), '--global-config=False'], apply_config=True)
        self.assertEqual(args.global_config, 'False')
        self.assertEqual(args.indent_size, 2)

    def test_config_false_with_local_space(self):
        if False:
            while True:
                i = 10
        args = autopep8.parse_args([os.path.join(FAKE_CONFIGURATION, 'foo.py'), '--global-config', 'False'], apply_config=True)
        self.assertEqual(args.global_config, 'False')
        self.assertEqual(args.indent_size, 2)

    def test_local_pycodestyle_config_line_length(self):
        if False:
            print('Hello World!')
        args = autopep8.parse_args([os.path.join(FAKE_PYCODESTYLE_CONFIGURATION, 'foo.py'), '--global-config={}'.format(os.devnull)], apply_config=True)
        self.assertEqual(args.max_line_length, 40)

    def test_config_false_with_local_autocomplete(self):
        if False:
            print('Hello World!')
        args = autopep8.parse_args([os.path.join(FAKE_CONFIGURATION, 'foo.py'), '--g', 'False'], apply_config=True)
        self.assertEqual(args.global_config, 'False')
        self.assertEqual(args.indent_size, 2)

    def test_config_false_without_local(self):
        if False:
            for i in range(10):
                print('nop')
        args = autopep8.parse_args(['/nowhere/foo.py', '--global-config={}'.format(os.devnull)], apply_config=True)
        self.assertEqual(args.indent_size, 4)

    def test_global_config_with_locals(self):
        if False:
            print('Hello World!')
        with temporary_file_context('[pep8]\nindent-size=3\n') as filename:
            args = autopep8.parse_args([os.path.join(FAKE_CONFIGURATION, 'foo.py'), '--global-config={}'.format(filename)], apply_config=True)
            self.assertEqual(args.indent_size, 2)

    def test_global_config_ignore_locals(self):
        if False:
            return 10
        with temporary_file_context('[pep8]\nindent-size=3\n') as filename:
            args = autopep8.parse_args([os.path.join(FAKE_CONFIGURATION, 'foo.py'), '--global-config={}'.format(filename), '--ignore-local-config'], apply_config=True)
            self.assertEqual(args.indent_size, 3)

    def test_global_config_without_locals(self):
        if False:
            return 10
        with temporary_file_context('[pep8]\nindent-size=3\n') as filename:
            args = autopep8.parse_args(['/nowhere/foo.py', '--global-config={}'.format(filename)], apply_config=True)
            self.assertEqual(args.indent_size, 3)

    def test_config_local_int_value(self):
        if False:
            for i in range(10):
                print('nop')
        with temporary_file_context('[pep8]\naggressive=1\n') as filename:
            args = autopep8.parse_args([os.path.join(FAKE_CONFIGURATION, 'foo.py'), '--global-config={}'.format(filename)], apply_config=True)
            self.assertEqual(args.aggressive, 1)

    def test_config_local_inclue_invalid_key(self):
        if False:
            i = 10
            return i + 15
        configstr = '[pep8]\ncount=True\naggressive=1\n'
        with temporary_file_context(configstr) as filename:
            args = autopep8.parse_args([os.path.join(FAKE_CONFIGURATION, 'foo.py'), '--global-config={}'.format(filename)], apply_config=True)
            self.assertEqual(args.aggressive, 1)

    def test_pyproject_toml_config_local_int_value(self):
        if False:
            return 10
        with temporary_file_context('[tool.autopep8]\naggressive=2\n') as filename:
            args = autopep8.parse_args([os.path.join(FAKE_CONFIGURATION, 'foo.py'), '--ignore-local-config', '--global-config={}'.format(filename)], apply_config=True)
            self.assertEqual(args.aggressive, 2)

class ConfigurationFileTests(unittest.TestCase):

    def test_pyproject_toml_with_flake8_config(self):
        if False:
            return 10
        'override to flake8 config'
        line = 'a =  1\n'
        dot_flake8 = '[pep8]\naggressive=0\n'
        pyproject_toml = '[tool.autopep8]\naggressvie=2\nignore="E,W"\n'
        with temporary_project_directory() as dirname:
            with open(os.path.join(dirname, 'pyproject.toml'), 'w') as fp:
                fp.write(pyproject_toml)
            with open(os.path.join(dirname, '.flake8'), 'w') as fp:
                fp.write(dot_flake8)
            target_filename = os.path.join(dirname, 'foo.py')
            with open(target_filename, 'w') as fp:
                fp.write(line)
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [target_filename], stdout=PIPE)
            self.assertEqual(p.communicate()[0].decode('utf-8'), line)
            self.assertEqual(p.returncode, autopep8.EXIT_CODE_OK)

    def test_pyproject_toml_with_verbose_option(self):
        if False:
            for i in range(10):
                print('nop')
        'override to flake8 config'
        line = 'a =  1\n'
        verbose_line = 'enable pyproject.toml config: key=ignore, value=E,W\n'
        pyproject_toml = '[tool.autopep8]\naggressvie=2\nignore="E,W"\n'
        with temporary_project_directory() as dirname:
            with open(os.path.join(dirname, 'pyproject.toml'), 'w') as fp:
                fp.write(pyproject_toml)
            target_filename = os.path.join(dirname, 'foo.py')
            with open(target_filename, 'w') as fp:
                fp.write(line)
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [target_filename, '-vvv'], stdout=PIPE)
            output = p.communicate()[0].decode('utf-8')
            self.assertTrue(line in output)
            self.assertTrue(verbose_line in output)
            self.assertEqual(p.returncode, autopep8.EXIT_CODE_OK)

    def test_pyproject_toml_with_iterable_value(self):
        if False:
            while True:
                i = 10
        line = 'a =  1\n'
        pyproject_toml = '[tool.autopep8]\naggressvie=2\nignore=["E","W"]\n'
        with temporary_project_directory() as dirname:
            with open(os.path.join(dirname, 'pyproject.toml'), 'w') as fp:
                fp.write(pyproject_toml)
            target_filename = os.path.join(dirname, 'foo.py')
            with open(target_filename, 'w') as fp:
                fp.write(line)
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [target_filename], stdout=PIPE)
            output = p.communicate()[0].decode('utf-8')
            self.assertTrue(line in output)
            self.assertEqual(p.returncode, autopep8.EXIT_CODE_OK)

    def test_setupcfg_with_flake8_config(self):
        if False:
            i = 10
            return i + 15
        line = 'a =  1\n'
        fixed = 'a = 1\n'
        setupcfg_flake8 = '[flake8]\njobs=auto\n'
        with temporary_project_directory() as dirname:
            with open(os.path.join(dirname, 'setup.cfg'), 'w') as fp:
                fp.write(setupcfg_flake8)
            target_filename = os.path.join(dirname, 'foo.py')
            with open(target_filename, 'w') as fp:
                fp.write(line)
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [target_filename, '-v'], stdout=PIPE)
            output = p.communicate()[0].decode('utf-8')
            self.assertTrue(fixed in output)
            self.assertTrue('ignore config: jobs=auto' in output)
            self.assertEqual(p.returncode, autopep8.EXIT_CODE_OK)

    def test_setupcfg_with_pycodestyle_config(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'a =  1\n'
        fixed = 'a = 1\n'
        setupcfg_flake8 = '[pycodestyle]\ndiff=True\nignore="E,W"\n'
        with temporary_project_directory() as dirname:
            with open(os.path.join(dirname, 'setup.cfg'), 'w') as fp:
                fp.write(setupcfg_flake8)
            target_filename = os.path.join(dirname, 'foo.py')
            with open(target_filename, 'w') as fp:
                fp.write(line)
            p = Popen(list(AUTOPEP8_CMD_TUPLE) + [target_filename, '-v'], stdout=PIPE)
            output = p.communicate()[0].decode('utf-8')
            self.assertTrue(fixed in output)
            self.assertEqual(p.returncode, autopep8.EXIT_CODE_OK)

class ExperimentalSystemTests(unittest.TestCase):
    maxDiff = None

    def test_e501_experimental_basic(self):
        if False:
            i = 10
            return i + 15
        line = 'print(111, 111, 111, 111, 222, 222, 222, 222, 222, 222, 222, 222, 222, 333, 333, 333, 333)\n'
        fixed = 'print(111, 111, 111, 111, 222, 222, 222, 222,\n      222, 222, 222, 222, 222, 333, 333, 333, 333)\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_commas_and_colons(self):
        if False:
            return 10
        line = "foobar = {'aaaaaaaaaaaa': 'bbbbbbbbbbbbbbbb', 'dddddd': 'eeeeeeeeeeeeeeee', 'ffffffffffff': 'gggggggg'}\n"
        fixed = "foobar = {'aaaaaaaaaaaa': 'bbbbbbbbbbbbbbbb',\n          'dddddd': 'eeeeeeeeeeeeeeee', 'ffffffffffff': 'gggggggg'}\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_inline_comments(self):
        if False:
            while True:
                i = 10
        line = "'                                                          '  # Long inline comments should be moved above.\nif True:\n    '                                                          '  # Long inline comments should be moved above.\n"
        fixed = "# Long inline comments should be moved above.\n'                                                          '\nif True:\n    # Long inline comments should be moved above.\n    '                                                          '\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_inline_comments_should_skip_multiline(self):
        if False:
            i = 10
            return i + 15
        line = "'''This should be left alone. -----------------------------------------------------\n\n'''  # foo\n\n'''This should be left alone. -----------------------------------------------------\n\n''' \\\n# foo\n\n'''This should be left alone. -----------------------------------------------------\n\n''' \\\n\\\n# foo\n"
        fixed = "'''This should be left alone. -----------------------------------------------------\n\n'''  # foo\n\n'''This should be left alone. -----------------------------------------------------\n\n'''  # foo\n\n'''This should be left alone. -----------------------------------------------------\n\n'''  # foo\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_inline_comments_should_skip_keywords(self):
        if False:
            return 10
        line = "'                                                          '  # noqa Long inline comments should be moved above.\nif True:\n    '                                                          '  # pylint: disable-msgs=E0001\n    '                                                          '  # pragma: no cover\n    '                                                          '  # pragma: no cover\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(line, result)

    def test_e501_experimental_with_inline_comments_should_skip_edge_cases(self):
        if False:
            i = 10
            return i + 15
        line = "if True:\n    x = \\\n        '                                                          '  # Long inline comments should be moved above.\n"
        fixed = "if True:\n    # Long inline comments should be moved above.\n    x = '                                                          '\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_basic_should_prefer_balanced_brackets(self):
        if False:
            i = 10
            return i + 15
        line = 'if True:\n    reconstructed = iradon(radon(image), filter="ramp", interpolation="nearest")\n'
        fixed = 'if True:\n    reconstructed = iradon(\n        radon(image),\n        filter="ramp", interpolation="nearest")\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_very_long_line(self):
        if False:
            return 10
        line = 'x = [3244234243234, 234234234324, 234234324, 23424234, 234234234, 234234, 234243, 234243, 234234234324, 234234324, 23424234, 234234234, 234234, 234243, 234243]\n'
        fixed = 'x = [3244234243234, 234234234324, 234234324, 23424234, 234234234, 234234, 234243,\n     234243, 234234234324, 234234324, 23424234, 234234234, 234234, 234243, 234243]\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_shorten_at_commas_skip(self):
        if False:
            i = 10
            return i + 15
        line = "parser.add_argument('source_corpus', help='corpus name/path relative to an nltk_data directory')\nparser.add_argument('target_corpus', help='corpus name/path relative to an nltk_data directory')\n"
        fixed = "parser.add_argument(\n    'source_corpus',\n    help='corpus name/path relative to an nltk_data directory')\nparser.add_argument(\n    'target_corpus',\n    help='corpus name/path relative to an nltk_data directory')\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_shorter_length(self):
        if False:
            print('Hello World!')
        line = "foooooooooooooooooo('abcdefghijklmnopqrstuvwxyz')\n"
        fixed = "foooooooooooooooooo(\n    'abcdefghijklmnopqrstuvwxyz')\n"
        with autopep8_context(line, options=['--max-line-length=40', '--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_indent(self):
        if False:
            print('Hello World!')
        line = '\ndef d():\n    print(111, 111, 111, 111, 222, 222, 222, 222, 222, 222, 222, 222, 222, 333, 333, 333, 333)\n'
        fixed = '\ndef d():\n    print(111, 111, 111, 111, 222, 222, 222, 222,\n          222, 222, 222, 222, 222, 333, 333, 333, 333)\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_alone_with_indentation(self):
        if False:
            i = 10
            return i + 15
        line = 'if True:\n    print(111, 111, 111, 111, 222, 222, 222, 222, 222, 222, 222, 222, 222, 333, 333, 333, 333)\n'
        fixed = 'if True:\n    print(111, 111, 111, 111, 222, 222, 222, 222,\n          222, 222, 222, 222, 222, 333, 333, 333, 333)\n'
        with autopep8_context(line, options=['--select=E501', '--experimental']) as result:
            self.assertEqual(fixed, result)

    @unittest.skip('Not sure why space is not removed anymore')
    def test_e501_experimental_alone_with_tuple(self):
        if False:
            while True:
                i = 10
        line = "fooooooooooooooooooooooooooooooo000000000000000000000000 = [1,\n                                                            ('TransferTime', 'FLOAT')\n                                                           ]\n"
        fixed = "fooooooooooooooooooooooooooooooo000000000000000000000000 = [\n    1, ('TransferTime', 'FLOAT')]\n"
        with autopep8_context(line, options=['--select=E501', '--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_should_not_try_to_break_at_every_paren_in_arithmetic(self):
        if False:
            return 10
        line = "term3 = w6 * c5 * (8.0 * psi4 * (11.0 - 24.0 * t2) - 28 * psi3 * (1 - 6.0 * t2) + psi2 * (1 - 32 * t2) - psi * (2.0 * t2) + t4) / 720.0\nthis_should_be_shortened = ('                                                                 ', '            ')\n"
        fixed = "term3 = w6 * c5 * (8.0 * psi4 * (11.0 - 24.0 * t2) - 28 * psi3 * (1 - 6.0 * t2) +\n                   psi2 * (1 - 32 * t2) - psi * (2.0 * t2) + t4) / 720.0\nthis_should_be_shortened = (\n    '                                                                 ',\n    '            ')\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_arithmetic_operator_with_indent(self):
        if False:
            print('Hello World!')
        line = 'def d():\n    111 + 111 + 111 + 111 + 111 + 222 + 222 + 222 + 222 + 222 + 222 + 222 + 222 + 222 + 333 + 333 + 333 + 333\n'
        fixed = 'def d():\n    111 + 111 + 111 + 111 + 111 + 222 + 222 + 222 + 222 + \\\n        222 + 222 + 222 + 222 + 222 + 333 + 333 + 333 + 333\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_more_complicated(self):
        if False:
            while True:
                i = 10
        line = "blahblah = os.environ.get('blahblah') or os.environ.get('blahblahblah') or os.environ.get('blahblahblahblah')\n"
        fixed = "blahblah = os.environ.get('blahblah') or os.environ.get(\n    'blahblahblah') or os.environ.get('blahblahblahblah')\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_skip_even_more_complicated(self):
        if False:
            while True:
                i = 10
        line = 'if True:\n    if True:\n        if True:\n            blah = blah.blah_blah_blah_bla_bl(blahb.blah, blah.blah,\n                                              blah=blah.label, blah_blah=blah_blah,\n                                              blah_blah2=blah_blah)\n'
        fixed = 'if True:\n    if True:\n        if True:\n            blah = blah.blah_blah_blah_bla_bl(\n                blahb.blah, blah.blah, blah=blah.label, blah_blah=blah_blah,\n                blah_blah2=blah_blah)\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_logical_fix(self):
        if False:
            while True:
                i = 10
        line = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx(aaaaaaaaaaaaaaaaaaaaaaa,\n                             bbbbbbbbbbbbbbbbbbbbbbbbbbbb, cccccccccccccccccccccccccccc, dddddddddddddddddddddddd)\n'
        fixed = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx(\n    aaaaaaaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbbbbbbbbbbbb,\n    cccccccccccccccccccccccccccc, dddddddddddddddddddddddd)\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_logical_fix_and_physical_fix(self):
        if False:
            return 10
        line = '# ------ ------------------------------------------------------------------------\nxxxxxxxxxxxxxxxxxxxxxxxxxxxx(aaaaaaaaaaaaaaaaaaaaaaa,\n                             bbbbbbbbbbbbbbbbbbbbbbbbbbbb, cccccccccccccccccccccccccccc, dddddddddddddddddddddddd)\n'
        fixed = '# ------ -----------------------------------------------------------------\nxxxxxxxxxxxxxxxxxxxxxxxxxxxx(\n    aaaaaaaaaaaaaaaaaaaaaaa,\n    bbbbbbbbbbbbbbbbbbbbbbbbbbbb,\n    cccccccccccccccccccccccccccc,\n    dddddddddddddddddddddddd)\n'
        with autopep8_context(line, options=['--experimental', '--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_logical_fix_and_adjacent_strings(self):
        if False:
            i = 10
            return i + 15
        line = 'print(\'a-----------------------\' \'b-----------------------\' \'c-----------------------\'\n      \'d-----------------------\'\'e\'"f"r"g")\n'
        fixed = 'print(\n    \'a-----------------------\'\n    \'b-----------------------\'\n    \'c-----------------------\'\n    \'d-----------------------\'\n    \'e\'\n    "f"\n    r"g")\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_multiple_lines(self):
        if False:
            while True:
                i = 10
        line = 'foo_bar_zap_bing_bang_boom(111, 111, 111, 111, 222, 222, 222, 222, 222, 222, 222, 222, 222, 333, 333,\n                           111, 111, 111, 111, 222, 222, 222, 222, 222, 222, 222, 222, 222, 333, 333)\n'
        fixed = 'foo_bar_zap_bing_bang_boom(\n    111, 111, 111, 111, 222, 222, 222, 222, 222, 222, 222, 222, 222, 333, 333,\n    111, 111, 111, 111, 222, 222, 222, 222, 222, 222, 222, 222, 222, 333, 333)\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_do_not_break_on_keyword(self):
        if False:
            print('Hello World!')
        line = "if True:\n    long_variable_name = tempfile.mkstemp(prefix='abcdefghijklmnopqrstuvwxyz0123456789')\n"
        fixed = "if True:\n    long_variable_name = tempfile.mkstemp(\n        prefix='abcdefghijklmnopqrstuvwxyz0123456789')\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_do_not_begin_line_with_comma(self):
        if False:
            print('Hello World!')
        line = 'def dummy():\n    if True:\n        if True:\n            if True:\n                object = ModifyAction( [MODIFY70.text, OBJECTBINDING71.text, COLON72.text], MODIFY70.getLine(), MODIFY70.getCharPositionInLine() )\n'
        fixed = 'def dummy():\n    if True:\n        if True:\n            if True:\n                object = ModifyAction(\n                    [MODIFY70.text, OBJECTBINDING71.text, COLON72.text],\n                    MODIFY70.getLine(),\n                    MODIFY70.getCharPositionInLine())\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_should_not_break_on_dot(self):
        if False:
            while True:
                i = 10
        line = 'if True:\n    if True:\n        raise xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(\'xxxxxxxxxxxxxxxxx "{d}" xxxxxxxxxxxxxx\'.format(d=\'xxxxxxxxxxxxxxx\'))\n'
        fixed = 'if True:\n    if True:\n        raise xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(\n            \'xxxxxxxxxxxxxxxxx "{d}" xxxxxxxxxxxxxx\'.format(\n                d=\'xxxxxxxxxxxxxxx\'))\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_comment(self):
        if False:
            i = 10
            return i + 15
        line = '123\nif True:\n    if True:\n        if True:\n            if True:\n                if True:\n                    if True:\n                        # This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n                        pass\n\n# http://foo.bar/abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-\n\n# The following is ugly commented-out code and should not be touched.\n#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx = 1\n'
        fixed = '123\nif True:\n    if True:\n        if True:\n            if True:\n                if True:\n                    if True:\n                        # This is a long comment that should be wrapped. I will\n                        # wrap it using textwrap to be within 72 characters.\n                        pass\n\n# http://foo.bar/abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-abc-\n\n# The following is ugly commented-out code and should not be touched.\n# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx = 1\n'
        with autopep8_context(line, options=['--experimental', '--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_comment_should_not_modify_docstring(self):
        if False:
            i = 10
            return i + 15
        line = 'def foo():\n    """\n                        # This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n    """\n'
        with autopep8_context(line, options=['--experimental', '--aggressive']) as result:
            self.assertEqual(line, result)

    def test_e501_experimental_should_only_modify_last_comment(self):
        if False:
            print('Hello World!')
        line = '123\nif True:\n    if True:\n        if True:\n            if True:\n                if True:\n                    if True:\n                        # This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n                        # 1. This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n                        # 2. This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n                        # 3. This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n'
        fixed = '123\nif True:\n    if True:\n        if True:\n            if True:\n                if True:\n                    if True:\n                        # This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n                        # 1. This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n                        # 2. This is a long comment that should be wrapped. I will wrap it using textwrap to be within 72 characters.\n                        # 3. This is a long comment that should be wrapped. I\n                        # will wrap it using textwrap to be within 72\n                        # characters.\n'
        with autopep8_context(line, options=['--experimental', '--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_should_not_interfere_with_non_comment(self):
        if False:
            return 10
        line = '\n"""\n# not actually a comment %d. 12345678901234567890, 12345678901234567890, 12345678901234567890.\n""" % (0,)\n'
        with autopep8_context(line, options=['--experimental', '--aggressive']) as result:
            self.assertEqual(line, result)

    def test_e501_experimental_should_cut_comment_pattern(self):
        if False:
            while True:
                i = 10
        line = '123\n# -- Useless lines ----------------------------------------------------------------------\n321\n'
        fixed = '123\n# -- Useless lines -------------------------------------------------------\n321\n'
        with autopep8_context(line, options=['--experimental', '--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_function_should_not_break_on_colon(self):
        if False:
            print('Hello World!')
        line = '\nclass Useless(object):\n\n    def _table_field_is_plain_widget(self, widget):\n        if widget.__class__ == Widget or\\\n                (widget.__class__ == WidgetMeta and Widget in widget.__bases__):\n            return True\n\n        return False\n'
        fixed = '\nclass Useless(object):\n\n    def _table_field_is_plain_widget(self, widget):\n        if widget.__class__ == Widget or (\n                widget.__class__ == WidgetMeta and Widget in widget.__bases__):\n            return True\n\n        return False\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_experimental(self):
        if False:
            print('Hello World!')
        line = 'models = {\n    \'auth.group\': {\n        \'Meta\': {\'object_name\': \'Group\'},\n        \'permissions\': (\'django.db.models.fields.related.ManyToManyField\', [], {\'to\': "orm[\'auth.Permission\']", \'symmetrical\': \'False\', \'blank\': \'True\'})\n    },\n    \'auth.permission\': {\n        \'Meta\': {\'ordering\': "(\'content_type__app_label\', \'content_type__model\', \'codename\')", \'unique_together\': "((\'content_type\', \'codename\'),)", \'object_name\': \'Permission\'},\n        \'name\': (\'django.db.models.fields.CharField\', [], {\'max_length\': \'50\'})\n    },\n}\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(line, result)

    def test_e501_experimental_and_multiple_logical_lines(self):
        if False:
            i = 10
            return i + 15
        line = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx(aaaaaaaaaaaaaaaaaaaaaaa,\n                             bbbbbbbbbbbbbbbbbbbbbbbbbbbb, cccccccccccccccccccccccccccc, dddddddddddddddddddddddd)\nxxxxxxxxxxxxxxxxxxxxxxxxxxxx(aaaaaaaaaaaaaaaaaaaaaaa,\n                             bbbbbbbbbbbbbbbbbbbbbbbbbbbb, cccccccccccccccccccccccccccc, dddddddddddddddddddddddd)\n'
        fixed = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx(\n    aaaaaaaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbbbbbbbbbbbb,\n    cccccccccccccccccccccccccccc, dddddddddddddddddddddddd)\nxxxxxxxxxxxxxxxxxxxxxxxxxxxx(\n    aaaaaaaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbbbbbbbbbbbb,\n    cccccccccccccccccccccccccccc, dddddddddddddddddddddddd)\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_and_multiple_logical_lines_with_math(self):
        if False:
            while True:
                i = 10
        line = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx([-1 + 5 / -10,\n                                                                            100,\n                                                                            -3 - 4])\n'
        fixed = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(\n    [-1 + 5 / -10, 100, -3 - 4])\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_and_import(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'from . import (xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,\n               yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy)\n'
        fixed = 'from . import (\n    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,\n    yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy)\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_shorten_comment_with_experimental(self):
        if False:
            for i in range(10):
                print('nop')
        line = '# ------ -------------------------------------------------------------------------\n'
        fixed = '# ------ -----------------------------------------------------------------\n'
        with autopep8_context(line, options=['--experimental', '--aggressive']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_experimental_and_escaped_newline(self):
        if False:
            print('Hello World!')
        line = 'if True or \\\n    False:  # test test test test test test test test test test test test test test\n    pass\n'
        fixed = 'if True or \\\n        False:  # test test test test test test test test test test test test test test\n    pass\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_experimental_and_multiline_string(self):
        if False:
            print('Hello World!')
        line = "print('---------------------------------------------------------------------',\n      ('================================================', '====================='),\n      '''--------------------------------------------------------------------------------\n      ''')\n"
        fixed = "print(\n    '---------------------------------------------------------------------',\n    ('================================================',\n     '====================='),\n    '''--------------------------------------------------------------------------------\n      ''')\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_experimental_and_multiline_string_with_addition(self):
        if False:
            print('Hello World!')
        line = 'def f():\n    email_text += """<html>This is a really long docstring that goes over the column limit and is multi-line.<br><br>\n<b>Czar: </b>"""+despot["Nicholas"]+"""<br>\n<b>Minion: </b>"""+serf["Dmitri"]+"""<br>\n<b>Residence: </b>"""+palace["Winter"]+"""<br>\n</body>\n</html>"""\n'
        fixed = 'def f():\n    email_text += """<html>This is a really long docstring that goes over the column limit and is multi-line.<br><br>\n<b>Czar: </b>"""+despot["Nicholas"]+"""<br>\n<b>Minion: </b>"""+serf["Dmitri"]+"""<br>\n<b>Residence: </b>"""+palace["Winter"]+"""<br>\n</body>\n</html>"""\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_experimental_and_multiline_string_in_parens(self):
        if False:
            i = 10
            return i + 15
        line = 'def f():\n    email_text += ("""<html>This is a really long docstring that goes over the column limit and is multi-line.<br><br>\n<b>Czar: </b>"""+despot["Nicholas"]+"""<br>\n<b>Minion: </b>"""+serf["Dmitri"]+"""<br>\n<b>Residence: </b>"""+palace["Winter"]+"""<br>\n</body>\n</html>""")\n'
        fixed = 'def f():\n    email_text += (\n        """<html>This is a really long docstring that goes over the column limit and is multi-line.<br><br>\n<b>Czar: </b>"""+despot["Nicholas"]+"""<br>\n<b>Minion: </b>"""+serf["Dmitri"]+"""<br>\n<b>Residence: </b>"""+palace["Winter"]+"""<br>\n</body>\n</html>""")\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_experimental_and_indentation(self):
        if False:
            print('Hello World!')
        line = 'if True:\n    # comment here\n    print(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n          bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,cccccccccccccccccccccccccccccccccccccccccc)\n'
        fixed = 'if True:\n    # comment here\n    print(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n          bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,\n          cccccccccccccccccccccccccccccccccccccccccc)\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_multiple_keys_and_experimental(self):
        if False:
            i = 10
            return i + 15
        line = "one_two_three_four_five_six = {'one two three four five': 12345, 'asdfsdflsdkfjl sdflkjsdkfkjsfjsdlkfj sdlkfjlsfjs': '343',\n                               1: 1}\n"
        fixed = "one_two_three_four_five_six = {\n    'one two three four five': 12345,\n    'asdfsdflsdkfjl sdflkjsdkfkjsfjsdlkfj sdlkfjlsfjs': '343', 1: 1}\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_with_experimental_and_carriage_returns_only(self):
        if False:
            i = 10
            return i + 15
        'Make sure _find_logical() does not crash.'
        line = 'if True:\r    from aaaaaaaaaaaaaaaa import bbbbbbbbbbbbbbbbbbb\r    \r    ccccccccccc = None\r'
        fixed = 'if True:\r    from aaaaaaaaaaaaaaaa import bbbbbbbbbbbbbbbbbbb\r\r    ccccccccccc = None\r'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_should_ignore_imports(self):
        if False:
            while True:
                i = 10
        line = 'import logging, os, bleach, commonware, urllib2, json, time, requests, urlparse, re\n'
        with autopep8_context(line, options=['--select=E501', '--experimental']) as result:
            self.assertEqual(line, result)

    def test_e501_experimental_should_not_do_useless_things(self):
        if False:
            return 10
        line = "foo('                                                                            ')\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(line, result)

    def test_e501_experimental_with_percent(self):
        if False:
            while True:
                i = 10
        line = 'raise MultiProjectException("Ambiguous workspace: %s=%s, %s" % ( varname, varname_path, os.path.abspath(config_filename)))\n'
        fixed = 'raise MultiProjectException(\n    "Ambiguous workspace: %s=%s, %s" %\n    (varname, varname_path, os.path.abspath(config_filename)))\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_def(self):
        if False:
            return 10
        line = 'def foobar(sldfkjlsdfsdf, kksdfsdfsf,sdfsdfsdf, sdfsdfkdk, szdfsdfsdf, sdfsdfsdfsdlkfjsdlf, sdfsdfddf,sdfsdfsfd, sdfsdfdsf):\n    pass\n'
        fixed = 'def foobar(sldfkjlsdfsdf, kksdfsdfsf, sdfsdfsdf, sdfsdfkdk, szdfsdfsdf,\n           sdfsdfsdfsdlkfjsdlf, sdfsdfddf, sdfsdfsfd, sdfsdfdsf):\n    pass\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_tuple(self):
        if False:
            return 10
        line = "def f():\n    man_this_is_a_very_long_function_name(an_extremely_long_variable_name,\n                                          ('a string that is long: %s'%'bork'))\n"
        fixed = "def f():\n    man_this_is_a_very_long_function_name(\n        an_extremely_long_variable_name,\n        ('a string that is long: %s' % 'bork'))\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_tuple_in_list(self):
        if False:
            for i in range(10):
                print('nop')
        line = "def f(self):\n    self._xxxxxxxx(aaaaaa, bbbbbbbbb, cccccccccccccccccc,\n                   [('mmmmmmmmmm', self.yyyyyyyyyy.zzzzzzzz/_DDDDDD)], eee, 'ff')\n"
        fixed = "def f(self):\n    self._xxxxxxxx(\n        aaaaaa, bbbbbbbbb, cccccccccccccccccc,\n        [('mmmmmmmmmm', self.yyyyyyyyyy.zzzzzzzz / _DDDDDD)],\n        eee, 'ff')\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_complex_reformat(self):
        if False:
            print('Hello World!')
        line = 'bork(111, 111, 111, 111, 222, 222, 222, { \'foo\': 222, \'qux\': 222 }, (([\'hello\', \'world\'], [\'yo\', \'stella\', "how\'s", \'it\'], [\'going\']), {str(i): i for i in range(10)}, {\'bork\':((x, x**x) for x in range(10))}), 222, 222, 222, 222, 333, 333, 333, 333)\n'
        fixed = 'bork(\n    111, 111, 111, 111, 222, 222, 222, {\'foo\': 222, \'qux\': 222},\n    (([\'hello\', \'world\'],\n      [\'yo\', \'stella\', "how\'s", \'it\'],\n      [\'going\']),\n     {str(i): i for i in range(10)},\n     {\'bork\': ((x, x ** x) for x in range(10))}),\n    222, 222, 222, 222, 333, 333, 333, 333)\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_dot_calls(self):
        if False:
            i = 10
            return i + 15
        line = "if True:\n    logging.info('aaaaaa bbbbb dddddd ccccccc eeeeeee fffffff gg: %s',\n        xxxxxxxxxxxxxxxxx.yyyyyyyyyyyyyyyyyyyyy(zzzzzzzzzzzzzzzzz.jjjjjjjjjjjjjjjjj()))\n"
        fixed = "if True:\n    logging.info(\n        'aaaaaa bbbbb dddddd ccccccc eeeeeee fffffff gg: %s',\n        xxxxxxxxxxxxxxxxx.yyyyyyyyyyyyyyyyyyyyy(\n            zzzzzzzzzzzzzzzzz.jjjjjjjjjjjjjjjjj()))\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_avoid_breaking_at_empty_parentheses_if_possible(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'someverylongindenttionwhatnot().foo().bar().baz("and here is a long string 123456789012345678901234567890")\n'
        fixed = 'someverylongindenttionwhatnot().foo().bar().baz(\n    "and here is a long string 123456789012345678901234567890")\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_unicode(self):
        if False:
            return 10
        line = 'someverylongindenttionwhatnot().foo().bar().baz("and here is a l안녕하세요 123456789012345678901234567890")\n'
        fixed = 'someverylongindenttionwhatnot().foo().bar().baz(\n    "and here is a l안녕하세요 123456789012345678901234567890")\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_tuple_assignment(self):
        if False:
            while True:
                i = 10
        line = 'if True:\n    (xxxxxxx,) = xxxx.xxxxxxx.xxxxx(xxxxxxxxxxxx.xx).xxxxxx(xxxxxxxxxxxx.xxxx == xxxx.xxxx).xxxxx()\n'
        fixed = 'if True:\n    (xxxxxxx,) = xxxx.xxxxxxx.xxxxx(xxxxxxxxxxxx.xx).xxxxxx(\n        xxxxxxxxxxxx.xxxx == xxxx.xxxx).xxxxx()\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    @unittest.skip('To do')
    def test_e501_experimental_tuple_on_line(self):
        if False:
            while True:
                i = 10
        line = "def f():\n    self.aaaaaaaaa(bbbbbb, ccccccccc, dddddddddddddddd,\n                   ((x, y/eeeeeee) for x, y in self.outputs.total.iteritems()),\n                   fff, 'GG')\n"
        fixed = "def f():\n    self.aaaaaaaaa(\n        bbbbbb, ccccccccc, dddddddddddddddd,\n        ((x, y / eeeeeee) for x, y in self.outputs.total.iteritems()),\n        fff, 'GG')\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_tuple_on_line_two_space_indent(self):
        if False:
            print('Hello World!')
        line = "def f():\n  self.aaaaaaaaa(bbbbbb, ccccccccc, dddddddddddddddd,\n                 ((x, y/eeeeeee) for x, y in self.outputs.total.iteritems()),\n                 fff, 'GG')\n"
        fixed = "def f():\n  self.aaaaaaaaa(bbbbbb, ccccccccc, dddddddddddddddd,\n                 ((x, y/eeeeeee) for x, y in self.outputs.total.iteritems()),\n                 fff, 'GG')\n"
        with autopep8_context(line, options=['--experimental', '--indent-size=2']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_oversized_default_initializer(self):
        if False:
            return 10
        line = 'aaaaaaaaaaaaaaaaaaaaa(lllll,mmmmmmmm,nnn,fffffffffff,ggggggggggg,hhh,ddddddddddddd=eeeeeeeee,bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb=ccccccccccccccccccccccccccccccccccccccccccccccccc,bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb=cccccccccccccccccccccccccccccccccccccccccccccccc)\n'
        fixed = 'aaaaaaaaaaaaaaaaaaaaa(\n    lllll, mmmmmmmm, nnn, fffffffffff, ggggggggggg, hhh,\n    ddddddddddddd=eeeeeeeee,\n    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb=ccccccccccccccccccccccccccccccccccccccccccccccccc,\n    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb=cccccccccccccccccccccccccccccccccccccccccccccccc)\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_decorator(self):
        if False:
            print('Hello World!')
        line = "@foo(('xxxxxxxxxxxxxxxxxxxxxxxxxx', users.xxxxxxxxxxxxxxxxxxxxxxxxxx), ('yyyyyyyyyyyy', users.yyyyyyyyyyyy), ('zzzzzzzzzzzzzz', users.zzzzzzzzzzzzzz))\n"
        fixed = "@foo(('xxxxxxxxxxxxxxxxxxxxxxxxxx', users.xxxxxxxxxxxxxxxxxxxxxxxxxx),\n     ('yyyyyyyyyyyy', users.yyyyyyyyyyyy),\n     ('zzzzzzzzzzzzzz', users.zzzzzzzzzzzzzz))\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_long_class_name(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'class AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA(BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB):\n    pass\n'
        fixed = 'class AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA(\n        BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB):\n    pass\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_no_line_change(self):
        if False:
            return 10
        line = 'def f():\n    return \'<a href="javascript:;" class="copy-to-clipboard-button" data-clipboard-text="%s" title="copy url to clipboard">Copy Link</a>\' % url\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(line, result)

    def test_e501_experimental_splitting_small_arrays(self):
        if False:
            i = 10
            return i + 15
        line = "def foo():\n    unspecified[service] = ('# The %s brown fox jumped over the lazy, good for nothing '\n                            'dog until it grew tired and set its sights upon the cat!' % adj)\n"
        fixed = "def foo():\n    unspecified[service] = (\n        '# The %s brown fox jumped over the lazy, good for nothing '\n        'dog until it grew tired and set its sights upon the cat!' % adj)\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_no_splitting_in_func_call(self):
        if False:
            print('Hello World!')
        line = "def foo():\n    if True:\n        if True:\n            function.calls('%r (%s): aaaaaaaa bbbbbbbbbb ccccccc ddddddd eeeeee (%d, %d)',\n                           xxxxxx.yy, xxxxxx.yyyy, len(mmmmmmmmmmmmm['fnord']),\n                           len(mmmmmmmmmmmmm['asdfakjhdsfkj']))\n"
        fixed = "def foo():\n    if True:\n        if True:\n            function.calls(\n                '%r (%s): aaaaaaaa bbbbbbbbbb ccccccc ddddddd eeeeee (%d, %d)',\n                xxxxxx.yy, xxxxxx.yyyy, len(mmmmmmmmmmmmm['fnord']),\n                len(mmmmmmmmmmmmm['asdfakjhdsfkj']))\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_no_splitting_at_dot(self):
        if False:
            i = 10
            return i + 15
        line = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx = [yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy.MMMMMM_NNNNNNN_OOOOO,\n                                yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy.PPPPPP_QQQQQQQ_RRRRR,\n                                yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy.SSSSSS_TTTTTTT_UUUUU]\n'
        fixed = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx = [\n    yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy.MMMMMM_NNNNNNN_OOOOO,\n    yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy.PPPPPP_QQQQQQQ_RRRRR,\n    yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy.SSSSSS_TTTTTTT_UUUUU]\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_no_splitting_before_arg_list(self):
        if False:
            while True:
                i = 10
        line = "xxxxxxxxxxxx = [yyyyyy['yyyyyy'].get('zzzzzzzzzzz') for yyyyyy in x.get('aaaaaaaaaaa') if yyyyyy['yyyyyy'].get('zzzzzzzzzzz')]\n"
        fixed = "xxxxxxxxxxxx = [yyyyyy['yyyyyy'].get('zzzzzzzzzzz')\n                for yyyyyy in x.get('aaaaaaaaaaa')\n                if yyyyyy['yyyyyy'].get('zzzzzzzzzzz')]\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_dont_split_if_looks_bad(self):
        if False:
            while True:
                i = 10
        line = "def f():\n    if True:\n        BAD(('xxxxxxxxxxxxx', 42), 'I died for beauty, but was scarce / Adjusted in the tomb %s', yyyyyyyyyyyyy)\n"
        fixed = "def f():\n    if True:\n        BAD(('xxxxxxxxxxxxx', 42),\n            'I died for beauty, but was scarce / Adjusted in the tomb %s',\n            yyyyyyyyyyyyy)\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_list_comp(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'xxxxxxxxxxxs = [xxxxxxxxxxx for xxxxxxxxxxx in xxxxxxxxxxxs if not yyyyyyyyyyyy[xxxxxxxxxxx] or not yyyyyyyyyyyy[xxxxxxxxxxx].zzzzzzzzzz]\n'
        fixed = 'xxxxxxxxxxxs = [\n    xxxxxxxxxxx for xxxxxxxxxxx in xxxxxxxxxxxs\n    if not yyyyyyyyyyyy[xxxxxxxxxxx] or\n    not yyyyyyyyyyyy[xxxxxxxxxxx].zzzzzzzzzz]\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)
        line = 'def f():\n    xxxxxxxxxx = [f for f in yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy.zzzzzzzzzzzzzzzzzzzzzzzz.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]\n'
        fixed = 'def f():\n    xxxxxxxxxx = [\n        f\n        for f in\n        yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy.zzzzzzzzzzzzzzzzzzzzzzzz.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_dict(self):
        if False:
            for i in range(10):
                print('nop')
        line = "def f():\n    zzzzzzzzzzzzz = {\n        'aaaaaa/bbbbbb/ccccc/dddddddd/eeeeeeeee/fffffffffff/ggggggggg/hhhhhhhh.py':\n            yyyyyyyyyyy.xxxxxxxxxxx(\n                'aa/bbbbbbb/cc/ddddddd/eeeeeeeeeee/fffffffffff/ggggggggg/hhhhhhh/ggggg.py',\n                '00000000',\n                yyyyyyyyyyy.xxxxxxxxx.zzzz),\n    }\n"
        fixed = "def f():\n    zzzzzzzzzzzzz = {\n        'aaaaaa/bbbbbb/ccccc/dddddddd/eeeeeeeee/fffffffffff/ggggggggg/hhhhhhhh.py':\n        yyyyyyyyyyy.xxxxxxxxxxx(\n            'aa/bbbbbbb/cc/ddddddd/eeeeeeeeeee/fffffffffff/ggggggggg/hhhhhhh/ggggg.py',\n            '00000000', yyyyyyyyyyy.xxxxxxxxx.zzzz), }\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_indentation(self):
        if False:
            while True:
                i = 10
        line = "class Klass(object):\n\n    '''Class docstring.'''\n\n    def Quote(self, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5):\n        pass\n"
        fixed = "class Klass(object):\n\n  '''Class docstring.'''\n\n  def Quote(\n      self, parameter_1, parameter_2, parameter_3, parameter_4,\n          parameter_5):\n    pass\n"
        with autopep8_context(line, options=['--experimental', '--indent-size=2']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_long_function_call_elements(self):
        if False:
            i = 10
            return i + 15
        line = "def g():\n    pppppppppppppppppppppppppp1, pppppppppppppppppppppppp2 = (\n        zzzzzzzzzzzz.yyyyyyyyyyyyyy(aaaaaaaaa=10, bbbbbbbbbbbbbbbb='2:3',\n                                    cccccccc='{1:2}', dd=1, eeeee=0),\n        zzzzzzzzzzzz.yyyyyyyyyyyyyy(dd=7, aaaaaaaaa=16, bbbbbbbbbbbbbbbb='2:3',\n                                    cccccccc='{1:2}',\n                                    eeeee=xxxxxxxxxxxxxxxxx.wwwwwwwwwwwww.vvvvvvvvvvvvvvvvvvvvvvvvv))\n"
        fixed = "def g():\n    pppppppppppppppppppppppppp1, pppppppppppppppppppppppp2 = (\n        zzzzzzzzzzzz.yyyyyyyyyyyyyy(\n            aaaaaaaaa=10, bbbbbbbbbbbbbbbb='2:3', cccccccc='{1:2}', dd=1,\n            eeeee=0),\n        zzzzzzzzzzzz.yyyyyyyyyyyyyy(\n            dd=7, aaaaaaaaa=16, bbbbbbbbbbbbbbbb='2:3', cccccccc='{1:2}',\n            eeeee=xxxxxxxxxxxxxxxxx.wwwwwwwwwwwww.vvvvvvvvvvvvvvvvvvvvvvvvv))\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_long_nested_tuples_in_arrays(self):
        if False:
            i = 10
            return i + 15
        line = 'def f():\n    aaaaaaaaaaa.bbbbbbb([\n        (\'xxxxxxxxxx\', \'yyyyyy\', \'Heaven hath no wrath like love to hatred turned. Nor hell a fury like a woman scorned.\'),\n        (\'xxxxxxx\', \'yyyyyyyyyyy\', "To the last I grapple with thee. From hell\'s heart I stab at thee. For hate\'s sake I spit my last breath at thee!")])\n'
        fixed = 'def f():\n    aaaaaaaaaaa.bbbbbbb(\n        [(\'xxxxxxxxxx\', \'yyyyyy\',\n          \'Heaven hath no wrath like love to hatred turned. Nor hell a fury like a woman scorned.\'),\n         (\'xxxxxxx\', \'yyyyyyyyyyy\',\n          "To the last I grapple with thee. From hell\'s heart I stab at thee. For hate\'s sake I spit my last breath at thee!")])\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_func_call_open_paren_not_separated(self):
        if False:
            while True:
                i = 10
        line = "def f():\n    owned_list = [o for o in owned_list if self.display['zzzzzzzzzzzzzz'] in aaaaaaaaaaaaaaaaa.bbbbbbbbbbbbbbbbbbbb(o.qq, ccccccccccccccccccccccccccc.ddddddddd.eeeeeee)]\n"
        fixed = "def f():\n    owned_list = [\n        o for o in owned_list\n        if self.display['zzzzzzzzzzzzzz'] in aaaaaaaaaaaaaaaaa.bbbbbbbbbbbbbbbbbbbb(\n            o.qq, ccccccccccccccccccccccccccc.ddddddddd.eeeeeee)]\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_long_dotted_object(self):
        if False:
            i = 10
            return i + 15
        line = 'def f(self):\n  return self.xxxxxxxxxxxxxxx(aaaaaaa.bbbbb.ccccccc.ddd.eeeeee.fffffffff.ggggg.hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh)\n'
        fixed = 'def f(self):\n    return self.xxxxxxxxxxxxxxx(\n        aaaaaaa.bbbbb.ccccccc.ddd.eeeeee.fffffffff.ggggg.\n        hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh)\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    @unittest.skipIf(sys.version_info >= (3, 12), 'not detech in Python3.12+')
    def test_e501_experimental_parsing_dict_with_comments(self):
        if False:
            i = 10
            return i + 15
        line = "self.display['xxxxxxxxxxxx'] = [{'title': _('Library'),  #. This is the first comment.\n    'flag': aaaaaaaaaa.bbbbbbbbb.cccccccccc\n    }, {'title': _('Original'),  #. This is the second comment.\n    'flag': aaaaaaaaaa.bbbbbbbbb.dddddddddd\n    }, {'title': _('Unknown'),  #. This is the third comment.\n    'flag': aaaaaaaaaa.bbbbbbbbb.eeeeeeeeee}]\n"
        fixed = "self.display['xxxxxxxxxxxx'] = [{'title': _('Library'),  # . This is the first comment.\n                                 'flag': aaaaaaaaaa.bbbbbbbbb.cccccccccc\n                                 # . This is the second comment.\n                                 }, {'title': _('Original'),\n                                     'flag': aaaaaaaaaa.bbbbbbbbb.dddddddddd\n                                     # . This is the third comment.\n                                     }, {'title': _('Unknown'),\n                                         'flag': aaaaaaaaaa.bbbbbbbbb.eeeeeeeeee}]\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    @unittest.skipIf(sys.version_info >= (3, 12), 'not detech in Python3.12+')
    def test_e501_experimental_if_line_over_limit(self):
        if False:
            for i in range(10):
                print('nop')
        line = 'if not xxxxxxxxxxxx(aaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbb, cccccccccccccc, dddddddddddddddddddddd):\n    return 1\n'
        fixed = 'if not xxxxxxxxxxxx(\n        aaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbb, cccccccccccccc,\n        dddddddddddddddddddddd):\n    return 1\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_for_line_over_limit(self):
        if False:
            while True:
                i = 10
        line = 'for aaaaaaaaa in xxxxxxxxxxxx(aaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbb, cccccccccccccc, dddddddddddddddddddddd):\n    pass\n'
        fixed = 'for aaaaaaaaa in xxxxxxxxxxxx(\n        aaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbb, cccccccccccccc,\n        dddddddddddddddddddddd):\n    pass\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_while_line_over_limit(self):
        if False:
            while True:
                i = 10
        line = 'while xxxxxxxxxxxx(aaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbb, cccccccccccccc, dddddddddddddddddddddd):\n    pass\n'
        fixed = 'while xxxxxxxxxxxx(\n        aaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbb, cccccccccccccc,\n        dddddddddddddddddddddd):\n    pass\n'
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

    def test_e501_experimental_with_in(self):
        if False:
            i = 10
            return i + 15
        line = "if True:\n    if True:\n        if True:\n            if True:\n                if True:\n                    if True:\n                        if True:\n                            if True:\n                                if k_left in ('any', k_curr) and k_right in ('any', k_curr):\n                                    pass\n"
        fixed = "if True:\n    if True:\n        if True:\n            if True:\n                if True:\n                    if True:\n                        if True:\n                            if True:\n                                if k_left in (\n                                        'any', k_curr) and k_right in (\n                                        'any', k_curr):\n                                    pass\n"
        with autopep8_context(line, options=['--experimental']) as result:
            self.assertEqual(fixed, result)

def fix_e266(source):
    if False:
        for i in range(10):
            print('nop')
    with autopep8_context(source, options=['--select=E266']) as result:
        return result

def fix_e265_and_e266(source):
    if False:
        for i in range(10):
            print('nop')
    with autopep8_context(source, options=['--select=E265,E266']) as result:
        return result

@contextlib.contextmanager
def autopep8_context(line, options=None):
    if False:
        return 10
    if not options:
        options = []
    with temporary_file_context(line) as filename:
        options = autopep8.parse_args([filename] + list(options))
        yield autopep8.fix_file(filename=filename, options=options)

@contextlib.contextmanager
def autopep8_subprocess(line, options, timeout=None):
    if False:
        while True:
            i = 10
    with temporary_file_context(line) as filename:
        p = Popen(list(AUTOPEP8_CMD_TUPLE) + [filename] + options, stdout=PIPE)
        if timeout is None:
            (_stdout, _) = p.communicate()
        else:
            try:
                (_stdout, _) = p.communicate(timeout=timeout)
            except TypeError:
                while p.poll() is None and timeout > 0:
                    time.sleep(0.5)
                    timeout -= 0.5
                if p.poll() is None:
                    p.kill()
                    raise Exception('subprocess is timed out')
                (_stdout, _) = p.communicate()
        yield (_stdout.decode('utf-8'), p.returncode)

@contextlib.contextmanager
def temporary_file_context(text, suffix='', prefix=''):
    if False:
        print('Hello World!')
    temporary = mkstemp(suffix=suffix, prefix=prefix)
    os.close(temporary[0])
    with autopep8.open_with_encoding(temporary[1], encoding='utf-8', mode='w') as temp_file:
        temp_file.write(text)
    yield temporary[1]
    os.remove(temporary[1])

@contextlib.contextmanager
def readonly_temporary_file_context(text, suffix='', prefix=''):
    if False:
        return 10
    temporary = mkstemp(suffix=suffix, prefix=prefix)
    os.close(temporary[0])
    with autopep8.open_with_encoding(temporary[1], encoding='utf-8', mode='w') as temp_file:
        temp_file.write(text)
    os.chmod(temporary[1], stat.S_IRUSR)
    yield temporary[1]
    os.remove(temporary[1])

@contextlib.contextmanager
def temporary_project_directory(prefix='autopep8test'):
    if False:
        return 10
    temporary = mkdtemp(prefix=prefix)
    yield temporary
    shutil.rmtree(temporary)

@contextlib.contextmanager
def disable_stderr():
    if False:
        i = 10
        return i + 15
    sio = StringIO()
    with capture_stderr(sio):
        yield

@contextlib.contextmanager
def capture_stderr(sio):
    if False:
        return 10
    _tmp = sys.stderr
    sys.stderr = sio
    try:
        yield
    finally:
        sys.stderr = _tmp
if __name__ == '__main__':
    unittest.main()