"""Tests for yapf.yapf."""
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from io import StringIO
from yapf_third_party._ylib2to3.pgen2 import tokenize
from yapf.yapflib import errors
from yapf.yapflib import style
from yapf.yapflib import yapf_api
from yapftests import utils
from yapftests import yapf_test_helper
YAPF_BINARY = [sys.executable, '-m', 'yapf', '--no-local-style']

class FormatCodeTest(yapf_test_helper.YAPFTest):

    def _Check(self, unformatted_code, expected_formatted_code):
        if False:
            return 10
        (formatted_code, _) = yapf_api.FormatCode(unformatted_code, style_config='yapf')
        self.assertCodeEqual(expected_formatted_code, formatted_code)

    def testSimple(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent("        print('foo')\n    ")
        self._Check(unformatted_code, unformatted_code)

    def testNoEndingNewline(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        if True:\n          pass')
        expected_formatted_code = textwrap.dedent('        if True:\n          pass\n    ')
        self._Check(unformatted_code, expected_formatted_code)

class FormatFileTest(yapf_test_helper.YAPFTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.test_tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(self.test_tmpdir)

    def testFormatFile(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        if True:\n         pass\n    ')
        expected_formatted_code_pep8 = textwrap.dedent('        if True:\n            pass\n    ')
        expected_formatted_code_yapf = textwrap.dedent('        if True:\n          pass\n    ')
        with utils.TempFileContents(self.test_tmpdir, unformatted_code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='pep8')
            self.assertCodeEqual(expected_formatted_code_pep8, formatted_code)
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='yapf')
            self.assertCodeEqual(expected_formatted_code_yapf, formatted_code)

    def testDisableLinesPattern(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        if a:    b\n\n        # yapf: disable\n        if f:    g\n\n        if h:    i\n    ')
        expected_formatted_code = textwrap.dedent('        if a: b\n\n        # yapf: disable\n        if f:    g\n\n        if h:    i\n    ')
        with utils.TempFileContents(self.test_tmpdir, unformatted_code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='pep8')
            self.assertCodeEqual(expected_formatted_code, formatted_code)

    def testDisableAndReenableLinesPattern(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        if a:    b\n\n        # yapf: disable\n        if f:    g\n        # yapf: enable\n\n        if h:    i\n    ')
        expected_formatted_code = textwrap.dedent('        if a: b\n\n        # yapf: disable\n        if f:    g\n        # yapf: enable\n\n        if h: i\n    ')
        with utils.TempFileContents(self.test_tmpdir, unformatted_code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='pep8')
            self.assertCodeEqual(expected_formatted_code, formatted_code)

    def testFmtOnOff(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        if a:    b\n\n        # fmt: off\n        if f:    g\n        # fmt: on\n\n        if h:    i\n    ')
        expected_formatted_code = textwrap.dedent('        if a: b\n\n        # fmt: off\n        if f:    g\n        # fmt: on\n\n        if h: i\n    ')
        with utils.TempFileContents(self.test_tmpdir, unformatted_code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='pep8')
            self.assertCodeEqual(expected_formatted_code, formatted_code)

    def testDisablePartOfMultilineComment(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        if a:    b\n\n        # This is a multiline comment that disables YAPF.\n        # yapf: disable\n        if f:    g\n        # yapf: enable\n        # This is a multiline comment that enables YAPF.\n\n        if h:    i\n    ')
        expected_formatted_code = textwrap.dedent('        if a: b\n\n        # This is a multiline comment that disables YAPF.\n        # yapf: disable\n        if f:    g\n        # yapf: enable\n        # This is a multiline comment that enables YAPF.\n\n        if h: i\n    ')
        with utils.TempFileContents(self.test_tmpdir, unformatted_code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='pep8')
            self.assertCodeEqual(expected_formatted_code, formatted_code)
        code = textwrap.dedent('      def foo_function():\n          # some comment\n          # yapf: disable\n\n          foo(\n          bar,\n          baz\n          )\n\n          # yapf: enable\n    ')
        with utils.TempFileContents(self.test_tmpdir, code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='pep8')
            self.assertCodeEqual(code, formatted_code)

    def testEnabledDisabledSameComment(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        # yapf: disable\n        a(bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb, ccccccccccccccccccccccccccccccc, ddddddddddddddddddddddd, eeeeeeeeeeeeeeeeeeeeeeeeeee)\n        # yapf: enable\n        # yapf: disable\n        a(bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb, ccccccccccccccccccccccccccccccc, ddddddddddddddddddddddd, eeeeeeeeeeeeeeeeeeeeeeeeeee)\n        # yapf: enable\n    ')
        with utils.TempFileContents(self.test_tmpdir, code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='pep8')
            self.assertCodeEqual(code, formatted_code)

    def testFormatFileLinesSelection(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        if a:    b\n\n        if f:    g\n\n        if h:    i\n    ')
        expected_formatted_code_lines1and2 = textwrap.dedent('        if a: b\n\n        if f:    g\n\n        if h:    i\n    ')
        expected_formatted_code_lines3 = textwrap.dedent('        if a:    b\n\n        if f: g\n\n        if h:    i\n    ')
        with utils.TempFileContents(self.test_tmpdir, unformatted_code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='pep8', lines=[(1, 2)])
            self.assertCodeEqual(expected_formatted_code_lines1and2, formatted_code)
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='pep8', lines=[(3, 3)])
            self.assertCodeEqual(expected_formatted_code_lines3, formatted_code)

    def testFormatFileDiff(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        if True:\n         pass\n    ')
        with utils.TempFileContents(self.test_tmpdir, unformatted_code) as filepath:
            (diff, _, _) = yapf_api.FormatFile(filepath, print_diff=True)
            self.assertIn('+  pass', diff)

    def testFormatFileInPlace(self):
        if False:
            return 10
        unformatted_code = 'True==False\n'
        formatted_code = 'True == False\n'
        with utils.TempFileContents(self.test_tmpdir, unformatted_code) as filepath:
            (result, _, _) = yapf_api.FormatFile(filepath, in_place=True)
            self.assertEqual(result, None)
            with open(filepath) as fd:
                self.assertCodeEqual(formatted_code, fd.read())
            self.assertRaises(ValueError, yapf_api.FormatFile, filepath, in_place=True, print_diff=True)

    def testNoFile(self):
        if False:
            print('Hello World!')
        with self.assertRaises(IOError) as context:
            yapf_api.FormatFile('not_a_file.py')
        self.assertEqual(str(context.exception), "[Errno 2] No such file or directory: 'not_a_file.py'")

    def testCommentsUnformatted(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent("        foo = [# A list of things\n               # bork\n            'one',\n            # quark\n            'two'] # yapf: disable\n    ")
        with utils.TempFileContents(self.test_tmpdir, code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='pep8')
            self.assertCodeEqual(code, formatted_code)

    def testDisabledHorizontalFormattingOnNewLine(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        # yapf: disable\n        a = [\n        1]\n        # yapf: enable\n    ')
        with utils.TempFileContents(self.test_tmpdir, code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='pep8')
            self.assertCodeEqual(code, formatted_code)

    def testSplittingSemicolonStatements(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        def f():\n          x = y + 42 ; z = n * 42\n          if True: a += 1 ; b += 1; c += 1\n    ')
        expected_formatted_code = textwrap.dedent('        def f():\n            x = y + 42\n            z = n * 42\n            if True:\n                a += 1\n                b += 1\n                c += 1\n    ')
        with utils.TempFileContents(self.test_tmpdir, unformatted_code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='pep8')
            self.assertCodeEqual(expected_formatted_code, formatted_code)

    def testSemicolonStatementsDisabled(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        def f():\n          x = y + 42 ; z = n * 42  # yapf: disable\n          if True: a += 1 ; b += 1; c += 1\n    ')
        expected_formatted_code = textwrap.dedent('        def f():\n            x = y + 42 ; z = n * 42  # yapf: disable\n            if True:\n                a += 1\n                b += 1\n                c += 1\n    ')
        with utils.TempFileContents(self.test_tmpdir, unformatted_code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='pep8')
            self.assertCodeEqual(expected_formatted_code, formatted_code)

    def testDisabledSemiColonSeparatedStatements(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent('        # yapf: disable\n        if True: a ; b\n    ')
        with utils.TempFileContents(self.test_tmpdir, code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='pep8')
            self.assertCodeEqual(code, formatted_code)

    def testDisabledMultilineStringInDictionary(self):
        if False:
            return 10
        code = textwrap.dedent('        # yapf: disable\n\n        A = [\n            {\n                "aaaaaaaaaaaaaaaaaaa": \'\'\'\n        bbbbbbbbbbb: "ccccccccccc"\n        dddddddddddddd: 1\n        eeeeeeee: 0\n        ffffffffff: "ggggggg"\n        \'\'\',\n            },\n        ]\n    ')
        with utils.TempFileContents(self.test_tmpdir, code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='yapf')
            self.assertCodeEqual(code, formatted_code)

    def testDisabledWithPrecedingText(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        # TODO(fix formatting): yapf: disable\n\n        A = [\n            {\n                "aaaaaaaaaaaaaaaaaaa": \'\'\'\n        bbbbbbbbbbb: "ccccccccccc"\n        dddddddddddddd: 1\n        eeeeeeee: 0\n        ffffffffff: "ggggggg"\n        \'\'\',\n            },\n        ]\n    ')
        with utils.TempFileContents(self.test_tmpdir, code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='yapf')
            self.assertCodeEqual(code, formatted_code)

    def testCRLFLineEnding(self):
        if False:
            for i in range(10):
                print('nop')
        code = 'class _():\r\n  pass\r\n'
        with utils.TempFileContents(self.test_tmpdir, code) as filepath:
            (formatted_code, _, _) = yapf_api.FormatFile(filepath, style_config='yapf')
            self.assertCodeEqual(code, formatted_code)

class CommandLineTest(yapf_test_helper.YAPFTest):
    """Test how calling yapf from the command line acts."""

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.test_tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(cls.test_tmpdir)

    def assertYapfReformats(self, unformatted, expected, extra_options=None, env=None):
        if False:
            print('Hello World!')
        'Check that yapf reformats the given code as expected.\n\n    Invokes yapf in a subprocess, piping the unformatted code into its stdin.\n    Checks that the formatted output is as expected.\n\n    Arguments:\n      unformatted: unformatted code - input to yapf\n      expected: expected formatted code at the output of yapf\n      extra_options: iterable of extra command-line options to pass to yapf\n      env: dict of environment variables.\n    '
        cmdline = YAPF_BINARY + (extra_options or [])
        p = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        (reformatted_code, stderrdata) = p.communicate(unformatted.encode('utf-8-sig'))
        self.assertEqual(stderrdata, b'')
        self.assertMultiLineEqual(reformatted_code.decode('utf-8'), expected)

    def testInPlaceReformatting(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        def foo():\n          x = 37\n    ')
        expected_formatted_code = textwrap.dedent('        def foo():\n            x = 37\n    ')
        with utils.TempFileContents(self.test_tmpdir, unformatted_code, suffix='.py') as filepath:
            p = subprocess.Popen(YAPF_BINARY + ['--in-place', filepath])
            p.wait()
            with io.open(filepath, mode='r', newline='') as fd:
                reformatted_code = fd.read()
        self.assertEqual(reformatted_code, expected_formatted_code)

    def testInPlaceReformattingBlank(self):
        if False:
            while True:
                i = 10
        unformatted_code = '\n\n'
        expected_formatted_code = '\n'
        with utils.TempFileContents(self.test_tmpdir, unformatted_code, suffix='.py') as filepath:
            p = subprocess.Popen(YAPF_BINARY + ['--in-place', filepath])
            p.wait()
            with io.open(filepath, mode='r', encoding='utf-8', newline='') as fd:
                reformatted_code = fd.read()
        self.assertEqual(reformatted_code, expected_formatted_code)

    def testInPlaceReformattingWindowsNewLine(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = '\r\n\r\n'
        expected_formatted_code = '\r\n'
        with utils.TempFileContents(self.test_tmpdir, unformatted_code, suffix='.py') as filepath:
            p = subprocess.Popen(YAPF_BINARY + ['--in-place', filepath])
            p.wait()
            with io.open(filepath, mode='r', encoding='utf-8', newline='') as fd:
                reformatted_code = fd.read()
        self.assertEqual(reformatted_code, expected_formatted_code)

    def testInPlaceReformattingNoNewLine(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('def foo(): x = 37')
        expected_formatted_code = textwrap.dedent('        def foo():\n            x = 37\n    ')
        with utils.TempFileContents(self.test_tmpdir, unformatted_code, suffix='.py') as filepath:
            p = subprocess.Popen(YAPF_BINARY + ['--in-place', filepath])
            p.wait()
            with io.open(filepath, mode='r', newline='') as fd:
                reformatted_code = fd.read()
        self.assertEqual(reformatted_code, expected_formatted_code)

    def testInPlaceReformattingEmpty(self):
        if False:
            return 10
        unformatted_code = ''
        expected_formatted_code = ''
        with utils.TempFileContents(self.test_tmpdir, unformatted_code, suffix='.py') as filepath:
            p = subprocess.Popen(YAPF_BINARY + ['--in-place', filepath])
            p.wait()
            with io.open(filepath, mode='r', encoding='utf-8', newline='') as fd:
                reformatted_code = fd.read()
        self.assertEqual(reformatted_code, expected_formatted_code)

    def testPrintModified(self):
        if False:
            for i in range(10):
                print('nop')
        for (unformatted_code, has_change) in [('1==2', True), ('1 == 2', False)]:
            with utils.TempFileContents(self.test_tmpdir, unformatted_code, suffix='.py') as filepath:
                output = subprocess.check_output(YAPF_BINARY + ['--in-place', '--print-modified', filepath], text=True)
                check = self.assertIn if has_change else self.assertNotIn
                check(f'Formatted {filepath}', output)

    def testReadFromStdin(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        def foo():\n          x = 37\n    ')
        expected_formatted_code = textwrap.dedent('        def foo():\n            x = 37\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code)

    def testReadFromStdinWithEscapedStrings(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        s =   "foo\\nbar"\n    ')
        expected_formatted_code = textwrap.dedent('        s = "foo\\nbar"\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code)

    def testSetYapfStyle(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        def foo(): # trail\n            x = 37\n    ')
        expected_formatted_code = textwrap.dedent('        def foo():  # trail\n          x = 37\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--style=yapf'])

    def testSetCustomStyleBasedOnYapf(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        def foo(): # trail\n            x = 37\n    ')
        expected_formatted_code = textwrap.dedent('        def foo():    # trail\n          x = 37\n    ')
        style_file = textwrap.dedent('        [style]\n        based_on_style = yapf\n        spaces_before_comment = 4\n    ')
        with utils.TempFileContents(self.test_tmpdir, style_file) as stylepath:
            self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--style={0}'.format(stylepath)])

    def testSetCustomStyleSpacesBeforeComment(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        a_very_long_statement_that_extends_way_beyond # Comment\n        short # This is a shorter statement\n    ')
        expected_formatted_code = textwrap.dedent('        a_very_long_statement_that_extends_way_beyond # Comment\n        short                                         # This is a shorter statement\n    ')
        style_file = textwrap.dedent('        [style]\n        spaces_before_comment = 15, 20\n    ')
        with utils.TempFileContents(self.test_tmpdir, style_file) as stylepath:
            self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--style={0}'.format(stylepath)])

    def testReadSingleLineCodeFromStdin(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        if True: pass\n    ')
        expected_formatted_code = textwrap.dedent('        if True: pass\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code)

    def testEncodingVerification(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent("        '''The module docstring.'''\n        # -*- coding: utf-8 -*-\n        def f():\n            x = 37\n    ")
        with utils.NamedTempFile(suffix='.py', dirname=self.test_tmpdir) as (out, _):
            with utils.TempFileContents(self.test_tmpdir, unformatted_code, suffix='.py') as filepath:
                try:
                    subprocess.check_call(YAPF_BINARY + ['--diff', filepath], stdout=out)
                except subprocess.CalledProcessError as e:
                    self.assertEqual(e.returncode, 1)

    def testReformattingSpecificLines(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent("        def h():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n\n        def g():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n        ")
        expected_formatted_code = textwrap.dedent("        def h():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and\n                    xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n\n        def g():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n    ")
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '1-2'])

    def testOmitFormattingLinesBeforeDisabledFunctionComment(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        import sys\n\n        # Comment\n        def some_func(x):\n            x = ["badly" , "formatted","line" ]\n    ')
        expected_formatted_code = textwrap.dedent('        import sys\n\n        # Comment\n        def some_func(x):\n            x = ["badly", "formatted", "line"]\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '5-5'])

    def testReformattingSkippingLines(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        def h():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n        # yapf: disable\n        def g():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n        # yapf: enable\n    ")
        expected_formatted_code = textwrap.dedent("        def h():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and\n                    xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n\n        # yapf: disable\n        def g():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n        # yapf: enable\n    ")
        self.assertYapfReformats(unformatted_code, expected_formatted_code)

    def testReformattingSkippingToEndOfFile(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent("        def h():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n        # yapf: disable\n        def g():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n        def f():\n            def e():\n                while (xxxxxxxxxxxxxxxxxxxxx(yyyyyyyyyyyyy[zzzzz]) == 'aaaaaaaaaaa' and\n                       xxxxxxxxxxxxxxxxxxxxx(yyyyyyyyyyyyy[zzzzz].aaaaaaaa[0]) ==\n                       'bbbbbbb'):\n                    pass\n    ")
        expected_formatted_code = textwrap.dedent("        def h():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and\n                    xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n\n        # yapf: disable\n        def g():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n        def f():\n            def e():\n                while (xxxxxxxxxxxxxxxxxxxxx(yyyyyyyyyyyyy[zzzzz]) == 'aaaaaaaaaaa' and\n                       xxxxxxxxxxxxxxxxxxxxx(yyyyyyyyyyyyy[zzzzz].aaaaaaaa[0]) ==\n                       'bbbbbbb'):\n                    pass\n    ")
        self.assertYapfReformats(unformatted_code, expected_formatted_code)

    def testReformattingSkippingSingleLine(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent("        def h():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n        def g():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):  # yapf: disable\n                pass\n    ")
        expected_formatted_code = textwrap.dedent("        def h():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and\n                    xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n\n        def g():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):  # yapf: disable\n                pass\n    ")
        self.assertYapfReformats(unformatted_code, expected_formatted_code)

    def testDisableWholeDataStructure(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent("        A = set([\n            'hello',\n            'world',\n        ])  # yapf: disable\n    ")
        expected_formatted_code = textwrap.dedent("        A = set([\n            'hello',\n            'world',\n        ])  # yapf: disable\n    ")
        self.assertYapfReformats(unformatted_code, expected_formatted_code)

    def testDisableButAdjustIndentations(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        class SplitPenaltyTest(unittest.TestCase):\n\n          def testUnbreakable(self):\n            self._CheckPenalties(tree, [\n            ])  # yapf: disable\n    ')
        expected_formatted_code = textwrap.dedent('        class SplitPenaltyTest(unittest.TestCase):\n\n            def testUnbreakable(self):\n                self._CheckPenalties(tree, [\n                ])  # yapf: disable\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code)

    def testRetainingHorizontalWhitespace(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent("        def h():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n        def g():\n            if (xxxxxxxxxxxx.yyyyyyyy        (zzzzzzzzzzzzz  [0]) ==     'aaaaaaaaaaa' and    xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):  # yapf: disable\n                pass\n    ")
        expected_formatted_code = textwrap.dedent("        def h():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and\n                    xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n\n        def g():\n            if (xxxxxxxxxxxx.yyyyyyyy        (zzzzzzzzzzzzz  [0]) ==     'aaaaaaaaaaa' and    xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):  # yapf: disable\n                pass\n    ")
        self.assertYapfReformats(unformatted_code, expected_formatted_code)

    def testRetainingVerticalWhitespace(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        def h():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n        def g():\n\n\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n\n                pass\n    ")
        expected_formatted_code = textwrap.dedent("        def h():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and\n                    xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n        def g():\n\n\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n\n                pass\n    ")
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '1-2'])
        unformatted_code = textwrap.dedent('\n\n        if a:     b\n\n\n        if c:\n            to_much      + indent\n\n            same\n\n\n\n        #comment\n\n        #   trailing whitespace\n    ')
        expected_formatted_code = textwrap.dedent('        if a: b\n\n\n        if c:\n            to_much      + indent\n\n            same\n\n\n\n        #comment\n\n        #   trailing whitespace\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '3-3', '--lines', '13-13'])
        unformatted_code = textwrap.dedent("        '''\n        docstring\n\n        '''\n\n        import blah\n    ")
        self.assertYapfReformats(unformatted_code, unformatted_code, extra_options=['--lines', '2-2'])

    def testVerticalSpacingWithCommentWithContinuationMarkers(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        # \\\n        # \\\n        # \\\n\n        x = {\n        }\n    ')
        expected_formatted_code = textwrap.dedent('        # \\\n        # \\\n        # \\\n\n        x = {\n        }\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '1-1'])

    def testRetainingSemicolonsWhenSpecifyingLines(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        a = line_to_format\n        def f():\n            x = y + 42; z = n * 42\n            if True: a += 1 ; b += 1 ; c += 1\n    ')
        expected_formatted_code = textwrap.dedent('        a = line_to_format\n        def f():\n            x = y + 42; z = n * 42\n            if True: a += 1 ; b += 1 ; c += 1\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '1-1'])

    def testDisabledMultilineStrings(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        foo=42\n        def f():\n            email_text += """<html>This is a really long docstring that goes over the column limit and is multi-line.<br><br>\n        <b>Czar: </b>"""+despot["Nicholas"]+"""<br>\n        <b>Minion: </b>"""+serf["Dmitri"]+"""<br>\n        <b>Residence: </b>"""+palace["Winter"]+"""<br>\n        </body>\n        </html>"""\n    ')
        expected_formatted_code = textwrap.dedent('        foo = 42\n        def f():\n            email_text += """<html>This is a really long docstring that goes over the column limit and is multi-line.<br><br>\n        <b>Czar: </b>"""+despot["Nicholas"]+"""<br>\n        <b>Minion: </b>"""+serf["Dmitri"]+"""<br>\n        <b>Residence: </b>"""+palace["Winter"]+"""<br>\n        </body>\n        </html>"""\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '1-1'])

    def testDisableWhenSpecifyingLines(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent("        # yapf: disable\n        A = set([\n            'hello',\n            'world',\n        ])\n        # yapf: enable\n        B = set([\n            'hello',\n            'world',\n        ])  # yapf: disable\n    ")
        expected_formatted_code = textwrap.dedent("        # yapf: disable\n        A = set([\n            'hello',\n            'world',\n        ])\n        # yapf: enable\n        B = set([\n            'hello',\n            'world',\n        ])  # yapf: disable\n    ")
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '1-10'])

    def testDisableFormattingInDataLiteral(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        def horrible():\n          oh_god()\n          why_would_you()\n          [\n             'do',\n\n              'that',\n          ]\n\n        def still_horrible():\n            oh_god()\n            why_would_you()\n            [\n                'do',\n\n                'that'\n            ]\n    ")
        expected_formatted_code = textwrap.dedent("        def horrible():\n            oh_god()\n            why_would_you()\n            [\n               'do',\n\n                'that',\n            ]\n\n        def still_horrible():\n            oh_god()\n            why_would_you()\n            ['do', 'that']\n    ")
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '14-15'])

    def testRetainVerticalFormattingBetweenDisabledAndEnabledLines(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        class A(object):\n            def aaaaaaaaaaaaa(self):\n                c = bbbbbbbbb.ccccccccc('challenge', 0, 1, 10)\n                self.assertEqual(\n                    ('ddddddddddddddddddddddddd',\n             'eeeeeeeeeeeeeeeeeeeeeeeee.%s' %\n                     c.ffffffffffff),\n             gggggggggggg.hhhhhhhhh(c, c.ffffffffffff))\n                iiiii = jjjjjjjjjjjjjj.iiiii\n    ")
        expected_formatted_code = textwrap.dedent("        class A(object):\n            def aaaaaaaaaaaaa(self):\n                c = bbbbbbbbb.ccccccccc('challenge', 0, 1, 10)\n                self.assertEqual(('ddddddddddddddddddddddddd',\n                                  'eeeeeeeeeeeeeeeeeeeeeeeee.%s' % c.ffffffffffff),\n                                 gggggggggggg.hhhhhhhhh(c, c.ffffffffffff))\n                iiiii = jjjjjjjjjjjjjj.iiiii\n    ")
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '4-7'])

    def testRetainVerticalFormattingBetweenDisabledLines(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        class A(object):\n            def aaaaaaaaaaaaa(self):\n                pass\n\n\n            def bbbbbbbbbbbbb(self):  # 5\n                pass\n    ')
        expected_formatted_code = textwrap.dedent('        class A(object):\n            def aaaaaaaaaaaaa(self):\n                pass\n\n\n            def bbbbbbbbbbbbb(self):  # 5\n                pass\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '4-4'])

    def testFormatLinesSpecifiedInMiddleOfExpression(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent("        class A(object):\n            def aaaaaaaaaaaaa(self):\n                c = bbbbbbbbb.ccccccccc('challenge', 0, 1, 10)\n                self.assertEqual(\n                    ('ddddddddddddddddddddddddd',\n             'eeeeeeeeeeeeeeeeeeeeeeeee.%s' %\n                     c.ffffffffffff),\n             gggggggggggg.hhhhhhhhh(c, c.ffffffffffff))\n                iiiii = jjjjjjjjjjjjjj.iiiii\n    ")
        expected_formatted_code = textwrap.dedent("        class A(object):\n            def aaaaaaaaaaaaa(self):\n                c = bbbbbbbbb.ccccccccc('challenge', 0, 1, 10)\n                self.assertEqual(('ddddddddddddddddddddddddd',\n                                  'eeeeeeeeeeeeeeeeeeeeeeeee.%s' % c.ffffffffffff),\n                                 gggggggggggg.hhhhhhhhh(c, c.ffffffffffff))\n                iiiii = jjjjjjjjjjjjjj.iiiii\n    ")
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '5-6'])

    def testCommentFollowingMultilineString(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent("        def foo():\n            '''First line.\n            Second line.\n            '''  # comment\n            x = '''hello world'''  # second comment\n            return 42  # another comment\n    ")
        expected_formatted_code = textwrap.dedent("        def foo():\n            '''First line.\n            Second line.\n            '''  # comment\n            x = '''hello world'''  # second comment\n            return 42  # another comment\n    ")
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '1-1'])

    def testDedentClosingBracket(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        def overly_long_function_name(first_argument_on_the_same_line,\n        second_argument_makes_the_line_too_long):\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def overly_long_function_name(first_argument_on_the_same_line,\n                                      second_argument_makes_the_line_too_long):\n            pass\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--style=pep8'])
        unformatted_code = textwrap.dedent('        def overly_long_function_name(\n          first_argument_on_the_same_line,\n          second_argument_makes_the_line_too_long):\n          pass\n    ')
        expected_formatted_fb_code = textwrap.dedent('        def overly_long_function_name(\n            first_argument_on_the_same_line, second_argument_makes_the_line_too_long\n        ):\n            pass\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_fb_code, extra_options=['--style=facebook'])

    def testCoalesceBrackets(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('       some_long_function_name_foo(\n           {\n               \'first_argument_of_the_thing\': id,\n               \'second_argument_of_the_thing\': "some thing"\n           }\n       )\n    ')
        expected_formatted_code = textwrap.dedent('       some_long_function_name_foo({\n           \'first_argument_of_the_thing\': id,\n           \'second_argument_of_the_thing\': "some thing"\n       })\n    ')
        with utils.NamedTempFile(dirname=self.test_tmpdir, mode='w') as (f, name):
            f.write(textwrap.dedent('          [style]\n          column_limit=82\n          coalesce_brackets = True\n      '))
            f.flush()
            self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--style={0}'.format(name)])

    def testPseudoParenSpaces(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        def   foo():\n          def bar():\n            return {msg_id: author for author, msg_id in reader}\n    ')
        expected_formatted_code = textwrap.dedent('        def foo():\n          def bar():\n            return {msg_id: author for author, msg_id in reader}\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '1-1', '--style', 'yapf'])

    def testMultilineCommentFormattingDisabled(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        # This is a comment\n        FOO = {\n            aaaaaaaa.ZZZ: [\n                bbbbbbbbbb.Pop(),\n                # Multiline comment.\n                # Line two.\n                bbbbbbbbbb.Pop(),\n            ],\n            'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx':\n                ('yyyyy', zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz),\n            '#': lambda x: x  # do nothing\n        }\n    ")
        expected_formatted_code = textwrap.dedent("        # This is a comment\n        FOO = {\n            aaaaaaaa.ZZZ: [\n                bbbbbbbbbb.Pop(),\n                # Multiline comment.\n                # Line two.\n                bbbbbbbbbb.Pop(),\n            ],\n            'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx':\n                ('yyyyy', zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz),\n            '#': lambda x: x  # do nothing\n        }\n    ")
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '1-1', '--style', 'yapf'])

    def testTrailingCommentsWithDisabledFormatting(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        import os\n\n        SCOPES = [\n            'hello world'  # This is a comment.\n        ]\n    ")
        expected_formatted_code = textwrap.dedent("        import os\n\n        SCOPES = [\n            'hello world'  # This is a comment.\n        ]\n    ")
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '1-1', '--style', 'yapf'])

    def testUseTabs(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        def foo_function():\n         if True:\n          pass\n    ')
        expected_formatted_code = 'def foo_function():\n\tif True:\n\t\tpass\n'
        style_contents = textwrap.dedent('        [style]\n        based_on_style = yapf\n        use_tabs = true\n        indent_width = 1\n    ')
        with utils.TempFileContents(self.test_tmpdir, style_contents) as stylepath:
            self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--style={0}'.format(stylepath)])

    def testUseTabsWith(self):
        if False:
            return 10
        unformatted_code = "def f():\n  return ['hello', 'world',]\n"
        expected_formatted_code = "def f():\n\treturn [\n\t    'hello',\n\t    'world',\n\t]\n"
        style_contents = textwrap.dedent('        [style]\n        based_on_style = yapf\n        use_tabs = true\n        indent_width = 1\n    ')
        with utils.TempFileContents(self.test_tmpdir, style_contents) as stylepath:
            self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--style={0}'.format(stylepath)])

    def testUseTabsContinuationAlignStyleFixed(self):
        if False:
            print('Hello World!')
        unformatted_code = "def foo_function(arg1, arg2, arg3):\n  return ['hello', 'world',]\n"
        expected_formatted_code = "def foo_function(\n\t\targ1, arg2, arg3):\n\treturn [\n\t\t\t'hello',\n\t\t\t'world',\n\t]\n"
        style_contents = textwrap.dedent('        [style]\n        based_on_style = yapf\n        use_tabs = true\n        column_limit=32\n        indent_width=4\n        continuation_indent_width=8\n        continuation_align_style = fixed\n    ')
        with utils.TempFileContents(self.test_tmpdir, style_contents) as stylepath:
            self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--style={0}'.format(stylepath)])

    def testUseTabsContinuationAlignStyleVAlignRight(self):
        if False:
            while True:
                i = 10
        unformatted_code = "def foo_function(arg1, arg2, arg3):\n  return ['hello', 'world',]\n"
        expected_formatted_code = "def foo_function(arg1, arg2,\n\t\t\t\t\targ3):\n\treturn [\n\t\t\t'hello',\n\t\t\t'world',\n\t]\n"
        style_contents = textwrap.dedent('        [style]\n        based_on_style = yapf\n        use_tabs = true\n        column_limit = 32\n        indent_width = 4\n        continuation_indent_width = 8\n        continuation_align_style = valign-right\n    ')
        with utils.TempFileContents(self.test_tmpdir, style_contents) as stylepath:
            self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--style={0}'.format(stylepath)])

    def testUseSpacesContinuationAlignStyleFixed(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent("        def foo_function(arg1, arg2, arg3):\n          return ['hello', 'world',]\n    ")
        expected_formatted_code = textwrap.dedent("        def foo_function(\n                arg1, arg2, arg3):\n            return [\n                    'hello',\n                    'world',\n            ]\n    ")
        style_contents = textwrap.dedent('        [style]\n        based_on_style = yapf\n        column_limit = 32\n        indent_width = 4\n        continuation_indent_width = 8\n        continuation_align_style = fixed\n    ')
        with utils.TempFileContents(self.test_tmpdir, style_contents) as stylepath:
            self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--style={0}'.format(stylepath)])

    def testUseSpacesContinuationAlignStyleVAlignRight(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent("        def foo_function(arg1, arg2, arg3):\n          return ['hello', 'world',]\n    ")
        expected_formatted_code = textwrap.dedent("        def foo_function(arg1, arg2,\n                            arg3):\n            return [\n                    'hello',\n                    'world',\n            ]\n    ")
        style_contents = textwrap.dedent('        [style]\n        based_on_style = yapf\n        column_limit = 32\n        indent_width = 4\n        continuation_indent_width = 8\n        continuation_align_style = valign-right\n    ')
        with utils.TempFileContents(self.test_tmpdir, style_contents) as stylepath:
            self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--style={0}'.format(stylepath)])

    def testStyleOutputRoundTrip(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        def foo_function():\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def foo_function():\n            pass\n    ')
        with utils.NamedTempFile(dirname=self.test_tmpdir) as (stylefile, stylepath):
            p = subprocess.Popen(YAPF_BINARY + ['--style-help'], stdout=stylefile, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            (_, stderrdata) = p.communicate()
            self.assertEqual(stderrdata, b'')
            self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--style={0}'.format(stylepath)])

    def testSpacingBeforeComments(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        A = 42\n\n\n        # A comment\n        def x():\n            pass\n        def _():\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        A = 42\n\n\n        # A comment\n        def x():\n            pass\n        def _():\n            pass\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '1-2'])

    def testSpacingBeforeCommentsInDicts(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        A=42\n\n        X = {\n            # 'Valid' statuses.\n            PASSED:  # Passed\n                'PASSED',\n            FAILED:  # Failed\n                'FAILED',\n            TIMED_OUT:  # Timed out.\n                'FAILED',\n            BORKED:  # Broken.\n                'BROKEN'\n        }\n    ")
        expected_formatted_code = textwrap.dedent("        A = 42\n\n        X = {\n            # 'Valid' statuses.\n            PASSED:  # Passed\n                'PASSED',\n            FAILED:  # Failed\n                'FAILED',\n            TIMED_OUT:  # Timed out.\n                'FAILED',\n            BORKED:  # Broken.\n                'BROKEN'\n        }\n    ")
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--style', 'yapf', '--lines', '1-1'])

    def testDisableWithLinesOption(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        # yapf_lines_bug.py\n        # yapf: disable\n        def outer_func():\n            def inner_func():\n                return\n            return\n        # yapf: enable\n    ')
        expected_formatted_code = textwrap.dedent('        # yapf_lines_bug.py\n        # yapf: disable\n        def outer_func():\n            def inner_func():\n                return\n            return\n        # yapf: enable\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--lines', '1-8'])

    def testDisableWithLineRanges(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        # yapf: disable\n        a = [\n            1,\n            2,\n\n            3\n        ]\n    ')
        expected_formatted_code = textwrap.dedent('        # yapf: disable\n        a = [\n            1,\n            2,\n\n            3\n        ]\n    ')
        self.assertYapfReformats(unformatted_code, expected_formatted_code, extra_options=['--style', 'yapf', '--lines', '1-100'])

class BadInputTest(yapf_test_helper.YAPFTest):
    """Test yapf's behaviour when passed bad input."""

    def testBadSyntax(self):
        if False:
            while True:
                i = 10
        code = '  a = 1\n'
        self.assertRaises(errors.YapfError, yapf_api.FormatCode, code)

    def testBadCode(self):
        if False:
            for i in range(10):
                print('nop')
        code = 'x = """hello\n'
        self.assertRaises(errors.YapfError, yapf_api.FormatCode, code)

class DiffIndentTest(yapf_test_helper.YAPFTest):

    @staticmethod
    def _OwnStyle():
        if False:
            i = 10
            return i + 15
        my_style = style.CreatePEP8Style()
        my_style['INDENT_WIDTH'] = 3
        my_style['CONTINUATION_INDENT_WIDTH'] = 3
        return my_style

    def _Check(self, unformatted_code, expected_formatted_code):
        if False:
            while True:
                i = 10
        (formatted_code, _) = yapf_api.FormatCode(unformatted_code, style_config=style.SetGlobalStyle(self._OwnStyle()))
        self.assertEqual(expected_formatted_code, formatted_code)

    def testSimple(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent("        for i in range(5):\n         print('bar')\n    ")
        expected_formatted_code = textwrap.dedent("        for i in range(5):\n           print('bar')\n    ")
        self._Check(unformatted_code, expected_formatted_code)

class HorizontallyAlignedTrailingCommentsTest(yapf_test_helper.YAPFTest):

    @staticmethod
    def _OwnStyle():
        if False:
            return 10
        my_style = style.CreatePEP8Style()
        my_style['SPACES_BEFORE_COMMENT'] = [15, 25, 35]
        return my_style

    def _Check(self, unformatted_code, expected_formatted_code):
        if False:
            return 10
        (formatted_code, _) = yapf_api.FormatCode(unformatted_code, style_config=style.SetGlobalStyle(self._OwnStyle()))
        self.assertCodeEqual(expected_formatted_code, formatted_code)

    def testSimple(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent("        foo = '1' # Aligned at first list value\n\n        foo = '2__<15>' # Aligned at second list value\n\n        foo = '3____________<25>' # Aligned at third list value\n\n        foo = '4______________________<35>' # Aligned beyond list values\n    ")
        expected_formatted_code = textwrap.dedent("        foo = '1'     # Aligned at first list value\n\n        foo = '2__<15>'         # Aligned at second list value\n\n        foo = '3____________<25>'         # Aligned at third list value\n\n        foo = '4______________________<35>' # Aligned beyond list values\n    ")
        self._Check(unformatted_code, expected_formatted_code)

    def testBlock(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        func(1)     # Line 1\n        func(2) # Line 2\n        # Line 3\n        func(3)                             # Line 4\n                                            # Line 5\n                                            # Line 6\n    ')
        expected_formatted_code = textwrap.dedent('        func(1)       # Line 1\n        func(2)       # Line 2\n                      # Line 3\n        func(3)       # Line 4\n                      # Line 5\n                      # Line 6\n    ')
        self._Check(unformatted_code, expected_formatted_code)

    def testBlockWithLongLine(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        func(1)     # Line 1\n        func___________________(2) # Line 2\n        # Line 3\n        func(3)                             # Line 4\n                                            # Line 5\n                                            # Line 6\n    ')
        expected_formatted_code = textwrap.dedent('        func(1)                           # Line 1\n        func___________________(2)        # Line 2\n                                          # Line 3\n        func(3)                           # Line 4\n                                          # Line 5\n                                          # Line 6\n    ')
        self._Check(unformatted_code, expected_formatted_code)

    def testBlockFuncSuffix(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        func(1)     # Line 1\n        func(2) # Line 2\n        # Line 3\n        func(3)                             # Line 4\n                                        # Line 5\n                                    # Line 6\n\n        def Func():\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        func(1)       # Line 1\n        func(2)       # Line 2\n                      # Line 3\n        func(3)       # Line 4\n                      # Line 5\n                      # Line 6\n\n\n        def Func():\n            pass\n    ')
        self._Check(unformatted_code, expected_formatted_code)

    def testBlockCommentSuffix(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        func(1)     # Line 1\n        func(2) # Line 2\n        # Line 3\n        func(3)                             # Line 4\n                                        # Line 5 - SpliceComments makes this part of the previous block\n                                    # Line 6\n\n                                            # Aligned with prev comment block\n    ')
        expected_formatted_code = textwrap.dedent('        func(1)       # Line 1\n        func(2)       # Line 2\n                      # Line 3\n        func(3)       # Line 4\n                      # Line 5 - SpliceComments makes this part of the previous block\n                      # Line 6\n\n                      # Aligned with prev comment block\n    ')
        self._Check(unformatted_code, expected_formatted_code)

    def testBlockIndentedFuncSuffix(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        if True:\n            func(1)     # Line 1\n            func(2) # Line 2\n            # Line 3\n            func(3)                             # Line 4\n                                                # Line 5 - SpliceComments makes this a new block\n                                                # Line 6\n\n                                                # Aligned with Func\n\n            def Func():\n                pass\n    ')
        expected_formatted_code = textwrap.dedent('        if True:\n            func(1)   # Line 1\n            func(2)   # Line 2\n                      # Line 3\n            func(3)   # Line 4\n\n            # Line 5 - SpliceComments makes this a new block\n            # Line 6\n\n            # Aligned with Func\n\n\n            def Func():\n                pass\n    ')
        self._Check(unformatted_code, expected_formatted_code)

    def testBlockIndentedCommentSuffix(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        if True:\n            func(1)     # Line 1\n            func(2) # Line 2\n            # Line 3\n            func(3)                             # Line 4\n                                                # Line 5\n                                                # Line 6\n\n                                                # Not aligned\n    ')
        expected_formatted_code = textwrap.dedent('        if True:\n            func(1)   # Line 1\n            func(2)   # Line 2\n                      # Line 3\n            func(3)   # Line 4\n                      # Line 5\n                      # Line 6\n\n            # Not aligned\n    ')
        self._Check(unformatted_code, expected_formatted_code)

    def testBlockMultiIndented(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        if True:\n            if True:\n                if True:\n                    func(1)     # Line 1\n                    func(2) # Line 2\n                    # Line 3\n                    func(3)                             # Line 4\n                                                        # Line 5\n                                                        # Line 6\n\n                                                        # Not aligned\n    ')
        expected_formatted_code = textwrap.dedent('        if True:\n            if True:\n                if True:\n                    func(1)     # Line 1\n                    func(2)     # Line 2\n                                # Line 3\n                    func(3)     # Line 4\n                                # Line 5\n                                # Line 6\n\n                    # Not aligned\n    ')
        self._Check(unformatted_code, expected_formatted_code)

    def testArgs(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        def MyFunc(\n            arg1,   # Desc 1\n            arg2,   # Desc 2\n            a_longer_var_name,  # Desc 3\n            arg4,\n            arg5,   # Desc 5\n            arg6,\n        ):\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        def MyFunc(\n            arg1,               # Desc 1\n            arg2,               # Desc 2\n            a_longer_var_name,  # Desc 3\n            arg4,\n            arg5,               # Desc 5\n            arg6,\n        ):\n            pass\n    ')
        self._Check(unformatted_code, expected_formatted_code)

    def testDisableBlock(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        a() # comment 1\n        b() # comment 2\n\n        # yapf: disable\n        c() # comment 3\n        d()   # comment 4\n        # yapf: enable\n\n        e() # comment 5\n        f() # comment 6\n    ')
        expected_formatted_code = textwrap.dedent('        a()           # comment 1\n        b()           # comment 2\n\n        # yapf: disable\n        c() # comment 3\n        d()   # comment 4\n        # yapf: enable\n\n        e()           # comment 5\n        f()           # comment 6\n    ')
        self._Check(unformatted_code, expected_formatted_code)

    def testDisabledLine(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        short # comment 1\n        do_not_touch1 # yapf: disable\n        do_not_touch2   # yapf: disable\n        a_longer_statement # comment 2\n    ')
        expected_formatted_code = textwrap.dedent('        short                   # comment 1\n        do_not_touch1 # yapf: disable\n        do_not_touch2   # yapf: disable\n        a_longer_statement      # comment 2\n    ')
        self._Check(unformatted_code, expected_formatted_code)

class _SpacesAroundDictListTupleTestImpl(yapf_test_helper.YAPFTest):

    @staticmethod
    def _OwnStyle():
        if False:
            i = 10
            return i + 15
        my_style = style.CreatePEP8Style()
        my_style['DISABLE_ENDING_COMMA_HEURISTIC'] = True
        my_style['SPLIT_ALL_COMMA_SEPARATED_VALUES'] = False
        my_style['SPLIT_ARGUMENTS_WHEN_COMMA_TERMINATED'] = False
        return my_style

    def _Check(self, unformatted_code, expected_formatted_code):
        if False:
            i = 10
            return i + 15
        (formatted_code, _) = yapf_api.FormatCode(unformatted_code, style_config=style.SetGlobalStyle(self._OwnStyle()))
        self.assertEqual(expected_formatted_code, formatted_code)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.maxDiff = None

class SpacesAroundDictTest(_SpacesAroundDictListTupleTestImpl):

    @classmethod
    def _OwnStyle(cls):
        if False:
            for i in range(10):
                print('nop')
        style = super(SpacesAroundDictTest, cls)._OwnStyle()
        style['SPACES_AROUND_DICT_DELIMITERS'] = True
        return style

    def testStandard(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        {1 : 2}\n        {k:v for k, v in other.items()}\n        {k for k in [1, 2, 3]}\n\n        # The following statements should not change\n        {}\n        {1 : 2} # yapf: disable\n\n        # yapf: disable\n        {1 : 2}\n        # yapf: enable\n\n        # Dict settings should not impact lists or tuples\n        [1, 2]\n        (3, 4)\n    ')
        expected_formatted_code = textwrap.dedent('        { 1: 2 }\n        { k: v for k, v in other.items() }\n        { k for k in [1, 2, 3] }\n\n        # The following statements should not change\n        {}\n        {1 : 2} # yapf: disable\n\n        # yapf: disable\n        {1 : 2}\n        # yapf: enable\n\n        # Dict settings should not impact lists or tuples\n        [1, 2]\n        (3, 4)\n    ')
        self._Check(unformatted_code, expected_formatted_code)

class SpacesAroundListTest(_SpacesAroundDictListTupleTestImpl):

    @classmethod
    def _OwnStyle(cls):
        if False:
            while True:
                i = 10
        style = super(SpacesAroundListTest, cls)._OwnStyle()
        style['SPACES_AROUND_LIST_DELIMITERS'] = True
        return style

    def testStandard(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        [a,b,c]\n        [4,5,]\n        [6, [7, 8], 9]\n        [v for v in [1,2,3] if v & 1]\n\n        # The following statements should not change\n        index[0]\n        index[a, b]\n        []\n        [v for v in [1,2,3] if v & 1] # yapf: disable\n\n        # yapf: disable\n        [a,b,c]\n        [4,5,]\n        # yapf: enable\n\n        # List settings should not impact dicts or tuples\n        {a: b}\n        (1, 2)\n    ')
        expected_formatted_code = textwrap.dedent('        [ a, b, c ]\n        [ 4, 5, ]\n        [ 6, [ 7, 8 ], 9 ]\n        [ v for v in [ 1, 2, 3 ] if v & 1 ]\n\n        # The following statements should not change\n        index[0]\n        index[a, b]\n        []\n        [v for v in [1,2,3] if v & 1] # yapf: disable\n\n        # yapf: disable\n        [a,b,c]\n        [4,5,]\n        # yapf: enable\n\n        # List settings should not impact dicts or tuples\n        {a: b}\n        (1, 2)\n    ')
        self._Check(unformatted_code, expected_formatted_code)

class SpacesAroundTupleTest(_SpacesAroundDictListTupleTestImpl):

    @classmethod
    def _OwnStyle(cls):
        if False:
            print('Hello World!')
        style = super(SpacesAroundTupleTest, cls)._OwnStyle()
        style['SPACES_AROUND_TUPLE_DELIMITERS'] = True
        return style

    def testStandard(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        (0, 1)\n        (2, 3)\n        (4, 5, 6,)\n        func((7, 8), 9)\n\n        # The following statements should not change\n        func(1, 2)\n        (this_func or that_func)(3, 4)\n        if (True and False): pass\n        ()\n\n        (0, 1) # yapf: disable\n\n        # yapf: disable\n        (0, 1)\n        (2, 3)\n        # yapf: enable\n\n        # Tuple settings should not impact dicts or lists\n        {a: b}\n        [3, 4]\n    ')
        expected_formatted_code = textwrap.dedent('        ( 0, 1 )\n        ( 2, 3 )\n        ( 4, 5, 6, )\n        func(( 7, 8 ), 9)\n\n        # The following statements should not change\n        func(1, 2)\n        (this_func or that_func)(3, 4)\n        if (True and False): pass\n        ()\n\n        (0, 1) # yapf: disable\n\n        # yapf: disable\n        (0, 1)\n        (2, 3)\n        # yapf: enable\n\n        # Tuple settings should not impact dicts or lists\n        {a: b}\n        [3, 4]\n    ')
        self._Check(unformatted_code, expected_formatted_code)
if __name__ == '__main__':
    unittest.main()