"""Tests for yapf.style."""
import os
import shutil
import tempfile
import textwrap
import unittest
from yapf.yapflib import style
from yapftests import utils
from yapftests import yapf_test_helper

class UtilsTest(yapf_test_helper.YAPFTest):

    def testContinuationAlignStyleStringConverter(self):
        if False:
            while True:
                i = 10
        for cont_align_space in ('', 'space', '"space"', "'space'"):
            self.assertEqual(style._ContinuationAlignStyleStringConverter(cont_align_space), 'SPACE')
        for cont_align_fixed in ('fixed', '"fixed"', "'fixed'"):
            self.assertEqual(style._ContinuationAlignStyleStringConverter(cont_align_fixed), 'FIXED')
        for cont_align_valignright in ('valign-right', '"valign-right"', "'valign-right'", 'valign_right', '"valign_right"', "'valign_right'"):
            self.assertEqual(style._ContinuationAlignStyleStringConverter(cont_align_valignright), 'VALIGN-RIGHT')
        with self.assertRaises(ValueError) as ctx:
            style._ContinuationAlignStyleStringConverter('blahblah')
        self.assertIn("unknown continuation align style: 'blahblah'", str(ctx.exception))

    def testStringListConverter(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(style._StringListConverter('foo, bar'), ['foo', 'bar'])
        self.assertEqual(style._StringListConverter('foo,bar'), ['foo', 'bar'])
        self.assertEqual(style._StringListConverter('  foo'), ['foo'])
        self.assertEqual(style._StringListConverter('joe  ,foo,  bar'), ['joe', 'foo', 'bar'])

    def testBoolConverter(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(style._BoolConverter('true'), True)
        self.assertEqual(style._BoolConverter('1'), True)
        self.assertEqual(style._BoolConverter('false'), False)
        self.assertEqual(style._BoolConverter('0'), False)

    def testIntListConverter(self):
        if False:
            return 10
        self.assertEqual(style._IntListConverter('1, 2, 3'), [1, 2, 3])
        self.assertEqual(style._IntListConverter('[ 1, 2, 3 ]'), [1, 2, 3])
        self.assertEqual(style._IntListConverter('[ 1, 2, 3, ]'), [1, 2, 3])

    def testIntOrIntListConverter(self):
        if False:
            print('Hello World!')
        self.assertEqual(style._IntOrIntListConverter('10'), 10)
        self.assertEqual(style._IntOrIntListConverter('1, 2, 3'), [1, 2, 3])

def _LooksLikeGoogleStyle(cfg):
    if False:
        return 10
    return cfg['COLUMN_LIMIT'] == 80 and cfg['SPLIT_COMPLEX_COMPREHENSION']

def _LooksLikePEP8Style(cfg):
    if False:
        print('Hello World!')
    return cfg['COLUMN_LIMIT'] == 79

def _LooksLikeFacebookStyle(cfg):
    if False:
        return 10
    return cfg['DEDENT_CLOSING_BRACKETS']

def _LooksLikeYapfStyle(cfg):
    if False:
        for i in range(10):
            print('nop')
    return cfg['SPLIT_BEFORE_DOT']

class PredefinedStylesByNameTest(yapf_test_helper.YAPFTest):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        style.SetGlobalStyle(style.CreatePEP8Style())

    def testDefault(self):
        if False:
            print('Hello World!')
        cfg = style.CreateStyleFromConfig(None)
        self.assertTrue(_LooksLikePEP8Style(cfg))

    def testPEP8ByName(self):
        if False:
            for i in range(10):
                print('nop')
        for pep8_name in ('PEP8', 'pep8', 'Pep8'):
            cfg = style.CreateStyleFromConfig(pep8_name)
            self.assertTrue(_LooksLikePEP8Style(cfg))

    def testGoogleByName(self):
        if False:
            while True:
                i = 10
        for google_name in ('google', 'Google', 'GOOGLE'):
            cfg = style.CreateStyleFromConfig(google_name)
            self.assertTrue(_LooksLikeGoogleStyle(cfg))

    def testYapfByName(self):
        if False:
            return 10
        for yapf_name in ('yapf', 'YAPF'):
            cfg = style.CreateStyleFromConfig(yapf_name)
            self.assertTrue(_LooksLikeYapfStyle(cfg))

    def testFacebookByName(self):
        if False:
            for i in range(10):
                print('nop')
        for fb_name in ('facebook', 'FACEBOOK', 'Facebook'):
            cfg = style.CreateStyleFromConfig(fb_name)
            self.assertTrue(_LooksLikeFacebookStyle(cfg))

class StyleFromFileTest(yapf_test_helper.YAPFTest):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.test_tmpdir = tempfile.mkdtemp()
        style.SetGlobalStyle(style.CreatePEP8Style())

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        shutil.rmtree(cls.test_tmpdir)

    def testDefaultBasedOnStyle(self):
        if False:
            print('Hello World!')
        cfg = textwrap.dedent('        [style]\n        continuation_indent_width = 20\n    ')
        with utils.TempFileContents(self.test_tmpdir, cfg) as filepath:
            cfg = style.CreateStyleFromConfig(filepath)
            self.assertTrue(_LooksLikePEP8Style(cfg))
            self.assertEqual(cfg['CONTINUATION_INDENT_WIDTH'], 20)

    def testDefaultBasedOnPEP8Style(self):
        if False:
            while True:
                i = 10
        cfg = textwrap.dedent('        [style]\n        based_on_style = pep8\n        continuation_indent_width = 40\n    ')
        with utils.TempFileContents(self.test_tmpdir, cfg) as filepath:
            cfg = style.CreateStyleFromConfig(filepath)
            self.assertTrue(_LooksLikePEP8Style(cfg))
            self.assertEqual(cfg['CONTINUATION_INDENT_WIDTH'], 40)

    def testDefaultBasedOnGoogleStyle(self):
        if False:
            return 10
        cfg = textwrap.dedent('        [style]\n        based_on_style = google\n        continuation_indent_width = 20\n    ')
        with utils.TempFileContents(self.test_tmpdir, cfg) as filepath:
            cfg = style.CreateStyleFromConfig(filepath)
            self.assertTrue(_LooksLikeGoogleStyle(cfg))
            self.assertEqual(cfg['CONTINUATION_INDENT_WIDTH'], 20)

    def testDefaultBasedOnFacebookStyle(self):
        if False:
            return 10
        cfg = textwrap.dedent('        [style]\n        based_on_style = facebook\n        continuation_indent_width = 20\n    ')
        with utils.TempFileContents(self.test_tmpdir, cfg) as filepath:
            cfg = style.CreateStyleFromConfig(filepath)
            self.assertTrue(_LooksLikeFacebookStyle(cfg))
            self.assertEqual(cfg['CONTINUATION_INDENT_WIDTH'], 20)

    def testBoolOptionValue(self):
        if False:
            for i in range(10):
                print('nop')
        cfg = textwrap.dedent('        [style]\n        based_on_style = pep8\n        SPLIT_BEFORE_NAMED_ASSIGNS=False\n        split_before_logical_operator = true\n    ')
        with utils.TempFileContents(self.test_tmpdir, cfg) as filepath:
            cfg = style.CreateStyleFromConfig(filepath)
            self.assertTrue(_LooksLikePEP8Style(cfg))
            self.assertEqual(cfg['SPLIT_BEFORE_NAMED_ASSIGNS'], False)
            self.assertEqual(cfg['SPLIT_BEFORE_LOGICAL_OPERATOR'], True)

    def testStringListOptionValue(self):
        if False:
            print('Hello World!')
        cfg = textwrap.dedent('        [style]\n        based_on_style = pep8\n        I18N_FUNCTION_CALL = N_, V_, T_\n    ')
        with utils.TempFileContents(self.test_tmpdir, cfg) as filepath:
            cfg = style.CreateStyleFromConfig(filepath)
            self.assertTrue(_LooksLikePEP8Style(cfg))
            self.assertEqual(cfg['I18N_FUNCTION_CALL'], ['N_', 'V_', 'T_'])

    def testErrorNoStyleFile(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(style.StyleConfigError, 'is not a valid style or file path'):
            style.CreateStyleFromConfig('/8822/xyznosuchfile')

    def testErrorNoStyleSection(self):
        if False:
            print('Hello World!')
        cfg = textwrap.dedent('        [s]\n        indent_width=2\n    ')
        with utils.TempFileContents(self.test_tmpdir, cfg) as filepath:
            with self.assertRaisesRegex(style.StyleConfigError, 'Unable to find section'):
                style.CreateStyleFromConfig(filepath)

    def testErrorUnknownStyleOption(self):
        if False:
            i = 10
            return i + 15
        cfg = textwrap.dedent('        [style]\n        indent_width=2\n        hummus=2\n    ')
        with utils.TempFileContents(self.test_tmpdir, cfg) as filepath:
            with self.assertRaisesRegex(style.StyleConfigError, 'Unknown style option'):
                style.CreateStyleFromConfig(filepath)

    def testPyprojectTomlNoYapfSection(self):
        if False:
            i = 10
            return i + 15
        filepath = os.path.join(self.test_tmpdir, 'pyproject.toml')
        _ = open(filepath, 'w')
        with self.assertRaisesRegex(style.StyleConfigError, 'Unable to find section'):
            style.CreateStyleFromConfig(filepath)

    def testPyprojectTomlParseYapfSection(self):
        if False:
            while True:
                i = 10
        cfg = textwrap.dedent('        [tool.yapf]\n        based_on_style = "pep8"\n        continuation_indent_width = 40\n    ')
        filepath = os.path.join(self.test_tmpdir, 'pyproject.toml')
        with open(filepath, 'w') as f:
            f.write(cfg)
        cfg = style.CreateStyleFromConfig(filepath)
        self.assertTrue(_LooksLikePEP8Style(cfg))
        self.assertEqual(cfg['CONTINUATION_INDENT_WIDTH'], 40)

class StyleFromDict(yapf_test_helper.YAPFTest):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        style.SetGlobalStyle(style.CreatePEP8Style())

    def testDefaultBasedOnStyle(self):
        if False:
            for i in range(10):
                print('nop')
        config_dict = {'based_on_style': 'pep8', 'indent_width': 2, 'blank_line_before_nested_class_or_def': True}
        cfg = style.CreateStyleFromConfig(config_dict)
        self.assertTrue(_LooksLikePEP8Style(cfg))
        self.assertEqual(cfg['INDENT_WIDTH'], 2)

    def testDefaultBasedOnStyleBadDict(self):
        if False:
            while True:
                i = 10
        self.assertRaisesRegex(style.StyleConfigError, 'Unknown style option', style.CreateStyleFromConfig, {'based_on_styl': 'pep8'})
        self.assertRaisesRegex(style.StyleConfigError, 'not a valid', style.CreateStyleFromConfig, {'INDENT_WIDTH': 'FOUR'})

class StyleFromCommandLine(yapf_test_helper.YAPFTest):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        style.SetGlobalStyle(style.CreatePEP8Style())

    def testDefaultBasedOnStyle(self):
        if False:
            for i in range(10):
                print('nop')
        cfg = style.CreateStyleFromConfig('{based_on_style: pep8, indent_width: 2, blank_line_before_nested_class_or_def: True}')
        self.assertTrue(_LooksLikePEP8Style(cfg))
        self.assertEqual(cfg['INDENT_WIDTH'], 2)

    def testDefaultBasedOnStyleNotStrict(self):
        if False:
            i = 10
            return i + 15
        cfg = style.CreateStyleFromConfig('{based_on_style : pep8, indent_width=2 blank_line_before_nested_class_or_def:True}')
        self.assertTrue(_LooksLikePEP8Style(cfg))
        self.assertEqual(cfg['INDENT_WIDTH'], 2)

    def testDefaultBasedOnExplicitlyUnicodeTypeString(self):
        if False:
            while True:
                i = 10
        cfg = style.CreateStyleFromConfig('{}')
        self.assertIsInstance(cfg, dict)

    def testDefaultBasedOnDetaultTypeString(self):
        if False:
            while True:
                i = 10
        cfg = style.CreateStyleFromConfig('{}')
        self.assertIsInstance(cfg, dict)

    def testDefaultBasedOnStyleBadString(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaisesRegex(style.StyleConfigError, 'Unknown style option', style.CreateStyleFromConfig, '{based_on_styl: pep8}')
        self.assertRaisesRegex(style.StyleConfigError, 'not a valid', style.CreateStyleFromConfig, '{INDENT_WIDTH: FOUR}')
        self.assertRaisesRegex(style.StyleConfigError, 'Invalid style dict', style.CreateStyleFromConfig, '{based_on_style: pep8')

class StyleHelp(yapf_test_helper.YAPFTest):

    def testHelpKeys(self):
        if False:
            print('Hello World!')
        settings = sorted(style.Help())
        expected = sorted(style._style)
        self.assertListEqual(settings, expected)
if __name__ == '__main__':
    unittest.main()