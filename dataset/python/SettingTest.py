import os
import unittest
from collections import OrderedDict

from coalib.bearlib.languages import Language
from coalib.settings.Setting import (
    Setting, path, path_list, url, typed_dict, typed_list, typed_ordered_dict,
    glob, glob_list,
    language,
    float_list, bool_list, int_list, str_list,
)
from coalib.parsing.DefaultArgParser import PathArg
from coalib.parsing.Globbing import glob_escape
from coalib.results.SourcePosition import SourcePosition


class SettingTest(unittest.TestCase):

    def test_constructor_signature(self):
        self.assertRaises(TypeError, Setting, '', 2, 2)
        self.assertRaises(TypeError, Setting, '', '', '', from_cli=5)
        self.assertRaisesRegex(TypeError, 'to_append',
                               Setting, '', '', '', to_append=10)
        self.assertRaises(TypeError, Setting, 'a', 'b', list_delimiters=5)
        self.assertRaises(TypeError, Setting, 'a', 'b', list_delimiters=None)

    def test_empty_key(self):
        with self.assertRaisesRegex(
                ValueError, 'An empty key is not allowed for a setting'):
            Setting('', 2)

    def test_path(self):
        self.uut = Setting(
            'key', ' 22\n', '.' + os.path.sep, strip_whitespaces=True)
        self.assertEqual(path(self.uut),
                         os.path.abspath(os.path.join('.', '22')))

        abspath = PathArg(os.path.abspath('.'))
        self.uut = Setting('key', abspath)
        self.assertEqual(path(self.uut), abspath)

        self.uut = Setting('key', ' 22', '')
        self.assertRaises(ValueError, path, self.uut)
        self.assertEqual(path(self.uut,
                              origin='test' + os.path.sep),
                         os.path.abspath(os.path.join('test', '22')))

        self.uut = Setting('key', '22\n',
                           SourcePosition('.' + os.path.sep),
                           strip_whitespaces=True)
        self.assertEqual(path(self.uut),
                         os.path.abspath(os.path.join('.', '22')))

    def test_glob(self):
        self.uut = Setting('key', '.',
                           origin=os.path.join('test (1)', 'somefile'))
        self.assertEqual(glob(self.uut),
                         glob_escape(os.path.abspath('test (1)')))

        self.uut = Setting('key', '.',
                           origin=SourcePosition(
                                  os.path.join('test (1)', 'somefile')))
        self.assertEqual(glob(self.uut),
                         glob_escape(os.path.abspath('test (1)')))

    def test_path_list(self):
        abspath = os.path.abspath('.')
        # Need to escape backslashes since we use list conversion
        self.uut = Setting('key', '., ' + abspath.replace('\\', '\\\\'),
                           origin=os.path.join('test', 'somefile'))
        self.assertEqual(path_list(self.uut),
                         [os.path.abspath(os.path.join('test', '.')), abspath])

        self.uut = Setting('key', '., ' + abspath.replace('\\', '\\\\'),
                           origin=SourcePosition(
                                  os.path.join('test', 'somefile')))
        self.assertEqual(path_list(self.uut),
                         [os.path.abspath(os.path.join('test', '.')), abspath])

    def test_url(self):
        uut = Setting('key', 'http://google.com')
        self.assertEqual(url(uut), 'http://google.com')

        with self.assertRaises(ValueError):
            uut = Setting('key', 'abc')
            url(uut)

    def test_glob_list(self):
        abspath = glob_escape(os.path.abspath('.'))
        # Need to escape backslashes since we use list conversion
        self.uut = Setting('key', '., ' + abspath.replace('\\', '\\\\'),
                           origin=os.path.join('test (1)', 'somefile'))
        self.assertEqual(
            glob_list(self.uut),
            [glob_escape(os.path.abspath(os.path.join('test (1)', '.'))),
             abspath])

        self.uut = Setting('key', '.,' + abspath.replace('\\', '\\\\'),
                           origin=SourcePosition(
                                  os.path.join('test (1)', 'somefile')))
        self.assertEqual(glob_list(self.uut),
                         [glob_escape(os.path.abspath(
                                      os.path.join('test (1)', '.'))), abspath])

    def test_language(self):
        self.uut = Setting('key', 'python 3.4')
        result = language(self.uut)
        self.assertIsInstance(result, Language)
        self.assertEqual(str(result), 'Python 3.4')

    def test_language_invalid(self):
        self.uut = Setting('key', 'not a language')
        with self.assertRaisesRegexp(
                ValueError,
                'Language `not a language` is not a valid language name or '
                'not recognized by coala.'):
            language(self.uut)

    def test_typed_list(self):
        self.uut = Setting('key', '1, 2, 3')
        self.assertEqual(typed_list(int)(self.uut),
                         [1, 2, 3])

        with self.assertRaises(ValueError):
            self.uut = Setting('key', '1, a, 3')
            typed_list(int)(self.uut)

        self.assertRegex(repr(typed_list(int)),
                         'typed_list\\(int\\) at \\(0x[a-fA-F0-9]+\\)')

    def test_int_list(self):
        self.uut = Setting('key', '1, 2, 3')
        self.assertEqual(int_list(self.uut), [1, 2, 3])

        with self.assertRaises(ValueError):
            self.uut = Setting('key', '1, a, 3')
            int_list(self.uut)

        self.assertRegex(
            repr(int_list), 'typed_list\\(int\\) at \\(0x[a-fA-F0-9]+\\)')

    def test_str_list(self):
        self.uut = Setting('key', 'a, b, c')
        self.assertEqual(str_list(self.uut), ['a', 'b', 'c'])
        self.assertRegex(
            repr(str_list), 'typed_list\\(str\\) at \\(0x[a-fA-F0-9]+\\)')

    def test_float_list(self):
        self.uut = Setting('key', '0.8, 1.3, 5.87')
        self.assertEqual(float_list(self.uut), [0.8, 1.3, 5.87])

        with self.assertRaises(ValueError):
            self.uut = Setting('key', '1.987, a, 3')
            float_list(self.uut)

        self.assertRegex(
            repr(float_list), 'typed_list\\(float\\) at \\(0x[a-fA-F0-9]+\\)')

    def test_bool_list(self):
        self.uut = Setting('key', 'true, nope, yeah')
        self.assertEqual(bool_list(self.uut), [True, False, True])

        with self.assertRaises(ValueError):
            self.uut = Setting('key', 'true, false, 78, 89.0')
            bool_list(self.uut)

        self.assertRegex(
            repr(bool_list), 'typed_list\\(bool\\) at \\(0x[a-fA-F0-9]+\\)')

    def test_typed_dict(self):
        self.uut = Setting('key', '1, 2: t, 3')
        self.assertEqual(typed_dict(int, str, None)(self.uut),
                         {1: None, 2: 't', 3: None})

        with self.assertRaises(ValueError):
            self.uut = Setting('key', '1, a, 3')
            typed_dict(int, str, '')(self.uut)

        self.assertRegex(
            repr(typed_dict(int, str, None)),
            'typed_dict\\(int, str, default=None\\) at \\(0x[a-fA-F0-9]+\\)'
        )

    def test_typed_ordered_dict(self):
        self.uut = Setting('key', '1, 2: t, 3')
        self.assertEqual(typed_ordered_dict(int, str, None)(self.uut),
                         OrderedDict([(1, None), (2, 't'), (3, None)]))

        with self.assertRaises(ValueError):
            self.uut = Setting('key', '1, a, 3')
            typed_ordered_dict(int, str, '')(self.uut)

        self.assertRegex(
            repr(typed_ordered_dict(int, str, None)),
            # Start ignoring LineLengthBear, PyCodeStyleBear
            'typed_ordered_dict\\(int, str, default=None\\) at \\(0x[a-fA-F0-9]+\\)'
            # Stop ignoring
        )

    def test_inherited_conversions(self):
        self.uut = Setting('key', ' 22\n', '.', strip_whitespaces=True)
        self.assertEqual(str(self.uut), '22')
        self.assertEqual(int(self.uut), 22)
        self.assertRaises(ValueError, bool, self.uut)

    def test_value_getter(self):
        with self.assertRaisesRegex(ValueError, 'This property is invalid'):
            self.uut = Setting('key', '22\n', '.', to_append=True)
            self.uut.value

        with self.assertRaisesRegex(ValueError,
                                    'Iteration on this object is invalid'):
            self.uut = Setting('key', '1, 2, 3', '.', to_append=True)
            list(self.uut)

    def test_line_number(self):
        self.uut = Setting('key', '22\n', origin=SourcePosition('filename', 3))
        self.assertEqual(self.uut.line_number, 3)

        with self.assertRaisesRegex(TypeError,
                                    "Instantiated with str 'origin' "
                                    'which does not have line numbers. '
                                    'Use SourcePosition for line numbers.'):
            self.uut = Setting('key', '22\n', origin='filename')
            self.uut.line_number

    def test_end_line_number(self):
        with self.assertRaisesRegex(TypeError,
                                    "Instantiated with str 'origin' "
                                    'which does not have line numbers. '
                                    'Use SourcePosition for line numbers.'):
            self.uut = Setting('key', '22\n', origin='filename')
            self.uut.end_line_number
