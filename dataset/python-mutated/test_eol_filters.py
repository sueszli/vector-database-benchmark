"""Tests for eol conversion."""
from bzrlib import errors
from bzrlib.filters import _get_filter_stack_for
from bzrlib.filters.eol import _to_crlf_converter, _to_lf_converter
from bzrlib.tests import TestCase
_sample_file1 = 'hello\nworld\r\n'

class TestEolFilters(TestCase):

    def test_to_lf(self):
        if False:
            while True:
                i = 10
        result = _to_lf_converter([_sample_file1])
        self.assertEqual(['hello\nworld\n'], result)

    def test_to_crlf(self):
        if False:
            return 10
        result = _to_crlf_converter([_sample_file1])
        self.assertEqual(['hello\r\nworld\r\n'], result)

class TestEolRulesSpecifications(TestCase):

    def test_exact_value(self):
        if False:
            for i in range(10):
                print('nop')
        "'eol = exact' should have no content filters"
        prefs = (('eol', 'exact'),)
        self.assertEqual([], _get_filter_stack_for(prefs))

    def test_other_known_values(self):
        if False:
            return 10
        'These known eol values have corresponding filters.'
        known_values = ('lf', 'crlf', 'native', 'native-with-crlf-in-repo', 'lf-with-crlf-in-repo', 'crlf-with-crlf-in-repo')
        for value in known_values:
            prefs = (('eol', value),)
            self.assertNotEqual([], _get_filter_stack_for(prefs))

    def test_unknown_value(self):
        if False:
            while True:
                i = 10
        '\n        Unknown eol values should raise an error.\n        '
        prefs = (('eol', 'unknown-value'),)
        self.assertRaises(errors.BzrError, _get_filter_stack_for, prefs)

    def test_eol_missing_altogether_is_ok(self):
        if False:
            return 10
        '\n        Not having eol in the set of preferences should be ok.\n        '
        prefs = (('eol', None),)
        self.assertEqual([], _get_filter_stack_for(prefs))