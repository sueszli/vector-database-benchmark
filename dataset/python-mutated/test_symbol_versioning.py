"""Symbol versioning tests."""
import warnings
from bzrlib import symbol_versioning
from bzrlib.symbol_versioning import deprecated_function, deprecated_in, deprecated_method
from bzrlib.tests import TestCase

@deprecated_function(deprecated_in((0, 7, 0)))
def sample_deprecated_function():
    if False:
        while True:
            i = 10
    'Deprecated function docstring.'
    return 1
a_deprecated_list = symbol_versioning.deprecated_list(deprecated_in((0, 9, 0)), 'a_deprecated_list', ['one'], extra="Don't use me")
a_deprecated_dict = symbol_versioning.DeprecatedDict(deprecated_in((0, 14, 0)), 'a_deprecated_dict', dict(a=42), advice='Pull the other one!')

class TestDeprecationWarnings(TestCase):

    def capture_warning(self, message, category, stacklevel=None):
        if False:
            for i in range(10):
                print('nop')
        self._warnings.append((message, category, stacklevel))

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestDeprecationWarnings, self).setUp()
        self._warnings = []

    @deprecated_method(deprecated_in((0, 7, 0)))
    def deprecated_method(self):
        if False:
            return 10
        'Deprecated method docstring.\n\n        This might explain stuff.\n        '
        return 1

    @staticmethod
    @deprecated_function(deprecated_in((0, 7, 0)))
    def deprecated_static():
        if False:
            print('Hello World!')
        'Deprecated static.'
        return 1

    def test_deprecated_static(self):
        if False:
            while True:
                i = 10
        expected_warning = ('bzrlib.tests.test_symbol_versioning.deprecated_static was deprecated in version 0.7.0.', DeprecationWarning, 2)
        expected_docstring = 'Deprecated static.\n\nThis function was deprecated in version 0.7.0.\n'
        self.check_deprecated_callable(expected_warning, expected_docstring, 'deprecated_static', 'bzrlib.tests.test_symbol_versioning', self.deprecated_static)

    def test_deprecated_method(self):
        if False:
            i = 10
            return i + 15
        expected_warning = ('bzrlib.tests.test_symbol_versioning.TestDeprecationWarnings.deprecated_method was deprecated in version 0.7.0.', DeprecationWarning, 2)
        expected_docstring = 'Deprecated method docstring.\n\n        This might explain stuff.\n        \n        This method was deprecated in version 0.7.0.\n        '
        self.check_deprecated_callable(expected_warning, expected_docstring, 'deprecated_method', 'bzrlib.tests.test_symbol_versioning', self.deprecated_method)

    def test_deprecated_function(self):
        if False:
            print('Hello World!')
        expected_warning = ('bzrlib.tests.test_symbol_versioning.sample_deprecated_function was deprecated in version 0.7.0.', DeprecationWarning, 2)
        expected_docstring = 'Deprecated function docstring.\n\nThis function was deprecated in version 0.7.0.\n'
        self.check_deprecated_callable(expected_warning, expected_docstring, 'sample_deprecated_function', 'bzrlib.tests.test_symbol_versioning', sample_deprecated_function)

    def test_deprecated_list(self):
        if False:
            while True:
                i = 10
        expected_warning = ("Modifying a_deprecated_list was deprecated in version 0.9.0. Don't use me", DeprecationWarning, 3)
        old_warning_method = symbol_versioning.warn
        try:
            symbol_versioning.set_warning_method(self.capture_warning)
            self.assertEqual(['one'], a_deprecated_list)
            self.assertEqual([], self._warnings)
            a_deprecated_list.append('foo')
            self.assertEqual([expected_warning], self._warnings)
            self.assertEqual(['one', 'foo'], a_deprecated_list)
            a_deprecated_list.extend(['bar', 'baz'])
            self.assertEqual([expected_warning] * 2, self._warnings)
            self.assertEqual(['one', 'foo', 'bar', 'baz'], a_deprecated_list)
            a_deprecated_list.insert(1, 'xxx')
            self.assertEqual([expected_warning] * 3, self._warnings)
            self.assertEqual(['one', 'xxx', 'foo', 'bar', 'baz'], a_deprecated_list)
            a_deprecated_list.remove('foo')
            self.assertEqual([expected_warning] * 4, self._warnings)
            self.assertEqual(['one', 'xxx', 'bar', 'baz'], a_deprecated_list)
            val = a_deprecated_list.pop()
            self.assertEqual([expected_warning] * 5, self._warnings)
            self.assertEqual('baz', val)
            self.assertEqual(['one', 'xxx', 'bar'], a_deprecated_list)
            val = a_deprecated_list.pop(1)
            self.assertEqual([expected_warning] * 6, self._warnings)
            self.assertEqual('xxx', val)
            self.assertEqual(['one', 'bar'], a_deprecated_list)
        finally:
            symbol_versioning.set_warning_method(old_warning_method)

    def test_deprecated_dict(self):
        if False:
            i = 10
            return i + 15
        expected_warning = ('access to a_deprecated_dict was deprecated in version 0.14.0. Pull the other one!', DeprecationWarning, 2)
        old_warning_method = symbol_versioning.warn
        try:
            symbol_versioning.set_warning_method(self.capture_warning)
            self.assertEqual(len(a_deprecated_dict), 1)
            self.assertEqual([expected_warning], self._warnings)
            a_deprecated_dict['b'] = 42
            self.assertEqual(a_deprecated_dict['b'], 42)
            self.assertTrue('b' in a_deprecated_dict)
            del a_deprecated_dict['b']
            self.assertFalse('b' in a_deprecated_dict)
            self.assertEqual([expected_warning] * 6, self._warnings)
        finally:
            symbol_versioning.set_warning_method(old_warning_method)

    def check_deprecated_callable(self, expected_warning, expected_docstring, expected_name, expected_module, deprecated_callable):
        if False:
            print('Hello World!')
        if __doc__ is None:
            expected_docstring = expected_docstring.split('\n')[-2].lstrip()
        old_warning_method = symbol_versioning.warn
        try:
            symbol_versioning.set_warning_method(self.capture_warning)
            self.assertEqual(1, deprecated_callable())
            self.assertEqual([expected_warning], self._warnings)
            deprecated_callable()
            self.assertEqual([expected_warning, expected_warning], self._warnings)
            self.assertEqualDiff(expected_docstring, deprecated_callable.__doc__)
            self.assertEqualDiff(expected_name, deprecated_callable.__name__)
            self.assertEqualDiff(expected_module, deprecated_callable.__module__)
            self.assertTrue(deprecated_callable.is_deprecated)
        finally:
            symbol_versioning.set_warning_method(old_warning_method)

    def test_deprecated_passed(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(True, symbol_versioning.deprecated_passed(None))
        self.assertEqual(True, symbol_versioning.deprecated_passed(True))
        self.assertEqual(True, symbol_versioning.deprecated_passed(False))
        self.assertEqual(False, symbol_versioning.deprecated_passed(symbol_versioning.DEPRECATED_PARAMETER))

    def test_deprecation_string(self):
        if False:
            while True:
                i = 10
        'We can get a deprecation string for a method or function.'
        self.assertEqual('bzrlib.tests.test_symbol_versioning.TestDeprecationWarnings.test_deprecation_string was deprecated in version 0.11.0.', symbol_versioning.deprecation_string(self.test_deprecation_string, deprecated_in((0, 11, 0))))
        self.assertEqual('bzrlib.symbol_versioning.deprecated_function was deprecated in version 0.11.0.', symbol_versioning.deprecation_string(symbol_versioning.deprecated_function, deprecated_in((0, 11, 0))))

class TestSuppressAndActivate(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestSuppressAndActivate, self).setUp()
        existing_filters = list(warnings.filters)

        def restore():
            if False:
                for i in range(10):
                    print('nop')
            warnings.filters[:] = existing_filters
        self.addCleanup(restore)
        warnings.resetwarnings()

    def assertFirstWarning(self, action, category):
        if False:
            while True:
                i = 10
        'Test the first warning in the filters is correct'
        first = warnings.filters[0]
        self.assertEqual((action, category), (first[0], first[2]))

    def test_suppress_deprecation_warnings(self):
        if False:
            while True:
                i = 10
        'suppress_deprecation_warnings sets DeprecationWarning to ignored.'
        symbol_versioning.suppress_deprecation_warnings()
        self.assertFirstWarning('ignore', DeprecationWarning)

    def test_set_restore_filters(self):
        if False:
            return 10
        original_filters = warnings.filters[:]
        symbol_versioning.suppress_deprecation_warnings()()
        self.assertEqual(original_filters, warnings.filters)

    def test_suppress_deprecation_with_warning_filter(self):
        if False:
            return 10
        "don't suppress if we already have a filter"
        warnings.filterwarnings('error', category=Warning)
        self.assertFirstWarning('error', Warning)
        self.assertEqual(1, len(warnings.filters))
        symbol_versioning.suppress_deprecation_warnings(override=False)
        self.assertFirstWarning('error', Warning)
        self.assertEqual(1, len(warnings.filters))

    def test_suppress_deprecation_with_filter(self):
        if False:
            i = 10
            return i + 15
        "don't suppress if we already have a filter"
        warnings.filterwarnings('error', category=DeprecationWarning)
        self.assertFirstWarning('error', DeprecationWarning)
        self.assertEqual(1, len(warnings.filters))
        symbol_versioning.suppress_deprecation_warnings(override=False)
        self.assertFirstWarning('error', DeprecationWarning)
        self.assertEqual(1, len(warnings.filters))
        symbol_versioning.suppress_deprecation_warnings(override=True)
        self.assertFirstWarning('ignore', DeprecationWarning)
        self.assertEqual(2, len(warnings.filters))

    def test_activate_deprecation_no_error(self):
        if False:
            print('Hello World!')
        symbol_versioning.activate_deprecation_warnings()
        self.assertFirstWarning('default', DeprecationWarning)

    def test_activate_deprecation_with_error(self):
        if False:
            while True:
                i = 10
        warnings.filterwarnings('error', category=Warning)
        self.assertFirstWarning('error', Warning)
        self.assertEqual(1, len(warnings.filters))
        symbol_versioning.activate_deprecation_warnings(override=False)
        self.assertFirstWarning('error', Warning)
        self.assertEqual(1, len(warnings.filters))

    def test_activate_deprecation_with_DW_error(self):
        if False:
            while True:
                i = 10
        warnings.filterwarnings('error', category=DeprecationWarning)
        self.assertFirstWarning('error', DeprecationWarning)
        self.assertEqual(1, len(warnings.filters))
        symbol_versioning.activate_deprecation_warnings(override=False)
        self.assertFirstWarning('error', DeprecationWarning)
        self.assertEqual(1, len(warnings.filters))
        symbol_versioning.activate_deprecation_warnings(override=True)
        self.assertFirstWarning('default', DeprecationWarning)
        self.assertEqual(2, len(warnings.filters))