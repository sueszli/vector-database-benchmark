"""Tests for the lazy loaders."""
import sys
from unittest import mock
import ddt
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.test import QiskitTestCase
from qiskit.utils import LazyImportTester, LazySubprocessTester

def available_importer(**kwargs):
    if False:
        return 10
    'A LazyImportTester that should succeed.'
    return LazyImportTester('site', **kwargs)

def unavailable_importer(**kwargs):
    if False:
        while True:
            i = 10
    'A LazyImportTester that should fail.'
    return LazyImportTester('_qiskit_this_module_does_not_exist_', **kwargs)

def available_process(**kwargs):
    if False:
        i = 10
        return i + 15
    'A LazySubprocessTester that should fail.'
    return LazySubprocessTester([sys.executable, '-c', 'import sys; sys.exit(0)'], **kwargs)

def unavailable_process(**kwargs):
    if False:
        while True:
            i = 10
    'A LazySubprocessTester that should fail.'
    return LazySubprocessTester([sys.executable, '-c', 'import sys; sys.exit(1)'], **kwargs)

def mock_availability_test(feature):
    if False:
        i = 10
        return i + 15
    'Context manager that mocks out the availability checker for a given dependency checker.  The\n    context manager returns the mocked-out method.'
    return mock.patch.object(type(feature), '_is_available', wraps=feature._is_available)

@ddt.ddt
class TestLazyDependencyTester(QiskitTestCase):
    """Tests for the lazy loaders.  Within this class, we parameterise the test cases with
    generators, rather than the mocks themselves.  That allows us to easily generate clean
    instances, and means that creation doesn't happen during test collection."""

    @ddt.data(available_importer, available_process)
    def test_evaluates_correctly_true(self, test_generator):
        if False:
            print('Hello World!')
        'Test that the available loaders evaluate True in various Boolean contexts.'
        self.assertTrue(test_generator())
        self.assertTrue(bool(test_generator()))
        if not test_generator():
            self.fail('did not evaluate true')

    @ddt.data(unavailable_importer, unavailable_process)
    def test_evaluates_correctly_false(self, test_generator):
        if False:
            while True:
                i = 10
        'Test that the available loaders evaluate False in various Boolean contexts.'
        self.assertFalse(test_generator())
        self.assertFalse(bool(test_generator()))
        if test_generator():
            self.fail('did not evaluate false')

    @ddt.data(available_importer, available_process, unavailable_importer, unavailable_process)
    def test_check_occurs_once(self, test_generator):
        if False:
            return 10
        'Check that the test of availability is only performed once.'
        feature = test_generator()
        with mock_availability_test(feature) as check:
            check.assert_not_called()
            if feature:
                pass
            check.assert_called_once()
            if feature:
                feature.require_now('no message')
                feature.require_in_call(lambda : None)()
                feature.require_in_call('no message')(lambda : None)()
                feature.require_in_instance(type('Dummy', (), {}))()
                feature.require_in_instance('no message')(type('Dummy', (), {}))()
            check.assert_called_once()

    @ddt.data(available_importer, available_process, unavailable_importer, unavailable_process)
    def test_callback_occurs_once(self, test_generator):
        if False:
            i = 10
            return i + 15
        'Check that the callback is only called once.'
        callback = mock.MagicMock()
        feature = test_generator(callback=callback)
        callback.assert_not_called()
        if feature:
            pass
        callback.assert_called_once_with(bool(feature))
        callback.reset_mock()
        if feature:
            feature.require_now('no message')
            feature.require_in_call(lambda : None)()
            feature.require_in_call('no message')(lambda : None)()
            feature.require_in_instance(type('Dummy', (), {}))()
            feature.require_in_instance('no message')(type('Dummy', (), {}))()
        callback.assert_not_called()

    @ddt.data(available_importer, available_process)
    def test_require_now_silently_succeeds_for_available_tests(self, test_generator):
        if False:
            for i in range(10):
                print('nop')
        'Test that the available loaders silently do nothing when they are required.'
        feature = test_generator()
        with mock_availability_test(feature) as check:
            check.assert_not_called()
            feature.require_now('no message')
            check.assert_called_once()

    @ddt.data(available_importer, available_process)
    def test_require_in_call_silently_succeeds_for_available_tests(self, test_generator):
        if False:
            return 10
        'Test that the available loaders silently do nothing when they are required in the\n        decorator form.'
        with self.subTest('direct decorator'):
            feature = test_generator()
            with mock_availability_test(feature) as check:
                check.assert_not_called()

                @feature.require_in_call
                def decorated():
                    if False:
                        for i in range(10):
                            print('nop')
                    pass
                check.assert_not_called()
                decorated()
                check.assert_called_once()
        with self.subTest('named decorator'):
            feature = test_generator()
            with mock_availability_test(feature) as check:
                check.assert_not_called()

                @feature.require_in_call('sentinel name')
                def decorated():
                    if False:
                        for i in range(10):
                            print('nop')
                    pass
                check.assert_not_called()
                decorated()
                check.assert_called_once()

    @ddt.data(available_importer, available_process)
    def test_require_in_instance_silently_succeeds_for_available_tests(self, test_generator):
        if False:
            while True:
                i = 10
        'Test that the available loaders silently do nothing when they are required in the\n        decorator form.'
        with self.subTest('direct decorator'):
            feature = test_generator()
            with mock_availability_test(feature) as check:
                check.assert_not_called()

                @feature.require_in_instance
                class Dummy:
                    """Dummy class."""
                check.assert_not_called()
                Dummy()
                check.assert_called_once()
        with self.subTest('named decorator'):
            feature = test_generator()
            with mock_availability_test(feature) as check:
                check.assert_not_called()

                @feature.require_in_instance('sentinel name')
                class Dummy:
                    """Dummy class."""
                check.assert_not_called()
                Dummy()
                check.assert_called_once()

    @ddt.data(unavailable_importer, unavailable_process)
    def test_require_now_raises_for_unavailable_tests(self, test_generator):
        if False:
            print('Hello World!')
        'Test that the unavailable loaders loudly raise when they are required.'
        feature = test_generator()
        with mock_availability_test(feature) as check:
            check.assert_not_called()
            with self.assertRaisesRegex(MissingOptionalLibraryError, 'sentinel message'):
                feature.require_now('sentinel message')
            check.assert_called_once()

    @ddt.data(unavailable_importer, unavailable_process)
    def test_require_in_call_raises_for_unavailable_tests(self, test_generator):
        if False:
            print('Hello World!')
        'Test that the unavailable loaders loudly raise when the inner functions of decorators are\n        called, and not before, and raise each time they are called.'
        with self.subTest('direct decorator'):
            feature = test_generator()
            with mock_availability_test(feature) as check:
                check.assert_not_called()

                @feature.require_in_call
                def decorated():
                    if False:
                        print('Hello World!')
                    pass
                check.assert_not_called()
                with self.assertRaisesRegex(MissingOptionalLibraryError, 'decorated'):
                    decorated()
                check.assert_called_once()
                with self.assertRaisesRegex(MissingOptionalLibraryError, 'decorated'):
                    decorated()
                check.assert_called_once()
        with self.subTest('named decorator'):
            feature = test_generator()
            with mock_availability_test(feature) as check:
                check.assert_not_called()

                @feature.require_in_call('sentinel message')
                def decorated():
                    if False:
                        i = 10
                        return i + 15
                    pass
                check.assert_not_called()
                with self.assertRaisesRegex(MissingOptionalLibraryError, 'sentinel message'):
                    decorated()
                check.assert_called_once()
                with self.assertRaisesRegex(MissingOptionalLibraryError, 'sentinel message'):
                    decorated()
                check.assert_called_once()

    @ddt.data(unavailable_importer, unavailable_process)
    def test_require_in_instance_raises_for_unavailable_tests(self, test_generator):
        if False:
            return 10
        'Test that the unavailable loaders loudly raise when the inner classes of decorators are\n        instantiated, and not before, and raise each time they are instantiated.'
        with self.subTest('direct decorator'):
            feature = test_generator()
            with mock_availability_test(feature) as check:
                check.assert_not_called()

                @feature.require_in_instance
                class Dummy:
                    """Dummy class."""
                check.assert_not_called()
                with self.assertRaisesRegex(MissingOptionalLibraryError, 'Dummy'):
                    Dummy()
                check.assert_called_once()
                with self.assertRaisesRegex(MissingOptionalLibraryError, 'Dummy'):
                    Dummy()
                check.assert_called_once()
        with self.subTest('named decorator'):
            feature = test_generator()
            with mock_availability_test(feature) as check:
                check.assert_not_called()

                @feature.require_in_instance('sentinel message')
                class Dummy:
                    """Dummy class."""
                check.assert_not_called()
                with self.assertRaisesRegex(MissingOptionalLibraryError, 'sentinel message'):
                    Dummy()
                check.assert_called_once()
                with self.assertRaisesRegex(MissingOptionalLibraryError, 'sentinel message'):
                    Dummy()
                check.assert_called_once()

    def test_import_allows_multiple_modules_successful(self):
        if False:
            print('Hello World!')
        'Check that the import tester can accept an iterable of modules.'
        feature = LazyImportTester(['site', 'sys'])
        with mock_availability_test(feature) as check:
            check.assert_not_called()
            self.assertTrue(feature)
            check.assert_called_once()

    def test_import_allows_multiple_modules_failure(self):
        if False:
            print('Hello World!')
        'Check that the import tester can accept an iterable of modules, and will .'
        feature = LazyImportTester(['site', 'sys', '_qiskit_module_does_not_exist_'])
        with mock_availability_test(feature) as check:
            check.assert_not_called()
            self.assertFalse(feature)
            check.assert_called_once()

    def test_import_allows_attributes_successful(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that the import tester can accept a dictionary mapping module names to attributes,\n        and that these can be fetched.'
        name_map = {'_qiskit_dummy_module_1_': ('attr1', 'attr2'), '_qiskit_dummy_module_2_': ('thing1', 'thing2')}
        mock_modules = {}
        for (module, attributes) in name_map.items():

            class Module:
                """Dummy module."""
                unaccessed_attribute = mock.PropertyMock()
            for attribute in attributes:
                setattr(Module, attribute, mock.PropertyMock())
            mock_modules[module] = Module()
        feature = LazyImportTester(name_map)
        with mock.patch.dict(sys.modules, **mock_modules):
            self.assertTrue(feature)
        for (module, attributes) in name_map.items():
            mock_module = mock_modules[module]
            for attribute in attributes:
                vars(type(mock_module))[attribute].assert_called()
            vars(type(mock_module))['unaccessed_attribute'].assert_not_called()

    def test_import_allows_attributes_failure(self):
        if False:
            return 10
        'Check that the import tester can accept a dictionary mapping module names to attributes,\n        and that these are recognised when they are missing.'
        name_map = {'sys': ('executable', 'path'), 'builtins': ('list', '_qiskit_dummy_attribute_')}
        feature = LazyImportTester(name_map)
        self.assertFalse(feature)

    def test_import_fails_with_no_modules(self):
        if False:
            i = 10
            return i + 15
        'Catch programmer errors with no modules to test.'
        with self.assertRaises(ValueError):
            LazyImportTester([])

    def test_subprocess_fails_with_no_command(self):
        if False:
            for i in range(10):
                print('nop')
        'Catch programmer errors with no command to test.'
        with self.assertRaises(ValueError):
            LazySubprocessTester([])