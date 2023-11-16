import os
import sys
import json
import unittest
from coalib.bearlib.abstractions.ExternalBearWrap import external_bear_wrap
from coalib.results.Result import Result
from coalib.settings.Section import Section
from coalib.results.SourceRange import SourceRange
from coalib.results.RESULT_SEVERITY import RESULT_SEVERITY
from coalib.settings.FunctionMetadata import FunctionMetadata

def get_testfile_path(name):
    if False:
        i = 10
        return i + 15
    '\n    Gets the full path to a testfile inside the same directory.\n\n    :param name: The filename of the testfile to get the full path for.\n    :return:     The full path to given testfile name.\n    '
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)

class ExternalBearWrapComponentTest(unittest.TestCase):

    class Dummy:
        pass

    class TestBear:

        @staticmethod
        def create_arguments():
            if False:
                for i in range(10):
                    print('nop')
            return (os.path.join(os.path.dirname(__file__), 'test_external_bear.py'),)

    class WrongArgsBear:

        @staticmethod
        def create_arguments():
            if False:
                return 10
            return 1

    def setUp(self):
        if False:
            while True:
                i = 10
        self._old_python_path = os.environ.get('PYTHONPATH')
        os.environ['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
        self.section = Section('TEST_SECTION')
        self.test_program_path = get_testfile_path('test_external_bear.py')
        self.testfile_path = get_testfile_path('test_file.txt')
        with open(self.testfile_path, mode='r') as fl:
            self.testfile_content = fl.read().splitlines(keepends=True)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        if self._old_python_path:
            os.environ['PYTHONPATH'] = self._old_python_path

    def test_decorator_invalid_parameters(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError) as cm:
            external_bear_wrap('exec', invalid_arg=88)
        self.assertEqual(str(cm.exception), "Invalid keyword arguments provided: 'invalid_arg'")

    def test_decorator_invalid_parameter_types(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            external_bear_wrap(executable=1337)

    def test_get_executable(self):
        if False:
            return 10
        uut = external_bear_wrap('exec')(self.TestBear)
        self.assertEqual(uut.get_executable(), 'exec')

    def test_create_arguments_fail(self):
        if False:
            i = 10
            return i + 15
        uut = external_bear_wrap('exec')(self.Dummy)
        self.assertEqual(uut.create_arguments(), ())

    def test_create_arguments_non_iterable(self):
        if False:
            return 10
        uut = external_bear_wrap('exec')(self.WrongArgsBear)(self.section, None)
        with self.assertRaises(TypeError):
            list(uut.run(self.testfile_path, self.testfile_content))

    def test_invalid_output(self):
        if False:
            return 10
        broken_json = json.dumps([{'broken': 'JSON'}])[:-1]
        uut = external_bear_wrap('exec')(self.Dummy)(self.section, None)
        with self.assertRaises(ValueError):
            list(uut.parse_output(broken_json, 'some_file'))

    def test_setting_desc(self):
        if False:
            return 10
        uut = external_bear_wrap('exec', settings={'asetting': ('', bool), 'bsetting': ('', bool, True), 'csetting': ('My desc.', bool, False), 'dsetting': ('Another desc', bool), 'esetting': ('', int, None)})(self.Dummy)
        metadata = uut.get_metadata()
        self.assertEqual(metadata.non_optional_params['asetting'][0], FunctionMetadata.str_nodesc)
        self.assertEqual(metadata.optional_params['bsetting'][0], FunctionMetadata.str_nodesc + ' ' + FunctionMetadata.str_optional.format(True))
        self.assertEqual(metadata.optional_params['csetting'][0], 'My desc.' + ' ' + FunctionMetadata.str_optional.format(False))
        self.assertEqual(metadata.non_optional_params['dsetting'][0], 'Another desc')
        self.assertEqual(metadata.optional_params['esetting'][0], FunctionMetadata.str_nodesc + ' ' + FunctionMetadata.str_optional.format(None))

    def test_optional_settings(self):
        if False:
            return 10
        uut = external_bear_wrap(sys.executable, settings={'set_normal_severity': ('', bool), 'set_sample_dbg_msg': ('', bool, False), 'not_set_different_msg': ('', bool, True)})(self.TestBear)(self.section, None)
        results = list(uut.run(self.testfile_path, self.testfile_content, set_normal_severity=False))
        expected = [Result(origin=uut, message='This is wrong', affected_code=(SourceRange.from_values(self.testfile_path, 1),), severity=RESULT_SEVERITY.MAJOR), Result(origin=uut, message='This is wrong too', affected_code=(SourceRange.from_values(self.testfile_path, 3),), severity=RESULT_SEVERITY.INFO)]
        self.assertEqual(results, expected)
        results = list(uut.run(self.testfile_path, self.testfile_content, set_normal_severity=True))
        expected = [Result(origin=uut, message='This is wrong', affected_code=(SourceRange.from_values(self.testfile_path, 1),), severity=RESULT_SEVERITY.NORMAL), Result(origin=uut, message='This is wrong too', affected_code=(SourceRange.from_values(self.testfile_path, 3),), severity=RESULT_SEVERITY.NORMAL)]
        self.assertEqual(results, expected)

    def test_settings(self):
        if False:
            print('Hello World!')
        uut = external_bear_wrap(sys.executable, settings={'set_normal_severity': ('', bool), 'set_sample_dbg_msg': ('', bool, False), 'not_set_different_msg': ('', bool, True)})(self.TestBear)(self.section, None)
        results = list(uut.run(self.testfile_path, self.testfile_content, set_normal_severity=False, set_sample_dbg_msg=True, not_set_different_msg=False))
        expected = [Result(origin=uut, message='This is wrong', affected_code=(SourceRange.from_values(self.testfile_path, 1),), severity=RESULT_SEVERITY.MAJOR, debug_msg='Sample debug message'), Result(origin=uut, message='Different message', affected_code=(SourceRange.from_values(self.testfile_path, 3),), severity=RESULT_SEVERITY.INFO)]
        self.assertEqual(results, expected)