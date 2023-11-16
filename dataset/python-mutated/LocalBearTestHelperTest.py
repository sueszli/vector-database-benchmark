from queue import Queue
import sys
import unittest
from tests.test_bears.TestBear import TestBear
from tests.test_bears.AspectsGeneralTestBear import AspectsGeneralTestBear
from tests.test_bears.TestBearDep import TestDepBearBDependsA, TestDepBearCDependsB, TestDepBearDependsAAndAA
from coalib.bearlib.abstractions.Linter import linter
from tests.test_bears.LineCountTestBear import LineCountTestBear
from coala_utils.ContextManagers import prepare_file, retrieve_stderr
from coalib.results.Result import Result
from coalib.results.RESULT_SEVERITY import RESULT_SEVERITY
from coalib.settings.Section import Section
from coalib.settings.Setting import Setting
from coalib.testing.LocalBearTestHelper import verify_local_bear, execute_bear
from coalib.testing.LocalBearTestHelper import LocalBearTestHelper as Helper
from coalib.bearlib.aspects import AspectList, get as get_aspect
files = ('Everything is invalid/valid/raises error',)
invalidTest = verify_local_bear(TestBear, valid_files=(), invalid_files=files, settings={'result': True})
validTest = verify_local_bear(TestBear, valid_files=files, invalid_files=())
min_files = ('This is valid.',)
max_files = ('This is particularly an invalid file',)
AspectTest = verify_local_bear(AspectsGeneralTestBear, valid_files=min_files, invalid_files=max_files, aspects=AspectList([get_aspect('LineLength')('Unknown', max_line_length=20)]))
PriorityAspectsTest = verify_local_bear(AspectsGeneralTestBear, valid_files=min_files, invalid_files=max_files, aspects=AspectList([get_aspect('LineLength')('Unknown', max_line_length=60)]), settings={'max_line_length': 20})

class LocalBearCheckResultsTest(Helper):

    def setUp(self):
        if False:
            return 10
        section = Section('')
        section.append(Setting('result', 'a, b'))
        self.uut = TestBear(section, Queue())

    def test_order_ignored(self):
        if False:
            while True:
                i = 10
        self.check_results(self.uut, ['a', 'b'], ['b', 'a'], check_order=False)

    def test_require_order(self):
        if False:
            return 10
        with self.assertRaises(AssertionError):
            self.check_results(self.uut, ['a', 'b'], ['b', 'a'], check_order=True)

    def test_result_inequality(self):
        if False:
            print('Hello World!')
        with self.assertRaises(AssertionError):
            self.check_results(self.uut, ['a', 'b'], ['a', 'b', None], check_order=True)

    def test_good_assertComparableObjectsEqual(self):
        if False:
            return 10
        self.uut = LineCountTestBear(Section('name'), Queue())
        file_content = 'a\nb\nc'
        with prepare_file(file_content.splitlines(), filename=None, create_tempfile=True) as (file, fname):
            self.check_results(self.uut, file_content.splitlines(), [Result.from_values(origin='LineCountTestBear', message='This file has 3 lines.', severity=RESULT_SEVERITY.INFO, file=fname)], filename=fname, create_tempfile=False)

    def test_bad_assertComparableObjectsEqual(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(AssertionError) as cm:
            self.uut = LineCountTestBear(Section('name'), Queue())
            file_content = 'a\nb\nc'
            with prepare_file(file_content.splitlines(), filename=None, create_tempfile=True) as (file, fname):
                self.check_results(self.uut, file_content.splitlines(), [Result.from_values(origin='LineCountTestBea', message='This file has 2 lines.', severity=RESULT_SEVERITY.INFO, file=fname)], filename=fname, create_tempfile=False)
        self.assertEqual("'LineCountTestBear' != 'LineCountTestBea'\n- LineCountTestBear\n?                 -\n+ LineCountTestBea\n : origin mismatch.\n\n'This file has 3 lines.' != 'This file has 2 lines.'\n- This file has 3 lines.\n?               ^\n+ This file has 2 lines.\n?               ^\n : message_base mismatch.\n\n", str(cm.exception))

class LocalBearTestCheckLineResultCountTest(Helper):

    def setUp(self):
        if False:
            print('Hello World!')
        section = Section('')
        section.append(Setting('result', True))
        self.uut = TestBear(section, Queue())

    def test_check_line_result_count(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_line_result_count(self.uut, ['a', '', 'b', '   ', '# abc', '1'], [1, 1, 1])

class LocalBearTestDependency(Helper):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.section = Section('')

    def test_check_results_bear_with_dependency(self):
        if False:
            while True:
                i = 10
        bear = TestDepBearBDependsA(self.section, Queue())
        self.check_results(bear, [], [['settings1', 'settings2', 'settings3', 'settings4']], settings={'settings1': 'settings1', 'settings2': 'settings2', 'settings3': 'settings3', 'settings4': 'settings4'})

    def test_check_results_bear_with_2_deep_dependency(self):
        if False:
            print('Hello World!')
        bear = TestDepBearCDependsB(self.section, Queue())
        self.check_results(bear, [], [['settings1', 'settings2', 'settings3', 'settings4', 'settings5', 'settings6']], settings={'settings1': 'settings1', 'settings2': 'settings2', 'settings3': 'settings3', 'settings4': 'settings4', 'settings5': 'settings5', 'settings6': 'settings6'})

    def test_check_results_bear_with_two_dependencies(self):
        if False:
            while True:
                i = 10
        bear = TestDepBearDependsAAndAA(self.section, Queue())
        self.check_results(bear, [], [['settings1', 'settings2', 'settings3', 'settings4']], settings={'settings1': 'settings1', 'settings2': 'settings2', 'settings3': 'settings3', 'settings4': 'settings4'})

class LocalBearTestHelper(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        section = Section('')
        section.append(Setting('exception', True))
        self.uut = TestBear(section, Queue())

    def test_stdout_stderr_on_linter_test_fail(self):
        if False:
            return 10

        class TestLinter:

            @staticmethod
            def process_output(output, filename, file):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            @staticmethod
            def create_arguments(filename, file, config_file):
                if False:
                    print('Hello World!')
                code = '\n'.join(['import sys', "print('hello stdout')", "print('hello stderr', file=sys.stderr)"])
                return ('-c', code)
        uut = linter(sys.executable, use_stdout=True, use_stderr=True)(TestLinter)(Section('TEST_SECTION'), Queue())
        try:
            with execute_bear(uut, 'filename', ['file']) as result:
                raise AssertionError
        except AssertionError as ex:
            self.assertIn('Program arguments:\n(\'-c\', "import sys\\nprint(\'hello stdout\')\\nprint(\'hello stderr\', file=sys.stderr)")\nThe program yielded the following output:\n\nStdout:\nhello stdout\n\nStderr:\nhello stderr', str(ex))
        uut = linter(sys.executable, use_stdout=True)(TestLinter)(Section('TEST_SECTION'), Queue())
        try:
            with execute_bear(uut, 'filename', ['file']) as result:
                raise AssertionError
        except AssertionError as ex:
            self.assertIn('Program arguments:\n(\'-c\', "import sys\\nprint(\'hello stdout\')\\nprint(\'hello stderr\', file=sys.stderr)")\nThe program yielded the following output:\n\nhello stdout', str(ex))

    def test_exception(self):
        if False:
            return 10
        with self.assertRaises(AssertionError), execute_bear(self.uut, 'Luke', files[0]) as result:
            pass

class VerifyLocalBearTest(unittest.TestCase):

    def test_timeout_deprecation_warning(self):
        if False:
            for i in range(10):
                print('nop')
        with retrieve_stderr() as stderr:
            verify_local_bear(TestBear, valid_files=(), invalid_files=files, timeout=50)
            self.assertIn('timeout is ignored as the timeout set in the repo configuration will be sufficient', stderr.getvalue())