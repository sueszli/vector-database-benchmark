"""Unit test for Google Test fail_fast.

A user can specify if a Google Test program should continue test execution
after a test failure via the GTEST_FAIL_FAST environment variable or the
--gtest_fail_fast flag. The default value of the flag can also be changed
by Bazel fail fast environment variable TESTBRIDGE_TEST_RUNNER_FAIL_FAST.

This script tests such functionality by invoking googletest-failfast-unittest_
(a program written with Google Test) with different environments and command
line flags.
"""
import os
from googletest.test import gtest_test_utils
BAZEL_FAIL_FAST_ENV_VAR = 'TESTBRIDGE_TEST_RUNNER_FAIL_FAST'
FAIL_FAST_ENV_VAR = 'GTEST_FAIL_FAST'
FAIL_FAST_FLAG = 'gtest_fail_fast'
RUN_DISABLED_FLAG = 'gtest_also_run_disabled_tests'
FILTER_FLAG = 'gtest_filter'
COMMAND = gtest_test_utils.GetTestExecutablePath('googletest-failfast-unittest_')
LIST_TESTS_FLAG = '--gtest_list_tests'
SUPPORTS_DEATH_TESTS = 'HasDeathTest' in gtest_test_utils.Subprocess([COMMAND, LIST_TESTS_FLAG]).output
environ = os.environ.copy()

def SetEnvVar(env_var, value):
    if False:
        i = 10
        return i + 15
    "Sets the env variable to 'value'; unsets it when 'value' is None."
    if value is not None:
        environ[env_var] = value
    elif env_var in environ:
        del environ[env_var]

def RunAndReturnOutput(test_suite=None, fail_fast=None, run_disabled=False):
    if False:
        for i in range(10):
            print('nop')
    'Runs the test program and returns its output.'
    args = []
    xml_path = os.path.join(gtest_test_utils.GetTempDir(), '.GTestFailFastUnitTest.xml')
    args += ['--gtest_output=xml:' + xml_path]
    if fail_fast is not None:
        if isinstance(fail_fast, str):
            args += ['--%s=%s' % (FAIL_FAST_FLAG, fail_fast)]
        elif fail_fast:
            args += ['--%s' % FAIL_FAST_FLAG]
        else:
            args += ['--no%s' % FAIL_FAST_FLAG]
    if test_suite:
        args += ['--%s=%s.*' % (FILTER_FLAG, test_suite)]
    if run_disabled:
        args += ['--%s' % RUN_DISABLED_FLAG]
    txt_out = gtest_test_utils.Subprocess([COMMAND] + args, env=environ).output
    with open(xml_path) as xml_file:
        return (txt_out, xml_file.read())

class GTestFailFastUnitTest(gtest_test_utils.TestCase):
    """Tests the env variable or the command line flag for fail_fast."""

    def testDefaultBehavior(self):
        if False:
            while True:
                i = 10
        'Tests the behavior of not specifying the fail_fast.'
        (txt, _) = RunAndReturnOutput()
        self.assertIn('22 FAILED TEST', txt)

    def testGoogletestFlag(self):
        if False:
            while True:
                i = 10
        (txt, _) = RunAndReturnOutput(test_suite='HasSimpleTest', fail_fast=True)
        self.assertIn('1 FAILED TEST', txt)
        self.assertIn('[  SKIPPED ] 3 tests', txt)
        (txt, _) = RunAndReturnOutput(test_suite='HasSimpleTest', fail_fast=False)
        self.assertIn('4 FAILED TEST', txt)
        self.assertNotIn('[  SKIPPED ]', txt)

    def testGoogletestEnvVar(self):
        if False:
            i = 10
            return i + 15
        'Tests the behavior of specifying fail_fast via Googletest env var.'
        try:
            SetEnvVar(FAIL_FAST_ENV_VAR, '1')
            (txt, _) = RunAndReturnOutput('HasSimpleTest')
            self.assertIn('1 FAILED TEST', txt)
            self.assertIn('[  SKIPPED ] 3 tests', txt)
            SetEnvVar(FAIL_FAST_ENV_VAR, '0')
            (txt, _) = RunAndReturnOutput('HasSimpleTest')
            self.assertIn('4 FAILED TEST', txt)
            self.assertNotIn('[  SKIPPED ]', txt)
        finally:
            SetEnvVar(FAIL_FAST_ENV_VAR, None)

    def testBazelEnvVar(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests the behavior of specifying fail_fast via Bazel testbridge.'
        try:
            SetEnvVar(BAZEL_FAIL_FAST_ENV_VAR, '1')
            (txt, _) = RunAndReturnOutput('HasSimpleTest')
            self.assertIn('1 FAILED TEST', txt)
            self.assertIn('[  SKIPPED ] 3 tests', txt)
            SetEnvVar(BAZEL_FAIL_FAST_ENV_VAR, '0')
            (txt, _) = RunAndReturnOutput('HasSimpleTest')
            self.assertIn('4 FAILED TEST', txt)
            self.assertNotIn('[  SKIPPED ]', txt)
        finally:
            SetEnvVar(BAZEL_FAIL_FAST_ENV_VAR, None)

    def testFlagOverridesEnvVar(self):
        if False:
            i = 10
            return i + 15
        'Tests precedence of flag over env var.'
        try:
            SetEnvVar(FAIL_FAST_ENV_VAR, '0')
            (txt, _) = RunAndReturnOutput('HasSimpleTest', True)
            self.assertIn('1 FAILED TEST', txt)
            self.assertIn('[  SKIPPED ] 3 tests', txt)
        finally:
            SetEnvVar(FAIL_FAST_ENV_VAR, None)

    def testGoogletestEnvVarOverridesBazelEnvVar(self):
        if False:
            return 10
        'Tests that the Googletest native env var over Bazel testbridge.'
        try:
            SetEnvVar(BAZEL_FAIL_FAST_ENV_VAR, '0')
            SetEnvVar(FAIL_FAST_ENV_VAR, '1')
            (txt, _) = RunAndReturnOutput('HasSimpleTest')
            self.assertIn('1 FAILED TEST', txt)
            self.assertIn('[  SKIPPED ] 3 tests', txt)
        finally:
            SetEnvVar(FAIL_FAST_ENV_VAR, None)
            SetEnvVar(BAZEL_FAIL_FAST_ENV_VAR, None)

    def testEventListener(self):
        if False:
            i = 10
            return i + 15
        (txt, _) = RunAndReturnOutput(test_suite='HasSkipTest', fail_fast=True)
        self.assertIn('1 FAILED TEST', txt)
        self.assertIn('[  SKIPPED ] 3 tests', txt)
        for (expected_count, callback) in [(1, 'OnTestSuiteStart'), (5, 'OnTestStart'), (5, 'OnTestEnd'), (5, 'OnTestPartResult'), (1, 'OnTestSuiteEnd')]:
            self.assertEqual(expected_count, txt.count(callback), 'Expected %d calls to callback %s match count on output: %s ' % (expected_count, callback, txt))
        (txt, _) = RunAndReturnOutput(test_suite='HasSkipTest', fail_fast=False)
        self.assertIn('3 FAILED TEST', txt)
        self.assertIn('[  SKIPPED ] 1 test', txt)
        for (expected_count, callback) in [(1, 'OnTestSuiteStart'), (5, 'OnTestStart'), (5, 'OnTestEnd'), (5, 'OnTestPartResult'), (1, 'OnTestSuiteEnd')]:
            self.assertEqual(expected_count, txt.count(callback), 'Expected %d calls to callback %s match count on output: %s ' % (expected_count, callback, txt))

    def assertXmlResultCount(self, result, count, xml):
        if False:
            i = 10
            return i + 15
        self.assertEqual(count, xml.count('result="%s"' % result), 'Expected \'result="%s"\' match count of %s: %s ' % (result, count, xml))

    def assertXmlStatusCount(self, status, count, xml):
        if False:
            while True:
                i = 10
        self.assertEqual(count, xml.count('status="%s"' % status), 'Expected \'status="%s"\' match count of %s: %s ' % (status, count, xml))

    def assertFailFastXmlAndTxtOutput(self, fail_fast, test_suite, passed_count, failure_count, skipped_count, suppressed_count, run_disabled=False):
        if False:
            print('Hello World!')
        'Assert XML and text output of a test execution.'
        (txt, xml) = RunAndReturnOutput(test_suite, fail_fast, run_disabled)
        if failure_count > 0:
            self.assertIn('%s FAILED TEST' % failure_count, txt)
        if suppressed_count > 0:
            self.assertIn('%s DISABLED TEST' % suppressed_count, txt)
        if skipped_count > 0:
            self.assertIn('[  SKIPPED ] %s tests' % skipped_count, txt)
        self.assertXmlStatusCount('run', passed_count + failure_count + skipped_count, xml)
        self.assertXmlStatusCount('notrun', suppressed_count, xml)
        self.assertXmlResultCount('completed', passed_count + failure_count, xml)
        self.assertXmlResultCount('skipped', skipped_count, xml)
        self.assertXmlResultCount('suppressed', suppressed_count, xml)

    def assertFailFastBehavior(self, test_suite, passed_count, failure_count, skipped_count, suppressed_count, run_disabled=False):
        if False:
            while True:
                i = 10
        'Assert --fail_fast via flag.'
        for fail_fast in ('true', '1', 't', True):
            self.assertFailFastXmlAndTxtOutput(fail_fast, test_suite, passed_count, failure_count, skipped_count, suppressed_count, run_disabled)

    def assertNotFailFastBehavior(self, test_suite, passed_count, failure_count, skipped_count, suppressed_count, run_disabled=False):
        if False:
            i = 10
            return i + 15
        'Assert --nofail_fast via flag.'
        for fail_fast in ('false', '0', 'f', False):
            self.assertFailFastXmlAndTxtOutput(fail_fast, test_suite, passed_count, failure_count, skipped_count, suppressed_count, run_disabled)

    def testFlag_HasFixtureTest(self):
        if False:
            i = 10
            return i + 15
        'Tests the behavior of fail_fast and TEST_F.'
        self.assertFailFastBehavior(test_suite='HasFixtureTest', passed_count=1, failure_count=1, skipped_count=3, suppressed_count=0)
        self.assertNotFailFastBehavior(test_suite='HasFixtureTest', passed_count=1, failure_count=4, skipped_count=0, suppressed_count=0)

    def testFlag_HasSimpleTest(self):
        if False:
            while True:
                i = 10
        'Tests the behavior of fail_fast and TEST.'
        self.assertFailFastBehavior(test_suite='HasSimpleTest', passed_count=1, failure_count=1, skipped_count=3, suppressed_count=0)
        self.assertNotFailFastBehavior(test_suite='HasSimpleTest', passed_count=1, failure_count=4, skipped_count=0, suppressed_count=0)

    def testFlag_HasParametersTest(self):
        if False:
            i = 10
            return i + 15
        'Tests the behavior of fail_fast and TEST_P.'
        self.assertFailFastBehavior(test_suite='HasParametersSuite/HasParametersTest', passed_count=0, failure_count=1, skipped_count=3, suppressed_count=0)
        self.assertNotFailFastBehavior(test_suite='HasParametersSuite/HasParametersTest', passed_count=0, failure_count=4, skipped_count=0, suppressed_count=0)

    def testFlag_HasDisabledTest(self):
        if False:
            i = 10
            return i + 15
        'Tests the behavior of fail_fast and Disabled test cases.'
        self.assertFailFastBehavior(test_suite='HasDisabledTest', passed_count=1, failure_count=1, skipped_count=2, suppressed_count=1, run_disabled=False)
        self.assertNotFailFastBehavior(test_suite='HasDisabledTest', passed_count=1, failure_count=3, skipped_count=0, suppressed_count=1, run_disabled=False)

    def testFlag_HasDisabledRunDisabledTest(self):
        if False:
            while True:
                i = 10
        'Tests the behavior of fail_fast and Disabled test cases enabled.'
        self.assertFailFastBehavior(test_suite='HasDisabledTest', passed_count=1, failure_count=1, skipped_count=3, suppressed_count=0, run_disabled=True)
        self.assertNotFailFastBehavior(test_suite='HasDisabledTest', passed_count=1, failure_count=4, skipped_count=0, suppressed_count=0, run_disabled=True)

    def testFlag_HasDisabledSuiteTest(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests the behavior of fail_fast and Disabled test suites.'
        self.assertFailFastBehavior(test_suite='DISABLED_HasDisabledSuite', passed_count=0, failure_count=0, skipped_count=0, suppressed_count=5, run_disabled=False)
        self.assertNotFailFastBehavior(test_suite='DISABLED_HasDisabledSuite', passed_count=0, failure_count=0, skipped_count=0, suppressed_count=5, run_disabled=False)

    def testFlag_HasDisabledSuiteRunDisabledTest(self):
        if False:
            print('Hello World!')
        'Tests the behavior of fail_fast and Disabled test suites enabled.'
        self.assertFailFastBehavior(test_suite='DISABLED_HasDisabledSuite', passed_count=1, failure_count=1, skipped_count=3, suppressed_count=0, run_disabled=True)
        self.assertNotFailFastBehavior(test_suite='DISABLED_HasDisabledSuite', passed_count=1, failure_count=4, skipped_count=0, suppressed_count=0, run_disabled=True)
    if SUPPORTS_DEATH_TESTS:

        def testFlag_HasDeathTest(self):
            if False:
                for i in range(10):
                    print('nop')
            'Tests the behavior of fail_fast and death tests.'
            self.assertFailFastBehavior(test_suite='HasDeathTest', passed_count=1, failure_count=1, skipped_count=3, suppressed_count=0)
            self.assertNotFailFastBehavior(test_suite='HasDeathTest', passed_count=1, failure_count=4, skipped_count=0, suppressed_count=0)
if __name__ == '__main__':
    gtest_test_utils.Main()