"""Unit test for Google Test's break-on-failure mode.

A user can ask Google Test to seg-fault when an assertion fails, using
either the GTEST_BREAK_ON_FAILURE environment variable or the
--gtest_break_on_failure flag.  This script tests such functionality
by invoking googletest-break-on-failure-unittest_ (a program written with
Google Test) with different environments and command line flags.
"""
import os
import gtest_test_utils
IS_WINDOWS = os.name == 'nt'
BREAK_ON_FAILURE_ENV_VAR = 'GTEST_BREAK_ON_FAILURE'
BREAK_ON_FAILURE_FLAG = 'gtest_break_on_failure'
THROW_ON_FAILURE_ENV_VAR = 'GTEST_THROW_ON_FAILURE'
CATCH_EXCEPTIONS_ENV_VAR = 'GTEST_CATCH_EXCEPTIONS'
EXE_PATH = gtest_test_utils.GetTestExecutablePath('googletest-break-on-failure-unittest_')
environ = gtest_test_utils.environ
SetEnvVar = gtest_test_utils.SetEnvVar
SetEnvVar(gtest_test_utils.PREMATURE_EXIT_FILE_ENV_VAR, None)

def Run(command):
    if False:
        print('Hello World!')
    'Runs a command; returns 1 if it was killed by a signal, or 0 otherwise.'
    p = gtest_test_utils.Subprocess(command, env=environ)
    if p.terminated_by_signal:
        return 1
    else:
        return 0

class GTestBreakOnFailureUnitTest(gtest_test_utils.TestCase):
    """Tests using the GTEST_BREAK_ON_FAILURE environment variable or
  the --gtest_break_on_failure flag to turn assertion failures into
  segmentation faults.
  """

    def RunAndVerify(self, env_var_value, flag_value, expect_seg_fault):
        if False:
            for i in range(10):
                print('nop')
        'Runs googletest-break-on-failure-unittest_ and verifies that it does\n    (or does not) have a seg-fault.\n\n    Args:\n      env_var_value:    value of the GTEST_BREAK_ON_FAILURE environment\n                        variable; None if the variable should be unset.\n      flag_value:       value of the --gtest_break_on_failure flag;\n                        None if the flag should not be present.\n      expect_seg_fault: 1 if the program is expected to generate a seg-fault;\n                        0 otherwise.\n    '
        SetEnvVar(BREAK_ON_FAILURE_ENV_VAR, env_var_value)
        if env_var_value is None:
            env_var_value_msg = ' is not set'
        else:
            env_var_value_msg = '=' + env_var_value
        if flag_value is None:
            flag = ''
        elif flag_value == '0':
            flag = '--%s=0' % BREAK_ON_FAILURE_FLAG
        else:
            flag = '--%s' % BREAK_ON_FAILURE_FLAG
        command = [EXE_PATH]
        if flag:
            command.append(flag)
        if expect_seg_fault:
            should_or_not = 'should'
        else:
            should_or_not = 'should not'
        has_seg_fault = Run(command)
        SetEnvVar(BREAK_ON_FAILURE_ENV_VAR, None)
        msg = 'when %s%s, an assertion failure in "%s" %s cause a seg-fault.' % (BREAK_ON_FAILURE_ENV_VAR, env_var_value_msg, ' '.join(command), should_or_not)
        self.assert_(has_seg_fault == expect_seg_fault, msg)

    def testDefaultBehavior(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests the behavior of the default mode.'
        self.RunAndVerify(env_var_value=None, flag_value=None, expect_seg_fault=0)

    def testEnvVar(self):
        if False:
            return 10
        'Tests using the GTEST_BREAK_ON_FAILURE environment variable.'
        self.RunAndVerify(env_var_value='0', flag_value=None, expect_seg_fault=0)
        self.RunAndVerify(env_var_value='1', flag_value=None, expect_seg_fault=1)

    def testFlag(self):
        if False:
            return 10
        'Tests using the --gtest_break_on_failure flag.'
        self.RunAndVerify(env_var_value=None, flag_value='0', expect_seg_fault=0)
        self.RunAndVerify(env_var_value=None, flag_value='1', expect_seg_fault=1)

    def testFlagOverridesEnvVar(self):
        if False:
            i = 10
            return i + 15
        'Tests that the flag overrides the environment variable.'
        self.RunAndVerify(env_var_value='0', flag_value='0', expect_seg_fault=0)
        self.RunAndVerify(env_var_value='0', flag_value='1', expect_seg_fault=1)
        self.RunAndVerify(env_var_value='1', flag_value='0', expect_seg_fault=0)
        self.RunAndVerify(env_var_value='1', flag_value='1', expect_seg_fault=1)

    def testBreakOnFailureOverridesThrowOnFailure(self):
        if False:
            while True:
                i = 10
        'Tests that gtest_break_on_failure overrides gtest_throw_on_failure.'
        SetEnvVar(THROW_ON_FAILURE_ENV_VAR, '1')
        try:
            self.RunAndVerify(env_var_value=None, flag_value='1', expect_seg_fault=1)
        finally:
            SetEnvVar(THROW_ON_FAILURE_ENV_VAR, None)
    if IS_WINDOWS:

        def testCatchExceptionsDoesNotInterfere(self):
            if False:
                print('Hello World!')
            "Tests that gtest_catch_exceptions doesn't interfere."
            SetEnvVar(CATCH_EXCEPTIONS_ENV_VAR, '1')
            try:
                self.RunAndVerify(env_var_value='1', flag_value='1', expect_seg_fault=1)
            finally:
                SetEnvVar(CATCH_EXCEPTIONS_ENV_VAR, None)
if __name__ == '__main__':
    gtest_test_utils.Main()