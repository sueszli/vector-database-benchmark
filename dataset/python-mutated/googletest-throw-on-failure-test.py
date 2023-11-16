"""Tests Google Test's throw-on-failure mode with exceptions disabled.

This script invokes googletest-throw-on-failure-test_ (a program written with
Google Test) with different environments and command line flags.
"""
import os
import gtest_test_utils
THROW_ON_FAILURE = 'gtest_throw_on_failure'
EXE_PATH = gtest_test_utils.GetTestExecutablePath('googletest-throw-on-failure-test_')

def SetEnvVar(env_var, value):
    if False:
        return 10
    'Sets an environment variable to a given value; unsets it when the\n  given value is None.\n  '
    env_var = env_var.upper()
    if value is not None:
        os.environ[env_var] = value
    elif env_var in os.environ:
        del os.environ[env_var]

def Run(command):
    if False:
        while True:
            i = 10
    "Runs a command; returns True/False if its exit code is/isn't 0."
    print('Running "%s". . .' % ' '.join(command))
    p = gtest_test_utils.Subprocess(command)
    return p.exited and p.exit_code == 0

class ThrowOnFailureTest(gtest_test_utils.TestCase):
    """Tests the throw-on-failure mode."""

    def RunAndVerify(self, env_var_value, flag_value, should_fail):
        if False:
            return 10
        'Runs googletest-throw-on-failure-test_ and verifies that it does\n    (or does not) exit with a non-zero code.\n\n    Args:\n      env_var_value:    value of the GTEST_BREAK_ON_FAILURE environment\n                        variable; None if the variable should be unset.\n      flag_value:       value of the --gtest_break_on_failure flag;\n                        None if the flag should not be present.\n      should_fail:      True iff the program is expected to fail.\n    '
        SetEnvVar(THROW_ON_FAILURE, env_var_value)
        if env_var_value is None:
            env_var_value_msg = ' is not set'
        else:
            env_var_value_msg = '=' + env_var_value
        if flag_value is None:
            flag = ''
        elif flag_value == '0':
            flag = '--%s=0' % THROW_ON_FAILURE
        else:
            flag = '--%s' % THROW_ON_FAILURE
        command = [EXE_PATH]
        if flag:
            command.append(flag)
        if should_fail:
            should_or_not = 'should'
        else:
            should_or_not = 'should not'
        failed = not Run(command)
        SetEnvVar(THROW_ON_FAILURE, None)
        msg = 'when %s%s, an assertion failure in "%s" %s cause a non-zero exit code.' % (THROW_ON_FAILURE, env_var_value_msg, ' '.join(command), should_or_not)
        self.assert_(failed == should_fail, msg)

    def testDefaultBehavior(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests the behavior of the default mode.'
        self.RunAndVerify(env_var_value=None, flag_value=None, should_fail=False)

    def testThrowOnFailureEnvVar(self):
        if False:
            i = 10
            return i + 15
        'Tests using the GTEST_THROW_ON_FAILURE environment variable.'
        self.RunAndVerify(env_var_value='0', flag_value=None, should_fail=False)
        self.RunAndVerify(env_var_value='1', flag_value=None, should_fail=True)

    def testThrowOnFailureFlag(self):
        if False:
            while True:
                i = 10
        'Tests using the --gtest_throw_on_failure flag.'
        self.RunAndVerify(env_var_value=None, flag_value='0', should_fail=False)
        self.RunAndVerify(env_var_value=None, flag_value='1', should_fail=True)

    def testThrowOnFailureFlagOverridesEnvVar(self):
        if False:
            print('Hello World!')
        'Tests that --gtest_throw_on_failure overrides GTEST_THROW_ON_FAILURE.'
        self.RunAndVerify(env_var_value='0', flag_value='0', should_fail=False)
        self.RunAndVerify(env_var_value='0', flag_value='1', should_fail=True)
        self.RunAndVerify(env_var_value='1', flag_value='0', should_fail=False)
        self.RunAndVerify(env_var_value='1', flag_value='1', should_fail=True)
if __name__ == '__main__':
    gtest_test_utils.Main()