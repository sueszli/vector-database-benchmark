"""Verifies that Google Test correctly determines whether to use colors."""
import os
import gtest_test_utils
IS_WINDOWS = os.name == 'nt'
COLOR_ENV_VAR = 'GTEST_COLOR'
COLOR_FLAG = 'gtest_color'
COMMAND = gtest_test_utils.GetTestExecutablePath('googletest-color-test_')

def SetEnvVar(env_var, value):
    if False:
        print('Hello World!')
    "Sets the env variable to 'value'; unsets it when 'value' is None."
    if value is not None:
        os.environ[env_var] = value
    elif env_var in os.environ:
        del os.environ[env_var]

def UsesColor(term, color_env_var, color_flag):
    if False:
        i = 10
        return i + 15
    'Runs googletest-color-test_ and returns its exit code.'
    SetEnvVar('TERM', term)
    SetEnvVar(COLOR_ENV_VAR, color_env_var)
    if color_flag is None:
        args = []
    else:
        args = ['--%s=%s' % (COLOR_FLAG, color_flag)]
    p = gtest_test_utils.Subprocess([COMMAND] + args)
    return not p.exited or p.exit_code

class GTestColorTest(gtest_test_utils.TestCase):

    def testNoEnvVarNoFlag(self):
        if False:
            return 10
        "Tests the case when there's neither GTEST_COLOR nor --gtest_color."
        if not IS_WINDOWS:
            self.assert_(not UsesColor('dumb', None, None))
            self.assert_(not UsesColor('emacs', None, None))
            self.assert_(not UsesColor('xterm-mono', None, None))
            self.assert_(not UsesColor('unknown', None, None))
            self.assert_(not UsesColor(None, None, None))
        self.assert_(UsesColor('linux', None, None))
        self.assert_(UsesColor('cygwin', None, None))
        self.assert_(UsesColor('xterm', None, None))
        self.assert_(UsesColor('xterm-color', None, None))
        self.assert_(UsesColor('xterm-256color', None, None))

    def testFlagOnly(self):
        if False:
            while True:
                i = 10
        "Tests the case when there's --gtest_color but not GTEST_COLOR."
        self.assert_(not UsesColor('dumb', None, 'no'))
        self.assert_(not UsesColor('xterm-color', None, 'no'))
        if not IS_WINDOWS:
            self.assert_(not UsesColor('emacs', None, 'auto'))
        self.assert_(UsesColor('xterm', None, 'auto'))
        self.assert_(UsesColor('dumb', None, 'yes'))
        self.assert_(UsesColor('xterm', None, 'yes'))

    def testEnvVarOnly(self):
        if False:
            while True:
                i = 10
        "Tests the case when there's GTEST_COLOR but not --gtest_color."
        self.assert_(not UsesColor('dumb', 'no', None))
        self.assert_(not UsesColor('xterm-color', 'no', None))
        if not IS_WINDOWS:
            self.assert_(not UsesColor('dumb', 'auto', None))
        self.assert_(UsesColor('xterm-color', 'auto', None))
        self.assert_(UsesColor('dumb', 'yes', None))
        self.assert_(UsesColor('xterm-color', 'yes', None))

    def testEnvVarAndFlag(self):
        if False:
            return 10
        'Tests the case when there are both GTEST_COLOR and --gtest_color.'
        self.assert_(not UsesColor('xterm-color', 'no', 'no'))
        self.assert_(UsesColor('dumb', 'no', 'yes'))
        self.assert_(UsesColor('xterm-color', 'no', 'auto'))

    def testAliasesOfYesAndNo(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests using aliases in specifying --gtest_color.'
        self.assert_(UsesColor('dumb', None, 'true'))
        self.assert_(UsesColor('dumb', None, 'YES'))
        self.assert_(UsesColor('dumb', None, 'T'))
        self.assert_(UsesColor('dumb', None, '1'))
        self.assert_(not UsesColor('xterm', None, 'f'))
        self.assert_(not UsesColor('xterm', None, 'false'))
        self.assert_(not UsesColor('xterm', None, '0'))
        self.assert_(not UsesColor('xterm', None, 'unknown'))
if __name__ == '__main__':
    gtest_test_utils.Main()