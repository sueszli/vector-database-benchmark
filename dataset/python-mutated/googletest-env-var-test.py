"""Verifies that Google Test correctly parses environment variables."""
import os
import gtest_test_utils
IS_WINDOWS = os.name == 'nt'
IS_LINUX = os.name == 'posix' and os.uname()[0] == 'Linux'
COMMAND = gtest_test_utils.GetTestExecutablePath('googletest-env-var-test_')
environ = os.environ.copy()

def AssertEq(expected, actual):
    if False:
        while True:
            i = 10
    if expected != actual:
        print('Expected: %s' % (expected,))
        print('  Actual: %s' % (actual,))
        raise AssertionError

def SetEnvVar(env_var, value):
    if False:
        print('Hello World!')
    "Sets the env variable to 'value'; unsets it when 'value' is None."
    if value is not None:
        environ[env_var] = value
    elif env_var in environ:
        del environ[env_var]

def GetFlag(flag):
    if False:
        while True:
            i = 10
    'Runs googletest-env-var-test_ and returns its output.'
    args = [COMMAND]
    if flag is not None:
        args += [flag]
    return gtest_test_utils.Subprocess(args, env=environ).output

def TestFlag(flag, test_val, default_val):
    if False:
        i = 10
        return i + 15
    'Verifies that the given flag is affected by the corresponding env var.'
    env_var = 'GTEST_' + flag.upper()
    SetEnvVar(env_var, test_val)
    AssertEq(test_val, GetFlag(flag))
    SetEnvVar(env_var, None)
    AssertEq(default_val, GetFlag(flag))

class GTestEnvVarTest(gtest_test_utils.TestCase):

    def testEnvVarAffectsFlag(self):
        if False:
            while True:
                i = 10
        'Tests that environment variable should affect the corresponding flag.'
        TestFlag('break_on_failure', '1', '0')
        TestFlag('color', 'yes', 'auto')
        TestFlag('filter', 'FooTest.Bar', '*')
        SetEnvVar('XML_OUTPUT_FILE', None)
        TestFlag('output', 'xml:tmp/foo.xml', '')
        TestFlag('print_time', '0', '1')
        TestFlag('repeat', '999', '1')
        TestFlag('throw_on_failure', '1', '0')
        TestFlag('death_test_style', 'threadsafe', 'fast')
        TestFlag('catch_exceptions', '0', '1')
        if IS_LINUX:
            TestFlag('death_test_use_fork', '1', '0')
            TestFlag('stack_trace_depth', '0', '100')

    def testXmlOutputFile(self):
        if False:
            return 10
        'Tests that $XML_OUTPUT_FILE affects the output flag.'
        SetEnvVar('GTEST_OUTPUT', None)
        SetEnvVar('XML_OUTPUT_FILE', 'tmp/bar.xml')
        AssertEq('xml:tmp/bar.xml', GetFlag('output'))

    def testXmlOutputFileOverride(self):
        if False:
            while True:
                i = 10
        'Tests that $XML_OUTPUT_FILE is overridden by $GTEST_OUTPUT.'
        SetEnvVar('GTEST_OUTPUT', 'xml:tmp/foo.xml')
        SetEnvVar('XML_OUTPUT_FILE', 'tmp/bar.xml')
        AssertEq('xml:tmp/foo.xml', GetFlag('output'))
if __name__ == '__main__':
    gtest_test_utils.Main()