"""Unit test utilities for Google C++ Testing and Mocking Framework."""
import os
import sys
IS_WINDOWS = os.name == 'nt'
IS_CYGWIN = os.name == 'posix' and 'CYGWIN' in os.uname()[0]
IS_OS2 = os.name == 'os2'
import atexit
import shutil
import tempfile
import unittest as _test_module
try:
    import subprocess
    _SUBPROCESS_MODULE_AVAILABLE = True
except:
    import popen2
    _SUBPROCESS_MODULE_AVAILABLE = False
GTEST_OUTPUT_VAR_NAME = 'GTEST_OUTPUT'
PREMATURE_EXIT_FILE_ENV_VAR = 'TEST_PREMATURE_EXIT_FILE'
environ = os.environ.copy()

def SetEnvVar(env_var, value):
    if False:
        for i in range(10):
            print('nop')
    'Sets/unsets an environment variable to a given value.'
    if value is not None:
        environ[env_var] = value
    elif env_var in environ:
        del environ[env_var]
TestCase = _test_module.TestCase
_flag_map = {'source_dir': os.path.dirname(sys.argv[0]), 'build_dir': os.path.dirname(sys.argv[0])}
_gtest_flags_are_parsed = False

def _ParseAndStripGTestFlags(argv):
    if False:
        for i in range(10):
            print('nop')
    'Parses and strips Google Test flags from argv.  This is idempotent.'
    global _gtest_flags_are_parsed
    if _gtest_flags_are_parsed:
        return
    _gtest_flags_are_parsed = True
    for flag in _flag_map:
        if flag.upper() in os.environ:
            _flag_map[flag] = os.environ[flag.upper()]
        i = 1
        while i < len(argv):
            prefix = '--' + flag + '='
            if argv[i].startswith(prefix):
                _flag_map[flag] = argv[i][len(prefix):]
                del argv[i]
                break
            else:
                i += 1

def GetFlag(flag):
    if False:
        return 10
    'Returns the value of the given flag.'
    _ParseAndStripGTestFlags(sys.argv)
    return _flag_map[flag]

def GetSourceDir():
    if False:
        return 10
    'Returns the absolute path of the directory where the .py files are.'
    return os.path.abspath(GetFlag('source_dir'))

def GetBuildDir():
    if False:
        i = 10
        return i + 15
    'Returns the absolute path of the directory where the test binaries are.'
    return os.path.abspath(GetFlag('build_dir'))
_temp_dir = None

def _RemoveTempDir():
    if False:
        while True:
            i = 10
    if _temp_dir:
        shutil.rmtree(_temp_dir, ignore_errors=True)
atexit.register(_RemoveTempDir)

def GetTempDir():
    if False:
        return 10
    global _temp_dir
    if not _temp_dir:
        _temp_dir = tempfile.mkdtemp()
    return _temp_dir

def GetTestExecutablePath(executable_name, build_dir=None):
    if False:
        for i in range(10):
            print('nop')
    "Returns the absolute path of the test binary given its name.\n\n  The function will print a message and abort the program if the resulting file\n  doesn't exist.\n\n  Args:\n    executable_name: name of the test binary that the test script runs.\n    build_dir:       directory where to look for executables, by default\n                     the result of GetBuildDir().\n\n  Returns:\n    The absolute path of the test binary.\n  "
    path = os.path.abspath(os.path.join(build_dir or GetBuildDir(), executable_name))
    if (IS_WINDOWS or IS_CYGWIN or IS_OS2) and (not path.endswith('.exe')):
        path += '.exe'
    if not os.path.exists(path):
        message = 'Unable to find the test binary "%s". Please make sure to provide\na path to the binary via the --build_dir flag or the BUILD_DIR\nenvironment variable.' % path
        (print >> sys.stderr, message)
        sys.exit(1)
    return path

def GetExitStatus(exit_code):
    if False:
        for i in range(10):
            print('nop')
    "Returns the argument to exit(), or -1 if exit() wasn't called.\n\n  Args:\n    exit_code: the result value of os.system(command).\n  "
    if os.name == 'nt':
        return exit_code
    elif os.WIFEXITED(exit_code):
        return os.WEXITSTATUS(exit_code)
    else:
        return -1

class Subprocess:

    def __init__(self, command, working_dir=None, capture_stderr=True, env=None):
        if False:
            for i in range(10):
                print('nop')
        "Changes into a specified directory, if provided, and executes a command.\n\n    Restores the old directory afterwards.\n\n    Args:\n      command:        The command to run, in the form of sys.argv.\n      working_dir:    The directory to change into.\n      capture_stderr: Determines whether to capture stderr in the output member\n                      or to discard it.\n      env:            Dictionary with environment to pass to the subprocess.\n\n    Returns:\n      An object that represents outcome of the executed process. It has the\n      following attributes:\n        terminated_by_signal   True iff the child process has been terminated\n                               by a signal.\n        signal                 Sygnal that terminated the child process.\n        exited                 True iff the child process exited normally.\n        exit_code              The code with which the child process exited.\n        output                 Child process's stdout and stderr output\n                               combined in a string.\n    "
        if _SUBPROCESS_MODULE_AVAILABLE:
            if capture_stderr:
                stderr = subprocess.STDOUT
            else:
                stderr = subprocess.PIPE
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr, cwd=working_dir, universal_newlines=True, env=env)
            self.output = p.communicate()[0]
            self._return_code = p.returncode
        else:
            old_dir = os.getcwd()

            def _ReplaceEnvDict(dest, src):
                if False:
                    i = 10
                    return i + 15
                for key in dest.keys():
                    del dest[key]
                dest.update(src)
            if env is not None:
                old_environ = os.environ.copy()
                _ReplaceEnvDict(os.environ, env)
            try:
                if working_dir is not None:
                    os.chdir(working_dir)
                if capture_stderr:
                    p = popen2.Popen4(command)
                else:
                    p = popen2.Popen3(command)
                p.tochild.close()
                self.output = p.fromchild.read()
                ret_code = p.wait()
            finally:
                os.chdir(old_dir)
                if env is not None:
                    _ReplaceEnvDict(os.environ, old_environ)
            if os.WIFSIGNALED(ret_code):
                self._return_code = -os.WTERMSIG(ret_code)
            else:
                self._return_code = os.WEXITSTATUS(ret_code)
        if self._return_code < 0:
            self.terminated_by_signal = True
            self.exited = False
            self.signal = -self._return_code
        else:
            self.terminated_by_signal = False
            self.exited = True
            self.exit_code = self._return_code

def Main():
    if False:
        for i in range(10):
            print('nop')
    'Runs the unit test.'
    _ParseAndStripGTestFlags(sys.argv)
    if GTEST_OUTPUT_VAR_NAME in os.environ:
        del os.environ[GTEST_OUTPUT_VAR_NAME]
    _test_module.main()