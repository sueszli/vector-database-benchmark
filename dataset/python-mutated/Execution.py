""" Program execution related stuff.

Basically a layer for os, subprocess, shutil to come together. It can find
binaries (needed for exec) and run them capturing outputs.
"""
import os
from contextlib import contextmanager
from nuitka.__past__ import subprocess
from nuitka.Tracing import general
from .Download import getCachedDownloadedMinGW64
from .FileOperations import getExternalUsePath
from .Utils import getArchitecture, isWin32OrPosixWindows, isWin32Windows
_executable_command_cache = {}

def _getExecutablePath(filename, search_path):
    if False:
        i = 10
        return i + 15
    if isWin32OrPosixWindows() and (not filename.lower().endswith(('.exe', '.cmd'))):
        filename += '.exe'
    path_elements = search_path.split(os.pathsep)
    for path_element in path_elements:
        path_element = path_element.strip('"')
        path_element = os.path.expanduser(path_element)
        candidate = None
        if os.path.isfile(path_element):
            if os.path.normcase(os.path.basename(path_element)) == os.path.normcase(filename):
                candidate = path_element
        else:
            full = os.path.join(path_element, filename)
            if os.path.exists(full):
                candidate = full
        if candidate is not None:
            if os.access(candidate, os.X_OK):
                return candidate

def getExecutablePath(filename, extra_dir=None):
    if False:
        while True:
            i = 10
    'Find an execute in PATH environment.'
    search_path = os.environ.get('PATH', '')
    if extra_dir is not None:
        search_path = extra_dir + os.pathsep + search_path
    key = (filename, search_path)
    if key not in _executable_command_cache:
        _executable_command_cache[key] = _getExecutablePath(filename, search_path)
    return _executable_command_cache[key]

def isExecutableCommand(command):
    if False:
        while True:
            i = 10
    return getExecutablePath(command) is not None

class NuitkaCalledProcessError(subprocess.CalledProcessError):

    def __init__(self, exit_code, cmd, output, stderr):
        if False:
            for i in range(10):
                print('nop')
        subprocess.CalledProcessError(self, exit_code, cmd)
        self.stderr = stderr
        self.output = output
        self.cmd = cmd
        self.returncode = exit_code

    def __str__(self):
        if False:
            return 10
        result = subprocess.CalledProcessError.__str__(self)
        if self.output:
            result += ' Output was %r.' % self.output.strip()
        if self.stderr:
            result += ' Error was %r.' % self.stderr.strip()
        return result

def check_output(*popenargs, **kwargs):
    if False:
        while True:
            i = 10
    "Call a process and check result code.\n\n    This is for Python 2.6 compatibility, which doesn't have that in its\n    standard library.\n\n    Note: We use same name as in Python stdlib, violating our rules to\n    make it more recognizable what this does.\n    "
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    if 'stderr' not in kwargs:
        kwargs['stderr'] = subprocess.PIPE
    process = subprocess.Popen(*popenargs, stdout=subprocess.PIPE, **kwargs)
    (output, stderr) = process.communicate()
    exit_code = process.poll()
    if exit_code:
        cmd = kwargs.get('args')
        if cmd is None:
            cmd = popenargs[0]
        raise NuitkaCalledProcessError(exit_code, cmd, output=output, stderr=stderr)
    return output

def check_call(*popenargs, **kwargs):
    if False:
        print('Hello World!')
    'Call a process and check result code.\n\n    Note: This catches the error, and makes it nicer, and an error\n    exit. So this is for tooling only.\n\n    Note: We use same name as in Python stdlib, violating our rules to\n    make it more recognizable what this does.\n    '
    logger = kwargs.pop('logger', None)
    if logger is not None:
        logger.info("Executing command '%s'." % popenargs[0])
    try:
        subprocess.check_call(*popenargs, **kwargs)
    except OSError:
        general.sysexit("Error, failed to execute '%s'. Is it installed?" % popenargs[0])

def callProcess(*popenargs, **kwargs):
    if False:
        i = 10
        return i + 15
    'Call a process and return result code.'
    logger = kwargs.pop('logger', None)
    if logger is not None:
        logger.info("Executing command '%s'." % popenargs[0])
    return subprocess.call(*popenargs, **kwargs)

@contextmanager
def withEnvironmentPathAdded(env_var_name, *paths):
    if False:
        while True:
            i = 10
    assert os.path.sep not in env_var_name
    paths = [path for path in paths if path]
    path = os.pathsep.join(paths)
    if path:
        if str is not bytes and type(path) is bytes:
            path = path.decode('utf8')
        if env_var_name in os.environ:
            old_path = os.environ[env_var_name]
            os.environ[env_var_name] += os.pathsep + path
        else:
            old_path = None
            os.environ[env_var_name] = path
    yield
    if path:
        if old_path is None:
            del os.environ[env_var_name]
        else:
            os.environ[env_var_name] = old_path

@contextmanager
def withEnvironmentVarOverridden(env_var_name, value):
    if False:
        i = 10
        return i + 15
    'Change an environment and restore it after context.'
    if env_var_name in os.environ:
        old_value = os.environ[env_var_name]
    else:
        old_value = None
    if value is not None:
        os.environ[env_var_name] = value
    elif old_value is not None:
        del os.environ[env_var_name]
    yield
    if old_value is None:
        if value is not None:
            del os.environ[env_var_name]
    else:
        os.environ[env_var_name] = old_value

@contextmanager
def withEnvironmentVarsOverridden(mapping):
    if False:
        for i in range(10):
            print('nop')
    'Change multiple environment variables and restore them after context.'
    old_values = {}
    for (env_var_name, value) in mapping.items():
        if env_var_name in os.environ:
            old_values[env_var_name] = os.environ[env_var_name]
        else:
            old_values[env_var_name] = None
        if value is not None:
            os.environ[env_var_name] = value
        elif old_values[env_var_name] is not None:
            del os.environ[env_var_name]
    yield
    for (env_var_name, value) in mapping.items():
        if old_values[env_var_name] is None:
            if value is not None:
                del os.environ[env_var_name]
        else:
            os.environ[env_var_name] = old_values[env_var_name]

def wrapCommandForDebuggerForExec(*args):
    if False:
        while True:
            i = 10
    'Wrap a command for system debugger to call exec\n\n    Args:\n        args: (list of str) args for call to be debugged\n    Returns:\n        args tuple with debugger command inserted\n\n    Notes:\n        Currently only gdb and lldb are supported, but adding more\n        debuggers would be very welcome.\n    '
    gdb_path = getExecutablePath('gdb')
    lldb_path = None
    if isWin32Windows() and gdb_path is None:
        from nuitka.Options import assumeYesForDownloads
        mingw64_gcc_path = getCachedDownloadedMinGW64(target_arch=getArchitecture(), assume_yes_for_downloads=assumeYesForDownloads())
        with withEnvironmentPathAdded('PATH', os.path.dirname(mingw64_gcc_path)):
            lldb_path = getExecutablePath('lldb')
    if gdb_path is None and lldb_path is None:
        lldb_path = getExecutablePath('lldb')
        if lldb_path is None:
            general.sysexit("Error, no 'gdb' or 'lldb' binary found in path.")
    if gdb_path is not None:
        args = (gdb_path, 'gdb', '-ex=run', '-ex=where', '-ex=quit', '--args') + args
    else:
        args = (lldb_path, 'lldb', '-o', 'run', '-o', 'bt', '-o', 'quit', '--') + args
    return args

def wrapCommandForDebuggerForSubprocess(*args):
    if False:
        i = 10
        return i + 15
    'Wrap a command for system debugger with subprocess module.\n\n    Args:\n        args: (list of str) args for call to be debugged\n    Returns:\n        args tuple with debugger command inserted\n\n    Notes:\n        Currently only gdb and lldb are supported, but adding more\n        debuggers would be very welcome.\n    '
    args = wrapCommandForDebuggerForExec(*args)
    args = args[0:1] + args[2:]
    return args

def getNullOutput():
    if False:
        print('Hello World!')
    try:
        return subprocess.NULLDEV
    except AttributeError:
        return open(os.devnull, 'wb')

def getNullInput():
    if False:
        while True:
            i = 10
    try:
        return subprocess.NULLDEV
    except AttributeError:
        subprocess.NULLDEV = open(os.devnull, 'rb')
        return subprocess.NULLDEV

def executeToolChecked(logger, command, absence_message, stderr_filter=None, optional=False):
    if False:
        print('Hello World!')
    'Execute external tool, checking for success and no error outputs, returning result.'
    command = list(command)
    tool = command[0]
    if not isExecutableCommand(tool):
        if optional:
            logger.warning(absence_message)
            return (0, b'', b'')
        else:
            logger.sysexit(absence_message)
    command[0] = getExecutablePath(tool)
    process = subprocess.Popen(command, stdin=getNullInput(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    (stdout, stderr) = process.communicate()
    result = process.poll()
    if stderr_filter is not None:
        (new_result, stderr) = stderr_filter(stderr)
        if new_result is not None:
            result = new_result
    if result != 0:
        logger.sysexit("Error, call to '%s' failed: %s -> %s." % (tool, command, stderr))
    elif stderr:
        logger.sysexit("Error, call to '%s' gave warnings: %s -> %s." % (tool, command, stderr))
    return stdout

def createProcess(command, env=None, stdin=False, stdout=None, stderr=None, shell=False, external_cwd=False, new_group=False):
    if False:
        return 10
    if not env:
        env = os.environ
    kw_args = {}
    if new_group:
        if isWin32Windows():
            kw_args['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw_args['preexec_fn'] = os.setsid
    process = subprocess.Popen(command, stdin=subprocess.PIPE if stdin not in (False, None) else getNullInput(), stdout=subprocess.PIPE if stdout is None else stdout, stderr=subprocess.PIPE if stderr is None else stderr, shell=shell, close_fds=not isWin32Windows(), env=env, cwd=getExternalUsePath(os.getcwd()) if external_cwd else None, **kw_args)
    return process

def executeProcess(command, env=None, stdin=False, shell=False, external_cwd=False, timeout=None):
    if False:
        i = 10
        return i + 15
    process = createProcess(command=command, env=env, stdin=stdin, shell=shell, external_cwd=external_cwd)
    if stdin is True:
        process_input = None
    elif stdin is not False:
        process_input = stdin
    else:
        process_input = None
    kw_args = {}
    if timeout is not None:
        if 'timeout' in subprocess.Popen.communicate.__code__.co_varnames:
            kw_args['timeout'] = timeout
    (stdout, stderr) = process.communicate(input=process_input)
    exit_code = process.wait()
    return (stdout, stderr, exit_code)