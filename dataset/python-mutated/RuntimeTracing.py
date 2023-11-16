""" Runtime tracing

At this time we detect DLLs used by a program with this code, such
that we can check if it loads things outside of the program, but we
can also use this to determine what to include, so some plugins will
be using this.

"""
import os
import re
import sys
from nuitka.freezer.DependsExe import getDependsExePath, parseDependsExeOutput
from nuitka.utils.Execution import callProcess, executeProcess, isExecutableCommand, withEnvironmentVarOverridden
from nuitka.utils.FileOperations import deleteFile
from nuitka.utils.Utils import isFreeBSD, isMacOS, isWin32Windows
from .Common import traceExecutedCommand

def _getRuntimeTraceOfLoadedFilesWin32(logger, command, required):
    if False:
        return 10
    path = command[0]
    output_filename = path + '.depends'
    command = (getDependsExePath(), '-c', '-ot%s' % output_filename, '-f1', '-pb', '-pa1', '-ps1', '-pp1', '-po1', '-ph1', '-pl1', '-pt1', '-pe1', '-pg1', '-pf1', '-pc1') + tuple(command)
    try:
        callProcess(command, timeout=5 * 60)
    except Exception as e:
        if e.__class__.__name__ == 'TimeoutExpired':
            if required:
                logger.sysexit('Timeout encountered when running dependency walker.')
            logger.warning('Timeout encountered when running dependency walker.')
            return []
        else:
            raise
    result = parseDependsExeOutput(output_filename)
    deleteFile(output_filename, must_exist=False)
    return result

def _takeSystemCallTraceOutput(logger, path, command):
    if False:
        print('Hello World!')
    tracing_tool = command[0] if command[0] != 'sudo' else command[1]
    result = []
    with withEnvironmentVarOverridden('LD_PRELOAD', None):
        if os.environ.get('NUITKA_TRACE_COMMANDS', '0') != '0':
            traceExecutedCommand('Tracing with:', command)
        (_stdout_strace, stderr_strace, exit_strace) = executeProcess(command, stdin=False, timeout=5 * 60)
        if exit_strace != 0:
            if str is not bytes:
                stderr_strace = stderr_strace.decode('utf8')
            logger.warning(stderr_strace)
            logger.sysexit("Failed to run '%s'." % tracing_tool)
        if b'dtrace: system integrity protection is on' in stderr_strace:
            logger.sysexit('System integrity protection prevents system call tracing.')
        with open(path + '.' + tracing_tool, 'wb') as f:
            f.write(stderr_strace)
        for line in stderr_strace.split(b'\n'):
            if exit_strace != 0:
                logger.my_print(line)
            if not line:
                continue
            if b'ENOENT' in line:
                continue
            if line.startswith((b'write(', b'write_nocancel(')):
                continue
            if line.startswith((b'stat(', b'newfstatat(')) and b'S_IFDIR' in line:
                continue
            if line.startswith(b'stat64(') and b'= -1' in line:
                continue
            result.extend((os.path.abspath(match) for match in re.findall(b'"(.*?)(?:\\\\0)?"', line)))
        if str is not bytes:
            result = [s.decode('utf8') for s in result]
    return result

def _getRuntimeTraceOfLoadedFilesDtruss(logger, command):
    if False:
        print('Hello World!')
    if not isExecutableCommand('dtruss'):
        logger.sysexit("Error, needs 'dtruss' on your system to scan used libraries.")
    if not isExecutableCommand('sudo'):
        logger.sysexit("Error, needs 'sudo' on your system to scan used libraries.")
    binary_path = os.path.abspath(command[0])
    command = ('sudo', 'dtruss', binary_path) + tuple(command[1:])
    return _takeSystemCallTraceOutput(logger=logger, command=command, path=binary_path)

def _getRuntimeTraceOfLoadedFilesStrace(logger, command):
    if False:
        while True:
            i = 10
    if not isExecutableCommand('strace'):
        logger.sysexit("Error, needs 'strace' on your system to scan used libraries.")
    binary_path = os.path.abspath(command[0])
    command = ('strace', '-e', 'file', '-s4096', binary_path) + tuple(command[1:])
    return _takeSystemCallTraceOutput(logger=logger, command=command, path=binary_path)
_supports_taking_runtime_traces = None

def doesSupportTakingRuntimeTrace():
    if False:
        print('Hello World!')
    if not isMacOS():
        return True
    if str is bytes:
        return False
    global _supports_taking_runtime_traces
    if _supports_taking_runtime_traces is None:
        command = ('sudo', 'dtruss', 'echo')
        (_stdout, stderr, exit_code) = executeProcess(command, stdin=False, timeout=5 * 60)
        _supports_taking_runtime_traces = exit_code == 0 and b'dtrace: system integrity protection is on' not in stderr
    return _supports_taking_runtime_traces

def getRuntimeTraceOfLoadedFiles(logger, command, required=False):
    if False:
        for i in range(10):
            print('nop')
    'Returns the files loaded when executing a binary.'
    path = command[0]
    if not os.path.exists(path):
        logger.sysexit("Error, cannot find '%s' ('%s')." % (path, os.path.abspath(path)))
    result = []
    if isWin32Windows():
        result = _getRuntimeTraceOfLoadedFilesWin32(logger=logger, command=command, required=required)
    elif isMacOS() or isFreeBSD():
        result = _getRuntimeTraceOfLoadedFilesDtruss(logger=logger, command=command)
    elif os.name == 'posix':
        result = _getRuntimeTraceOfLoadedFilesStrace(logger=logger, command=command)
    result = tuple(sorted(set(result)))
    return result

def main():
    if False:
        print('Hello World!')
    from nuitka.Tracing import tools_logger
    for filename in getRuntimeTraceOfLoadedFiles(logger=tools_logger, command=sys.argv[1:]):
        print(filename)
if __name__ == '__main__':
    main()