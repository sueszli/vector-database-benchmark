""" Spawning processes.

This is to replace the standard spawn implementation with one that tracks the
progress, and gives warnings about things taking very long.
"""
import os
import sys
import threading
from nuitka.Tracing import my_print, scons_logger
from nuitka.utils.Execution import executeProcess
from nuitka.utils.FileOperations import getReportPath
from nuitka.utils.Timing import TimerReport
from .SconsCaching import runClCache
from .SconsProgress import closeSconsProgressBar, reportSlowCompilation, updateSconsProgressBar
from .SconsUtils import decodeData

class SubprocessThread(threading.Thread):

    def __init__(self, cmdline, env):
        if False:
            return 10
        threading.Thread.__init__(self)
        self.cmdline = cmdline
        self.env = env
        self.data = None
        self.err = None
        self.exit_code = None
        self.exception = None
        self.timer_report = TimerReport(message='Running %s took %%.2f seconds' % repr(self.cmdline).replace('%', '%%'), min_report_time=360, logger=scons_logger)

    def run(self):
        if False:
            print('Hello World!')
        try:
            with self.timer_report:
                (self.data, self.err, self.exit_code) = executeProcess(command=self.cmdline, env=self.env)
        except Exception as e:
            self.exception = e

    def getProcessResult(self):
        if False:
            print('Hello World!')
        return (self.data, self.err, self.exit_code, self.exception)

def _runProcessMonitored(env, cmdline, os_env):
    if False:
        while True:
            i = 10
    thread = SubprocessThread(cmdline, os_env)
    thread.start()
    thread.join(360)
    if thread.is_alive():
        reportSlowCompilation(env, cmdline, thread.timer_report.getTimer().getDelta())
    thread.join()
    updateSconsProgressBar()
    return thread.getProcessResult()

def _filterMsvcLinkOutput(env, module_mode, data, exit_code):
    if False:
        for i in range(10):
            print('nop')
    data = data.rstrip()
    if module_mode:
        data = b'\r\n'.join((line for line in data.split(b'\r\n') if b'   Creating library' not in line if not (module_mode and b'.exp' in line)))
    if env.lto_mode and exit_code == 0:
        if len(data.split(b'\r\n')) == 2:
            data = b''
    if env.pgo_mode == 'use' and exit_code == 0:
        data = b''
    return data

def _raiseCorruptedObjectFilesExit(cache_name):
    if False:
        for i in range(10):
            print('nop')
    'Error exit due to corrupt object files and point to cache cleanup.'
    scons_logger.sysexit("Error, the C linker reported a corrupt object file. You may need to run\nNuitka with '--clean-cache=%s' once to repair it, or else will\nsurely happen again." % cache_name)

def _getNoSuchCommandErrorMessage():
    if False:
        print('Hello World!')
    import ctypes
    return ctypes.WinError(3).args[1]

def _getWindowsSpawnFunction(env, module_mode, source_files):
    if False:
        return 10

    def spawnWindowsCommand(sh, escape, cmd, args, os_env):
        if False:
            print('Hello World!')
        'Our own spawn implementation for use on Windows.'
        if cmd == 'del':
            assert len(args) == 2
            os.unlink(args[1])
            return 0

        def removeTrailingSlashQuote(arg):
            if False:
                for i in range(10):
                    print('nop')
            if arg.endswith('\\"'):
                return arg[:-1] + '\\"'
            else:
                return arg
        new_args = ' '.join((removeTrailingSlashQuote(arg) for arg in args[1:]))
        cmdline = cmd + ' ' + new_args
        if cmd == '<clcache>':
            (data, err, rv) = runClCache(args, os_env)
        else:
            (data, err, rv, exception) = _runProcessMonitored(env, cmdline, os_env)
            if exception:
                closeSconsProgressBar()
                raise exception
        if cmd == 'link':
            data = _filterMsvcLinkOutput(env=env, module_mode=module_mode, data=data, exit_code=rv)
        elif cmd in ('cl', '<clcache>'):
            data = data[data.find(b'\r\n') + 2:]
            source_base_names = [os.path.basename(source_file) for source_file in source_files]

            def check(line):
                if False:
                    for i in range(10):
                        print('nop')
                return line in (b'', b'Generating Code...') or line in source_base_names
            data = b'\r\n'.join((line for line in data.split(b'\r\n') if not check(line))) + b'\r\n'
        if data is not None and data.rstrip():
            my_print('Unexpected output from this command:', style='yellow')
            my_print(cmdline, style='yellow')
            if str is not bytes:
                data = decodeData(data)
            my_print(data, style='yellow', end='' if data.endswith('\n') else '\n')
        if err:
            if str is not bytes:
                err = decodeData(err)
            err = '\r\n'.join((line for line in err.split('\r\n') if not isIgnoredError(line) if not (env.mingw_mode and env.lto_mode and (line == _getNoSuchCommandErrorMessage()))))
            if err:
                if 'corrupt file' in err:
                    _raiseCorruptedObjectFilesExit(cache_name='clcache')
                if 'Bad magic value' in err:
                    _raiseCorruptedObjectFilesExit(cache_name='ccache')
                err += '\r\n'
                my_print(err, style='yellow', end='')
        return rv
    return spawnWindowsCommand

def _formatForOutput(arg):
    if False:
        i = 10
        return i + 15
    arg = arg.strip('"')
    slash = '\\'
    special = '"$()'
    arg = arg.replace(slash + slash, slash)
    for c in special:
        arg = arg.replace(slash + c, c)
    if arg.startswith('-I'):
        prefix = '-I'
        arg = arg[2:]
    else:
        prefix = ''
    return prefix + getReportPath(arg)

def isIgnoredError(line):
    if False:
        for i in range(10):
            print('nop')
    if "function `posix_tmpnam':" in line:
        return True
    if "function `posix_tempnam':" in line:
        return True
    if "the use of `tmpnam_r' is dangerous" in line:
        return True
    if "the use of `tempnam' is dangerous" in line:
        return True
    if line.startswith(('Objects/structseq.c:', 'Python/import.c:')):
        return True
    if line == "In function 'load_next',":
        return True
    if 'at Python/import.c' in line:
        return True
    if 'overriding recipe for target' in line:
        return True
    if 'ignoring old recipe for target' in line:
        return True
    if 'Error 1 (ignored)' in line:
        return True
    if line == 'bytearrayobject.o (symbol from plugin): warning: memset used with constant zero length parameter; this could be due to transposed parameters':
        return True
    if 'Dwarf Error:' in line:
        return True
    if line.startswith('mingw32-make:') and line.endswith('Error 1 (ignored)'):
        return True
    return False

def subprocess_spawn(args):
    if False:
        for i in range(10):
            print('nop')
    (sh, _cmd, args, env) = args
    (_stdout, stderr, exit_code) = executeProcess(command=[sh, '-c', ' '.join(args)], env=env)
    if str is not bytes:
        stderr = decodeData(stderr)
    ignore_next = False
    for line in stderr.splitlines():
        if ignore_next:
            ignore_next = False
            continue
        if 'Bad magic value' in line:
            _raiseCorruptedObjectFilesExit(cache_name='ccache')
        if isIgnoredError(line):
            ignore_next = True
            continue
        if exit_code != 0 and 'terminated with signal 11' in line:
            exit_code = -11
        my_print(line, style='yellow', file=sys.stderr)
    return exit_code

class SpawnThread(threading.Thread):

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        threading.Thread.__init__(self)
        self.args = args
        self.timer_report = TimerReport(message='Running %s took %%.2f seconds' % (' '.join((_formatForOutput(arg) for arg in self.args[2])).replace('%', '%%'),), min_report_time=360, logger=scons_logger)
        self.result = None
        self.exception = None

    def run(self):
        if False:
            i = 10
            return i + 15
        try:
            with self.timer_report:
                self.result = subprocess_spawn(self.args)
        except Exception as e:
            self.exception = e

    def getSpawnResult(self):
        if False:
            return 10
        return (self.result, self.exception)

def _runSpawnMonitored(env, sh, cmd, args, os_env):
    if False:
        print('Hello World!')
    thread = SpawnThread(sh, cmd, args, os_env)
    thread.start()
    thread.join(360)
    if thread.is_alive():
        reportSlowCompilation(env, cmd, thread.timer_report.getTimer().getDelta())
    thread.join()
    updateSconsProgressBar()
    return thread.getSpawnResult()

def _getWrappedSpawnFunction(env):
    if False:
        return 10

    def spawnCommand(sh, escape, cmd, args, os_env):
        if False:
            print('Hello World!')
        if '"__constants_data.o"' in args or '"__constants_data.os"' in args:
            os_env = dict(os_env)
            os_env['CCACHE_DISABLE'] = '1'
        (result, exception) = _runSpawnMonitored(env, sh, cmd, args, os_env)
        if exception:
            closeSconsProgressBar()
            raise exception
        if result == -11:
            scons_logger.sysexit("Error, the C compiler '%s' crashed with segfault. Consider upgrading it or using '--clang' option." % env.the_compiler)
        return result
    return spawnCommand

def enableSpawnMonitoring(env, module_mode, source_files):
    if False:
        for i in range(10):
            print('nop')
    if os.name == 'nt':
        env['SPAWN'] = _getWindowsSpawnFunction(env=env, module_mode=module_mode, source_files=source_files)
    else:
        env['SPAWN'] = _getWrappedSpawnFunction(env=env)