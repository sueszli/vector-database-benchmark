"""
    :codeauthor: Pedro Algarvio (pedro@algarvio.me)


    salt.utils.vt
    ~~~~~~~~~~~~~

    Virtual Terminal

    This code has been heavily inspired by Python's subprocess code, the `non
    blocking version of it`__, some minor online snippets about TTY handling
    with python including `Python's own ``pty`` source code`__ and `Pexpect`__
    which has already surpassed some of the pitfalls that some systems would
    get us into.

    .. __: http://code.activestate.com/recipes/440554/
    .. __: https://github.com/python/cpython/blob/3.3/Lib/pty.py
    .. __: https://github.com/pexpect/pexpect

"""
import errno
import functools
import logging
import os
import select
import signal
import subprocess
import sys
import time
import salt.utils.crypt
import salt.utils.data
import salt.utils.stringutils
from salt._logging import LOG_LEVELS
mswindows = sys.platform == 'win32'
try:
    import msvcrt
    import win32api
    import win32con
    import win32process
    from win32file import ReadFile, WriteFile
    from win32pipe import PeekNamedPipe
except ImportError:
    import fcntl
    import pty
    import struct
    import termios
log = logging.getLogger(__name__)

class TerminalException(Exception):
    """
    Terminal specific exception
    """

def setwinsize(child, rows=80, cols=80):
    if False:
        for i in range(10):
            print('nop')
    '\n    This sets the terminal window size of the child tty. This will\n    cause a SIGWINCH signal to be sent to the child. This does not\n    change the physical window size. It changes the size reported to\n    TTY-aware applications like vi or curses -- applications that\n    respond to the SIGWINCH signal.\n\n    Thank you for the shortcut PEXPECT\n    '
    TIOCSWINSZ = getattr(termios, 'TIOCSWINSZ', -2146929561)
    if TIOCSWINSZ == 2148037735:
        TIOCSWINSZ = -2146929561
    packed = struct.pack(b'HHHH', rows, cols, 0, 0)
    fcntl.ioctl(child, TIOCSWINSZ, packed)

def getwinsize(child):
    if False:
        return 10
    '\n    This returns the terminal window size of the child tty. The return\n    value is a tuple of (rows, cols).\n\n    Thank you for the shortcut PEXPECT\n    '
    TIOCGWINSZ = getattr(termios, 'TIOCGWINSZ', 1074295912)
    packed = struct.pack(b'HHHH', 0, 0, 0, 0)
    ioctl = fcntl.ioctl(child, TIOCGWINSZ, packed)
    return struct.unpack(b'HHHH', ioctl)[0:2]

class Terminal:
    """
    I'm a virtual terminal
    """

    def __init__(self, args=None, executable=None, shell=False, cwd=None, env=None, preexec_fn=None, rows=None, cols=None, log_stdin=None, log_stdin_level='debug', log_stdout=None, log_stdout_level='debug', log_stderr=None, log_stderr_level='debug', log_sanitize=None, stream_stdout=None, stream_stderr=None, force_receive_encoding=__salt_system_encoding__):
        if False:
            return 10
        if not args and (not executable):
            raise TerminalException('You need to pass at least one of "args", "executable" ')
        self.args = args
        self.executable = executable
        self.shell = shell
        self.cwd = cwd
        self.env = env
        self.preexec_fn = preexec_fn
        self.receive_encoding = force_receive_encoding
        if rows is None and cols is None:
            (rows, cols) = self.__detect_parent_terminal_size()
        elif rows is not None and cols is None:
            (_, cols) = self.__detect_parent_terminal_size()
        elif rows is None and cols is not None:
            (rows, _) = self.__detect_parent_terminal_size()
        self.rows = rows
        self.cols = cols
        self.pid = None
        self.stdin = None
        self.stdout = None
        self.stderr = None
        self.child_fd = None
        self.child_fde = None
        self.partial_data_stdout = b''
        self.partial_data_stderr = b''
        self.closed = True
        self.flag_eof_stdout = False
        self.flag_eof_stderr = False
        self.terminated = True
        self.exitstatus = None
        self.signalstatus = None
        self.status = None
        if stream_stdout is True:
            self.stream_stdout = sys.stdout
        elif stream_stdout is False:
            self.stream_stdout = None
        elif stream_stdout is not None:
            if not hasattr(stream_stdout, 'write') or not hasattr(stream_stdout, 'flush') or (not hasattr(stream_stdout, 'close')):
                raise TerminalException("'stream_stdout' needs to have at least 3 methods, 'write()', 'flush()' and 'close()'.")
            self.stream_stdout = stream_stdout
        else:
            raise TerminalException("Don't know how to handle '{}' as the VT's 'stream_stdout' parameter.".format(stream_stdout))
        if stream_stderr is True:
            self.stream_stderr = sys.stderr
        elif stream_stderr is False:
            self.stream_stderr = None
        elif stream_stderr is not None:
            if not hasattr(stream_stderr, 'write') or not hasattr(stream_stderr, 'flush') or (not hasattr(stream_stderr, 'close')):
                raise TerminalException("'stream_stderr' needs to have at least 3 methods, 'write()', 'flush()' and 'close()'.")
            self.stream_stderr = stream_stderr
        else:
            raise TerminalException("Don't know how to handle '{}' as the VT's 'stream_stderr' parameter.".format(stream_stderr))
        try:
            self._spawn()
        except Exception as err:
            log.warning('Failed to spawn the VT: %s', err, exc_info_on_loglevel=logging.DEBUG)
            raise TerminalException('Failed to spawn the VT. Error: {}'.format(err))
        log.debug('Child Forked! PID: %s  STDOUT_FD: %s  STDERR_FD: %s', self.pid, self.child_fd, self.child_fde)
        if log_sanitize:
            if not isinstance(log_sanitize, str):
                raise RuntimeError("'log_sanitize' needs to be a str type")
            self.log_sanitize = log_sanitize
        else:
            self.log_sanitize = None
        terminal_command = ' '.join(self.args)
        if self.log_sanitize:
            terminal_command = terminal_command.replace(self.log_sanitize, '*' * 6)
        if 'decode("base64")' in terminal_command or 'base64.b64decode(' in terminal_command:
            log.debug('VT: Salt-SSH SHIM Terminal Command executed. Logged to TRACE')
            log.trace('Terminal Command: %s', terminal_command)
        else:
            log.debug('Terminal Command: %s', terminal_command)
        self.stdin_logger_level = LOG_LEVELS.get(log_stdin_level, log_stdin_level)
        if log_stdin is True:
            self.stdin_logger = logging.getLogger('{}.{}.PID-{}.STDIN'.format(__name__, self.__class__.__name__, self.pid))
        elif log_stdin is not None:
            if not isinstance(log_stdin, logging.Logger):
                raise RuntimeError("'log_stdin' needs to subclass `logging.Logger`")
            self.stdin_logger = log_stdin
        else:
            self.stdin_logger = None
        self.stdout_logger_level = LOG_LEVELS.get(log_stdout_level, log_stdout_level)
        if log_stdout is True:
            self.stdout_logger = logging.getLogger('{}.{}.PID-{}.STDOUT'.format(__name__, self.__class__.__name__, self.pid))
        elif log_stdout is not None:
            if not isinstance(log_stdout, logging.Logger):
                raise RuntimeError("'log_stdout' needs to subclass `logging.Logger`")
            self.stdout_logger = log_stdout
        else:
            self.stdout_logger = None
        self.stderr_logger_level = LOG_LEVELS.get(log_stderr_level, log_stderr_level)
        if log_stderr is True:
            self.stderr_logger = logging.getLogger('{}.{}.PID-{}.STDERR'.format(__name__, self.__class__.__name__, self.pid))
        elif log_stderr is not None:
            if not isinstance(log_stderr, logging.Logger):
                raise RuntimeError("'log_stderr' needs to subclass `logging.Logger`")
            self.stderr_logger = log_stderr
        else:
            self.stderr_logger = None

    def send(self, data):
        if False:
            print('Hello World!')
        '\n        Send data to the terminal. You are responsible to send any required\n        line feeds.\n        '
        return self._send(data)

    def sendline(self, data, linesep=os.linesep):
        if False:
            while True:
                i = 10
        '\n        Send the provided data to the terminal appending a line feed.\n        '
        return self.send('{}{}'.format(data, linesep))

    def recv(self, maxsize=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Receive data from the terminal as a (``stdout``, ``stderr``) tuple. If\n        any of those is ``None`` we can no longer communicate with the\n        terminal's child process.\n        "
        if maxsize is None:
            maxsize = 1024
        elif maxsize < 1:
            maxsize = 1
        return self._recv(maxsize)

    def close(self, terminate=True, kill=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Close the communication with the terminal's child.\n        If ``terminate`` is ``True`` then additionally try to terminate the\n        terminal, and if ``kill`` is also ``True``, kill the terminal if\n        terminating it was not enough.\n        "
        if not self.closed:
            if self.child_fd is not None:
                os.close(self.child_fd)
                self.child_fd = None
            if self.child_fde is not None:
                os.close(self.child_fde)
                self.child_fde = None
            time.sleep(0.1)
            if terminate:
                if not self.terminate(kill):
                    raise TerminalException('Failed to terminate child process.')
            self.closed = True

    @property
    def has_unread_data(self):
        if False:
            i = 10
            return i + 15
        return self.flag_eof_stderr is False or self.flag_eof_stdout is False

    def _translate_newlines(self, data):
        if False:
            return 10
        if data is None or not data:
            return
        return data.replace('\r\n', os.linesep)

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            for i in range(10):
                print('nop')
        self.close(terminate=True, kill=True)
        if self.isalive():
            self.wait()
    if mswindows:

        def _execute(self):
            if False:
                for i in range(10):
                    print('nop')
            raise NotImplementedError

        def _spawn(self):
            if False:
                for i in range(10):
                    print('nop')
            raise NotImplementedError

        def _recv(self, maxsize):
            if False:
                return 10
            raise NotImplementedError

        def _send(self, data):
            if False:
                while True:
                    i = 10
            raise NotImplementedError

        def send_signal(self, sig):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Send a signal to the process\n            '
            if sig == signal.SIGTERM:
                self.terminate()
            elif sig == signal.CTRL_C_EVENT:
                os.kill(self.pid, signal.CTRL_C_EVENT)
            elif sig == signal.CTRL_BREAK_EVENT:
                os.kill(self.pid, signal.CTRL_BREAK_EVENT)
            else:
                raise ValueError('Unsupported signal: {}'.format(sig))

        def terminate(self, force=False):
            if False:
                i = 10
                return i + 15
            '\n            Terminates the process\n            '
            try:
                win32api.TerminateProcess(self._handle, 1)
            except OSError:
                ecode = win32process.GetExitCodeProcess(self._handle)
                if ecode == win32con.STILL_ACTIVE:
                    raise
                self.exitstatus = ecode
        kill = terminate
    else:

        def _spawn(self):
            if False:
                for i in range(10):
                    print('nop')
            if not isinstance(self.args, str) and self.shell is True:
                self.args = ' '.join(self.args)
            (parent, child) = pty.openpty()
            (err_parent, err_child) = os.pipe()
            child_name = os.ttyname(child)
            proc = subprocess.Popen(self.args, preexec_fn=functools.partial(self._preexec, child_name, self.rows, self.cols, self.preexec_fn), shell=self.shell, executable=self.executable, cwd=self.cwd, stdin=child, stdout=child, stderr=err_child, env=self.env, close_fds=True)
            os.close(child)
            os.close(err_child)
            self.child_fd = parent
            self.child_fde = err_parent
            self.pid = proc.pid
            self.proc = proc
            self.closed = False
            self.terminated = False

        @staticmethod
        def _preexec(child_name, rows=80, cols=80, preexec_fn=None):
            if False:
                while True:
                    i = 10
            try:
                tty_fd = os.open('/dev/tty', os.O_RDWR | os.O_NOCTTY)
                if tty_fd >= 0:
                    os.close(tty_fd)
            except Exception:
                pass
            try:
                os.setsid()
            except OSError:
                pass
            try:
                tty_fd = os.open('/dev/tty', os.O_RDWR | os.O_NOCTTY)
                if tty_fd >= 0:
                    os.close(tty_fd)
                    raise TerminalException('Could not open child pty, {}'.format(child_name))
            except Exception:
                pass
            tty_fd = os.open(child_name, os.O_RDWR)
            setwinsize(tty_fd, rows, cols)
            if tty_fd < 0:
                raise TerminalException('Could not open child pty, {}'.format(child_name))
            else:
                os.close(tty_fd)
            if os.name != 'posix':
                tty_fd = os.open('/dev/tty', os.O_WRONLY)
                if tty_fd < 0:
                    raise TerminalException('Could not open controlling tty, /dev/tty')
                else:
                    os.close(tty_fd)
            salt.utils.crypt.reinit_crypto()
            if preexec_fn is not None:
                preexec_fn()

        def _send(self, data):
            if False:
                for i in range(10):
                    print('nop')
            if self.child_fd is None:
                return None
            if not select.select([], [self.child_fd], [], 0)[1]:
                return 0
            try:
                if self.stdin_logger:
                    self.stdin_logger.log(self.stdin_logger_level, data)
                written = os.write(self.child_fd, data.encode(__salt_system_encoding__))
            except OSError as why:
                if why.errno == errno.EPIPE:
                    os.close(self.child_fd)
                    self.child_fd = None
                    return
                raise
            return written

        def _recv(self, maxsize):
            if False:
                print('Hello World!')
            rfds = []
            if self.child_fd:
                rfds.append(self.child_fd)
            if self.child_fde:
                rfds.append(self.child_fde)
            if not self.isalive():
                if not rfds:
                    self.close()
                    return (None, None)
                (rlist, _, _) = select.select(rfds, [], [], 0)
                if not rlist:
                    self.flag_eof_stdout = self.flag_eof_stderr = True
                    log.debug('End of file(EOL). Brain-dead platform.')
                    if self.partial_data_stdout or self.partial_data_stderr:
                        self.close()
                        return (salt.utils.stringutils.to_unicode(self.partial_data_stdout), salt.utils.stringutils.to_unicode(self.partial_data_stderr))
                    self.close()
                    return (None, None)
            stderr = ''
            stdout = ''
            if self.child_fd:
                fd_flags = fcntl.fcntl(self.child_fd, fcntl.F_GETFL)
            if self.child_fde:
                fde_flags = fcntl.fcntl(self.child_fde, fcntl.F_GETFL)
            if self.child_fd:
                fcntl.fcntl(self.child_fd, fcntl.F_SETFL, fd_flags | os.O_NONBLOCK)
            if self.child_fde:
                fcntl.fcntl(self.child_fde, fcntl.F_SETFL, fde_flags | os.O_NONBLOCK)
            (rlist, _, _) = select.select(rfds, [], [], 0)
            if not rlist:
                if not self.isalive():
                    self.flag_eof_stdout = self.flag_eof_stderr = True
                    log.debug('End of file(EOL). Very slow platform.')
                    return (None, None)

            def read_and_decode_fd(fd, maxsize, partial_data_attr=None):
                if False:
                    for i in range(10):
                        print('nop')
                bytes_read = getattr(self, partial_data_attr, b'')
                bytes_read += os.read(fd, maxsize if not bytes_read else 1)
                try:
                    decoded_data = self._translate_newlines(salt.utils.stringutils.to_unicode(bytes_read, self.receive_encoding))
                    if partial_data_attr is not None:
                        setattr(self, partial_data_attr, b'')
                    return (decoded_data, False)
                except UnicodeDecodeError as ex:
                    max_multibyte_character_length = 4
                    if ex.start > len(bytes_read) - max_multibyte_character_length and ex.end == len(bytes_read):
                        if partial_data_attr is not None:
                            setattr(self, partial_data_attr, bytes_read)
                        else:
                            raise
                        return ('', True)
                    else:
                        raise
            if self.child_fde in rlist and (not self.flag_eof_stderr):
                try:
                    (stderr, partial_data) = read_and_decode_fd(self.child_fde, maxsize, 'partial_data_stderr')
                    if not stderr and (not partial_data):
                        self.flag_eof_stderr = True
                        stderr = None
                    else:
                        if self.stream_stderr:
                            self.stream_stderr.write(stderr)
                            self.stream_stderr.flush()
                        if self.stderr_logger:
                            stripped = stderr.rstrip()
                            if self.log_sanitize:
                                stripped = stripped.replace(self.log_sanitize, '*' * 6)
                            if stripped.startswith(os.linesep):
                                stripped = stripped[len(os.linesep):]
                            if stripped:
                                self.stderr_logger.log(self.stderr_logger_level, stripped)
                except OSError:
                    os.close(self.child_fde)
                    self.child_fde = None
                    self.flag_eof_stderr = True
                    stderr = None
                finally:
                    if self.child_fde is not None:
                        fcntl.fcntl(self.child_fde, fcntl.F_SETFL, fde_flags)
            if self.child_fd in rlist and (not self.flag_eof_stdout):
                try:
                    (stdout, partial_data) = read_and_decode_fd(self.child_fd, maxsize, 'partial_data_stdout')
                    if not stdout and (not partial_data):
                        self.flag_eof_stdout = True
                        stdout = None
                    else:
                        if self.stream_stdout:
                            self.stream_stdout.write(salt.utils.stringutils.to_str(stdout))
                            self.stream_stdout.flush()
                        if self.stdout_logger:
                            stripped = stdout.rstrip()
                            if self.log_sanitize:
                                stripped = stripped.replace(self.log_sanitize, '*' * 6)
                            if stripped.startswith(os.linesep):
                                stripped = stripped[len(os.linesep):]
                            if stripped:
                                self.stdout_logger.log(self.stdout_logger_level, stripped)
                except OSError:
                    os.close(self.child_fd)
                    self.child_fd = None
                    self.flag_eof_stdout = True
                    stdout = None
                finally:
                    if self.child_fd is not None:
                        fcntl.fcntl(self.child_fd, fcntl.F_SETFL, fd_flags)
            return (stdout, stderr)

        def __detect_parent_terminal_size(self):
            if False:
                return 10
            try:
                TIOCGWINSZ = getattr(termios, 'TIOCGWINSZ', 1074295912)
                packed = struct.pack(b'HHHH', 0, 0, 0, 0)
                ioctl = fcntl.ioctl(sys.stdin.fileno(), TIOCGWINSZ, packed)
                return struct.unpack(b'HHHH', ioctl)[0:2]
            except OSError:
                return (24, 80)

        def getwinsize(self):
            if False:
                while True:
                    i = 10
            '\n            This returns the terminal window size of the child tty. The return\n            value is a tuple of (rows, cols).\n\n            Thank you for the shortcut PEXPECT\n            '
            if self.child_fd is None:
                raise TerminalException("Can't check the size of the terminal since we're not connected to the child process.")
            return getwinsize(self.child_fd)

        def setwinsize(self, child, rows=80, cols=80):
            if False:
                while True:
                    i = 10
            setwinsize(self.child_fd, rows, cols)

        def isalive(self, _waitpid=os.waitpid, _wnohang=os.WNOHANG, _wifexited=os.WIFEXITED, _wexitstatus=os.WEXITSTATUS, _wifsignaled=os.WIFSIGNALED, _wifstopped=os.WIFSTOPPED, _wtermsig=os.WTERMSIG, _os_error=os.error, _errno_echild=errno.ECHILD, _terminal_exception=TerminalException):
            if False:
                while True:
                    i = 10
            '\n            This tests if the child process is running or not. This is\n            non-blocking. If the child was terminated then this will read the\n            exitstatus or signalstatus of the child. This returns True if the\n            child process appears to be running or False if not. It can take\n            literally SECONDS for Solaris to return the right status.\n            '
            if self.terminated:
                return False
            if self.has_unread_data is False:
                waitpid_options = 0
            else:
                waitpid_options = _wnohang
            try:
                (pid, status) = _waitpid(self.pid, waitpid_options)
            except ChildProcessError:
                polled = self.proc.poll()
                if polled is None:
                    return True
                pid = self.pid
                status = polled
            except _os_error:
                err = sys.exc_info()[1]
                if err.errno == _errno_echild:
                    raise _terminal_exception('isalive() encountered condition where "terminated" is 0, but there was no child process. Did someone else call waitpid() on our process?')
                else:
                    raise
            if pid == 0:
                try:
                    (pid, status) = _waitpid(self.pid, waitpid_options)
                except _os_error as exc:
                    if exc.errno == _errno_echild:
                        raise _terminal_exception('isalive() encountered condition that should never happen. There was no child process. Did someone else call waitpid() on our process?')
                    else:
                        raise
                if pid == 0:
                    return True
            if pid == 0:
                return True
            if _wifexited(status):
                self.status = status
                self.exitstatus = _wexitstatus(status)
                self.signalstatus = None
                self.terminated = True
            elif _wifsignaled(status):
                self.status = status
                self.exitstatus = None
                self.signalstatus = _wtermsig(status)
                self.terminated = True
            elif _wifstopped(status):
                raise _terminal_exception('isalive() encountered condition where child process is stopped. This is not supported. Is some other process attempting job control with our child pid?')
            return False

        def terminate(self, force=False):
            if False:
                while True:
                    i = 10
            '\n            This forces a child process to terminate. It starts nicely with\n            SIGHUP and SIGINT. If "force" is True then moves onto SIGKILL. This\n            returns True if the child was terminated. This returns False if the\n            child could not be terminated.\n            '
            if not self.closed:
                self.close(terminate=False)
            if not self.isalive():
                return True
            try:
                self.send_signal(signal.SIGHUP)
                time.sleep(0.1)
                if not self.isalive():
                    return True
                self.send_signal(signal.SIGCONT)
                time.sleep(0.1)
                if not self.isalive():
                    return True
                self.send_signal(signal.SIGINT)
                time.sleep(0.1)
                if not self.isalive():
                    return True
                if force:
                    self.send_signal(signal.SIGKILL)
                    time.sleep(0.1)
                    if not self.isalive():
                        return True
                    else:
                        return False
                return False
            except OSError:
                time.sleep(0.1)
                if not self.isalive():
                    return True
                else:
                    return False

        def wait(self):
            if False:
                while True:
                    i = 10
            '\n            This waits until the child exits internally consuming any remaining\n            output from the child, thus, no blocking forever because the child\n            has unread data.\n            '
            if self.isalive():
                while self.isalive():
                    (stdout, stderr) = self.recv()
                    if stdout is None:
                        break
                    if stderr is None:
                        break
            else:
                raise TerminalException('Cannot wait for dead child process.')
            return self.exitstatus

        def send_signal(self, sig):
            if False:
                return 10
            '\n            Send a signal to the process\n            '
            os.kill(self.pid, sig)

        def kill(self):
            if False:
                return 10
            '\n            Kill the process with SIGKILL\n            '
            self.send_signal(signal.SIGKILL)