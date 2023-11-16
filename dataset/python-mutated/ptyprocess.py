import codecs
import errno
import fcntl
import io
import os
import pty
import resource
import signal
import struct
import sys
import termios
import time
try:
    import builtins
except ImportError:
    import __builtin__ as builtins
from pty import STDIN_FILENO, CHILD
from .util import which, PtyProcessError
_platform = sys.platform.lower()
_is_solaris = _platform.startswith('solaris') or _platform.startswith('sunos')
if _is_solaris:
    use_native_pty_fork = False
    from . import _fork_pty
else:
    use_native_pty_fork = True
PY3 = sys.version_info[0] >= 3
if PY3:

    def _byte(i):
        if False:
            return 10
        return bytes([i])
else:

    def _byte(i):
        if False:
            i = 10
            return i + 15
        return chr(i)

    class FileNotFoundError(OSError):
        pass

    class TimeoutError(OSError):
        pass
(_EOF, _INTR) = (None, None)

def _make_eof_intr():
    if False:
        for i in range(10):
            print('nop')
    'Set constants _EOF and _INTR.\n    \n    This avoids doing potentially costly operations on module load.\n    '
    global _EOF, _INTR
    if _EOF is not None and _INTR is not None:
        return
    try:
        from termios import VEOF, VINTR
        fd = None
        for name in ('stdin', 'stdout'):
            stream = getattr(sys, '__%s__' % name, None)
            if stream is None or not hasattr(stream, 'fileno'):
                continue
            try:
                fd = stream.fileno()
            except ValueError:
                continue
        if fd is None:
            raise ValueError('No stream has a fileno')
        intr = ord(termios.tcgetattr(fd)[6][VINTR])
        eof = ord(termios.tcgetattr(fd)[6][VEOF])
    except (ImportError, OSError, IOError, ValueError, termios.error):
        try:
            from termios import CEOF, CINTR
            (intr, eof) = (CINTR, CEOF)
        except ImportError:
            (intr, eof) = (3, 4)
    _INTR = _byte(intr)
    _EOF = _byte(eof)

def _setecho(fd, state):
    if False:
        for i in range(10):
            print('nop')
    errmsg = 'setecho() may not be called on this platform (it may still be possible to enable/disable echo when spawning the child process)'
    try:
        attr = termios.tcgetattr(fd)
    except termios.error as err:
        if err.args[0] == errno.EINVAL:
            raise IOError(err.args[0], '%s: %s.' % (err.args[1], errmsg))
        raise
    if state:
        attr[3] = attr[3] | termios.ECHO
    else:
        attr[3] = attr[3] & ~termios.ECHO
    try:
        termios.tcsetattr(fd, termios.TCSANOW, attr)
    except IOError as err:
        if err.args[0] == errno.EINVAL:
            raise IOError(err.args[0], '%s: %s.' % (err.args[1], errmsg))
        raise

def _setwinsize(fd, rows, cols):
    if False:
        for i in range(10):
            print('nop')
    TIOCSWINSZ = getattr(termios, 'TIOCSWINSZ', -2146929561)
    s = struct.pack('HHHH', rows, cols, 0, 0)
    fcntl.ioctl(fd, TIOCSWINSZ, s)

class PtyProcess(object):
    """This class represents a process running in a pseudoterminal.
    
    The main constructor is the :meth:`spawn` classmethod.
    """
    string_type = bytes
    if PY3:
        linesep = os.linesep.encode('ascii')
        crlf = '\r\n'.encode('ascii')

        @staticmethod
        def write_to_stdout(b):
            if False:
                i = 10
                return i + 15
            try:
                return sys.stdout.buffer.write(b)
            except AttributeError:
                return sys.stdout.write(b.decode('ascii', 'replace'))
    else:
        linesep = os.linesep
        crlf = '\r\n'
        write_to_stdout = sys.stdout.write
    encoding = None
    argv = None
    env = None
    launch_dir = None

    def __init__(self, pid, fd):
        if False:
            for i in range(10):
                print('nop')
        _make_eof_intr()
        self.pid = pid
        self.fd = fd
        readf = io.open(fd, 'rb', buffering=0)
        writef = io.open(fd, 'wb', buffering=0, closefd=False)
        self.fileobj = io.BufferedRWPair(readf, writef)
        self.terminated = False
        self.closed = False
        self.exitstatus = None
        self.signalstatus = None
        self.status = None
        self.flag_eof = False
        self.delayafterclose = 0.1
        self.delayafterterminate = 0.1

    @classmethod
    def spawn(cls, argv, cwd=None, env=None, echo=True, preexec_fn=None, dimensions=(24, 80), pass_fds=()):
        if False:
            while True:
                i = 10
        'Start the given command in a child process in a pseudo terminal.\n\n        This does all the fork/exec type of stuff for a pty, and returns an\n        instance of PtyProcess.\n\n        If preexec_fn is supplied, it will be called with no arguments in the\n        child process before exec-ing the specified command.\n        It may, for instance, set signal handlers to SIG_DFL or SIG_IGN.\n\n        Dimensions of the psuedoterminal used for the subprocess can be\n        specified as a tuple (rows, cols), or the default (24, 80) will be used.\n\n        By default, all file descriptors except 0, 1 and 2 are closed. This\n        behavior can be overridden with pass_fds, a list of file descriptors to\n        keep open between the parent and the child.\n        '
        if not isinstance(argv, (list, tuple)):
            raise TypeError('Expected a list or tuple for argv, got %r' % argv)
        argv = argv[:]
        command = argv[0]
        command_with_path = which(command)
        if command_with_path is None:
            raise FileNotFoundError('The command was not found or was not ' + 'executable: %s.' % command)
        command = command_with_path
        argv[0] = command
        (exec_err_pipe_read, exec_err_pipe_write) = os.pipe()
        if use_native_pty_fork:
            (pid, fd) = pty.fork()
        else:
            (pid, fd) = _fork_pty.fork_pty()
        if pid == CHILD:
            try:
                _setwinsize(STDIN_FILENO, *dimensions)
            except IOError as err:
                if err.args[0] not in (errno.EINVAL, errno.ENOTTY):
                    raise
            if not echo:
                try:
                    _setecho(STDIN_FILENO, False)
                except (IOError, termios.error) as err:
                    if err.args[0] not in (errno.EINVAL, errno.ENOTTY):
                        raise
            os.close(exec_err_pipe_read)
            fcntl.fcntl(exec_err_pipe_write, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
            max_fd = min(1048576, resource.getrlimit(resource.RLIMIT_NOFILE)[0])
            spass_fds = sorted(set(pass_fds) | {exec_err_pipe_write})
            for pair in zip([2] + spass_fds, spass_fds + [max_fd]):
                os.closerange(pair[0] + 1, pair[1])
            if cwd is not None:
                os.chdir(cwd)
            if preexec_fn is not None:
                try:
                    preexec_fn()
                except Exception as e:
                    ename = type(e).__name__
                    tosend = '{}:0:{}'.format(ename, str(e))
                    if PY3:
                        tosend = tosend.encode('utf-8')
                    os.write(exec_err_pipe_write, tosend)
                    os.close(exec_err_pipe_write)
                    os._exit(1)
            try:
                if env is None:
                    os.execv(command, argv)
                else:
                    os.execvpe(command, argv, env)
            except OSError as err:
                tosend = 'OSError:{}:{}'.format(err.errno, str(err))
                if PY3:
                    tosend = tosend.encode('utf-8')
                os.write(exec_err_pipe_write, tosend)
                os.close(exec_err_pipe_write)
                os._exit(os.EX_OSERR)
        inst = cls(pid, fd)
        inst.argv = argv
        if env is not None:
            inst.env = env
        if cwd is not None:
            inst.launch_dir = cwd
        os.close(exec_err_pipe_write)
        exec_err_data = os.read(exec_err_pipe_read, 4096)
        os.close(exec_err_pipe_read)
        if len(exec_err_data) != 0:
            try:
                (errclass, errno_s, errmsg) = exec_err_data.split(b':', 2)
                exctype = getattr(builtins, errclass.decode('ascii'), Exception)
                exception = exctype(errmsg.decode('utf-8', 'replace'))
                if exctype is OSError:
                    exception.errno = int(errno_s)
            except:
                raise Exception('Subprocess failed, got bad error data: %r' % exec_err_data)
            else:
                raise exception
        try:
            inst.setwinsize(*dimensions)
        except IOError as err:
            if err.args[0] not in (errno.EINVAL, errno.ENOTTY, errno.ENXIO):
                raise
        return inst

    def __repr__(self):
        if False:
            while True:
                i = 10
        clsname = type(self).__name__
        if self.argv is not None:
            args = [repr(self.argv)]
            if self.env is not None:
                args.append('env=%r' % self.env)
            if self.launch_dir is not None:
                args.append('cwd=%r' % self.launch_dir)
            return '{}.spawn({})'.format(clsname, ', '.join(args))
        else:
            return '{}(pid={}, fd={})'.format(clsname, self.pid, self.fd)

    @staticmethod
    def _coerce_send_string(s):
        if False:
            i = 10
            return i + 15
        if not isinstance(s, bytes):
            return s.encode('utf-8')
        return s

    @staticmethod
    def _coerce_read_string(s):
        if False:
            i = 10
            return i + 15
        return s

    def __del__(self):
        if False:
            while True:
                i = 10
        'This makes sure that no system resources are left open. Python only\n        garbage collects Python objects. OS file descriptors are not Python\n        objects, so they must be handled explicitly. If the child file\n        descriptor was opened outside of this class (passed to the constructor)\n        then this does not close it. '
        if not self.closed:
            try:
                self.close()
            except:
                pass

    def fileno(self):
        if False:
            while True:
                i = 10
        'This returns the file descriptor of the pty for the child.\n        '
        return self.fd

    def close(self, force=True):
        if False:
            i = 10
            return i + 15
        'This closes the connection with the child application. Note that\n        calling close() more than once is valid. This emulates standard Python\n        behavior with files. Set force to True if you want to make sure that\n        the child is terminated (SIGKILL is sent if the child ignores SIGHUP\n        and SIGINT). '
        if not self.closed:
            self.flush()
            self.fileobj.close()
            time.sleep(self.delayafterclose)
            if self.isalive():
                if not self.terminate(force):
                    raise PtyProcessError('Could not terminate the child.')
            self.fd = -1
            self.closed = True

    def flush(self):
        if False:
            print('Hello World!')
        'This does nothing. It is here to support the interface for a\n        File-like object. '
        pass

    def isatty(self):
        if False:
            print('Hello World!')
        'This returns True if the file descriptor is open and connected to a\n        tty(-like) device, else False.\n\n        On SVR4-style platforms implementing streams, such as SunOS and HP-UX,\n        the child pty may not appear as a terminal device.  This means\n        methods such as setecho(), setwinsize(), getwinsize() may raise an\n        IOError. '
        return os.isatty(self.fd)

    def waitnoecho(self, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        'This waits until the terminal ECHO flag is set False. This returns\n        True if the echo mode is off. This returns False if the ECHO flag was\n        not set False before the timeout. This can be used to detect when the\n        child is waiting for a password. Usually a child application will turn\n        off echo mode when it is waiting for the user to enter a password. For\n        example, instead of expecting the "password:" prompt you can wait for\n        the child to set ECHO off::\n\n            p = pexpect.spawn(\'ssh user@example.com\')\n            p.waitnoecho()\n            p.sendline(mypassword)\n\n        If timeout==None then this method to block until ECHO flag is False.\n        '
        if timeout is not None:
            end_time = time.time() + timeout
        while True:
            if not self.getecho():
                return True
            if timeout < 0 and timeout is not None:
                return False
            if timeout is not None:
                timeout = end_time - time.time()
            time.sleep(0.1)

    def getecho(self):
        if False:
            print('Hello World!')
        'This returns the terminal echo mode. This returns True if echo is\n        on or False if echo is off. Child applications that are expecting you\n        to enter a password often set ECHO False. See waitnoecho().\n\n        Not supported on platforms where ``isatty()`` returns False.  '
        try:
            attr = termios.tcgetattr(self.fd)
        except termios.error as err:
            errmsg = 'getecho() may not be called on this platform'
            if err.args[0] == errno.EINVAL:
                raise IOError(err.args[0], '%s: %s.' % (err.args[1], errmsg))
            raise
        self.echo = bool(attr[3] & termios.ECHO)
        return self.echo

    def setecho(self, state):
        if False:
            i = 10
            return i + 15
        "This sets the terminal echo mode on or off. Note that anything the\n        child sent before the echo will be lost, so you should be sure that\n        your input buffer is empty before you call setecho(). For example, the\n        following will work as expected::\n\n            p = pexpect.spawn('cat') # Echo is on by default.\n            p.sendline('1234') # We expect see this twice from the child...\n            p.expect(['1234']) # ... once from the tty echo...\n            p.expect(['1234']) # ... and again from cat itself.\n            p.setecho(False) # Turn off tty echo\n            p.sendline('abcd') # We will set this only once (echoed by cat).\n            p.sendline('wxyz') # We will set this only once (echoed by cat)\n            p.expect(['abcd'])\n            p.expect(['wxyz'])\n\n        The following WILL NOT WORK because the lines sent before the setecho\n        will be lost::\n\n            p = pexpect.spawn('cat')\n            p.sendline('1234')\n            p.setecho(False) # Turn off tty echo\n            p.sendline('abcd') # We will set this only once (echoed by cat).\n            p.sendline('wxyz') # We will set this only once (echoed by cat)\n            p.expect(['1234'])\n            p.expect(['1234'])\n            p.expect(['abcd'])\n            p.expect(['wxyz'])\n\n\n        Not supported on platforms where ``isatty()`` returns False.\n        "
        _setecho(self.fd, state)
        self.echo = state

    def read(self, size=1024):
        if False:
            i = 10
            return i + 15
        "Read and return at most ``size`` bytes from the pty.\n\n        Can block if there is nothing to read. Raises :exc:`EOFError` if the\n        terminal was closed.\n        \n        Unlike Pexpect's ``read_nonblocking`` method, this doesn't try to deal\n        with the vagaries of EOF on platforms that do strange things, like IRIX\n        or older Solaris systems. It handles the errno=EIO pattern used on\n        Linux, and the empty-string return used on BSD platforms and (seemingly)\n        on recent Solaris.\n        "
        try:
            s = self.fileobj.read1(size)
        except (OSError, IOError) as err:
            if err.args[0] == errno.EIO:
                self.flag_eof = True
                raise EOFError('End Of File (EOF). Exception style platform.')
            raise
        if s == b'':
            self.flag_eof = True
            raise EOFError('End Of File (EOF). Empty string style platform.')
        return s

    def readline(self):
        if False:
            return 10
        'Read one line from the pseudoterminal, and return it as unicode.\n\n        Can block if there is nothing to read. Raises :exc:`EOFError` if the\n        terminal was closed.\n        '
        try:
            s = self.fileobj.readline()
        except (OSError, IOError) as err:
            if err.args[0] == errno.EIO:
                self.flag_eof = True
                raise EOFError('End Of File (EOF). Exception style platform.')
            raise
        if s == b'':
            self.flag_eof = True
            raise EOFError('End Of File (EOF). Empty string style platform.')
        return s

    def _writeb(self, b, flush=True):
        if False:
            return 10
        n = self.fileobj.write(b)
        if flush:
            self.fileobj.flush()
        return n

    def write(self, s, flush=True):
        if False:
            for i in range(10):
                print('nop')
        'Write bytes to the pseudoterminal.\n        \n        Returns the number of bytes written.\n        '
        return self._writeb(s, flush=flush)

    def sendcontrol(self, char):
        if False:
            return 10
        "Helper method that wraps send() with mnemonic access for sending control\n        character to the child (such as Ctrl-C or Ctrl-D).  For example, to send\n        Ctrl-G (ASCII 7, bell, '\x07')::\n\n            child.sendcontrol('g')\n\n        See also, sendintr() and sendeof().\n        "
        char = char.lower()
        a = ord(char)
        if 97 <= a <= 122:
            a = a - ord('a') + 1
            byte = _byte(a)
            return (self._writeb(byte), byte)
        d = {'@': 0, '`': 0, '[': 27, '{': 27, '\\': 28, '|': 28, ']': 29, '}': 29, '^': 30, '~': 30, '_': 31, '?': 127}
        if char not in d:
            return (0, b'')
        byte = _byte(d[char])
        return (self._writeb(byte), byte)

    def sendeof(self):
        if False:
            print('Hello World!')
        'This sends an EOF to the child. This sends a character which causes\n        the pending parent output buffer to be sent to the waiting child\n        program without waiting for end-of-line. If it is the first character\n        of the line, the read() in the user program returns 0, which signifies\n        end-of-file. This means to work as expected a sendeof() has to be\n        called at the beginning of a line. This method does not send a newline.\n        It is the responsibility of the caller to ensure the eof is sent at the\n        beginning of a line. '
        return (self._writeb(_EOF), _EOF)

    def sendintr(self):
        if False:
            while True:
                i = 10
        'This sends a SIGINT to the child. It does not require\n        the SIGINT to be the first character on a line. '
        return (self._writeb(_INTR), _INTR)

    def eof(self):
        if False:
            return 10
        'This returns True if the EOF exception was ever raised.\n        '
        return self.flag_eof

    def terminate(self, force=False):
        if False:
            for i in range(10):
                print('nop')
        'This forces a child process to terminate. It starts nicely with\n        SIGHUP and SIGINT. If "force" is True then moves onto SIGKILL. This\n        returns True if the child was terminated. This returns False if the\n        child could not be terminated. '
        if not self.isalive():
            return True
        try:
            self.kill(signal.SIGHUP)
            time.sleep(self.delayafterterminate)
            if not self.isalive():
                return True
            self.kill(signal.SIGCONT)
            time.sleep(self.delayafterterminate)
            if not self.isalive():
                return True
            self.kill(signal.SIGINT)
            time.sleep(self.delayafterterminate)
            if not self.isalive():
                return True
            if force:
                self.kill(signal.SIGKILL)
                time.sleep(self.delayafterterminate)
                if not self.isalive():
                    return True
                else:
                    return False
            return False
        except OSError:
            time.sleep(self.delayafterterminate)
            if not self.isalive():
                return True
            else:
                return False

    def wait(self):
        if False:
            while True:
                i = 10
        'This waits until the child exits. This is a blocking call. This will\n        not read any data from the child, so this will block forever if the\n        child has unread output and has terminated. In other words, the child\n        may have printed output then called exit(), but, the child is\n        technically still alive until its output is read by the parent. '
        if self.isalive():
            (pid, status) = os.waitpid(self.pid, 0)
        else:
            return self.exitstatus
        self.exitstatus = os.WEXITSTATUS(status)
        if os.WIFEXITED(status):
            self.status = status
            self.exitstatus = os.WEXITSTATUS(status)
            self.signalstatus = None
            self.terminated = True
        elif os.WIFSIGNALED(status):
            self.status = status
            self.exitstatus = None
            self.signalstatus = os.WTERMSIG(status)
            self.terminated = True
        elif os.WIFSTOPPED(status):
            raise PtyProcessError('Called wait() on a stopped child ' + 'process. This is not supported. Is some other ' + 'process attempting job control with our child pid?')
        return self.exitstatus

    def isalive(self):
        if False:
            return 10
        'This tests if the child process is running or not. This is\n        non-blocking. If the child was terminated then this will read the\n        exitstatus or signalstatus of the child. This returns True if the child\n        process appears to be running or False if not. It can take literally\n        SECONDS for Solaris to return the right status. '
        if self.terminated:
            return False
        if self.flag_eof:
            waitpid_options = 0
        else:
            waitpid_options = os.WNOHANG
        try:
            (pid, status) = os.waitpid(self.pid, waitpid_options)
        except OSError as e:
            if e.errno == errno.ECHILD:
                raise PtyProcessError('isalive() encountered condition ' + 'where "terminated" is 0, but there was no child ' + 'process. Did someone else call waitpid() ' + 'on our process?')
            else:
                raise
        if pid == 0:
            try:
                (pid, status) = os.waitpid(self.pid, waitpid_options)
            except OSError as e:
                if e.errno == errno.ECHILD:
                    raise PtyProcessError('isalive() encountered condition ' + 'that should never happen. There was no child ' + 'process. Did someone else call waitpid() ' + 'on our process?')
                else:
                    raise
            if pid == 0:
                return True
        if pid == 0:
            return True
        if os.WIFEXITED(status):
            self.status = status
            self.exitstatus = os.WEXITSTATUS(status)
            self.signalstatus = None
            self.terminated = True
        elif os.WIFSIGNALED(status):
            self.status = status
            self.exitstatus = None
            self.signalstatus = os.WTERMSIG(status)
            self.terminated = True
        elif os.WIFSTOPPED(status):
            raise PtyProcessError('isalive() encountered condition ' + 'where child process is stopped. This is not ' + 'supported. Is some other process attempting ' + 'job control with our child pid?')
        return False

    def kill(self, sig):
        if False:
            for i in range(10):
                print('nop')
        'Send the given signal to the child application.\n\n        In keeping with UNIX tradition it has a misleading name. It does not\n        necessarily kill the child unless you send the right signal. See the\n        :mod:`signal` module for constants representing signal numbers.\n        '
        if self.isalive():
            os.kill(self.pid, sig)

    def getwinsize(self):
        if False:
            i = 10
            return i + 15
        'Return the window size of the pseudoterminal as a tuple (rows, cols).\n        '
        TIOCGWINSZ = getattr(termios, 'TIOCGWINSZ', 1074295912)
        s = struct.pack('HHHH', 0, 0, 0, 0)
        x = fcntl.ioctl(self.fd, TIOCGWINSZ, s)
        return struct.unpack('HHHH', x)[0:2]

    def setwinsize(self, rows, cols):
        if False:
            return 10
        'Set the terminal window size of the child tty.\n\n        This will cause a SIGWINCH signal to be sent to the child. This does not\n        change the physical window size. It changes the size reported to\n        TTY-aware applications like vi or curses -- applications that respond to\n        the SIGWINCH signal.\n        '
        return _setwinsize(self.fd, rows, cols)

class PtyProcessUnicode(PtyProcess):
    """Unicode wrapper around a process running in a pseudoterminal.

    This class exposes a similar interface to :class:`PtyProcess`, but its read
    methods return unicode, and its :meth:`write` accepts unicode.
    """
    if PY3:
        string_type = str
    else:
        string_type = unicode

    def __init__(self, pid, fd, encoding='utf-8', codec_errors='strict'):
        if False:
            for i in range(10):
                print('nop')
        super(PtyProcessUnicode, self).__init__(pid, fd)
        self.encoding = encoding
        self.codec_errors = codec_errors
        self.decoder = codecs.getincrementaldecoder(encoding)(errors=codec_errors)

    def read(self, size=1024):
        if False:
            for i in range(10):
                print('nop')
        'Read at most ``size`` bytes from the pty, return them as unicode.\n\n        Can block if there is nothing to read. Raises :exc:`EOFError` if the\n        terminal was closed.\n\n        The size argument still refers to bytes, not unicode code points.\n        '
        b = super(PtyProcessUnicode, self).read(size)
        return self.decoder.decode(b, final=False)

    def readline(self):
        if False:
            print('Hello World!')
        'Read one line from the pseudoterminal, and return it as unicode.\n\n        Can block if there is nothing to read. Raises :exc:`EOFError` if the\n        terminal was closed.\n        '
        b = super(PtyProcessUnicode, self).readline()
        return self.decoder.decode(b, final=False)

    def write(self, s):
        if False:
            return 10
        'Write the unicode string ``s`` to the pseudoterminal.\n\n        Returns the number of bytes written.\n        '
        b = s.encode(self.encoding)
        return super(PtyProcessUnicode, self).write(b)