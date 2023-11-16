import os
import sys
import time
import pty
import tty
import errno
import signal
from contextlib import contextmanager
import pipenv.vendor.ptyprocess as ptyprocess
from pipenv.vendor.ptyprocess.ptyprocess import use_native_pty_fork
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .spawnbase import SpawnBase
from .utils import which, split_command_line, select_ignore_interrupts, poll_ignore_interrupts

@contextmanager
def _wrap_ptyprocess_err():
    if False:
        print('Hello World!')
    'Turn ptyprocess errors into our own ExceptionPexpect errors'
    try:
        yield
    except ptyprocess.PtyProcessError as e:
        raise ExceptionPexpect(*e.args)
PY3 = sys.version_info[0] >= 3

class spawn(SpawnBase):
    """This is the main class interface for Pexpect. Use this class to start
    and control child applications. """
    use_native_pty_fork = use_native_pty_fork

    def __init__(self, command, args=[], timeout=30, maxread=2000, searchwindowsize=None, logfile=None, cwd=None, env=None, ignore_sighup=False, echo=True, preexec_fn=None, encoding=None, codec_errors='strict', dimensions=None, use_poll=False):
        if False:
            while True:
                i = 10
        'This is the constructor. The command parameter may be a string that\n        includes a command and any arguments to the command. For example::\n\n            child = pexpect.spawn(\'/usr/bin/ftp\')\n            child = pexpect.spawn(\'/usr/bin/ssh user@example.com\')\n            child = pexpect.spawn(\'ls -latr /tmp\')\n\n        You may also construct it with a list of arguments like so::\n\n            child = pexpect.spawn(\'/usr/bin/ftp\', [])\n            child = pexpect.spawn(\'/usr/bin/ssh\', [\'user@example.com\'])\n            child = pexpect.spawn(\'ls\', [\'-latr\', \'/tmp\'])\n\n        After this the child application will be created and will be ready to\n        talk to. For normal use, see expect() and send() and sendline().\n\n        Remember that Pexpect does NOT interpret shell meta characters such as\n        redirect, pipe, or wild cards (``>``, ``|``, or ``*``). This is a\n        common mistake.  If you want to run a command and pipe it through\n        another command then you must also start a shell. For example::\n\n            child = pexpect.spawn(\'/bin/bash -c "ls -l | grep LOG > logs.txt"\')\n            child.expect(pexpect.EOF)\n\n        The second form of spawn (where you pass a list of arguments) is useful\n        in situations where you wish to spawn a command and pass it its own\n        argument list. This can make syntax more clear. For example, the\n        following is equivalent to the previous example::\n\n            shell_cmd = \'ls -l | grep LOG > logs.txt\'\n            child = pexpect.spawn(\'/bin/bash\', [\'-c\', shell_cmd])\n            child.expect(pexpect.EOF)\n\n        The maxread attribute sets the read buffer size. This is maximum number\n        of bytes that Pexpect will try to read from a TTY at one time. Setting\n        the maxread size to 1 will turn off buffering. Setting the maxread\n        value higher may help performance in cases where large amounts of\n        output are read back from the child. This feature is useful in\n        conjunction with searchwindowsize.\n\n        When the keyword argument *searchwindowsize* is None (default), the\n        full buffer is searched at each iteration of receiving incoming data.\n        The default number of bytes scanned at each iteration is very large\n        and may be reduced to collaterally reduce search cost.  After\n        :meth:`~.expect` returns, the full buffer attribute remains up to\n        size *maxread* irrespective of *searchwindowsize* value.\n\n        When the keyword argument ``timeout`` is specified as a number,\n        (default: *30*), then :class:`TIMEOUT` will be raised after the value\n        specified has elapsed, in seconds, for any of the :meth:`~.expect`\n        family of method calls.  When None, TIMEOUT will not be raised, and\n        :meth:`~.expect` may block indefinitely until match.\n\n\n        The logfile member turns on or off logging. All input and output will\n        be copied to the given file object. Set logfile to None to stop\n        logging. This is the default. Set logfile to sys.stdout to echo\n        everything to standard output. The logfile is flushed after each write.\n\n        Example log input and output to a file::\n\n            child = pexpect.spawn(\'some_command\')\n            fout = open(\'mylog.txt\',\'wb\')\n            child.logfile = fout\n\n        Example log to stdout::\n\n            # In Python 2:\n            child = pexpect.spawn(\'some_command\')\n            child.logfile = sys.stdout\n\n            # In Python 3, we\'ll use the ``encoding`` argument to decode data\n            # from the subprocess and handle it as unicode:\n            child = pexpect.spawn(\'some_command\', encoding=\'utf-8\')\n            child.logfile = sys.stdout\n\n        The logfile_read and logfile_send members can be used to separately log\n        the input from the child and output sent to the child. Sometimes you\n        don\'t want to see everything you write to the child. You only want to\n        log what the child sends back. For example::\n\n            child = pexpect.spawn(\'some_command\')\n            child.logfile_read = sys.stdout\n\n        You will need to pass an encoding to spawn in the above code if you are\n        using Python 3.\n\n        To separately log output sent to the child use logfile_send::\n\n            child.logfile_send = fout\n\n        If ``ignore_sighup`` is True, the child process will ignore SIGHUP\n        signals. The default is False from Pexpect 4.0, meaning that SIGHUP\n        will be handled normally by the child.\n\n        The delaybeforesend helps overcome a weird behavior that many users\n        were experiencing. The typical problem was that a user would expect() a\n        "Password:" prompt and then immediately call sendline() to send the\n        password. The user would then see that their password was echoed back\n        to them. Passwords don\'t normally echo. The problem is caused by the\n        fact that most applications print out the "Password" prompt and then\n        turn off stdin echo, but if you send your password before the\n        application turned off echo, then you get your password echoed.\n        Normally this wouldn\'t be a problem when interacting with a human at a\n        real keyboard. If you introduce a slight delay just before writing then\n        this seems to clear up the problem. This was such a common problem for\n        many users that I decided that the default pexpect behavior should be\n        to sleep just before writing to the child application. 1/20th of a\n        second (50 ms) seems to be enough to clear up the problem. You can set\n        delaybeforesend to None to return to the old behavior.\n\n        Note that spawn is clever about finding commands on your path.\n        It uses the same logic that "which" uses to find executables.\n\n        If you wish to get the exit status of the child you must call the\n        close() method. The exit or signal status of the child will be stored\n        in self.exitstatus or self.signalstatus. If the child exited normally\n        then exitstatus will store the exit return code and signalstatus will\n        be None. If the child was terminated abnormally with a signal then\n        signalstatus will store the signal value and exitstatus will be None::\n\n            child = pexpect.spawn(\'some_command\')\n            child.close()\n            print(child.exitstatus, child.signalstatus)\n\n        If you need more detail you can also read the self.status member which\n        stores the status returned by os.waitpid. You can interpret this using\n        os.WIFEXITED/os.WEXITSTATUS or os.WIFSIGNALED/os.TERMSIG.\n\n        The echo attribute may be set to False to disable echoing of input.\n        As a pseudo-terminal, all input echoed by the "keyboard" (send()\n        or sendline()) will be repeated to output.  For many cases, it is\n        not desirable to have echo enabled, and it may be later disabled\n        using setecho(False) followed by waitnoecho().  However, for some\n        platforms such as Solaris, this is not possible, and should be\n        disabled immediately on spawn.\n\n        If preexec_fn is given, it will be called in the child process before\n        launching the given command. This is useful to e.g. reset inherited\n        signal handlers.\n\n        The dimensions attribute specifies the size of the pseudo-terminal as\n        seen by the subprocess, and is specified as a two-entry tuple (rows,\n        columns). If this is unspecified, the defaults in ptyprocess will apply.\n\n        The use_poll attribute enables using select.poll() over select.select()\n        for socket handling. This is handy if your system could have > 1024 fds\n        '
        super(spawn, self).__init__(timeout=timeout, maxread=maxread, searchwindowsize=searchwindowsize, logfile=logfile, encoding=encoding, codec_errors=codec_errors)
        self.STDIN_FILENO = pty.STDIN_FILENO
        self.STDOUT_FILENO = pty.STDOUT_FILENO
        self.STDERR_FILENO = pty.STDERR_FILENO
        self.str_last_chars = 100
        self.cwd = cwd
        self.env = env
        self.echo = echo
        self.ignore_sighup = ignore_sighup
        self.__irix_hack = sys.platform.lower().startswith('irix')
        if command is None:
            self.command = None
            self.args = None
            self.name = '<pexpect factory incomplete>'
        else:
            self._spawn(command, args, preexec_fn, dimensions)
        self.use_poll = use_poll

    def __str__(self):
        if False:
            while True:
                i = 10
        'This returns a human-readable string that represents the state of\n        the object. '
        s = []
        s.append(repr(self))
        s.append('command: ' + str(self.command))
        s.append('args: %r' % (self.args,))
        s.append('buffer (last %s chars): %r' % (self.str_last_chars, self.buffer[-self.str_last_chars:]))
        s.append('before (last %s chars): %r' % (self.str_last_chars, self.before[-self.str_last_chars:] if self.before else ''))
        s.append('after: %r' % (self.after,))
        s.append('match: %r' % (self.match,))
        s.append('match_index: ' + str(self.match_index))
        s.append('exitstatus: ' + str(self.exitstatus))
        if hasattr(self, 'ptyproc'):
            s.append('flag_eof: ' + str(self.flag_eof))
        s.append('pid: ' + str(self.pid))
        s.append('child_fd: ' + str(self.child_fd))
        s.append('closed: ' + str(self.closed))
        s.append('timeout: ' + str(self.timeout))
        s.append('delimiter: ' + str(self.delimiter))
        s.append('logfile: ' + str(self.logfile))
        s.append('logfile_read: ' + str(self.logfile_read))
        s.append('logfile_send: ' + str(self.logfile_send))
        s.append('maxread: ' + str(self.maxread))
        s.append('ignorecase: ' + str(self.ignorecase))
        s.append('searchwindowsize: ' + str(self.searchwindowsize))
        s.append('delaybeforesend: ' + str(self.delaybeforesend))
        s.append('delayafterclose: ' + str(self.delayafterclose))
        s.append('delayafterterminate: ' + str(self.delayafterterminate))
        return '\n'.join(s)

    def _spawn(self, command, args=[], preexec_fn=None, dimensions=None):
        if False:
            print('Hello World!')
        'This starts the given command in a child process. This does all the\n        fork/exec type of stuff for a pty. This is called by __init__. If args\n        is empty then command will be parsed (split on spaces) and args will be\n        set to parsed arguments. '
        if isinstance(command, type(0)):
            raise ExceptionPexpect('Command is an int type. ' + 'If this is a file descriptor then maybe you want to ' + 'use fdpexpect.fdspawn which takes an existing ' + 'file descriptor instead of a command string.')
        if not isinstance(args, type([])):
            raise TypeError('The argument, args, must be a list.')
        if args == []:
            self.args = split_command_line(command)
            self.command = self.args[0]
        else:
            self.args = args[:]
            self.args.insert(0, command)
            self.command = command
        command_with_path = which(self.command, env=self.env)
        if command_with_path is None:
            raise ExceptionPexpect('The command was not found or was not ' + 'executable: %s.' % self.command)
        self.command = command_with_path
        self.args[0] = self.command
        self.name = '<' + ' '.join(self.args) + '>'
        assert self.pid is None, 'The pid member must be None.'
        assert self.command is not None, 'The command member must not be None.'
        kwargs = {'echo': self.echo, 'preexec_fn': preexec_fn}
        if self.ignore_sighup:

            def preexec_wrapper():
                if False:
                    i = 10
                    return i + 15
                'Set SIGHUP to be ignored, then call the real preexec_fn'
                signal.signal(signal.SIGHUP, signal.SIG_IGN)
                if preexec_fn is not None:
                    preexec_fn()
            kwargs['preexec_fn'] = preexec_wrapper
        if dimensions is not None:
            kwargs['dimensions'] = dimensions
        if self.encoding is not None:
            self.args = [a if isinstance(a, bytes) else a.encode(self.encoding) for a in self.args]
        self.ptyproc = self._spawnpty(self.args, env=self.env, cwd=self.cwd, **kwargs)
        self.pid = self.ptyproc.pid
        self.child_fd = self.ptyproc.fd
        self.terminated = False
        self.closed = False

    def _spawnpty(self, args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Spawn a pty and return an instance of PtyProcess.'
        return ptyprocess.PtyProcess.spawn(args, **kwargs)

    def close(self, force=True):
        if False:
            while True:
                i = 10
        'This closes the connection with the child application. Note that\n        calling close() more than once is valid. This emulates standard Python\n        behavior with files. Set force to True if you want to make sure that\n        the child is terminated (SIGKILL is sent if the child ignores SIGHUP\n        and SIGINT). '
        self.flush()
        with _wrap_ptyprocess_err():
            self.ptyproc.close(force=force)
        self.isalive()
        self.child_fd = -1
        self.closed = True

    def isatty(self):
        if False:
            for i in range(10):
                print('nop')
        'This returns True if the file descriptor is open and connected to a\n        tty(-like) device, else False.\n\n        On SVR4-style platforms implementing streams, such as SunOS and HP-UX,\n        the child pty may not appear as a terminal device.  This means\n        methods such as setecho(), setwinsize(), getwinsize() may raise an\n        IOError. '
        return os.isatty(self.child_fd)

    def waitnoecho(self, timeout=-1):
        if False:
            return 10
        'This waits until the terminal ECHO flag is set False. This returns\n        True if the echo mode is off. This returns False if the ECHO flag was\n        not set False before the timeout. This can be used to detect when the\n        child is waiting for a password. Usually a child application will turn\n        off echo mode when it is waiting for the user to enter a password. For\n        example, instead of expecting the "password:" prompt you can wait for\n        the child to set ECHO off::\n\n            p = pexpect.spawn(\'ssh user@example.com\')\n            p.waitnoecho()\n            p.sendline(mypassword)\n\n        If timeout==-1 then this method will use the value in self.timeout.\n        If timeout==None then this method to block until ECHO flag is False.\n        '
        if timeout == -1:
            timeout = self.timeout
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
            return 10
        'This returns the terminal echo mode. This returns True if echo is\n        on or False if echo is off. Child applications that are expecting you\n        to enter a password often set ECHO False. See waitnoecho().\n\n        Not supported on platforms where ``isatty()`` returns False.  '
        return self.ptyproc.getecho()

    def setecho(self, state):
        if False:
            for i in range(10):
                print('nop')
        "This sets the terminal echo mode on or off. Note that anything the\n        child sent before the echo will be lost, so you should be sure that\n        your input buffer is empty before you call setecho(). For example, the\n        following will work as expected::\n\n            p = pexpect.spawn('cat') # Echo is on by default.\n            p.sendline('1234') # We expect see this twice from the child...\n            p.expect(['1234']) # ... once from the tty echo...\n            p.expect(['1234']) # ... and again from cat itself.\n            p.setecho(False) # Turn off tty echo\n            p.sendline('abcd') # We will set this only once (echoed by cat).\n            p.sendline('wxyz') # We will set this only once (echoed by cat)\n            p.expect(['abcd'])\n            p.expect(['wxyz'])\n\n        The following WILL NOT WORK because the lines sent before the setecho\n        will be lost::\n\n            p = pexpect.spawn('cat')\n            p.sendline('1234')\n            p.setecho(False) # Turn off tty echo\n            p.sendline('abcd') # We will set this only once (echoed by cat).\n            p.sendline('wxyz') # We will set this only once (echoed by cat)\n            p.expect(['1234'])\n            p.expect(['1234'])\n            p.expect(['abcd'])\n            p.expect(['wxyz'])\n\n\n        Not supported on platforms where ``isatty()`` returns False.\n        "
        return self.ptyproc.setecho(state)

    def read_nonblocking(self, size=1, timeout=-1):
        if False:
            while True:
                i = 10
        "This reads at most size characters from the child application. It\n        includes a timeout. If the read does not complete within the timeout\n        period then a TIMEOUT exception is raised. If the end of file is read\n        then an EOF exception will be raised.  If a logfile is specified, a\n        copy is written to that log.\n\n        If timeout is None then the read may block indefinitely.\n        If timeout is -1 then the self.timeout value is used. If timeout is 0\n        then the child is polled and if there is no data immediately ready\n        then this will raise a TIMEOUT exception.\n\n        The timeout refers only to the amount of time to read at least one\n        character. This is not affected by the 'size' parameter, so if you call\n        read_nonblocking(size=100, timeout=30) and only one character is\n        available right away then one character will be returned immediately.\n        It will not wait for 30 seconds for another 99 characters to come in.\n\n        On the other hand, if there are bytes available to read immediately,\n        all those bytes will be read (up to the buffer size). So, if the\n        buffer size is 1 megabyte and there is 1 megabyte of data available\n        to read, the buffer will be filled, regardless of timeout.\n\n        This is a wrapper around os.read(). It uses select.select() or\n        select.poll() to implement the timeout. "
        if self.closed:
            raise ValueError('I/O operation on closed file.')
        if self.use_poll:

            def select(timeout):
                if False:
                    for i in range(10):
                        print('nop')
                return poll_ignore_interrupts([self.child_fd], timeout)
        else:

            def select(timeout):
                if False:
                    i = 10
                    return i + 15
                return select_ignore_interrupts([self.child_fd], [], [], timeout)[0]
        if select(0):
            try:
                incoming = super(spawn, self).read_nonblocking(size)
            except EOF:
                self.isalive()
                raise
            while len(incoming) < size and select(0):
                try:
                    incoming += super(spawn, self).read_nonblocking(size - len(incoming))
                except EOF:
                    self.isalive()
                    return incoming
            return incoming
        if timeout == -1:
            timeout = self.timeout
        if not self.isalive():
            if select(0):
                return super(spawn, self).read_nonblocking(size)
            self.flag_eof = True
            raise EOF('End Of File (EOF). Braindead platform.')
        elif self.__irix_hack:
            if timeout is not None and timeout < 2:
                timeout = 2
        if timeout != 0 and select(timeout):
            return super(spawn, self).read_nonblocking(size)
        if not self.isalive():
            self.flag_eof = True
            raise EOF('End of File (EOF). Very slow platform.')
        else:
            raise TIMEOUT('Timeout exceeded.')

    def write(self, s):
        if False:
            print('Hello World!')
        'This is similar to send() except that there is no return value.\n        '
        self.send(s)

    def writelines(self, sequence):
        if False:
            i = 10
            return i + 15
        'This calls write() for each element in the sequence. The sequence\n        can be any iterable object producing strings, typically a list of\n        strings. This does not add line separators. There is no return value.\n        '
        for s in sequence:
            self.write(s)

    def send(self, s):
        if False:
            return 10
        "Sends string ``s`` to the child process, returning the number of\n        bytes written. If a logfile is specified, a copy is written to that\n        log.\n\n        The default terminal input mode is canonical processing unless set\n        otherwise by the child process. This allows backspace and other line\n        processing to be performed prior to transmitting to the receiving\n        program. As this is buffered, there is a limited size of such buffer.\n\n        On Linux systems, this is 4096 (defined by N_TTY_BUF_SIZE). All\n        other systems honor the POSIX.1 definition PC_MAX_CANON -- 1024\n        on OSX, 256 on OpenSolaris, and 1920 on FreeBSD.\n\n        This value may be discovered using fpathconf(3)::\n\n            >>> from os import fpathconf\n            >>> print(fpathconf(0, 'PC_MAX_CANON'))\n            256\n\n        On such a system, only 256 bytes may be received per line. Any\n        subsequent bytes received will be discarded. BEL (``'\x07'``) is then\n        sent to output if IMAXBEL (termios.h) is set by the tty driver.\n        This is usually enabled by default.  Linux does not honor this as\n        an option -- it behaves as though it is always set on.\n\n        Canonical input processing may be disabled altogether by executing\n        a shell, then stty(1), before executing the final program::\n\n            >>> bash = pexpect.spawn('/bin/bash', echo=False)\n            >>> bash.sendline('stty -icanon')\n            >>> bash.sendline('base64')\n            >>> bash.sendline('x' * 5000)\n        "
        if self.delaybeforesend is not None:
            time.sleep(self.delaybeforesend)
        s = self._coerce_send_string(s)
        self._log(s, 'send')
        b = self._encoder.encode(s, final=False)
        return os.write(self.child_fd, b)

    def sendline(self, s=''):
        if False:
            print('Hello World!')
        'Wraps send(), sending string ``s`` to child process, with\n        ``os.linesep`` automatically appended. Returns number of bytes\n        written.  Only a limited number of bytes may be sent for each\n        line in the default terminal mode, see docstring of :meth:`send`.\n        '
        s = self._coerce_send_string(s)
        return self.send(s + self.linesep)

    def _log_control(self, s):
        if False:
            return 10
        'Write control characters to the appropriate log files'
        if self.encoding is not None:
            s = s.decode(self.encoding, 'replace')
        self._log(s, 'send')

    def sendcontrol(self, char):
        if False:
            return 10
        "Helper method that wraps send() with mnemonic access for sending control\n        character to the child (such as Ctrl-C or Ctrl-D).  For example, to send\n        Ctrl-G (ASCII 7, bell, '\x07')::\n\n            child.sendcontrol('g')\n\n        See also, sendintr() and sendeof().\n        "
        (n, byte) = self.ptyproc.sendcontrol(char)
        self._log_control(byte)
        return n

    def sendeof(self):
        if False:
            print('Hello World!')
        'This sends an EOF to the child. This sends a character which causes\n        the pending parent output buffer to be sent to the waiting child\n        program without waiting for end-of-line. If it is the first character\n        of the line, the read() in the user program returns 0, which signifies\n        end-of-file. This means to work as expected a sendeof() has to be\n        called at the beginning of a line. This method does not send a newline.\n        It is the responsibility of the caller to ensure the eof is sent at the\n        beginning of a line. '
        (n, byte) = self.ptyproc.sendeof()
        self._log_control(byte)

    def sendintr(self):
        if False:
            print('Hello World!')
        'This sends a SIGINT to the child. It does not require\n        the SIGINT to be the first character on a line. '
        (n, byte) = self.ptyproc.sendintr()
        self._log_control(byte)

    @property
    def flag_eof(self):
        if False:
            return 10
        return self.ptyproc.flag_eof

    @flag_eof.setter
    def flag_eof(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.ptyproc.flag_eof = value

    def eof(self):
        if False:
            return 10
        'This returns True if the EOF exception was ever raised.\n        '
        return self.flag_eof

    def terminate(self, force=False):
        if False:
            return 10
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
            print('Hello World!')
        'This waits until the child exits. This is a blocking call. This will\n        not read any data from the child, so this will block forever if the\n        child has unread output and has terminated. In other words, the child\n        may have printed output then called exit(), but, the child is\n        technically still alive until its output is read by the parent.\n\n        This method is non-blocking if :meth:`wait` has already been called\n        previously or :meth:`isalive` method returns False.  It simply returns\n        the previously determined exit status.\n        '
        ptyproc = self.ptyproc
        with _wrap_ptyprocess_err():
            exitstatus = ptyproc.wait()
        self.status = ptyproc.status
        self.exitstatus = ptyproc.exitstatus
        self.signalstatus = ptyproc.signalstatus
        self.terminated = True
        return exitstatus

    def isalive(self):
        if False:
            for i in range(10):
                print('nop')
        'This tests if the child process is running or not. This is\n        non-blocking. If the child was terminated then this will read the\n        exitstatus or signalstatus of the child. This returns True if the child\n        process appears to be running or False if not. It can take literally\n        SECONDS for Solaris to return the right status. '
        ptyproc = self.ptyproc
        with _wrap_ptyprocess_err():
            alive = ptyproc.isalive()
        if not alive:
            self.status = ptyproc.status
            self.exitstatus = ptyproc.exitstatus
            self.signalstatus = ptyproc.signalstatus
            self.terminated = True
        return alive

    def kill(self, sig):
        if False:
            return 10
        'This sends the given signal to the child application. In keeping\n        with UNIX tradition it has a misleading name. It does not necessarily\n        kill the child unless you send the right signal. '
        if self.isalive():
            os.kill(self.pid, sig)

    def getwinsize(self):
        if False:
            i = 10
            return i + 15
        'This returns the terminal window size of the child tty. The return\n        value is a tuple of (rows, cols). '
        return self.ptyproc.getwinsize()

    def setwinsize(self, rows, cols):
        if False:
            for i in range(10):
                print('nop')
        'This sets the terminal window size of the child tty. This will cause\n        a SIGWINCH signal to be sent to the child. This does not change the\n        physical window size. It changes the size reported to TTY-aware\n        applications like vi or curses -- applications that respond to the\n        SIGWINCH signal. '
        return self.ptyproc.setwinsize(rows, cols)

    def interact(self, escape_character=chr(29), input_filter=None, output_filter=None):
        if False:
            return 10
        'This gives control of the child process to the interactive user (the\n        human at the keyboard). Keystrokes are sent to the child process, and\n        the stdout and stderr output of the child process is printed. This\n        simply echos the child stdout and child stderr to the real stdout and\n        it echos the real stdin to the child stdin. When the user types the\n        escape_character this method will return None. The escape_character\n        will not be transmitted.  The default for escape_character is\n        entered as ``Ctrl - ]``, the very same as BSD telnet. To prevent\n        escaping, escape_character may be set to None.\n\n        If a logfile is specified, then the data sent and received from the\n        child process in interact mode is duplicated to the given log.\n\n        You may pass in optional input and output filter functions. These\n        functions should take bytes array and return bytes array too. Even\n        with ``encoding=\'utf-8\'`` support, meth:`interact` will always pass\n        input_filter and output_filter bytes. You may need to wrap your\n        function to decode and encode back to UTF-8.\n\n        The output_filter will be passed all the output from the child process.\n        The input_filter will be passed all the keyboard input from the user.\n        The input_filter is run BEFORE the check for the escape_character.\n\n        Note that if you change the window size of the parent the SIGWINCH\n        signal will not be passed through to the child. If you want the child\n        window size to change when the parent\'s window size changes then do\n        something like the following example::\n\n            import pipenv.vendor.pexpect as pexpect, struct, fcntl, termios, signal, sys\n            def sigwinch_passthrough (sig, data):\n                s = struct.pack("HHHH", 0, 0, 0, 0)\n                a = struct.unpack(\'hhhh\', fcntl.ioctl(sys.stdout.fileno(),\n                    termios.TIOCGWINSZ , s))\n                if not p.closed:\n                    p.setwinsize(a[0],a[1])\n\n            # Note this \'p\' is global and used in sigwinch_passthrough.\n            p = pexpect.spawn(\'/bin/bash\')\n            signal.signal(signal.SIGWINCH, sigwinch_passthrough)\n            p.interact()\n        '
        self.write_to_stdout(self.buffer)
        self.stdout.flush()
        self._buffer = self.buffer_type()
        mode = tty.tcgetattr(self.STDIN_FILENO)
        tty.setraw(self.STDIN_FILENO)
        if escape_character is not None and PY3:
            escape_character = escape_character.encode('latin-1')
        try:
            self.__interact_copy(escape_character, input_filter, output_filter)
        finally:
            tty.tcsetattr(self.STDIN_FILENO, tty.TCSAFLUSH, mode)

    def __interact_writen(self, fd, data):
        if False:
            print('Hello World!')
        'This is used by the interact() method.\n        '
        while data != b'' and self.isalive():
            n = os.write(fd, data)
            data = data[n:]

    def __interact_read(self, fd):
        if False:
            for i in range(10):
                print('nop')
        'This is used by the interact() method.\n        '
        return os.read(fd, 1000)

    def __interact_copy(self, escape_character=None, input_filter=None, output_filter=None):
        if False:
            while True:
                i = 10
        'This is used by the interact() method.\n        '
        while self.isalive():
            if self.use_poll:
                r = poll_ignore_interrupts([self.child_fd, self.STDIN_FILENO])
            else:
                (r, w, e) = select_ignore_interrupts([self.child_fd, self.STDIN_FILENO], [], [])
            if self.child_fd in r:
                try:
                    data = self.__interact_read(self.child_fd)
                except OSError as err:
                    if err.args[0] == errno.EIO:
                        break
                    raise
                if data == b'':
                    break
                if output_filter:
                    data = output_filter(data)
                self._log(data, 'read')
                os.write(self.STDOUT_FILENO, data)
            if self.STDIN_FILENO in r:
                data = self.__interact_read(self.STDIN_FILENO)
                if input_filter:
                    data = input_filter(data)
                i = -1
                if escape_character is not None:
                    i = data.rfind(escape_character)
                if i != -1:
                    data = data[:i]
                    if data:
                        self._log(data, 'send')
                    self.__interact_writen(self.child_fd, data)
                    break
                self._log(data, 'send')
                self.__interact_writen(self.child_fd, data)

def spawnu(*args, **kwargs):
    if False:
        while True:
            i = 10
    'Deprecated: pass encoding to spawn() instead.'
    kwargs.setdefault('encoding', 'utf-8')
    return spawn(*args, **kwargs)