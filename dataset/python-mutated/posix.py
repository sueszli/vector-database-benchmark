"""Interface for running subprocess-mode commands on posix systems."""
import array
import io
import os
import signal
import subprocess
import sys
import threading
import time
import xonsh.lazyasd as xl
import xonsh.lazyimps as xli
import xonsh.platform as xp
import xonsh.tools as xt
from xonsh.built_ins import XSH
from xonsh.procs.readers import BufferedFDParallelReader, NonBlockingFDReader, safe_fdclose
MODE_NUMS = ('1049', '47', '1047')

@xl.lazyobject
def START_ALTERNATE_MODE():
    if False:
        print('Hello World!')
    return frozenset((f'\x1b[?{i}h'.encode() for i in MODE_NUMS))

@xl.lazyobject
def END_ALTERNATE_MODE():
    if False:
        return 10
    return frozenset((f'\x1b[?{i}l'.encode() for i in MODE_NUMS))

@xl.lazyobject
def ALTERNATE_MODE_FLAGS():
    if False:
        print('Hello World!')
    return tuple(START_ALTERNATE_MODE) + tuple(END_ALTERNATE_MODE)

class PopenThread(threading.Thread):
    """A thread for running and managing subprocess. This allows reading
    from the stdin, stdout, and stderr streams in a non-blocking fashion.

    This takes the same arguments and keyword arguments as regular Popen.
    This requires that the captured_stdout and captured_stderr attributes
    to be set following instantiation.
    """

    def __init__(self, *args, stdin=None, stdout=None, stderr=None, **kwargs):
        if False:
            print('Hello World!')
        super().__init__()
        self.daemon = True
        self.lock = threading.RLock()
        env = XSH.env
        self.orig_stdin = stdin
        if stdin is None:
            self.stdin_fd = 0
        elif isinstance(stdin, int):
            self.stdin_fd = stdin
        else:
            self.stdin_fd = stdin.fileno()
        self.store_stdin = env.get('XONSH_STORE_STDIN')
        self.timeout = env.get('XONSH_PROC_FREQUENCY')
        self.in_alt_mode = False
        self.stdin_mode = None
        self._tc_cc_vsusp = b'\x1a'
        self._disable_suspend_keybind()
        self.orig_stdout = stdout
        self.stdout_fd = 1 if stdout is None else stdout.fileno()
        self._set_pty_size()
        self.orig_stderr = stderr
        self.proc = None
        self.old_int_handler = self.old_winch_handler = None
        self.old_tstp_handler = self.old_quit_handler = None
        if xt.on_main_thread():
            self.old_int_handler = signal.signal(signal.SIGINT, self._signal_int)
            if xp.ON_POSIX:
                self.old_tstp_handler = signal.signal(signal.SIGTSTP, self._signal_tstp)
                self.old_quit_handler = signal.signal(signal.SIGQUIT, self._signal_quit)
            if xp.CAN_RESIZE_WINDOW:
                self.old_winch_handler = signal.signal(signal.SIGWINCH, self._signal_winch)
        if xp.ON_WINDOWS and stdout is not None:
            os.set_inheritable(stdout.fileno(), False)
        try:
            self.proc = proc = subprocess.Popen(*args, stdin=stdin, stdout=stdout, stderr=stderr, **kwargs)
        except Exception:
            self._clean_up()
            raise
        self.pid = proc.pid
        self.universal_newlines = uninew = proc.universal_newlines
        if uninew:
            self.encoding = enc = env.get('XONSH_ENCODING')
            self.encoding_errors = err = env.get('XONSH_ENCODING_ERRORS')
            self.stdin = io.BytesIO()
            self.stdout = io.TextIOWrapper(io.BytesIO(), encoding=enc, errors=err)
            self.stderr = io.TextIOWrapper(io.BytesIO(), encoding=enc, errors=err)
        else:
            self.encoding = self.encoding_errors = None
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
        self.suspended = False
        self.prevs_are_closed = False
        self.original_swapped_values = XSH.env.get_swapped_values()
        self.start()

    def run(self):
        if False:
            while True:
                i = 10
        'Runs the subprocess by performing a parallel read on stdin if allowed,\n        and copying bytes from captured_stdout to stdout and bytes from\n        captured_stderr to stderr.\n        '
        XSH.env.set_swapped_values(self.original_swapped_values)
        proc = self.proc
        spec = self._wait_and_getattr('spec')
        stdin = self.stdin
        if self.orig_stdin is None:
            origin = None
        elif xp.ON_POSIX and self.store_stdin:
            origin = self.orig_stdin
            origfd = origin if isinstance(origin, int) else origin.fileno()
            origin = BufferedFDParallelReader(origfd, buffer=stdin)
        else:
            origin = None
        stdout = self.stdout.buffer if self.universal_newlines else self.stdout
        capout = spec.captured_stdout
        if capout is None:
            procout = None
        else:
            procout = NonBlockingFDReader(capout.fileno(), timeout=self.timeout)
        stderr = self.stderr.buffer if self.universal_newlines else self.stderr
        caperr = spec.captured_stderr
        if caperr is None:
            procerr = None
        else:
            procerr = NonBlockingFDReader(caperr.fileno(), timeout=self.timeout)
        self._read_write(procout, stdout, sys.__stdout__)
        self._read_write(procerr, stderr, sys.__stderr__)
        i = j = cnt = 1
        while proc.poll() is None:
            if i + j == 0:
                cnt = min(cnt + 1, 1000)
                tout = self.timeout * cnt
                if procout is not None:
                    procout.timeout = tout
                if procerr is not None:
                    procerr.timeout = tout
            elif cnt == 1:
                pass
            else:
                cnt = 1
                if procout is not None:
                    procout.timeout = self.timeout
                if procerr is not None:
                    procerr.timeout = self.timeout
            i = self._read_write(procout, stdout, sys.__stdout__)
            j = self._read_write(procerr, stderr, sys.__stderr__)
            if self.suspended:
                break
        if self.suspended:
            return
        safe_fdclose(self.orig_stdout)
        safe_fdclose(self.orig_stderr)
        if xp.ON_WINDOWS:
            safe_fdclose(capout)
            safe_fdclose(caperr)
        while procout is not None and (not procout.is_fully_read()) or (procerr is not None and (not procerr.is_fully_read())):
            self._read_write(procout, stdout, sys.__stdout__)
            self._read_write(procerr, stderr, sys.__stderr__)
        if proc.poll() is None:
            proc.terminate()

    def _wait_and_getattr(self, name):
        if False:
            for i in range(10):
                print('nop')
        'make sure the instance has a certain attr, and return it.'
        while not hasattr(self, name):
            time.sleep(1e-07)
        return getattr(self, name)

    def _read_write(self, reader, writer, stdbuf):
        if False:
            return 10
        'Reads a chunk of bytes from a buffer and write into memory or back\n        down to the standard buffer, as appropriate. Returns the number of\n        successful reads.\n        '
        if reader is None:
            return 0
        i = -1
        for (i, chunk) in enumerate(iter(reader.read_queue, b'')):
            self._alt_mode_switch(chunk, writer, stdbuf)
        if i >= 0:
            writer.flush()
            stdbuf.flush()
        return i + 1

    def _alt_mode_switch(self, chunk, membuf, stdbuf):
        if False:
            for i in range(10):
                print('nop')
        "Enables recursively switching between normal capturing mode\n        and 'alt' mode, which passes through values to the standard\n        buffer. Pagers, text editors, curses applications, etc. use\n        alternate mode.\n        "
        (i, flag) = xt.findfirst(chunk, ALTERNATE_MODE_FLAGS)
        if flag is None:
            self._alt_mode_writer(chunk, membuf, stdbuf)
        else:
            j = i + len(flag)
            self._alt_mode_writer(chunk[:i], membuf, stdbuf)
            alt_mode = flag in START_ALTERNATE_MODE
            if alt_mode:
                self.in_alt_mode = alt_mode
                self._alt_mode_writer(flag, membuf, stdbuf)
                self._enable_cbreak_stdin()
            else:
                self._alt_mode_writer(flag, membuf, stdbuf)
                self.in_alt_mode = alt_mode
                self._disable_cbreak_stdin()
            self._alt_mode_switch(chunk[j:], membuf, stdbuf)

    def _alt_mode_writer(self, chunk, membuf, stdbuf):
        if False:
            i = 10
            return i + 15
        'Write bytes to the standard buffer if in alt mode or otherwise\n        to the in-memory buffer.\n        '
        if not chunk:
            pass
        elif self.in_alt_mode:
            stdbuf.buffer.write(chunk)
        else:
            with self.lock:
                p = membuf.tell()
                membuf.seek(0, io.SEEK_END)
                membuf.write(chunk)
                membuf.seek(p)

    def _signal_winch(self, signum, frame):
        if False:
            print('Hello World!')
        'Signal handler for SIGWINCH - window size has changed.'
        self.send_signal(signal.SIGWINCH)
        self._set_pty_size()

    def _set_pty_size(self):
        if False:
            return 10
        'Sets the window size of the child pty based on the window size of\n        our own controlling terminal.\n        '
        if xp.ON_WINDOWS or not os.isatty(self.stdout_fd):
            return
        buf = array.array('h', [0, 0, 0, 0])
        try:
            xli.fcntl.ioctl(1, xli.termios.TIOCGWINSZ, buf, True)
            xli.fcntl.ioctl(self.stdout_fd, xli.termios.TIOCSWINSZ, buf)
        except OSError:
            pass

    def _signal_int(self, signum, frame):
        if False:
            return 10
        'Signal handler for SIGINT - Ctrl+C may have been pressed.'
        self.send_signal(signal.CTRL_C_EVENT if xp.ON_WINDOWS else signum)
        if self.proc is not None and self.proc.poll() is not None:
            self._restore_sigint(frame=frame)
        if xt.on_main_thread() and (not xp.ON_WINDOWS):
            signal.pthread_kill(threading.get_ident(), signal.SIGINT)

    def _restore_sigint(self, frame=None):
        if False:
            print('Hello World!')
        old = self.old_int_handler
        if old is not None:
            if xt.on_main_thread():
                signal.signal(signal.SIGINT, old)
            self.old_int_handler = None
        if frame is not None:
            self._disable_cbreak_stdin()
            if old is not None and old is not self._signal_int:
                old(signal.SIGINT, frame)

    def _signal_tstp(self, signum, frame):
        if False:
            return 10
        'Signal handler for suspending SIGTSTP - Ctrl+Z may have been pressed.'
        self.suspended = True
        self.send_signal(signum)
        self._restore_sigtstp(frame=frame)

    def _restore_sigtstp(self, frame=None):
        if False:
            i = 10
            return i + 15
        old = self.old_tstp_handler
        if old is not None:
            if xt.on_main_thread():
                signal.signal(signal.SIGTSTP, old)
            self.old_tstp_handler = None
        if frame is not None:
            self._disable_cbreak_stdin()
        self._restore_suspend_keybind()

    def _disable_suspend_keybind(self):
        if False:
            print('Hello World!')
        if xp.ON_WINDOWS:
            return
        try:
            mode = xli.termios.tcgetattr(0)
            self._tc_cc_vsusp = mode[xp.CC][xli.termios.VSUSP]
            mode[xp.CC][xli.termios.VSUSP] = b'\x00'
            xli.termios.tcsetattr(0, xli.termios.TCSANOW, mode)
        except xli.termios.error:
            return

    def _restore_suspend_keybind(self):
        if False:
            return 10
        if xp.ON_WINDOWS:
            return
        try:
            mode = xli.termios.tcgetattr(0)
            mode[xp.CC][xli.termios.VSUSP] = self._tc_cc_vsusp
            xli.termios.tcsetattr(0, xli.termios.TCSANOW, mode)
        except xli.termios.error:
            pass

    def _signal_quit(self, signum, frame):
        if False:
            i = 10
            return i + 15
        'Signal handler for quiting SIGQUIT - Ctrl+\\ may have been pressed.'
        self.send_signal(signum)
        self._restore_sigquit(frame=frame)

    def _restore_sigquit(self, frame=None):
        if False:
            i = 10
            return i + 15
        old = self.old_quit_handler
        if old is not None:
            if xt.on_main_thread():
                signal.signal(signal.SIGQUIT, old)
            self.old_quit_handler = None
        if frame is not None:
            self._disable_cbreak_stdin()

    def _enable_cbreak_stdin(self):
        if False:
            print('Hello World!')
        if not xp.ON_POSIX:
            return
        try:
            self.stdin_mode = xli.termios.tcgetattr(self.stdin_fd)[:]
        except xli.termios.error:
            self.stdin_mode = None
            return
        new = self.stdin_mode[:]
        new[xp.LFLAG] &= ~(xli.termios.ECHO | xli.termios.ICANON)
        new[xp.CC][xli.termios.VMIN] = 1
        new[xp.CC][xli.termios.VTIME] = 0
        try:
            xli.termios.tcsetattr(self.stdin_fd, xli.termios.TCSANOW, new)
        except xli.termios.error:
            self._disable_cbreak_stdin()

    def _disable_cbreak_stdin(self):
        if False:
            while True:
                i = 10
        if not xp.ON_POSIX or self.stdin_mode is None:
            return
        new = self.stdin_mode[:]
        new[xp.LFLAG] |= xli.termios.ECHO | xli.termios.ICANON
        new[xp.CC][xli.termios.VMIN] = 1
        new[xp.CC][xli.termios.VTIME] = 0
        try:
            xli.termios.tcsetattr(self.stdin_fd, xli.termios.TCSANOW, new)
        except xli.termios.error:
            pass

    def poll(self):
        if False:
            return 10
        'Dispatches to Popen.returncode.'
        return self.proc.returncode

    def wait(self, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        'Dispatches to Popen.wait(), but also does process cleanup such as\n        joining this thread and replacing the original window size signal\n        handler.\n        '
        self._disable_cbreak_stdin()
        rtn = self.proc.wait(timeout=timeout)
        self.join()
        if self.old_winch_handler is not None and xt.on_main_thread():
            signal.signal(signal.SIGWINCH, self.old_winch_handler)
            self.old_winch_handler = None
        self._clean_up()
        return rtn

    def _clean_up(self):
        if False:
            for i in range(10):
                print('nop')
        self._restore_sigint()
        self._restore_sigtstp()
        self._restore_sigquit()

    @property
    def returncode(self):
        if False:
            print('Hello World!')
        'Process return code.'
        return self.proc.returncode

    @returncode.setter
    def returncode(self, value):
        if False:
            print('Hello World!')
        'Process return code.'
        self.proc.returncode = value

    @property
    def signal(self):
        if False:
            i = 10
            return i + 15
        'Process signal, or None.'
        s = getattr(self.proc, 'signal', None)
        if s is None:
            rtn = self.returncode
            if rtn is not None and rtn != 0:
                s = (-1 * rtn, rtn < 0 if xp.ON_WINDOWS else os.WCOREDUMP(rtn))
        return s

    @signal.setter
    def signal(self, value):
        if False:
            while True:
                i = 10
        'Process signal, or None.'
        self.proc.signal = value

    def send_signal(self, signal):
        if False:
            for i in range(10):
                print('nop')
        'Dispatches to Popen.send_signal().'
        dt = 0.0
        while self.proc is None and dt < self.timeout:
            time.sleep(1e-07)
            dt += 1e-07
        if self.proc is None:
            return
        try:
            rtn = self.proc.send_signal(signal)
        except ProcessLookupError:
            rtn = None
        return rtn

    def terminate(self):
        if False:
            print('Hello World!')
        'Dispatches to Popen.terminate().'
        return self.proc.terminate()

    def kill(self):
        if False:
            while True:
                i = 10
        'Dispatches to Popen.kill().'
        return self.proc.kill()