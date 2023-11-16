"""Subprocess specification and related utilities."""
import contextlib
import inspect
import io
import os
import pathlib
import re
import shlex
import signal
import stat
import subprocess
import sys
import xonsh.environ as xenv
import xonsh.jobs as xj
import xonsh.lazyasd as xl
import xonsh.lazyimps as xli
import xonsh.platform as xp
import xonsh.tools as xt
from xonsh.built_ins import XSH
from xonsh.procs.pipelines import STDOUT_CAPTURE_KINDS, CommandPipeline, HiddenCommandPipeline, resume_process
from xonsh.procs.posix import PopenThread
from xonsh.procs.proxies import ProcProxy, ProcProxyThread
from xonsh.procs.readers import ConsoleParallelReader

@xl.lazyobject
def RE_SHEBANG():
    if False:
        return 10
    return re.compile('#![ \\t]*(.+?)$')

def is_app_execution_alias(fname):
    if False:
        print('Hello World!')
    'App execution aliases behave strangly on Windows and Python.\n    Here we try to detect if a file is an app execution alias.\n    '
    fname = pathlib.Path(fname)
    try:
        return fname.stat().st_reparse_tag == stat.IO_REPARSE_TAG_APPEXECLINK
    except (AttributeError, OSError):
        return not os.path.exists(fname) and fname.name in os.listdir(fname.parent)

def _is_binary(fname, limit=80):
    if False:
        return 10
    try:
        with open(fname, 'rb') as f:
            for _ in range(limit):
                char = f.read(1)
                if char == b'\x00':
                    return True
                if char == b'\n':
                    return False
                if char == b'':
                    return
    except OSError as e:
        if xp.ON_WINDOWS and is_app_execution_alias(fname):
            return True
        raise e
    return False

def _un_shebang(x):
    if False:
        while True:
            i = 10
    if x == '/usr/bin/env':
        return []
    elif any((x.startswith(i) for i in ['/usr/bin', '/usr/local/bin', '/bin'])):
        x = os.path.basename(x)
    elif x.endswith('python') or x.endswith('python.exe'):
        x = 'python'
    if x == 'xonsh':
        return ['python', '-m', 'xonsh.main']
    return [x]

def get_script_subproc_command(fname, args):
    if False:
        print('Hello World!')
    'Given the name of a script outside the path, returns a list representing\n    an appropriate subprocess command to execute the script or None if\n    the argument is not readable or not a script. Raises PermissionError\n    if the script is not executable.\n    '
    if not os.access(fname, os.X_OK):
        if not xp.ON_CYGWIN:
            raise PermissionError
        w_path = os.getenv('PATH').split(':')
        w_fpath = list(map(lambda p: p + os.sep + fname, w_path))
        if not any(list(map(lambda c: os.access(c, os.X_OK), w_fpath))):
            raise PermissionError
    if xp.ON_POSIX and (not os.access(fname, os.R_OK)):
        return None
    elif _is_binary(fname):
        return None
    if xp.ON_WINDOWS:
        (_, ext) = os.path.splitext(fname)
        if ext.upper() in XSH.env.get('PATHEXT'):
            return [fname] + args
    with open(fname, 'rb') as f:
        first_line = f.readline().decode().strip()
    m = RE_SHEBANG.match(first_line)
    if m is None:
        interp = ['xonsh']
    else:
        interp = m.group(1).strip()
        if len(interp) > 0:
            interp = shlex.split(interp)
        else:
            interp = ['xonsh']
    if xp.ON_WINDOWS:
        o = []
        for i in interp:
            o.extend(_un_shebang(i))
        interp = o
    return interp + [fname] + args

@xl.lazyobject
def _REDIR_REGEX():
    if False:
        return 10
    name = '(o(?:ut)?|e(?:rr)?|a(?:ll)?|&?\\d?)'
    return re.compile(f'{name}(>?>|<){name}$')

@xl.lazyobject
def _MODES():
    if False:
        return 10
    return {'>>': 'a', '>': 'w', '<': 'r'}

@xl.lazyobject
def _WRITE_MODES():
    if False:
        i = 10
        return i + 15
    return frozenset({'w', 'a'})

@xl.lazyobject
def _REDIR_ALL():
    if False:
        for i in range(10):
            print('nop')
    return frozenset({'&', 'a', 'all'})

@xl.lazyobject
def _REDIR_ERR():
    if False:
        return 10
    return frozenset({'2', 'e', 'err'})

@xl.lazyobject
def _REDIR_OUT():
    if False:
        i = 10
        return i + 15
    return frozenset({'', '1', 'o', 'out'})

@xl.lazyobject
def _E2O_MAP():
    if False:
        print('Hello World!')
    return frozenset({f'{e}>{o}' for e in _REDIR_ERR for o in _REDIR_OUT if o != ''})

@xl.lazyobject
def _O2E_MAP():
    if False:
        i = 10
        return i + 15
    return frozenset({f'{o}>{e}' for e in _REDIR_ERR for o in _REDIR_OUT if o != ''})

def _is_redirect(x):
    if False:
        while True:
            i = 10
    return isinstance(x, str) and _REDIR_REGEX.match(x)

def safe_open(fname, mode, buffering=-1):
    if False:
        return 10
    'Safely attempts to open a file in for xonsh subprocs.'
    try:
        return open(fname, mode, buffering=buffering)
    except PermissionError as ex:
        raise xt.XonshError(f'xonsh: {fname}: permission denied') from ex
    except FileNotFoundError as ex:
        raise xt.XonshError(f'xonsh: {fname}: no such file or directory') from ex
    except Exception as ex:
        raise xt.XonshError(f'xonsh: {fname}: unable to open file') from ex

def safe_close(x):
    if False:
        i = 10
        return i + 15
    'Safely attempts to close an object.'
    if not isinstance(x, io.IOBase):
        return
    if x.closed:
        return
    try:
        x.close()
    except Exception:
        pass

def _parse_redirects(r, loc=None):
    if False:
        for i in range(10):
            print('nop')
    'returns origin, mode, destination tuple'
    (orig, mode, dest) = _REDIR_REGEX.match(r).groups()
    if dest.startswith('&'):
        try:
            dest = int(dest[1:])
            if loc is None:
                (loc, dest) = (dest, '')
            else:
                e = f'Unrecognized redirection command: {r}'
                raise xt.XonshError(e)
        except (ValueError, xt.XonshError):
            raise
        except Exception:
            pass
    mode = _MODES.get(mode, None)
    if mode == 'r' and (len(orig) > 0 or len(dest) > 0):
        raise xt.XonshError(f'Unrecognized redirection command: {r}')
    elif mode in _WRITE_MODES and len(dest) > 0:
        raise xt.XonshError(f'Unrecognized redirection command: {r}')
    return (orig, mode, dest)

def _redirect_streams(r, loc=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns stdin, stdout, stderr tuple of redirections.'
    stdin = stdout = stderr = None
    no_ampersand = r.replace('&', '')
    if no_ampersand in _E2O_MAP:
        stderr = subprocess.STDOUT
        return (stdin, stdout, stderr)
    elif no_ampersand in _O2E_MAP:
        stdout = 2
        return (stdin, stdout, stderr)
    (orig, mode, dest) = _parse_redirects(r)
    if mode == 'r':
        stdin = safe_open(loc, mode)
    elif mode in _WRITE_MODES:
        if orig in _REDIR_ALL:
            stdout = stderr = safe_open(loc, mode)
        elif orig in _REDIR_OUT:
            stdout = safe_open(loc, mode)
        elif orig in _REDIR_ERR:
            stderr = safe_open(loc, mode)
        else:
            raise xt.XonshError(f'Unrecognized redirection command: {r}')
    else:
        raise xt.XonshError(f'Unrecognized redirection command: {r}')
    return (stdin, stdout, stderr)

def default_signal_pauser(n, f):
    if False:
        i = 10
        return i + 15
    'Pauses a signal, as needed.'
    signal.pause()

def no_pg_xonsh_preexec_fn():
    if False:
        for i in range(10):
            print('nop')
    'Default subprocess preexec function for when there is no existing\n    pipeline group.\n    '
    os.setpgrp()
    signal.signal(signal.SIGTSTP, default_signal_pauser)

class SubprocSpec:
    """A container for specifying how a subprocess command should be
    executed.
    """
    kwnames = ('stdin', 'stdout', 'stderr', 'universal_newlines', 'close_fds')

    def __init__(self, cmd, cls=subprocess.Popen, stdin=None, stdout=None, stderr=None, universal_newlines=False, close_fds=False, captured=False, env=None):
        if False:
            print('Hello World!')
        "\n        Parameters\n        ----------\n        cmd : list of str\n            Command to be run.\n        cls : Popen-like\n            Class to run the subprocess with.\n        stdin : file-like\n            Popen file descriptor or flag for stdin.\n        stdout : file-like\n            Popen file descriptor or flag for stdout.\n        stderr : file-like\n            Popen file descriptor or flag for stderr.\n        universal_newlines : bool\n            Whether or not to use universal newlines.\n        close_fds : bool\n            Whether or not to close the file descriptors when the\n            process exits.\n        captured : bool or str, optional\n            The flag for if the subprocess is captured, may be one of:\n            False for $[], 'stdout' for $(), 'hiddenobject' for ![], or\n            'object' for !().\n        env : dict\n            Replacement environment to run the subporcess in.\n\n        Attributes\n        ----------\n        args : list of str\n            Arguments as originally supplied.\n        alias : list of str, callable, or None\n            The alias that was resolved for this command, if any.\n        binary_loc : str or None\n            Path to binary to execute.\n        is_proxy : bool\n            Whether or not the subprocess is or should be run as a proxy.\n        background : bool\n            Whether or not the subprocess should be started in the background.\n        threadable : bool\n            Whether or not the subprocess is able to be run in a background\n            thread, rather than the main thread.\n        pipeline_index : int or None\n            The index number of this sepc into the pipeline that is being setup.\n        last_in_pipeline : bool\n            Whether the subprocess is the last in the execution pipeline.\n        captured_stdout : file-like\n            Handle to captured stdin\n        captured_stderr : file-like\n            Handle to captured stderr\n        stack : list of FrameInfo namedtuples or None\n            The stack of the call-site of alias, if the alias requires it.\n            None otherwise.\n        "
        self._stdin = self._stdout = self._stderr = None
        self.cmd = list(cmd)
        self.cls = cls
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.universal_newlines = universal_newlines
        self.close_fds = close_fds
        self.captured = captured
        if env is not None:
            self.env = {k: v if not isinstance(v, list) or len(v) > 1 else v[0] for (k, v) in env.items()}
        else:
            self.env = None
        self.args = list(cmd)
        self.alias = None
        self.alias_name = None
        self.alias_stack = XSH.env.get('__ALIAS_STACK', '').split(':')
        self.binary_loc = None
        self.is_proxy = False
        self.background = False
        self.threadable = True
        self.pipeline_index = None
        self.last_in_pipeline = False
        self.captured_stdout = None
        self.captured_stderr = None
        self.stack = None

    def __str__(self):
        if False:
            i = 10
            return i + 15
        s = self.__class__.__name__ + '(' + str(self.cmd) + ', '
        s += self.cls.__name__ + ', '
        kws = [n + '=' + str(getattr(self, n)) for n in self.kwnames]
        s += ', '.join(kws) + ')'
        return s

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        s = self.__class__.__name__ + '(' + repr(self.cmd) + ', '
        s += self.cls.__name__ + ', '
        kws = [n + '=' + repr(getattr(self, n)) for n in self.kwnames]
        s += ', '.join(kws) + ')'
        return s

    @property
    def stdin(self):
        if False:
            i = 10
            return i + 15
        return self._stdin

    @stdin.setter
    def stdin(self, value):
        if False:
            while True:
                i = 10
        if self._stdin is None:
            self._stdin = value
        elif value is None:
            pass
        else:
            safe_close(value)
            msg = 'Multiple inputs for stdin for {0!r}'
            msg = msg.format(' '.join(self.args))
            raise xt.XonshError(msg)

    @property
    def stdout(self):
        if False:
            for i in range(10):
                print('nop')
        return self._stdout

    @stdout.setter
    def stdout(self, value):
        if False:
            i = 10
            return i + 15
        if self._stdout is None:
            self._stdout = value
        elif value is None:
            pass
        else:
            safe_close(value)
            msg = 'Multiple redirections for stdout for {0!r}'
            msg = msg.format(' '.join(self.args))
            raise xt.XonshError(msg)

    @property
    def stderr(self):
        if False:
            print('Hello World!')
        return self._stderr

    @stderr.setter
    def stderr(self, value):
        if False:
            i = 10
            return i + 15
        if self._stderr is None:
            self._stderr = value
        elif value is None:
            pass
        else:
            safe_close(value)
            msg = 'Multiple redirections for stderr for {0!r}'
            msg = msg.format(' '.join(self.args))
            raise xt.XonshError(msg)

    def run(self, *, pipeline_group=None):
        if False:
            i = 10
            return i + 15
        'Launches the subprocess and returns the object.'
        event_name = self._cmd_event_name()
        self._pre_run_event_fire(event_name)
        kwargs = {n: getattr(self, n) for n in self.kwnames}
        if callable(self.alias):
            kwargs['env'] = self.env or {}
            kwargs['env']['__ALIAS_NAME'] = self.alias_name or ''
            p = self.cls(self.alias, self.cmd, **kwargs)
        else:
            self.prep_env_subproc(kwargs)
            self.prep_preexec_fn(kwargs, pipeline_group=pipeline_group)
            self._fix_null_cmd_bytes()
            p = self._run_binary(kwargs)
        p.spec = self
        p.last_in_pipeline = self.last_in_pipeline
        p.captured_stdout = self.captured_stdout
        p.captured_stderr = self.captured_stderr
        self._post_run_event_fire(event_name, p)
        return p

    def _run_binary(self, kwargs):
        if False:
            return 10
        if not self.cmd[0]:
            raise xt.XonshError('xonsh: subprocess mode: command is empty')
        bufsize = 1
        try:
            if xp.ON_WINDOWS and self.binary_loc is not None:
                cmd = [self.binary_loc] + self.cmd[1:]
            else:
                cmd = self.cmd
            p = self.cls(cmd, bufsize=bufsize, **kwargs)
        except PermissionError as ex:
            e = 'xonsh: subprocess mode: permission denied: {0}'
            raise xt.XonshError(e.format(self.cmd[0])) from ex
        except FileNotFoundError as ex:
            cmd0 = self.cmd[0]
            if len(self.cmd) == 1 and cmd0.endswith('?'):
                with contextlib.suppress(OSError):
                    return self.cls(['man', cmd0.rstrip('?')], bufsize=bufsize, **kwargs)
            e = f'xonsh: subprocess mode: command not found: {repr(cmd0)}'
            env = XSH.env
            sug = xt.suggest_commands(cmd0, env)
            if len(sug.strip()) > 0:
                e += '\n' + xt.suggest_commands(cmd0, env)
            if XSH.env.get('XONSH_INTERACTIVE'):
                events = XSH.builtins.events
                events.on_command_not_found.fire(cmd=self.cmd)
            raise xt.XonshError(e) from ex
        return p

    def prep_env_subproc(self, kwargs):
        if False:
            i = 10
            return i + 15
        'Prepares the environment to use in the subprocess.'
        with XSH.env.swap(self.env) as env:
            denv = env.detype()
        if xp.ON_WINDOWS:
            denv['PROMPT'] = '$P$G'
        kwargs['env'] = denv

    def prep_preexec_fn(self, kwargs, pipeline_group=None):
        if False:
            for i in range(10):
                print('nop')
        "Prepares the 'preexec_fn' keyword argument"
        if not xp.ON_POSIX:
            return
        if not XSH.env.get('XONSH_INTERACTIVE'):
            return
        if pipeline_group is None or xp.ON_WSL1:
            xonsh_preexec_fn = no_pg_xonsh_preexec_fn
        else:

            def xonsh_preexec_fn():
                if False:
                    while True:
                        i = 10
                'Preexec function bound to a pipeline group.'
                os.setpgid(0, pipeline_group)
                signal.signal(signal.SIGTERM if xp.ON_WINDOWS else signal.SIGTSTP, default_signal_pauser)
        kwargs['preexec_fn'] = xonsh_preexec_fn

    def _fix_null_cmd_bytes(self):
        if False:
            i = 10
            return i + 15
        cmd = self.cmd
        for i in range(len(cmd)):
            if callable(cmd[i]):
                raise Exception(f'The command contains callable argument: {cmd[i]}')
            cmd[i] = cmd[i].replace('\x00', '\\0')

    def _cmd_event_name(self):
        if False:
            while True:
                i = 10
        if callable(self.alias):
            return getattr(self.alias, '__name__', repr(self.alias))
        elif self.binary_loc is None:
            return '<not-found>'
        else:
            return os.path.basename(self.binary_loc)

    def _pre_run_event_fire(self, name):
        if False:
            i = 10
            return i + 15
        events = XSH.builtins.events
        event_name = 'on_pre_spec_run_' + name
        if events.exists(event_name):
            event = getattr(events, event_name)
            event.fire(spec=self)

    def _post_run_event_fire(self, name, proc):
        if False:
            i = 10
            return i + 15
        events = XSH.builtins.events
        event_name = 'on_post_spec_run_' + name
        if events.exists(event_name):
            event = getattr(events, event_name)
            event.fire(spec=self, proc=proc)

    @classmethod
    def build(kls, cmd, *, cls=subprocess.Popen, **kwargs):
        if False:
            while True:
                i = 10
        'Creates an instance of the subprocess command, with any\n        modifications and adjustments based on the actual cmd that\n        was received.\n        '
        if not cmd:
            raise xt.XonshError('xonsh: subprocess mode: command is empty')
        spec = kls(cmd, cls=cls, **kwargs)
        spec.redirect_leading()
        spec.redirect_trailing()
        spec.resolve_alias()
        spec.resolve_binary_loc()
        spec.resolve_auto_cd()
        spec.resolve_executable_commands()
        spec.resolve_alias_cls()
        spec.resolve_stack()
        return spec

    def redirect_leading(self):
        if False:
            print('Hello World!')
        "Manage leading redirects such as with '< input.txt COMMAND'."
        while len(self.cmd) >= 3 and self.cmd[0] == '<':
            self.stdin = safe_open(self.cmd[1], 'r')
            self.cmd = self.cmd[2:]

    def redirect_trailing(self):
        if False:
            while True:
                i = 10
        'Manages trailing redirects.'
        while True:
            cmd = self.cmd
            if len(cmd) >= 3 and _is_redirect(cmd[-2]):
                streams = _redirect_streams(cmd[-2], cmd[-1])
                (self.stdin, self.stdout, self.stderr) = streams
                self.cmd = cmd[:-2]
            elif len(cmd) >= 2 and _is_redirect(cmd[-1]):
                streams = _redirect_streams(cmd[-1])
                (self.stdin, self.stdout, self.stderr) = streams
                self.cmd = cmd[:-1]
            else:
                break

    def resolve_alias(self):
        if False:
            i = 10
            return i + 15
        'Sets alias in command, if applicable.'
        cmd0 = self.cmd[0]
        if cmd0 in self.alias_stack:
            self.alias = None
            return
        if callable(cmd0):
            alias = cmd0
        else:
            alias = XSH.aliases.get(cmd0, None)
            if alias is not None:
                self.alias_name = cmd0
        self.alias = alias

    def resolve_binary_loc(self):
        if False:
            return 10
        'Sets the binary location'
        alias = self.alias
        if alias is None:
            cmd0 = self.cmd[0]
            binary_loc = xenv.locate_binary(cmd0)
            if binary_loc == cmd0 and cmd0 in self.alias_stack:
                raise Exception(f'Recursive calls to "{cmd0}" alias.')
        elif callable(alias):
            binary_loc = None
        else:
            binary_loc = xenv.locate_binary(alias[0])
        self.binary_loc = binary_loc

    def resolve_auto_cd(self):
        if False:
            i = 10
            return i + 15
        'Implements AUTO_CD functionality.'
        if not (self.alias is None and self.binary_loc is None and (len(self.cmd) == 1) and XSH.env.get('AUTO_CD') and os.path.isdir(self.cmd[0])):
            return
        self.cmd.insert(0, 'cd')
        self.alias = XSH.aliases.get('cd', None)

    def resolve_executable_commands(self):
        if False:
            i = 10
            return i + 15
        'Resolve command executables, if applicable.'
        alias = self.alias
        if alias is None:
            pass
        elif callable(alias):
            self.cmd.pop(0)
            return
        else:
            self.cmd = alias + self.cmd[1:]
            self.redirect_leading()
            self.redirect_trailing()
        if self.binary_loc is None:
            return
        try:
            scriptcmd = get_script_subproc_command(self.binary_loc, self.cmd[1:])
            if scriptcmd is not None:
                self.cmd = scriptcmd
        except PermissionError as ex:
            e = 'xonsh: subprocess mode: permission denied: {0}'
            raise xt.XonshError(e.format(self.cmd[0])) from ex

    def resolve_alias_cls(self):
        if False:
            print('Hello World!')
        'Determine which proxy class to run an alias with.'
        alias = self.alias
        if not callable(alias):
            return
        self.is_proxy = True
        env = XSH.env
        thable = env.get('THREAD_SUBPROCS') and getattr(alias, '__xonsh_threadable__', True)
        cls = ProcProxyThread if thable else ProcProxy
        self.cls = cls
        self.threadable = thable
        cpable = getattr(alias, '__xonsh_capturable__', self.captured)
        self.captured = cpable

    def resolve_stack(self):
        if False:
            i = 10
            return i + 15
        "Computes the stack for a callable alias's call-site, if needed."
        if not callable(self.alias):
            return
        sig = inspect.signature(self.alias)
        if len(sig.parameters) <= 5 and 'stack' not in sig.parameters:
            return
        stack = inspect.stack(context=0)
        assert stack[3][3] == 'run_subproc', 'xonsh stack has changed!'
        del stack[:5]
        self.stack = stack

def _safe_pipe_properties(fd, use_tty=False):
    if False:
        for i in range(10):
            print('nop')
    'Makes sure that a pipe file descriptor properties are reasonable.'
    if not use_tty:
        return
    props = xli.termios.tcgetattr(fd)
    props[1] = props[1] & ~xli.termios.ONLCR | xli.termios.ONLRET
    xli.termios.tcsetattr(fd, xli.termios.TCSANOW, props)
    winsize = None
    if sys.stdin.isatty():
        winsize = xli.fcntl.ioctl(sys.stdin.fileno(), xli.termios.TIOCGWINSZ, b'0000')
    elif sys.stdout.isatty():
        winsize = xli.fcntl.ioctl(sys.stdout.fileno(), xli.termios.TIOCGWINSZ, b'0000')
    elif sys.stderr.isatty():
        winsize = xli.fcntl.ioctl(sys.stderr.fileno(), xli.termios.TIOCGWINSZ, b'0000')
    if winsize is not None:
        xli.fcntl.ioctl(fd, xli.termios.TIOCSWINSZ, winsize)

def _update_last_spec(last):
    if False:
        for i in range(10):
            print('nop')
    env = XSH.env
    captured = last.captured
    last.last_in_pipeline = True
    if not captured:
        return
    callable_alias = callable(last.alias)
    if callable_alias:
        if last.cls is ProcProxy and captured == 'hiddenobject':
            return
    else:
        cmds_cache = XSH.commands_cache
        thable = env.get('THREAD_SUBPROCS') and (captured != 'hiddenobject' or env.get('XONSH_CAPTURE_ALWAYS')) and cmds_cache.predict_threadable(last.args) and cmds_cache.predict_threadable(last.cmd)
        if captured and thable:
            last.cls = PopenThread
        elif not thable:
            last.threadable = False
            if captured == 'object' or captured == 'hiddenobject':
                return
    use_tty = xp.ON_POSIX and (not callable_alias)
    if last.stdout is not None:
        last.universal_newlines = True
    elif captured in STDOUT_CAPTURE_KINDS:
        last.universal_newlines = False
        (r, w) = os.pipe()
        last.stdout = safe_open(w, 'wb')
        last.captured_stdout = safe_open(r, 'rb')
    elif XSH.stdout_uncaptured is not None:
        last.universal_newlines = True
        last.stdout = XSH.stdout_uncaptured
        last.captured_stdout = last.stdout
    elif xp.ON_WINDOWS and (not callable_alias):
        last.universal_newlines = True
        last.stdout = None
        last.captured_stdout = ConsoleParallelReader(1)
    else:
        last.universal_newlines = True
        (r, w) = xli.pty.openpty() if use_tty else os.pipe()
        _safe_pipe_properties(w, use_tty=use_tty)
        last.stdout = safe_open(w, 'w')
        _safe_pipe_properties(r, use_tty=use_tty)
        last.captured_stdout = safe_open(r, 'r')
    if last.stderr is not None:
        pass
    elif captured == 'stdout':
        pass
    elif captured == 'object':
        (r, w) = os.pipe()
        last.stderr = safe_open(w, 'w')
        last.captured_stderr = safe_open(r, 'r')
    elif XSH.stderr_uncaptured is not None:
        last.stderr = XSH.stderr_uncaptured
        last.captured_stderr = last.stderr
    elif xp.ON_WINDOWS and (not callable_alias):
        last.universal_newlines = True
        last.stderr = None
    else:
        (r, w) = xli.pty.openpty() if use_tty else os.pipe()
        _safe_pipe_properties(w, use_tty=use_tty)
        last.stderr = safe_open(w, 'w')
        _safe_pipe_properties(r, use_tty=use_tty)
        last.captured_stderr = safe_open(r, 'r')
    if isinstance(last.stdout, int) and last.stdout == 2:
        last._stdout = last.stderr
    if callable_alias and last.stderr == subprocess.STDOUT:
        last._stderr = last.stdout
        last.captured_stderr = last.captured_stdout

def cmds_to_specs(cmds, captured=False, envs=None):
    if False:
        for i in range(10):
            print('nop')
    'Converts a list of cmds to a list of SubprocSpec objects that are\n    ready to be executed.\n    '
    i = 0
    specs = []
    redirects = []
    for (i, cmd) in enumerate(cmds):
        if isinstance(cmd, str):
            redirects.append(cmd)
        else:
            env = envs[i] if envs is not None else None
            spec = SubprocSpec.build(cmd, captured=captured, env=env)
            spec.pipeline_index = i
            specs.append(spec)
            i += 1
    for (i, redirect) in enumerate(redirects):
        if redirect == '|':
            (r, w) = os.pipe()
            specs[i].stdout = w
            specs[i + 1].stdin = r
        elif redirect == '&' and i == len(redirects) - 1:
            specs[i].background = True
        else:
            raise xt.XonshError(f'unrecognized redirect {redirect!r}')
    if not XSH.env.get('XONSH_CAPTURE_ALWAYS'):
        specs_to_capture = specs if captured in STDOUT_CAPTURE_KINDS else specs[:-1]
        for spec in specs_to_capture:
            if spec.env is None:
                spec.env = {'XONSH_CAPTURE_ALWAYS': True}
            else:
                spec.env.setdefault('XONSH_CAPTURE_ALWAYS', True)
    _update_last_spec(specs[-1])
    return specs

def _should_set_title():
    if False:
        i = 10
        return i + 15
    return XSH.env.get('XONSH_INTERACTIVE') and XSH.shell is not None

def run_subproc(cmds, captured=False, envs=None):
    if False:
        i = 10
        return i + 15
    "Runs a subprocess, in its many forms. This takes a list of 'commands,'\n    which may be a list of command line arguments or a string, representing\n    a special connecting character.  For example::\n\n        $ ls | grep wakka\n\n    is represented by the following cmds::\n\n        [['ls'], '|', ['grep', 'wakka']]\n\n    Lastly, the captured argument affects only the last real command.\n    "
    if XSH.env.get('XONSH_TRACE_SUBPROC', False):
        tracer = XSH.env.get('XONSH_TRACE_SUBPROC_FUNC')
        if callable(tracer):
            tracer(cmds, captured=captured)
        else:
            print(f'TRACE SUBPROC: {cmds}, captured={captured}', file=sys.stderr)
    specs = cmds_to_specs(cmds, captured=captured, envs=envs)
    if _should_set_title():
        with XSH.env['PROMPT_FIELDS']['current_job'].update_current_cmds(cmds):
            XSH.env['PROMPT_FIELDS'].reset_key('current_job')
            XSH.shell.settitle()
            return _run_specs(specs, cmds)
    else:
        return _run_specs(specs, cmds)

def _run_specs(specs, cmds):
    if False:
        print('Hello World!')
    captured = specs[-1].captured
    if captured == 'hiddenobject':
        command = HiddenCommandPipeline(specs)
    else:
        command = CommandPipeline(specs)
    proc = command.proc
    background = command.spec.background
    if not all((x.is_proxy for x in specs)):
        xj.add_job({'cmds': cmds, 'pids': [i.pid for i in command.procs], 'obj': proc, 'bg': background, 'pipeline': command, 'pgrp': command.term_pgid})
    resume_process(proc)
    if captured == 'object':
        return command
    elif captured == 'hiddenobject':
        if not background:
            command.end()
        return command
    elif background:
        return
    elif captured == 'stdout':
        command.end()
        return command.output
    else:
        command.end()
        return