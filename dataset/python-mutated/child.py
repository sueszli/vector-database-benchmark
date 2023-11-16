import os
import sys
from collections import defaultdict
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, DefaultDict, Dict, Generator, List, Optional, Sequence, Tuple
import kitty.fast_data_types as fast_data_types
from .constants import handled_signals, is_freebsd, is_macos, kitten_exe, kitty_base_dir, shell_path, terminfo_dir
from .types import run_once
from .utils import log_error, which
try:
    from typing import TypedDict
except ImportError:
    TypedDict = dict
if TYPE_CHECKING:
    from .window import CwdRequest
if is_macos:
    from kitty.fast_data_types import cmdline_of_process as cmdline_
    from kitty.fast_data_types import cwd_of_process as _cwd
    from kitty.fast_data_types import environ_of_process as _environ_of_process
    from kitty.fast_data_types import process_group_map as _process_group_map

    def cwd_of_process(pid: int) -> str:
        if False:
            print('Hello World!')
        return os.path.realpath(_cwd(pid))

    def process_group_map() -> DefaultDict[int, List[int]]:
        if False:
            print('Hello World!')
        ans: DefaultDict[int, List[int]] = defaultdict(list)
        for (pid, pgid) in _process_group_map():
            ans[pgid].append(pid)
        return ans

    def cmdline_of_pid(pid: int) -> List[str]:
        if False:
            print('Hello World!')
        return cmdline_(pid)
else:

    def cmdline_of_pid(pid: int) -> List[str]:
        if False:
            i = 10
            return i + 15
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            return list(filter(None, f.read().decode('utf-8').split('\x00')))
    if is_freebsd:

        def cwd_of_process(pid: int) -> str:
            if False:
                return 10
            import subprocess
            cp = subprocess.run(['pwdx', str(pid)], capture_output=True)
            if cp.returncode != 0:
                raise ValueError(f'Failed to find cwd of process with pid: {pid}')
            ans = cp.stdout.decode('utf-8', 'replace').split()[1]
            return os.path.realpath(ans)
    else:

        def cwd_of_process(pid: int) -> str:
            if False:
                while True:
                    i = 10
            ans = f'/proc/{pid}/cwd'
            return os.path.realpath(ans)

    def _environ_of_process(pid: int) -> str:
        if False:
            i = 10
            return i + 15
        with open(f'/proc/{pid}/environ', 'rb') as f:
            return f.read().decode('utf-8')

    def process_group_map() -> DefaultDict[int, List[int]]:
        if False:
            for i in range(10):
                print('nop')
        ans: DefaultDict[int, List[int]] = defaultdict(list)
        for x in os.listdir('/proc'):
            try:
                pid = int(x)
            except Exception:
                continue
            try:
                with open(f'/proc/{x}/stat', 'rb') as f:
                    raw = f.read().decode('utf-8')
            except OSError:
                continue
            try:
                q = int(raw.split(' ', 5)[4])
            except Exception:
                continue
            ans[q].append(pid)
        return ans

@run_once
def checked_terminfo_dir() -> Optional[str]:
    if False:
        return 10
    return terminfo_dir if os.path.isdir(terminfo_dir) else None

def processes_in_group(grp: int) -> List[int]:
    if False:
        print('Hello World!')
    gmap: Optional[DefaultDict[int, List[int]]] = getattr(process_group_map, 'cached_map', None)
    if gmap is None:
        try:
            gmap = process_group_map()
        except Exception:
            gmap = defaultdict(list)
    return gmap.get(grp, [])

@contextmanager
def cached_process_data() -> Generator[None, None, None]:
    if False:
        while True:
            i = 10
    try:
        cm = process_group_map()
    except Exception:
        cm = defaultdict(list)
    setattr(process_group_map, 'cached_map', cm)
    try:
        yield
    finally:
        delattr(process_group_map, 'cached_map')

def parse_environ_block(data: str) -> Dict[str, str]:
    if False:
        i = 10
        return i + 15
    'Parse a C environ block of environment variables into a dictionary.'
    ret: Dict[str, str] = {}
    pos = 0
    while True:
        next_pos = data.find('\x00', pos)
        if next_pos <= pos:
            break
        equal_pos = data.find('=', pos, next_pos)
        if equal_pos > pos:
            key = data[pos:equal_pos]
            value = data[equal_pos + 1:next_pos]
            ret[key] = value
        pos = next_pos + 1
    return ret

def environ_of_process(pid: int) -> Dict[str, str]:
    if False:
        for i in range(10):
            print('nop')
    return parse_environ_block(_environ_of_process(pid))

def process_env() -> Dict[str, str]:
    if False:
        while True:
            i = 10
    ans = dict(os.environ)
    ssl_env_var = getattr(sys, 'kitty_ssl_env_var', None)
    if ssl_env_var is not None:
        ans.pop(ssl_env_var, None)
    ans.pop('XDG_ACTIVATION_TOKEN', None)
    return ans

def default_env() -> Dict[str, str]:
    if False:
        for i in range(10):
            print('nop')
    ans: Optional[Dict[str, str]] = getattr(default_env, 'env', None)
    if ans is None:
        return process_env()
    return ans

def set_default_env(val: Optional[Dict[str, str]]=None) -> None:
    if False:
        return 10
    env = process_env().copy()
    has_lctype = False
    if val:
        has_lctype = 'LC_CTYPE' in val
        env.update(val)
    setattr(default_env, 'env', env)
    setattr(default_env, 'lc_ctype_set_by_user', has_lctype)

def set_LANG_in_default_env(val: str) -> None:
    if False:
        i = 10
        return i + 15
    default_env().setdefault('LANG', val)

def openpty() -> Tuple[int, int]:
    if False:
        i = 10
        return i + 15
    (master, slave) = os.openpty()
    os.set_inheritable(slave, True)
    os.set_inheritable(master, False)
    fast_data_types.set_iutf8_fd(master, True)
    return (master, slave)

@run_once
def getpid() -> str:
    if False:
        while True:
            i = 10
    return str(os.getpid())

class ProcessDesc(TypedDict):
    cwd: Optional[str]
    pid: int
    cmdline: Optional[Sequence[str]]

class Child:
    child_fd: Optional[int] = None
    pid: Optional[int] = None
    forked = False

    def __init__(self, argv: Sequence[str], cwd: str, stdin: Optional[bytes]=None, env: Optional[Dict[str, str]]=None, cwd_from: Optional['CwdRequest']=None, is_clone_launch: str='', add_listen_on_env_var: bool=True):
        if False:
            print('Hello World!')
        self.is_clone_launch = is_clone_launch
        self.add_listen_on_env_var = add_listen_on_env_var
        self.argv = list(argv)
        if cwd_from:
            try:
                cwd = cwd_from.modify_argv_for_launch_with_cwd(self.argv, env) or cwd
            except Exception as err:
                log_error(f'Failed to read cwd of {cwd_from} with error: {err}')
        else:
            cwd = os.path.expandvars(os.path.expanduser(cwd or os.getcwd()))
        self.cwd = os.path.abspath(cwd)
        self.stdin = stdin
        self.env = env or {}
        self.final_env: Dict[str, str] = {}
        self.is_default_shell = bool(self.argv and self.argv[0] == shell_path)
        self.should_run_via_run_shell_kitten = is_macos and self.is_default_shell

    def get_final_env(self) -> Dict[str, str]:
        if False:
            return 10
        from kitty.options.utils import DELETE_ENV_VAR
        env = default_env().copy()
        opts = fast_data_types.get_options()
        boss = fast_data_types.get_boss()
        if is_macos and env.get('LC_CTYPE') == 'UTF-8' and (not getattr(sys, 'kitty_run_data').get('lc_ctype_before_python')) and (not getattr(default_env, 'lc_ctype_set_by_user', False)):
            del env['LC_CTYPE']
        env.update(self.env)
        env['TERM'] = opts.term
        env['COLORTERM'] = 'truecolor'
        env['KITTY_PID'] = getpid()
        env['KITTY_PUBLIC_KEY'] = boss.encryption_public_key
        if self.add_listen_on_env_var and boss.listening_on:
            env['KITTY_LISTEN_ON'] = boss.listening_on
        else:
            env.pop('KITTY_LISTEN_ON', None)
        if self.cwd:
            env['PWD'] = self.cwd
        tdir = checked_terminfo_dir()
        if tdir:
            env['TERMINFO'] = tdir
        env['KITTY_INSTALLATION_DIR'] = kitty_base_dir
        if opts.forward_stdio:
            env['KITTY_STDIO_FORWARDED'] = '3'
        self.unmodified_argv = list(self.argv)
        if not self.should_run_via_run_shell_kitten and 'disabled' not in opts.shell_integration:
            from .shell_integration import modify_shell_environ
            modify_shell_environ(opts, env, self.argv)
        env = {k: v for (k, v) in env.items() if v is not DELETE_ENV_VAR}
        if self.is_clone_launch:
            env['KITTY_IS_CLONE_LAUNCH'] = self.is_clone_launch
            self.is_clone_launch = '1'
        else:
            env.pop('KITTY_IS_CLONE_LAUNCH', None)
        return env

    def fork(self) -> Optional[int]:
        if False:
            return 10
        if self.forked:
            return None
        opts = fast_data_types.get_options()
        self.forked = True
        (master, slave) = openpty()
        (stdin, self.stdin) = (self.stdin, None)
        (ready_read_fd, ready_write_fd) = os.pipe()
        os.set_inheritable(ready_write_fd, False)
        os.set_inheritable(ready_read_fd, True)
        if stdin is not None:
            (stdin_read_fd, stdin_write_fd) = os.pipe()
            os.set_inheritable(stdin_write_fd, False)
            os.set_inheritable(stdin_read_fd, True)
        else:
            stdin_read_fd = stdin_write_fd = -1
        self.final_env = self.get_final_env()
        argv = list(self.argv)
        cwd = self.cwd
        if self.should_run_via_run_shell_kitten:
            import shlex
            ksi = ' '.join(opts.shell_integration)
            if ksi == 'invalid':
                ksi = 'enabled'
            argv = [kitten_exe(), 'run-shell', '--shell', shlex.join(argv), '--shell-integration', ksi]
            if is_macos:
                import pwd
                user = pwd.getpwuid(os.geteuid()).pw_name
                if cwd:
                    argv.append('--cwd=' + cwd)
                    cwd = os.path.expanduser('~')
                argv = ['/usr/bin/login', '-f', '-l', '-p', user] + argv
        self.final_exe = which(argv[0]) or argv[0]
        self.final_argv0 = argv[0]
        env = tuple((f'{k}={v}' for (k, v) in self.final_env.items()))
        pid = fast_data_types.spawn(self.final_exe, cwd, tuple(argv), env, master, slave, stdin_read_fd, stdin_write_fd, ready_read_fd, ready_write_fd, tuple(handled_signals), kitten_exe(), opts.forward_stdio)
        os.close(slave)
        self.pid = pid
        self.child_fd = master
        if stdin is not None:
            os.close(stdin_read_fd)
            fast_data_types.thread_write(stdin_write_fd, stdin)
        os.close(ready_read_fd)
        self.terminal_ready_fd = ready_write_fd
        if self.child_fd is not None:
            os.set_blocking(self.child_fd, False)
        return pid

    def __del__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        fd = getattr(self, 'terminal_ready_fd', -1)
        if fd > -1:
            os.close(fd)
        self.terminal_ready_fd = -1

    def mark_terminal_ready(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        os.close(self.terminal_ready_fd)
        self.terminal_ready_fd = -1

    def cmdline_of_pid(self, pid: int) -> List[str]:
        if False:
            while True:
                i = 10
        try:
            ans = cmdline_of_pid(pid)
        except Exception:
            ans = []
        if pid == self.pid and (not ans):
            ans = list(self.argv)
        return ans

    @property
    def foreground_processes(self) -> List[ProcessDesc]:
        if False:
            i = 10
            return i + 15
        if self.child_fd is None:
            return []
        try:
            pgrp = os.tcgetpgrp(self.child_fd)
            foreground_processes = processes_in_group(pgrp) if pgrp >= 0 else []

            def process_desc(pid: int) -> ProcessDesc:
                if False:
                    print('Hello World!')
                ans: ProcessDesc = {'pid': pid, 'cmdline': None, 'cwd': None}
                with suppress(Exception):
                    ans['cmdline'] = self.cmdline_of_pid(pid)
                with suppress(Exception):
                    ans['cwd'] = cwd_of_process(pid) or None
                return ans
            return [process_desc(x) for x in foreground_processes]
        except Exception:
            return []

    @property
    def cmdline(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        try:
            assert self.pid is not None
            return self.cmdline_of_pid(self.pid) or list(self.argv)
        except Exception:
            return list(self.argv)

    @property
    def foreground_cmdline(self) -> List[str]:
        if False:
            return 10
        try:
            assert self.pid_for_cwd is not None
            return self.cmdline_of_pid(self.pid_for_cwd) or self.cmdline
        except Exception:
            return self.cmdline

    @property
    def environ(self) -> Dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        try:
            assert self.pid is not None
            return environ_of_process(self.pid) or self.final_env.copy()
        except Exception:
            return self.final_env.copy()

    @property
    def current_cwd(self) -> Optional[str]:
        if False:
            return 10
        with suppress(Exception):
            assert self.pid is not None
            return cwd_of_process(self.pid)
        return None

    def get_pid_for_cwd(self, oldest: bool=False) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        with suppress(Exception):
            assert self.child_fd is not None
            pgrp = os.tcgetpgrp(self.child_fd)
            foreground_processes = processes_in_group(pgrp) if pgrp >= 0 else []
            if foreground_processes:
                return min(foreground_processes) if oldest else max(foreground_processes)
        return self.pid

    @property
    def pid_for_cwd(self) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        return self.get_pid_for_cwd()

    def get_foreground_cwd(self, oldest: bool=False) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        with suppress(Exception):
            pid = self.get_pid_for_cwd(oldest)
            if pid is not None:
                return cwd_of_process(pid) or None
        return None

    def get_foreground_exe(self, oldest: bool=False) -> Optional[str]:
        if False:
            while True:
                i = 10
        with suppress(Exception):
            pid = self.get_pid_for_cwd(oldest)
            if pid is not None:
                c = cmdline_of_pid(pid)
                if c:
                    return c[0]
        return None

    @property
    def foreground_cwd(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self.get_foreground_cwd()

    @property
    def foreground_environ(self) -> Dict[str, str]:
        if False:
            while True:
                i = 10
        pid = self.pid_for_cwd
        if pid is not None:
            with suppress(Exception):
                return environ_of_process(pid)
        pid = self.pid
        if pid is not None:
            with suppress(Exception):
                return environ_of_process(pid)
        return {}

    def send_signal_for_key(self, key_num: bytes) -> bool:
        if False:
            return 10
        import signal
        import termios
        if self.child_fd is None:
            return False
        t = termios.tcgetattr(self.child_fd)
        if not t[3] & termios.ISIG:
            return False
        cc = t[-1]
        if key_num == cc[termios.VINTR]:
            s = signal.SIGINT
        elif key_num == cc[termios.VSUSP]:
            s = signal.SIGTSTP
        elif key_num == cc[termios.VQUIT]:
            s = signal.SIGQUIT
        else:
            return False
        pgrp = os.tcgetpgrp(self.child_fd)
        os.killpg(pgrp, s)
        return True