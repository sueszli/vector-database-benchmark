import collections
import contextlib
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from shutil import get_terminal_size
from pipenv.utils.shell import temp_environ
from pipenv.vendor import shellingham
ShellDetectionFailure = shellingham.ShellDetectionFailure

def _build_info(value):
    if False:
        while True:
            i = 10
    return (os.path.splitext(os.path.basename(value))[0], value)

def detect_info(project):
    if False:
        for i in range(10):
            print('nop')
    if project.s.PIPENV_SHELL_EXPLICIT:
        return _build_info(project.s.PIPENV_SHELL_EXPLICIT)
    try:
        return shellingham.detect_shell()
    except (shellingham.ShellDetectionFailure, TypeError):
        if project.s.PIPENV_SHELL:
            return _build_info(project.s.PIPENV_SHELL)
    raise ShellDetectionFailure

def _get_activate_script(cmd, venv):
    if False:
        i = 10
        return i + 15
    'Returns the string to activate a virtualenv.\n\n    This is POSIX-only at the moment since the compat (pexpect-based) shell\n    does not work elsewhere anyway.\n    '
    if 'fish' in cmd:
        suffix = '.fish'
        command = 'source'
    elif 'csh' in cmd:
        suffix = '.csh'
        command = 'source'
    elif 'xonsh' in cmd:
        suffix = '.xsh'
        command = 'source'
    elif 'nu' in cmd:
        suffix = '.nu'
        command = 'overlay use'
    else:
        suffix = ''
        command = '.'
    venv_location = re.sub('([ &$()\\[\\]])', '\\\\\\1', str(venv))
    return f' {command} {venv_location}/bin/activate{suffix}'

def _handover(cmd, args):
    if False:
        for i in range(10):
            print('nop')
    args = [cmd] + args
    if os.name != 'nt':
        os.execvp(cmd, args)
    else:
        sys.exit(subprocess.call(args, shell=True, universal_newlines=True))

class Shell:

    def __init__(self, cmd):
        if False:
            print('Hello World!')
        self.cmd = cmd
        self.args = []

    def __repr__(self):
        if False:
            print('Hello World!')
        return '{type(self).__name__}(cmd={self.cmd!r})'

    @contextlib.contextmanager
    def inject_path(self, venv):
        if False:
            print('Hello World!')
        with temp_environ():
            os.environ['PATH'] = '{}{}{}'.format(os.pathsep.join((str(p.parent) for p in _iter_python(venv))), os.pathsep, os.environ['PATH'])
            yield

    def fork(self, venv, cwd, args):
        if False:
            return 10
        name = os.path.basename(venv)
        os.environ['VIRTUAL_ENV'] = str(venv)
        if 'PROMPT' in os.environ:
            os.environ['PROMPT'] = '({}) {}'.format(name, os.environ['PROMPT'])
        if 'PS1' in os.environ:
            os.environ['PS1'] = '({}) {}'.format(name, os.environ['PS1'])
        with self.inject_path(venv):
            os.chdir(cwd)
            _handover(self.cmd, self.args + list(args))

    def fork_compat(self, venv, cwd, args):
        if False:
            print('Hello World!')
        from .vendor import pexpect
        dims = get_terminal_size()
        with temp_environ():
            c = pexpect.spawn(self.cmd, ['-i'], dimensions=(dims.lines, dims.columns))
        c.sendline(_get_activate_script(self.cmd, venv))
        if args:
            c.sendline(' '.join(args))

        def sigwinch_passthrough(sig, data):
            if False:
                i = 10
                return i + 15
            dims = get_terminal_size()
            c.setwinsize(dims.lines, dims.columns)
        signal.signal(signal.SIGWINCH, sigwinch_passthrough)
        c.interact(escape_character=None)
        c.close()
        sys.exit(c.exitstatus)
POSSIBLE_ENV_PYTHON = [Path('bin', 'python'), Path('Scripts', 'python.exe')]

def _iter_python(venv):
    if False:
        print('Hello World!')
    for path in POSSIBLE_ENV_PYTHON:
        full_path = Path(venv, path)
        if full_path.is_file():
            yield full_path

class Bash(Shell):

    def _format_path(self, python):
        if False:
            print('Hello World!')
        return python.parent.as_posix()

    @contextlib.contextmanager
    def inject_path(self, venv):
        if False:
            i = 10
            return i + 15
        from tempfile import NamedTemporaryFile
        bashrc_path = Path.home().joinpath('.bashrc')
        with NamedTemporaryFile('w+') as rcfile:
            if bashrc_path.is_file():
                base_rc_src = f'source "{bashrc_path.as_posix()}"\n'
                rcfile.write(base_rc_src)
            export_path = 'export PATH="{}:$PATH"\n'.format(':'.join((self._format_path(python) for python in _iter_python(venv))))
            rcfile.write(export_path)
            rcfile.flush()
            self.args.extend(['--rcfile', rcfile.name])
            yield

class MsysBash(Bash):

    def _format_path(self, python):
        if False:
            i = 10
            return i + 15
        s = super()._format_path(python)
        if not python.drive:
            return s
        return f'/{s[0].lower()}{s[2:]}'

class CmderEmulatedShell(Shell):

    def fork(self, venv, cwd, args):
        if False:
            while True:
                i = 10
        if cwd:
            os.environ['CMDER_START'] = cwd
        super().fork(venv, cwd, args)

class CmderCommandPrompt(CmderEmulatedShell):

    def fork(self, venv, cwd, args):
        if False:
            for i in range(10):
                print('nop')
        rc = os.path.expandvars('%CMDER_ROOT%\\vendor\\init.bat')
        if os.path.exists(rc):
            self.args.extend(['/k', rc])
        super().fork(venv, cwd, args)

class CmderPowershell(Shell):

    def fork(self, venv, cwd, args):
        if False:
            i = 10
            return i + 15
        rc = os.path.expandvars('%CMDER_ROOT%\\vendor\\profile.ps1')
        if os.path.exists(rc):
            self.args.extend(['-ExecutionPolicy', 'Bypass', '-NoLogo', '-NoProfile', '-NoExit', '-Command', f"Invoke-Expression '. ''{rc}'''"])
        super().fork(venv, cwd, args)
SHELL_LOOKUP = collections.defaultdict(lambda : collections.defaultdict(lambda : Shell), {'bash': collections.defaultdict(lambda : Bash, {'msys': MsysBash}), 'cmd': collections.defaultdict(lambda : Shell, {'cmder': CmderCommandPrompt}), 'powershell': collections.defaultdict(lambda : Shell, {'cmder': CmderPowershell}), 'pwsh': collections.defaultdict(lambda : Shell, {'cmder': CmderPowershell})})

def _detect_emulator():
    if False:
        i = 10
        return i + 15
    keys = []
    if os.environ.get('CMDER_ROOT'):
        keys.append('cmder')
    if os.environ.get('MSYSTEM'):
        keys.append('msys')
    return ','.join(keys)

def choose_shell(project):
    if False:
        print('Hello World!')
    emulator = project.s.PIPENV_EMULATOR.lower() or _detect_emulator()
    (type_, command) = detect_info(project)
    shell_types = SHELL_LOOKUP[type_]
    for key in emulator.split(','):
        key = key.strip().lower()
        if key in shell_types:
            return shell_types[key](command)
    return shell_types[''](command)