"""Tools to help interface with foreign shells, such as Bash."""
import collections.abc as cabc
import functools
import os
import re
import shlex
import subprocess
import sys
import tempfile
import warnings
from xonsh.built_ins import XSH
from xonsh.lazyasd import lazyobject
from xonsh.platform import ON_CYGWIN, ON_MSYS, ON_WINDOWS
from xonsh.tools import ensure_string, to_bool
COMMAND = '{seterrprevcmd}\n{prevcmd}\necho __XONSH_ENV_BEG__\n{envcmd}\necho __XONSH_ENV_END__\necho __XONSH_ALIAS_BEG__\n{aliascmd}\necho __XONSH_ALIAS_END__\necho __XONSH_FUNCS_BEG__\n{funcscmd}\necho __XONSH_FUNCS_END__\n{postcmd}\n{seterrpostcmd}'
DEFAULT_BASH_FUNCSCMD = '# get function names from declare\ndeclstr=$(echo $(declare -F))\nread -r -a decls <<< $declstr\nfor((n=0;n<${#decls[@]};n++)); do\n  if (( $(($n % 3 )) == 2 )); then\n    echo -n "${decls[$n]} "\n  fi\ndone\necho'
DEFAULT_ZSH_FUNCSCMD = '# get function names\nfor name in ${(ok)functions}; do\n  echo -n "$name "\ndone\necho'

@lazyobject
def CANON_SHELL_NAMES():
    if False:
        return 10
    return {'bash': 'bash', '/bin/bash': 'bash', 'zsh': 'zsh', '/bin/zsh': 'zsh', '/usr/bin/zsh': 'zsh', 'cmd': 'cmd', 'cmd.exe': 'cmd'}

@lazyobject
def DEFAULT_ENVCMDS():
    if False:
        while True:
            i = 10
    return {'bash': 'env', 'zsh': 'env', 'cmd': 'set'}

@lazyobject
def DEFAULT_ALIASCMDS():
    if False:
        i = 10
        return i + 15
    return {'bash': 'alias', 'zsh': 'alias -L', 'cmd': ''}

@lazyobject
def DEFAULT_FUNCSCMDS():
    if False:
        for i in range(10):
            print('nop')
    return {'bash': DEFAULT_BASH_FUNCSCMD, 'zsh': DEFAULT_ZSH_FUNCSCMD, 'cmd': ''}

@lazyobject
def DEFAULT_SOURCERS():
    if False:
        print('Hello World!')
    return {'bash': 'source', 'zsh': 'source', 'cmd': 'call'}

@lazyobject
def DEFAULT_TMPFILE_EXT():
    if False:
        i = 10
        return i + 15
    return {'bash': '.sh', 'zsh': '.zsh', 'cmd': '.bat'}

@lazyobject
def DEFAULT_RUNCMD():
    if False:
        while True:
            i = 10
    return {'bash': '-c', 'zsh': '-c', 'cmd': '/C'}

@lazyobject
def DEFAULT_SETERRPREVCMD():
    if False:
        i = 10
        return i + 15
    return {'bash': 'set -e', 'zsh': 'set -e', 'cmd': '@echo off'}

@lazyobject
def DEFAULT_SETERRPOSTCMD():
    if False:
        return 10
    return {'bash': '', 'zsh': '', 'cmd': 'if errorlevel 1 exit 1'}

@functools.lru_cache
def foreign_shell_data(shell, interactive=True, login=False, envcmd=None, aliascmd=None, extra_args=(), currenv=None, safe=True, prevcmd='', postcmd='', funcscmd=None, sourcer=None, use_tmpfile=False, tmpfile_ext=None, runcmd=None, seterrprevcmd=None, seterrpostcmd=None, show=False, dryrun=False, files=()):
    if False:
        print('Hello World!')
    'Extracts data from a foreign (non-xonsh) shells. Currently this gets\n    the environment, aliases, and functions but may be extended in the future.\n\n    Parameters\n    ----------\n    shell : str\n        The name of the shell, such as \'bash\' or \'/bin/sh\'.\n    interactive : bool, optional\n        Whether the shell should be run in interactive mode.\n    login : bool, optional\n        Whether the shell should be a login shell.\n    envcmd : str or None, optional\n        The command to generate environment output with.\n    aliascmd : str or None, optional\n        The command to generate alias output with.\n    extra_args : tuple of str, optional\n        Additional command line options to pass into the shell.\n    currenv : tuple of items or None, optional\n        Manual override for the current environment.\n    safe : bool, optional\n        Flag for whether or not to safely handle exceptions and other errors.\n    prevcmd : str, optional\n        A command to run in the shell before anything else, useful for\n        sourcing and other commands that may require environment recovery.\n    postcmd : str, optional\n        A command to run after everything else, useful for cleaning up any\n        damage that the prevcmd may have caused.\n    funcscmd : str or None, optional\n        This is a command or script that can be used to determine the names\n        and locations of any functions that are native to the foreign shell.\n        This command should print *only* a JSON object that maps\n        function names to the filenames where the functions are defined.\n        If this is None, then a default script will attempted to be looked\n        up based on the shell name. Callable wrappers for these functions\n        will be returned in the aliases dictionary.\n    sourcer : str or None, optional\n        How to source a foreign shell file for purposes of calling functions\n        in that shell. If this is None, a default value will attempt to be\n        looked up based on the shell name.\n    use_tmpfile : bool, optional\n        This specifies if the commands are written to a tmp file or just\n        parsed directly to the shell\n    tmpfile_ext : str or None, optional\n        If tmpfile is True this sets specifies the extension used.\n    runcmd : str or None, optional\n        Command line switches to use when running the script, such as\n        -c for Bash and /C for cmd.exe.\n    seterrprevcmd : str or None, optional\n        Command that enables exit-on-error for the shell that is run at the\n        start of the script. For example, this is "set -e" in Bash. To disable\n        exit-on-error behavior, simply pass in an empty string.\n    seterrpostcmd : str or None, optional\n        Command that enables exit-on-error for the shell that is run at the end\n        of the script. For example, this is "if errorlevel 1 exit 1" in\n        cmd.exe. To disable exit-on-error behavior, simply pass in an\n        empty string.\n    show : bool, optional\n        Whether or not to display the script that will be run.\n    dryrun : bool, optional\n        Whether or not to actually run and process the command.\n    files : tuple of str, optional\n        Paths to source.\n\n    Returns\n    -------\n    env : dict\n        Dictionary of shell\'s environment. (None if the subproc command fails)\n    aliases : dict\n        Dictionary of shell\'s aliases, this includes foreign function\n        wrappers.(None if the subproc command fails)\n    '
    cmd = [shell]
    cmd.extend(extra_args)
    if interactive:
        cmd.append('-i')
    if login:
        cmd.append('-l')
    shkey = CANON_SHELL_NAMES[shell]
    envcmd = DEFAULT_ENVCMDS.get(shkey, 'env') if envcmd is None else envcmd
    aliascmd = DEFAULT_ALIASCMDS.get(shkey, 'alias') if aliascmd is None else aliascmd
    funcscmd = DEFAULT_FUNCSCMDS.get(shkey, 'echo {}') if funcscmd is None else funcscmd
    tmpfile_ext = DEFAULT_TMPFILE_EXT.get(shkey, 'sh') if tmpfile_ext is None else tmpfile_ext
    runcmd = DEFAULT_RUNCMD.get(shkey, '-c') if runcmd is None else runcmd
    seterrprevcmd = DEFAULT_SETERRPREVCMD.get(shkey, '') if seterrprevcmd is None else seterrprevcmd
    seterrpostcmd = DEFAULT_SETERRPOSTCMD.get(shkey, '') if seterrpostcmd is None else seterrpostcmd
    command = COMMAND.format(envcmd=envcmd, aliascmd=aliascmd, prevcmd=prevcmd, postcmd=postcmd, funcscmd=funcscmd, seterrprevcmd=seterrprevcmd, seterrpostcmd=seterrpostcmd).strip()
    if show:
        print(command)
    if dryrun:
        return (None, None)
    cmd.append(runcmd)
    if not use_tmpfile:
        cmd.append(command)
    else:
        tmpfile = tempfile.NamedTemporaryFile(suffix=tmpfile_ext, delete=False)
        tmpfile.write(command.encode('utf8'))
        tmpfile.close()
        cmd.append(tmpfile.name)
    if currenv is None and XSH.env:
        currenv = XSH.env.detype()
    elif currenv is not None:
        currenv = dict(currenv)
    try:
        s = subprocess.check_output(cmd, stderr=subprocess.PIPE, env=currenv, start_new_session=not ON_CYGWIN and (not ON_MSYS), text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        if not safe:
            raise
        return (None, None)
    finally:
        if use_tmpfile:
            os.remove(tmpfile.name)
    env = parse_env(s)
    aliases = parse_aliases(s, shell=shell, sourcer=sourcer, files=files, extra_args=extra_args)
    funcs = parse_funcs(s, shell=shell, sourcer=sourcer, files=files, extra_args=extra_args)
    aliases.update(funcs)
    return (env, aliases)

@lazyobject
def ENV_RE():
    if False:
        i = 10
        return i + 15
    return re.compile('__XONSH_ENV_BEG__\n(.*)__XONSH_ENV_END__', flags=re.DOTALL)

@lazyobject
def ENV_SPLIT_RE():
    if False:
        while True:
            i = 10
    return re.compile('^([^=]+)=([^=]*|[^\n]*)$', flags=re.DOTALL | re.MULTILINE)

def parse_env(s):
    if False:
        while True:
            i = 10
    'Parses the environment portion of string into a dict.'
    m = ENV_RE.search(s)
    if m is None:
        return {}
    g1 = m.group(1)
    g1 = g1[:-1] if g1.endswith('\n') else g1
    env = dict(ENV_SPLIT_RE.findall(g1))
    return env

@lazyobject
def ALIAS_RE():
    if False:
        return 10
    return re.compile('__XONSH_ALIAS_BEG__\n(.*)__XONSH_ALIAS_END__', flags=re.DOTALL)

@lazyobject
def FS_EXEC_ALIAS_RE():
    if False:
        i = 10
        return i + 15
    return re.compile(';|`|\\$\\(')

def parse_aliases(s, shell, sourcer=None, files=(), extra_args=()):
    if False:
        i = 10
        return i + 15
    'Parses the aliases portion of string into a dict.'
    m = ALIAS_RE.search(s)
    if m is None:
        return {}
    g1 = m.group(1)
    g1 = g1.replace('\\\n', ' ')
    items = [line.split('=', 1) for line in g1.splitlines() if line.startswith('alias ') and '=' in line]
    aliases = {}
    for (key, value) in items:
        try:
            key = key[6:]
            value = value.replace("'\\''", "'")
            if value[0] == "'" and value[-1] == "'":
                value = value[1:-1]
            if FS_EXEC_ALIAS_RE.search(value) is None:
                value = shlex.split(value)
            else:
                value = ForeignShellExecAlias(src=value, shell=shell, sourcer=sourcer, files=files, extra_args=extra_args)
        except ValueError as exc:
            warnings.warn(f'could not parse alias "{key}": {exc!r}', RuntimeWarning, stacklevel=2)
            continue
        aliases[key] = value
    return aliases

@lazyobject
def FUNCS_RE():
    if False:
        return 10
    return re.compile('__XONSH_FUNCS_BEG__\n(.+)\n__XONSH_FUNCS_END__', flags=re.DOTALL)

def parse_funcs(s, shell, sourcer=None, files=(), extra_args=()):
    if False:
        print('Hello World!')
    'Parses the funcs portion of a string into a dict of callable foreign\n    function wrappers.\n    '
    m = FUNCS_RE.search(s)
    if m is None:
        return {}
    g1 = m.group(1)
    if ON_WINDOWS:
        g1 = g1.replace(os.sep, os.altsep)
    funcnames = g1.split()
    funcs = {}
    for funcname in funcnames:
        if funcname.startswith('_'):
            continue
        wrapper = ForeignShellFunctionAlias(funcname=funcname, shell=shell, sourcer=sourcer, files=files, extra_args=extra_args)
        funcs[funcname] = wrapper
    return funcs

class ForeignShellBaseAlias:
    """This class is responsible for calling foreign shell functions as if
    they were aliases. This does not currently support taking stdin.
    """
    INPUT = 'echo ForeignShellBaseAlias {shell} {args}\n'

    def __init__(self, shell, sourcer=None, files=(), extra_args=()):
        if False:
            while True:
                i = 10
        '\n        Parameters\n        ----------\n        shell : str\n            Name or path to shell\n        sourcer : str or None, optional\n            Command to source foreign files with.\n        files : tuple of str, optional\n            Paths to source.\n        extra_args : tuple of str, optional\n            Additional command line options to pass into the shell.\n        '
        sourcer = DEFAULT_SOURCERS.get(shell, 'source') if sourcer is None else sourcer
        self.shell = shell
        self.sourcer = sourcer
        self.files = files
        self.extra_args = extra_args

    def _input_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        return {'shell': self.shell, 'extra_args': self.extra_args}

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(other, '_input_kwargs') or not callable(other._input_kwargs):
            return NotImplemented
        return self._input_kwargs() == other._input_kwargs()

    def __call__(self, args, stdin=None, stdout=None, stderr=None, spec=None, stack=None):
        if False:
            while True:
                i = 10
        (args, streaming) = self._is_streaming(args)
        input = self.INPUT.format(args=' '.join(args), **self._input_kwargs())
        if len(self.files) > 0:
            input = ''.join([f'{self.sourcer} "{f}"\n' for f in self.files]) + input
        cmd = [self.shell] + list(self.extra_args) + ['-ic', input]
        env = XSH.env
        denv = env.detype()
        if streaming:
            subprocess.check_call(cmd, env=denv)
            out = None
        else:
            out = subprocess.check_output(cmd, env=denv, stderr=subprocess.STDOUT)
            out = out.decode(encoding=env.get('XONSH_ENCODING'), errors=env.get('XONSH_ENCODING_ERRORS'))
            out = out.replace('\r\n', '\n')
        return out

    def __repr__(self):
        if False:
            print('Hello World!')
        return self.__class__.__name__ + '(' + ', '.join([f'{k}={v!r}' for (k, v) in sorted(self._input_kwargs().items())]) + ')'

    @staticmethod
    def _is_streaming(args):
        if False:
            for i in range(10):
                print('nop')
        'Test and modify args if --xonsh-stream is present.'
        if '--xonsh-nostream' not in args:
            return (args, True)
        args = list(args)
        args.remove('--xonsh-nostream')
        return (args, False)

class ForeignShellFunctionAlias(ForeignShellBaseAlias):
    """This class is responsible for calling foreign shell functions as if
    they were aliases. This does not currently support taking stdin.
    """
    INPUT = '{funcname} {args}\n'

    def __init__(self, funcname, shell, sourcer=None, files=(), extra_args=()):
        if False:
            while True:
                i = 10
        '\n        Parameters\n        ----------\n        funcname : str\n            function name\n        shell : str\n            Name or path to shell\n        sourcer : str or None, optional\n            Command to source foreign files with.\n        files : tuple of str, optional\n            Paths to source.\n        extra_args : tuple of str, optional\n            Additional command line options to pass into the shell.\n        '
        super().__init__(shell=shell, sourcer=sourcer, files=files, extra_args=extra_args)
        self.funcname = funcname

    def _input_kwargs(self):
        if False:
            print('Hello World!')
        inp = super()._input_kwargs()
        inp['funcname'] = self.funcname
        return inp

class ForeignShellExecAlias(ForeignShellBaseAlias):
    """Provides a callable alias for source code in a foreign shell."""
    INPUT = '{src} {args}\n'

    def __init__(self, src, shell, sourcer=None, files=(), extra_args=()):
        if False:
            while True:
                i = 10
        '\n        Parameters\n        ----------\n        src : str\n            Source code in the shell language\n        shell : str\n            Name or path to shell\n        sourcer : str or None, optional\n            Command to source foreign files with.\n        files : tuple of str, optional\n            Paths to source.\n        extra_args : tuple of str, optional\n            Additional command line options to pass into the shell.\n        '
        super().__init__(shell=shell, sourcer=sourcer, files=files, extra_args=extra_args)
        self.src = src.strip()

    def _input_kwargs(self):
        if False:
            return 10
        inp = super()._input_kwargs()
        inp['src'] = self.src
        return inp

@lazyobject
def VALID_SHELL_PARAMS():
    if False:
        for i in range(10):
            print('nop')
    return frozenset(['shell', 'interactive', 'login', 'envcmd', 'aliascmd', 'extra_args', 'currenv', 'safe', 'prevcmd', 'postcmd', 'funcscmd', 'sourcer'])

def ensure_shell(shell):
    if False:
        i = 10
        return i + 15
    'Ensures that a mapping follows the shell specification.'
    if not isinstance(shell, cabc.MutableMapping):
        shell = dict(shell)
    shell_keys = set(shell.keys())
    if not shell_keys <= VALID_SHELL_PARAMS:
        raise KeyError(f'unknown shell keys: {shell_keys - VALID_SHELL_PARAMS}')
    shell['shell'] = ensure_string(shell['shell']).lower()
    if 'interactive' in shell_keys:
        shell['interactive'] = to_bool(shell['interactive'])
    if 'login' in shell_keys:
        shell['login'] = to_bool(shell['login'])
    if 'envcmd' in shell_keys:
        shell['envcmd'] = None if shell['envcmd'] is None else ensure_string(shell['envcmd'])
    if 'aliascmd' in shell_keys:
        shell['aliascmd'] = None if shell['aliascmd'] is None else ensure_string(shell['aliascmd'])
    if 'extra_args' in shell_keys and (not isinstance(shell['extra_args'], tuple)):
        shell['extra_args'] = tuple(map(ensure_string, shell['extra_args']))
    if 'currenv' in shell_keys and (not isinstance(shell['currenv'], tuple)):
        ce = shell['currenv']
        if isinstance(ce, cabc.Mapping):
            ce = tuple(((ensure_string(k), v) for (k, v) in ce.items()))
        elif isinstance(ce, cabc.Sequence):
            ce = tuple(((ensure_string(k), v) for (k, v) in ce))
        else:
            raise RuntimeError('unrecognized type for currenv')
        shell['currenv'] = ce
    if 'safe' in shell_keys:
        shell['safe'] = to_bool(shell['safe'])
    if 'prevcmd' in shell_keys:
        shell['prevcmd'] = ensure_string(shell['prevcmd'])
    if 'postcmd' in shell_keys:
        shell['postcmd'] = ensure_string(shell['postcmd'])
    if 'funcscmd' in shell_keys:
        shell['funcscmd'] = None if shell['funcscmd'] is None else ensure_string(shell['funcscmd'])
    if 'sourcer' in shell_keys:
        shell['sourcer'] = None if shell['sourcer'] is None else ensure_string(shell['sourcer'])
    if 'seterrprevcmd' in shell_keys:
        shell['seterrprevcmd'] = None if shell['seterrprevcmd'] is None else ensure_string(shell['seterrprevcmd'])
    if 'seterrpostcmd' in shell_keys:
        shell['seterrpostcmd'] = None if shell['seterrpostcmd'] is None else ensure_string(shell['seterrpostcmd'])
    return shell

def load_foreign_envs(shells):
    if False:
        return 10
    'Loads environments from foreign shells.\n\n    Parameters\n    ----------\n    shells : sequence of dicts\n        An iterable of dicts that can be passed into foreign_shell_data() as\n        keyword arguments.\n\n    Returns\n    -------\n    env : dict\n        A dictionary of the merged environments.\n    '
    env = {}
    for shell in shells:
        shell = ensure_shell(shell)
        (shenv, _) = foreign_shell_data(**shell)
        if shenv:
            env.update(shenv)
    return env

def load_foreign_aliases(shells):
    if False:
        print('Hello World!')
    'Loads aliases from foreign shells.\n\n    Parameters\n    ----------\n    shells : sequence of dicts\n        An iterable of dicts that can be passed into foreign_shell_data() as\n        keyword arguments.\n\n    Returns\n    -------\n    aliases : dict\n        A dictionary of the merged aliases.\n    '
    aliases = {}
    xonsh_aliases = XSH.aliases
    for shell in shells:
        shell = ensure_shell(shell)
        (_, shaliases) = foreign_shell_data(**shell)
        if not XSH.env.get('FOREIGN_ALIASES_OVERRIDE'):
            shaliases = {} if shaliases is None else shaliases
            for alias in set(shaliases) & set(xonsh_aliases):
                del shaliases[alias]
                if XSH.env.get('XONSH_DEBUG') >= 1:
                    print(f"aliases: ignoring alias {alias!r} of shell {shell['shell']!r} which tries to override xonsh alias.", file=sys.stderr)
        aliases.update(shaliases)
    return aliases