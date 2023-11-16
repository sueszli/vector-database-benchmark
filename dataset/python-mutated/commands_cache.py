"""Module for caching command & alias names as well as for predicting whether
a command will be able to be run in the background.

A background predictor is a function that accepts a single argument list
and returns whether or not the process can be run in the background (returns
True) or must be run the foreground (returns False).
"""
import argparse
import collections.abc as cabc
import os
import pickle
import time
import typing as tp
from pathlib import Path
from xonsh.lazyasd import lazyobject
from xonsh.platform import ON_POSIX, ON_WINDOWS, pathbasename
from xonsh.tools import executables_in

class _Commands(tp.NamedTuple):
    mtime: float
    cmds: 'tuple[str, ...]'

class CommandsCache(cabc.Mapping):
    """A lazy cache representing the commands available on the file system.
    The keys are the command names and the values a tuple of (loc, has_alias)
    where loc is either a str pointing to the executable on the file system or
    None (if no executable exists) and has_alias is a boolean flag for whether
    the command has an alias.
    """
    CACHE_FILE = 'path-commands-cache.pickle'

    def __init__(self, env, aliases=None) -> None:
        if False:
            return 10
        self._paths_cache: dict[str, _Commands] = {}
        self._cmds_cache: dict[str, tuple[str, bool | None]] = {}
        self._alias_checksum: int | None = None
        self.threadable_predictors = default_threadable_predictors()
        self.env = env
        if aliases is None:
            from xonsh.aliases import Aliases, make_default_aliases
            self.aliases = Aliases(make_default_aliases())
        else:
            self.aliases = aliases
        self._cache_file = None

    @property
    def cache_file(self):
        if False:
            return 10
        'Keeping a property that lies on instance-attribute'
        env = self.env
        if self._cache_file is None:
            if 'XONSH_CACHE_DIR' in env and env.get('COMMANDS_CACHE_SAVE_INTERMEDIATE'):
                self._cache_file = Path(env['XONSH_CACHE_DIR']).joinpath(self.CACHE_FILE).resolve()
            else:
                self._cache_file = ''
        return self._cache_file

    def __contains__(self, key):
        if False:
            for i in range(10):
                print('nop')
        self.update_cache()
        return self.lazyin(key)

    def __iter__(self):
        if False:
            print('Hello World!')
        for (cmd, _) in self.iter_commands():
            yield cmd

    def iter_commands(self):
        if False:
            print('Hello World!')
        'Wrapper for handling windows path behaviour'
        for (cmd, (path, is_alias)) in self.all_commands.items():
            if ON_WINDOWS and path is not None:
                cmd = pathbasename(path)
            yield (cmd, (path, is_alias))

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.all_commands)

    def __getitem__(self, key) -> 'tuple[str, bool]':
        if False:
            for i in range(10):
                print('nop')
        self.update_cache()
        return self.lazyget(key)

    def is_empty(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns whether the cache is populated or not.'
        return len(self._cmds_cache) == 0

    def get_possible_names(self, name):
        if False:
            i = 10
            return i + 15
        'Generates the possible `PATHEXT` extension variants of a given executable\n        name on Windows as a list, conserving the ordering in `PATHEXT`.\n        Returns a list as `name` being the only item in it on other platforms.'
        if ON_WINDOWS:
            pathext = [''] + self.env.get('PATHEXT', [])
            name = name.upper()
            return [name + ext for ext in pathext]
        else:
            return [name]

    @staticmethod
    def remove_dups(paths):
        if False:
            print('Hello World!')
        cont = set()
        for p in map(os.path.realpath, paths):
            if p not in cont:
                cont.add(p)
                if os.path.isdir(p):
                    yield p

    def _check_changes(self, paths: tuple[str, ...]):
        if False:
            print('Hello World!')
        yield self._update_paths_cache(paths)
        al_hash = hash(frozenset(self.aliases))
        yield (al_hash != self._alias_checksum)
        self._alias_checksum = al_hash

    @property
    def all_commands(self):
        if False:
            print('Hello World!')
        self.update_cache()
        return self._cmds_cache

    def update_cache(self):
        if False:
            while True:
                i = 10
        env = self.env
        paths = tuple(reversed(tuple(self.remove_dups(env.get('PATH') or []))))
        if any(self._check_changes(paths)):
            all_cmds = {}
            for (cmd, path) in self._iter_binaries(paths):
                key = cmd.upper() if ON_WINDOWS else cmd
                all_cmds[key] = (path, None)
            for cmd in self.aliases:
                possibilities = self.get_possible_names(cmd)
                override_key = next((possible for possible in possibilities if possible in all_cmds), None)
                if override_key:
                    all_cmds[override_key] = (all_cmds[override_key][0], False)
                else:
                    key = cmd.upper() if ON_WINDOWS else cmd
                    all_cmds[key] = (cmd, True)
            self._cmds_cache = all_cmds
        return self._cmds_cache

    def _update_paths_cache(self, paths: tp.Sequence[str]) -> bool:
        if False:
            return 10
        'load cached results or update cache'
        if not self._paths_cache and self.cache_file and self.cache_file.exists():
            try:
                self._paths_cache = pickle.loads(self.cache_file.read_bytes()) or {}
            except Exception:
                self.cache_file.unlink(missing_ok=True)
        updated = False
        for path in paths:
            modified_time = os.path.getmtime(path)
            if not self.env.get('ENABLE_COMMANDS_CACHE', True) or path not in self._paths_cache or self._paths_cache[path].mtime != modified_time:
                updated = True
                self._paths_cache[path] = _Commands(modified_time, tuple(executables_in(path)))
        if updated and self.cache_file:
            self.cache_file.write_bytes(pickle.dumps(self._paths_cache))
        return updated

    def _iter_binaries(self, paths):
        if False:
            i = 10
            return i + 15
        for path in paths:
            for cmd in self._paths_cache[path].cmds:
                yield (cmd, os.path.join(path, cmd))

    def cached_name(self, name):
        if False:
            i = 10
            return i + 15
        'Returns the name that would appear in the cache, if it exists.'
        if name is None:
            return None
        cached = pathbasename(name) if os.pathsep in name else name
        if ON_WINDOWS:
            keys = self.get_possible_names(cached)
            cached = next((k for k in keys if k in self._cmds_cache), None)
        return cached

    def lazyin(self, key):
        if False:
            print('Hello World!')
        'Checks if the value is in the current cache without the potential to\n        update the cache. It just says whether the value is known *now*. This\n        may not reflect precisely what is on the $PATH.\n        '
        return self.cached_name(key) in self._cmds_cache

    def lazyiter(self):
        if False:
            print('Hello World!')
        'Returns an iterator over the current cache contents without the\n        potential to update the cache. This may not reflect what is on the\n        $PATH.\n        '
        return iter(self._cmds_cache)

    def lazylen(self):
        if False:
            print('Hello World!')
        'Returns the length of the current cache contents without the\n        potential to update the cache. This may not reflect precisely\n        what is on the $PATH.\n        '
        return len(self._cmds_cache)

    def lazyget(self, key, default=None):
        if False:
            print('Hello World!')
        'A lazy value getter.'
        return self._cmds_cache.get(self.cached_name(key), default)

    def locate_binary(self, name, ignore_alias=False):
        if False:
            for i in range(10):
                print('nop')
        'Locates an executable on the file system using the cache.\n\n        Parameters\n        ----------\n        name : str\n            name of binary to search for\n        ignore_alias : bool, optional\n            Force return of binary path even if alias of ``name`` exists\n            (default ``False``)\n        '
        self.update_cache()
        return self.lazy_locate_binary(name, ignore_alias)

    def lazy_locate_binary(self, name, ignore_alias=False):
        if False:
            i = 10
            return i + 15
        'Locates an executable in the cache, without checking its validity.\n\n        Parameters\n        ----------\n        name : str\n            name of binary to search for\n        ignore_alias : bool, optional\n            Force return of binary path even if alias of ``name`` exists\n            (default ``False``)\n        '
        possibilities = self.get_possible_names(name)
        if ON_WINDOWS:
            local_bin = next((fn for fn in possibilities if os.path.isfile(fn)), None)
            if local_bin:
                return os.path.abspath(local_bin)
        cached = next((cmd for cmd in possibilities if cmd in self._cmds_cache), None)
        if cached:
            (path, alias) = self._cmds_cache[cached]
            ispure = path == pathbasename(path)
            if alias and ignore_alias and ispure:
                return None
            else:
                return path
        elif os.path.isfile(name) and name != pathbasename(name):
            return name

    def is_only_functional_alias(self, name):
        if False:
            i = 10
            return i + 15
        'Returns whether or not a command is only a functional alias, and has\n        no underlying executable. For example, the "cd" command is only available\n        as a functional alias.\n        '
        self.update_cache()
        return self.lazy_is_only_functional_alias(name)

    def lazy_is_only_functional_alias(self, name) -> bool:
        if False:
            return 10
        'Returns whether or not a command is only a functional alias, and has\n        no underlying executable. For example, the "cd" command is only available\n        as a functional alias. This search is performed lazily.\n        '
        val = self._cmds_cache.get(name, None)
        if val is None:
            return False
        return val == (name, True) and self.locate_binary(name, ignore_alias=True) is None

    def predict_threadable(self, cmd):
        if False:
            print('Hello World!')
        'Predicts whether a command list is able to be run on a background\n        thread, rather than the main thread.\n        '
        predictor = self.get_predictor_threadable(cmd[0])
        return predictor(cmd[1:], self)

    def get_predictor_threadable(self, cmd0):
        if False:
            for i in range(10):
                print('nop')
        'Return the predictor whether a command list is able to be run on a\n        background thread, rather than the main thread.\n        '
        name = self.cached_name(cmd0)
        predictors = self.threadable_predictors
        if ON_WINDOWS:
            (path, _) = self.lazyget(name, (None, None))
            if path is None:
                return predict_true
            else:
                name = pathbasename(path)
            if name not in predictors:
                (pre, ext) = os.path.splitext(name)
                if pre in predictors:
                    predictors[name] = predictors[pre]
        if name not in predictors:
            predictors[name] = self.default_predictor(name, cmd0)
        predictor = predictors[name]
        return predictor

    def default_predictor(self, name, cmd0):
        if False:
            i = 10
            return i + 15
        'Default predictor, using predictor from original command if the\n        command is an alias, elseif build a predictor based on binary analysis\n        on POSIX, else return predict_true.\n        '
        if not os.path.isabs(cmd0) and os.sep not in cmd0:
            if cmd0 in self.aliases:
                return self.default_predictor_alias(cmd0)
        if ON_POSIX:
            return self.default_predictor_readbin(name, cmd0, timeout=0.1, failure=predict_true)
        else:
            return predict_true

    def default_predictor_alias(self, cmd0):
        if False:
            print('Hello World!')
        alias_recursion_limit = 10
        first_args = []
        while cmd0 in self.aliases:
            alias_name = self.aliases
            if isinstance(alias_name, (str, bytes)) or not isinstance(alias_name, cabc.Sequence):
                return predict_true
            for arg in alias_name[:0:-1]:
                first_args.insert(0, arg)
            if cmd0 == alias_name[0]:
                return predict_true
            cmd0 = alias_name[0]
            alias_recursion_limit -= 1
            if alias_recursion_limit == 0:
                return predict_true
        predictor_cmd0 = self.get_predictor_threadable(cmd0)
        return lambda cmd1: predictor_cmd0(first_args[::-1] + cmd1, self)

    def default_predictor_readbin(self, name, cmd0, timeout, failure):
        if False:
            for i in range(10):
                print('nop')
        'Make a default predictor by\n        analyzing the content of the binary. Should only works on POSIX.\n        Return failure if the analysis fails.\n        '
        fname = cmd0 if os.path.isabs(cmd0) else None
        fname = cmd0 if fname is None and os.sep in cmd0 else fname
        fname = self.lazy_locate_binary(name) if fname is None else fname
        if fname is None:
            return failure
        if not os.path.isfile(fname):
            return failure
        try:
            fd = os.open(fname, os.O_RDONLY | os.O_NONBLOCK)
        except Exception:
            return failure
        search_for = {(b'ncurses',): [False], (b'libgpm',): [False], (b'isatty', b'tcgetattr', b'tcsetattr'): [False, False, False]}
        tstart = time.time()
        block = b''
        while time.time() < tstart + timeout:
            previous_block = block
            try:
                block = os.read(fd, 2048)
            except Exception:
                os.close(fd)
                return failure
            if len(block) == 0:
                os.close(fd)
                return predict_true
            analyzed_block = previous_block + block
            for (k, v) in search_for.items():
                for i in range(len(k)):
                    if v[i]:
                        continue
                    if k[i] in analyzed_block:
                        v[i] = True
                if all(v):
                    os.close(fd)
                    return predict_false
        os.close(fd)
        return failure

def predict_true(_, __):
    if False:
        for i in range(10):
            print('nop')
    'Always say the process is threadable.'
    return True

def predict_false(_, __):
    if False:
        while True:
            i = 10
    'Never say the process is threadable.'
    return False

@lazyobject
def SHELL_PREDICTOR_PARSER():
    if False:
        print('Hello World!')
    p = argparse.ArgumentParser('shell', add_help=False)
    p.add_argument('-c', nargs='?', default=None)
    p.add_argument('filename', nargs='?', default=None)
    return p

def predict_shell(args, _):
    if False:
        return 10
    'Predict the backgroundability of the normal shell interface, which\n    comes down to whether it is being run in subproc mode.\n    '
    (ns, _) = SHELL_PREDICTOR_PARSER.parse_known_args(args)
    if ns.c is None and ns.filename is None:
        pred = False
    else:
        pred = True
    return pred

@lazyobject
def HELP_VER_PREDICTOR_PARSER():
    if False:
        while True:
            i = 10
    p = argparse.ArgumentParser('cmd', add_help=False)
    p.add_argument('-h', '--help', dest='help', nargs='?', action='store', default=None)
    p.add_argument('-v', '-V', '--version', dest='version', nargs='?', action='store', default=None)
    return p

def predict_help_ver(args, _):
    if False:
        i = 10
        return i + 15
    'Predict the backgroundability of commands that have help & version\n    switches: -h, --help, -v, -V, --version. If either of these options is\n    present, the command is assumed to print to stdout normally and is therefore\n    threadable. Otherwise, the command is assumed to not be threadable.\n    This is useful for commands, like top, that normally enter alternate mode\n    but may not in certain circumstances.\n    '
    (ns, _) = HELP_VER_PREDICTOR_PARSER.parse_known_args(args)
    pred = ns.help is not None or ns.version is not None
    return pred

@lazyobject
def HG_PREDICTOR_PARSER():
    if False:
        return 10
    p = argparse.ArgumentParser('hg', add_help=False)
    p.add_argument('command')
    p.add_argument('-i', '--interactive', action='store_true', default=False, dest='interactive')
    return p

def predict_hg(args, _):
    if False:
        return 10
    "Predict if mercurial is about to be run in interactive mode.\n    If it is interactive, predict False. If it isn't, predict True.\n    Also predict False for certain commands, such as split.\n    "
    (ns, _) = HG_PREDICTOR_PARSER.parse_known_args(args)
    if ns.command == 'split':
        return False
    else:
        return not ns.interactive

def predict_env(args, cmd_cache: CommandsCache):
    if False:
        i = 10
        return i + 15
    'Predict if env is launching a threadable command or not.\n    The launched command is extracted from env args, and the predictor of\n    lauched command is used.'
    for i in range(len(args)):
        if args[i] and args[i][0] != '-' and ('=' not in args[i]):
            return cmd_cache.predict_threadable(args[i:])
    return True

def default_threadable_predictors():
    if False:
        while True:
            i = 10
    'Generates a new defaultdict for known threadable predictors.\n    The default is to predict true.\n    '
    predictors = {'asciinema': predict_help_ver, 'aurman': predict_false, 'awk': predict_true, 'bash': predict_shell, 'cat': predict_false, 'clear': predict_false, 'cls': predict_false, 'cmd': predict_shell, 'cryptop': predict_false, 'cryptsetup': predict_true, 'csh': predict_shell, 'curl': predict_true, 'elvish': predict_shell, 'emacsclient': predict_false, 'env': predict_env, 'ex': predict_false, 'fish': predict_shell, 'gawk': predict_true, 'ghci': predict_help_ver, 'git': predict_true, 'gvim': predict_help_ver, 'hg': predict_hg, 'htop': predict_help_ver, 'ipython': predict_shell, 'julia': predict_shell, 'ksh': predict_shell, 'less': predict_help_ver, 'ls': predict_true, 'man': predict_help_ver, 'mc': predict_false, 'more': predict_help_ver, 'mutt': predict_help_ver, 'mvim': predict_help_ver, 'nano': predict_help_ver, 'nmcli': predict_true, 'nvim': predict_false, 'percol': predict_false, 'ponysay': predict_help_ver, 'psql': predict_false, 'push': predict_shell, 'pv': predict_false, 'python': predict_shell, 'python2': predict_shell, 'python3': predict_shell, 'ranger': predict_help_ver, 'repo': predict_help_ver, 'rview': predict_false, 'rvim': predict_false, 'rwt': predict_shell, 'scp': predict_false, 'sh': predict_shell, 'ssh': predict_false, 'startx': predict_false, 'sudo': predict_help_ver, 'sudoedit': predict_help_ver, 'systemctl': predict_true, 'tcsh': predict_shell, 'telnet': predict_false, 'top': predict_help_ver, 'tput': predict_false, 'udisksctl': predict_true, 'unzip': predict_true, 'vi': predict_false, 'view': predict_false, 'vim': predict_false, 'vimpager': predict_help_ver, 'weechat': predict_help_ver, 'wget': predict_true, 'xclip': predict_help_ver, 'xdg-open': predict_false, 'xo': predict_help_ver, 'xon.sh': predict_shell, 'xonsh': predict_shell, 'yes': predict_false, 'zip': predict_true, 'zipinfo': predict_true, 'zsh': predict_shell}
    return predictors