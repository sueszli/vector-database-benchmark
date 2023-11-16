"""This module provides the implementation for the retrieving completion results
from bash.
"""
import functools
import os
import pathlib
import platform
import re
import shlex
import shutil
import subprocess
import sys
import typing as tp
__version__ = '0.2.7'

@functools.lru_cache(1)
def _git_for_windows_path():
    if False:
        i = 10
        return i + 15
    'Returns the path to git for windows, if available and None otherwise.'
    import winreg
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\GitForWindows')
        (gfwp, _) = winreg.QueryValueEx(key, 'InstallPath')
    except FileNotFoundError:
        gfwp = None
    return gfwp

@functools.lru_cache(1)
def _windows_bash_command(env=None):
    if False:
        i = 10
        return i + 15
    'Determines the command for Bash on windows.'
    wbc = 'bash'
    path = None if env is None else env.get('PATH', None)
    bash_on_path = shutil.which('bash', path=path)
    if bash_on_path:
        try:
            out = subprocess.check_output([bash_on_path, '--version'], stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError:
            bash_works = False
        else:
            bash_works = out and 'pc-linux-gnu' not in out.splitlines()[0]
        if bash_works:
            wbc = bash_on_path
        else:
            gfwp = _git_for_windows_path()
            if gfwp:
                bashcmd = os.path.join(gfwp, 'bin\\bash.exe')
                if os.path.isfile(bashcmd):
                    wbc = bashcmd
    return wbc

def _bash_command(env=None):
    if False:
        return 10
    'Determines the command for Bash on the current plaform.'
    if platform.system() == 'Windows':
        bc = _windows_bash_command(env=None)
    else:
        bc = 'bash'
    return bc

def _bash_completion_paths_default():
    if False:
        for i in range(10):
            print('nop')
    'A possibly empty tuple with default paths to Bash completions known for\n    the current platform.\n    '
    platform_sys = platform.system()
    if platform_sys == 'Linux' or sys.platform == 'cygwin':
        bcd = ('/usr/share/bash-completion/bash_completion',)
    elif platform_sys == 'Darwin':
        bcd = ('/usr/local/share/bash-completion/bash_completion', '/usr/local/etc/bash_completion')
    elif platform_sys == 'Windows':
        gfwp = _git_for_windows_path()
        if gfwp:
            bcd = (os.path.join(gfwp, 'usr\\share\\bash-completion\\bash_completion'), os.path.join(gfwp, 'mingw64\\share\\git\\completion\\git-completion.bash'))
        else:
            bcd = ()
    else:
        bcd = ()
    return bcd
_BASH_COMPLETIONS_PATHS_DEFAULT: tuple[str, ...] = ()

def _get_bash_completions_source(paths=None):
    if False:
        i = 10
        return i + 15
    global _BASH_COMPLETIONS_PATHS_DEFAULT
    if paths is None:
        if _BASH_COMPLETIONS_PATHS_DEFAULT is None:
            _BASH_COMPLETIONS_PATHS_DEFAULT = _bash_completion_paths_default()
        paths = _BASH_COMPLETIONS_PATHS_DEFAULT
    for path in map(pathlib.Path, paths):
        if path.is_file():
            return f'source "{path.as_posix()}"'
    return None

def _bash_get_sep():
    if False:
        i = 10
        return i + 15
    'Returns the appropriate filepath separator char depending on OS and\n    xonsh options set\n    '
    if platform.system() == 'Windows':
        return os.altsep
    else:
        return os.sep
_BASH_PATTERN_NEED_QUOTES: tp.Optional[tp.Pattern] = None

def _bash_pattern_need_quotes():
    if False:
        for i in range(10):
            print('nop')
    global _BASH_PATTERN_NEED_QUOTES
    if _BASH_PATTERN_NEED_QUOTES is not None:
        return _BASH_PATTERN_NEED_QUOTES
    pattern = '\\s`\\$\\{\\}\\,\\*\\(\\)"\\\'\\?&'
    if platform.system() == 'Windows':
        pattern += '%'
    pattern = '[' + pattern + ']' + '|\\band\\b|\\bor\\b'
    _BASH_PATTERN_NEED_QUOTES = re.compile(pattern)
    return _BASH_PATTERN_NEED_QUOTES

def _bash_expand_path(s):
    if False:
        i = 10
        return i + 15
    'Takes a string path and expands ~ to home and environment vars.'
    (pre, char, post) = s.partition('=')
    if char:
        s = os.path.expanduser(pre) + char
        s += os.pathsep.join(map(os.path.expanduser, post.split(os.pathsep)))
    else:
        s = os.path.expanduser(s)
    return s

def _bash_quote_to_use(x):
    if False:
        return 10
    single = "'"
    double = '"'
    if single in x and double not in x:
        return double
    else:
        return single

def _bash_quote_paths(paths, start, end):
    if False:
        return 10
    out = set()
    space = ' '
    backslash = '\\'
    double_backslash = '\\\\'
    slash = _bash_get_sep()
    orig_start = start
    orig_end = end
    need_quotes = any((re.search(_bash_pattern_need_quotes(), x) or (backslash in x and slash != backslash) for x in paths))
    for s in paths:
        start = orig_start
        end = orig_end
        if start == '' and need_quotes:
            start = end = _bash_quote_to_use(s)
        if os.path.isdir(_bash_expand_path(s)):
            _tail = slash
        elif end == '' and (not s.endswith('=')):
            _tail = space
        else:
            _tail = ''
        if start != '' and 'r' not in start and (backslash in s):
            start = 'r%s' % start
        s = s + _tail
        if end != '':
            if 'r' not in start.lower():
                s = s.replace(backslash, double_backslash)
            if s.endswith(backslash) and (not s.endswith(double_backslash)):
                s += backslash
        if end in s:
            s = s.replace(end, ''.join(('\\%s' % i for i in end)))
        out.add(start + s + end)
    return (out, need_quotes)
BASH_COMPLETE_SCRIPT = '\n{source}\n\n# Override some functions in bash-completion, do not quote for readline\nquote_readline()\n{{\n    echo "$1"\n}}\n\n_quote_readline_by_ref()\n{{\n    if [[ $1 == \\\'* || $1 == \\"* ]]; then\n        # Leave out first character\n        printf -v $2 %s "${{1:1}}"\n    else\n        printf -v $2 %s "$1"\n    fi\n\n    [[ ${{!2}} == \\$* ]] && eval $2=${{!2}}\n}}\n\n\nfunction _get_complete_statement {{\n    complete -p {cmd} 2> /dev/null || echo "-F _minimal"\n}}\n\nfunction getarg {{\n    find=$1\n    shift 1\n    prev=""\n    for i in $* ; do\n        if [ "$prev" = "$find" ] ; then\n            echo $i\n        fi\n        prev=$i\n    done\n}}\n\n_complete_stmt=$(_get_complete_statement)\nif echo "$_complete_stmt" | grep --quiet -e "_minimal"\nthen\n    declare -f _completion_loader > /dev/null && _completion_loader {cmd}\n    _complete_stmt=$(_get_complete_statement)\nfi\n\n# Is -C (subshell) or -F (function) completion used?\nif [[ $_complete_stmt =~ "-C" ]] ; then\n    _func=$(eval getarg "-C" $_complete_stmt)\nelse\n    _func=$(eval getarg "-F" $_complete_stmt)\n    declare -f "$_func" > /dev/null || exit 1\nfi\n\necho "$_complete_stmt"\nexport COMP_WORDS=({line})\nexport COMP_LINE={comp_line}\nexport COMP_POINT=${{#COMP_LINE}}\nexport COMP_COUNT={end}\nexport COMP_CWORD={n}\n$_func {cmd} {prefix} {prev}\n\n# print out completions, right-stripped if they contain no internal spaces\nshopt -s extglob\nfor ((i=0;i<${{#COMPREPLY[*]}};i++))\ndo\n    no_spaces="${{COMPREPLY[i]//[[:space:]]}}"\n    no_trailing_spaces="${{COMPREPLY[i]%%+([[:space:]])}}"\n    if [[ "$no_spaces" == "$no_trailing_spaces" ]]; then\n        echo "$no_trailing_spaces"\n    else\n        echo "${{COMPREPLY[i]}}"\n    fi\ndone\n'

def bash_completions(prefix, line, begidx, endidx, env=None, paths=None, command=None, quote_paths=_bash_quote_paths, line_args=None, opening_quote='', closing_quote='', arg_index=None, **kwargs):
    if False:
        return 10
    "Completes based on results from BASH completion.\n\n    Parameters\n    ----------\n    prefix : str\n        The string to match\n    line : str\n        The line that prefix appears on.\n    begidx : int\n        The index in line that prefix starts on.\n    endidx : int\n        The index in line that prefix ends on.\n    env : Mapping, optional\n        The environment dict to execute the Bash subprocess in.\n    paths : list or tuple of str or None, optional\n        This is a list (or tuple) of strings that specifies where the\n        ``bash_completion`` script may be found. The first valid path will\n        be used. For better performance, bash-completion v2.x is recommended\n        since it lazy-loads individual completion scripts. For both\n        bash-completion v1.x and v2.x, paths of individual completion scripts\n        (like ``.../completes/ssh``) do not need to be included here. The\n        default values are platform dependent, but reasonable.\n    command : str or None, optional\n        The /path/to/bash to use. If None, it will be selected based on the\n        from the environment and platform.\n    quote_paths : callable, optional\n        A functions that quotes file system paths. You shouldn't normally need\n        this as the default is acceptable 99+% of the time. This function should\n        return a set of the new paths and a boolean for whether the paths were\n        quoted.\n    line_args : list of str, optional\n        A list of the args in the current line to be used instead of ``line.split()``.\n        This is usefull with a space in an argument, e.g. ``ls 'a dir/'<TAB>``.\n    opening_quote : str, optional\n        The current argument's opening quote. This is passed to the `quote_paths` function.\n    closing_quote : str, optional\n        The closing quote that **should** be used. This is also passed to the `quote_paths` function.\n    arg_index : int, optional\n        The current prefix's index in the args.\n\n    Returns\n    -------\n    rtn : set of str\n        Possible completions of prefix\n    lprefix : int\n        Length of the prefix to be replaced in the completion.\n    "
    source = _get_bash_completions_source(paths) or ''
    if prefix.startswith('$'):
        return (set(), 0)
    splt = line_args or line.split()
    cmd = splt[0]
    cmd = os.path.basename(cmd)
    prev = ''
    if arg_index is not None:
        n = arg_index
        if arg_index > 0:
            prev = splt[arg_index - 1]
    else:
        idx = n = 0
        for (n, tok) in enumerate(splt):
            if tok == prefix:
                idx = line.find(prefix, idx)
                if idx >= begidx:
                    break
            prev = tok
        if len(prefix) == 0:
            n += 1
    prefix_quoted = shlex.quote(prefix)
    script = BASH_COMPLETE_SCRIPT.format(source=source, line=' '.join((shlex.quote(p) for p in splt if p)), comp_line=shlex.quote(line), n=n, cmd=shlex.quote(cmd), end=endidx + 1, prefix=prefix_quoted, prev=shlex.quote(prev))
    if command is None:
        command = _bash_command(env=env)
    try:
        out = subprocess.check_output([command, '-c', script], text=True, stderr=subprocess.PIPE, env=env)
        if not out:
            raise ValueError
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return (set(), 0)
    out = out.splitlines()
    complete_stmt = out[0]
    out = set(out[1:])
    commprefix = os.path.commonprefix(list(out))
    if prefix.startswith('~') and commprefix and (prefix not in commprefix):
        home_ = os.path.expanduser('~')
        out = {f'~/{os.path.relpath(p, home_)}' for p in out}
        commprefix = f'~/{os.path.relpath(commprefix, home_)}'
    strip_len = 0
    strip_prefix = prefix.strip('"\'')
    while strip_len < len(strip_prefix) and strip_len < len(commprefix):
        if commprefix[strip_len] == strip_prefix[strip_len]:
            break
        strip_len += 1
    if '-o noquote' not in complete_stmt:
        (out, need_quotes) = quote_paths(out, opening_quote, closing_quote)
    if '-o nospace' in complete_stmt:
        out = {x.rstrip() for x in out}
    if '=' in prefix and '=' not in commprefix:
        strip_len = prefix.index('=') + 1
    elif ':' in prefix and ':' not in commprefix:
        strip_len = prefix.index(':') + 1
    return (out, max(len(prefix) - strip_len, 0))

def bash_complete_line(line, return_line=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Provides the completion from the end of the line.\n\n    Parameters\n    ----------\n    line : str\n        Line to complete\n    return_line : bool, optional\n        If true (default), will return the entire line, with the completion added.\n        If false, this will instead return the strings to append to the original line.\n    kwargs : optional\n        All other keyword arguments are passed to the bash_completions() function.\n\n    Returns\n    -------\n    rtn : set of str\n        Possible completions of prefix\n    '
    split = line.split()
    if len(split) > 1 and (not line.endswith(' ')):
        prefix = split[-1]
        begidx = len(line.rsplit(prefix)[0])
    else:
        prefix = ''
        begidx = len(line)
    endidx = len(line)
    (out, lprefix) = bash_completions(prefix, line, begidx, endidx, **kwargs)
    if return_line:
        preline = line[:-lprefix]
        rtn = {preline + o for o in out}
    else:
        rtn = {o[lprefix:] for o in out}
    return rtn

def _bc_main(args=None):
    if False:
        i = 10
        return i + 15
    'Runs complete_line() and prints the output.'
    from argparse import ArgumentParser
    p = ArgumentParser('bash_completions')
    p.add_argument('--return-line', action='store_true', dest='return_line', default=True, help='will return the entire line, with the completion added')
    p.add_argument('--no-return-line', action='store_false', dest='return_line', help='will instead return the strings to append to the original line')
    p.add_argument('line', help='line to complete')
    ns = p.parse_args(args=args)
    out = bash_complete_line(ns.line, return_line=ns.return_line)
    for o in sorted(out):
        print(o)
if __name__ == '__main__':
    _bc_main()