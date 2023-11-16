"""Directory stack and associated utilities for the xonsh shell.
https://www.gnu.org/software/bash/manual/html_node/Directory-Stack-Builtins.html
"""
import contextlib
import glob
import os
import subprocess
import typing as tp
from xonsh.built_ins import XSH
from xonsh.cli_utils import Annotated, Arg, ArgParserAlias
from xonsh.events import events
from xonsh.platform import ON_WINDOWS
from xonsh.tools import get_sep
DIRSTACK: list[str] = []
'A list containing the currently remembered directories.'
_unc_tempDrives: dict[str, str] = {}
' drive: sharePath for temp drive letters we create for UNC mapping'

@contextlib.contextmanager
def _win_reg_key(*paths, **kwargs):
    if False:
        print('Hello World!')
    import winreg
    key = winreg.OpenKey(*paths, **kwargs)
    yield key
    winreg.CloseKey(key)

def _query_win_reg_key(*paths):
    if False:
        i = 10
        return i + 15
    import winreg
    (*paths, name) = paths
    with contextlib.suppress(OSError):
        with _win_reg_key(*paths) as key:
            (wval, wtype) = winreg.QueryValueEx(key, name)
            return wval

@tp.no_type_check
def _unc_check_enabled() -> bool:
    if False:
        i = 10
        return i + 15
    'Check whether CMD.EXE is enforcing no-UNC-as-working-directory check.\n\n    Check can be disabled by setting {HKCU, HKLM}/SOFTWARE\\Microsoft\\Command Processor\\DisableUNCCheck:REG_DWORD=1\n\n    Returns:\n        True if `CMD.EXE` is enforcing the check (default Windows situation)\n        False if check is explicitly disabled.\n    '
    if not ON_WINDOWS:
        return False
    import winreg
    wval = _query_win_reg_key(winreg.HKEY_CURRENT_USER, 'software\\microsoft\\command processor', 'DisableUNCCheck')
    if wval is None:
        wval = _query_win_reg_key(winreg.HKEY_LOCAL_MACHINE, 'software\\microsoft\\command processor', 'DisableUNCCheck')
    return False if wval else True

def _is_unc_path(some_path) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'True if path starts with 2 backward (or forward, due to python path hacking) slashes.'
    return len(some_path) > 1 and some_path[0] == some_path[1] and (some_path[0] in (os.sep, os.altsep))

def _unc_map_temp_drive(unc_path) -> str:
    if False:
        i = 10
        return i + 15
    'Map a new temporary drive letter for each distinct share,\n    unless `CMD.EXE` is not insisting on non-UNC working directory.\n\n    Emulating behavior of `CMD.EXE` `pushd`, create a new mapped drive (starting from Z: towards A:, skipping existing\n     drive letters) for each new UNC path user selects.\n\n    Args:\n        unc_path: the path specified by user.  Assumed to be a UNC path of form \\\\<server>\\share...\n\n    Returns:\n        a replacement for `unc_path` to be used as the actual new working directory.\n        Note that the drive letter may be a the same as one already mapped if the server and share portion of `unc_path`\n         is the same as one still active on the stack.\n    '
    global _unc_tempDrives
    assert unc_path[1] in (os.sep, os.altsep), 'unc_path is UNC form of path'
    if not _unc_check_enabled():
        return unc_path
    (unc_share, rem_path) = os.path.splitdrive(unc_path)
    unc_share = unc_share.casefold()
    for d in _unc_tempDrives:
        if _unc_tempDrives[d] == unc_share:
            return os.path.join(d, rem_path)
    for dord in range(ord('z'), ord('a'), -1):
        d = chr(dord) + ':'
        if not os.path.isdir(d):
            subprocess.check_output(['NET', 'USE', d, unc_share], text=True)
            _unc_tempDrives[d] = unc_share
            return os.path.join(d, rem_path)
    raise RuntimeError(f'Failed to find a drive for UNC Path({unc_path})')

def _unc_unmap_temp_drive(left_drive, cwd):
    if False:
        while True:
            i = 10
    'Unmap a temporary drive letter if it is no longer needed.\n    Called after popping `DIRSTACK` and changing to new working directory, so we need stack *and*\n    new current working directory to be sure drive letter no longer needed.\n\n    Args:\n        left_drive: driveletter (and colon) of working directory we just left\n        cwd: full path of new current working directory\n    '
    global _unc_tempDrives
    if left_drive not in _unc_tempDrives:
        return
    for p in DIRSTACK + [cwd]:
        if p.casefold().startswith(left_drive):
            return
    _unc_tempDrives.pop(left_drive)
    subprocess.check_output(['NET', 'USE', left_drive, '/delete'], text=True)
events.doc('on_chdir', '\non_chdir(olddir: str, newdir: str) -> None\n\nFires when the current directory is changed for any reason.\n')

def _get_cwd():
    if False:
        i = 10
        return i + 15
    try:
        return os.getcwd()
    except OSError:
        return None

def _change_working_directory(newdir, follow_symlinks=False):
    if False:
        for i in range(10):
            print('nop')
    env = XSH.env
    old = env['PWD']
    new = os.path.join(old, newdir)
    if follow_symlinks:
        new = os.path.realpath(new)
    absnew = os.path.abspath(new)
    try:
        os.chdir(absnew)
    except OSError:
        if new.endswith(get_sep()):
            new = new[:-1]
        if os.path.basename(new) == '..':
            env['PWD'] = new
    else:
        if old is not None:
            env['OLDPWD'] = old
        if new is not None:
            env['PWD'] = absnew
    if old != env['PWD']:
        events.on_chdir.fire(olddir=old, newdir=env['PWD'])

def _try_cdpath(apath):
    if False:
        for i in range(10):
            print('nop')
    env = XSH.env
    cdpaths = env.get('CDPATH')
    for cdp in cdpaths:
        globber = XSH.expand_path(os.path.join(cdp, apath))
        for cdpath_prefixed_path in glob.iglob(globber):
            return cdpath_prefixed_path
    return apath

def cd(args, stdin=None):
    if False:
        for i in range(10):
            print('nop')
    "Changes the directory.\n\n    If no directory is specified (i.e. if `args` is None) then this\n    changes to the current user's home directory.\n    "
    env = XSH.env
    oldpwd = env.get('OLDPWD', None)
    cwd = env['PWD']
    follow_symlinks = False
    if len(args) > 0 and args[0] == '-P':
        follow_symlinks = True
        del args[0]
    if len(args) == 0:
        d = env.get('HOME', os.path.expanduser('~'))
    elif len(args) == 1:
        d = os.path.expanduser(args[0])
        if not os.path.isdir(d):
            if d == '-':
                if oldpwd is not None:
                    d = oldpwd
                else:
                    return ('', 'cd: no previous directory stored\n', 1)
            elif d.startswith('-'):
                try:
                    num = int(d[1:])
                except ValueError:
                    return ('', f'cd: Invalid destination: {d}\n', 1)
                if num == 0:
                    return (None, None, 0)
                elif num < 0:
                    return ('', f'cd: Invalid destination: {d}\n', 1)
                elif num > len(DIRSTACK):
                    e = 'cd: Too few elements in dirstack ({0} elements)\n'
                    return ('', e.format(len(DIRSTACK)), 1)
                else:
                    d = DIRSTACK[num - 1]
            else:
                d = _try_cdpath(d)
    else:
        return ('', f'cd takes 0 or 1 arguments, not {len(args)}. An additional `-P` flag can be passed in first position to follow symlinks.\n', 1)
    if not os.path.exists(d):
        return ('', f'cd: no such file or directory: {d}\n', 1)
    if not os.path.isdir(d):
        return ('', f'cd: {d} is not a directory\n', 1)
    if not os.access(d, os.X_OK):
        return ('', f'cd: permission denied: {d}\n', 1)
    if ON_WINDOWS and _is_unc_path(d) and _unc_check_enabled() and (not env.get('AUTO_PUSHD')):
        return ('', "cd: can't cd to UNC path on Windows, unless $AUTO_PUSHD set or reg entry " + 'HKCU\\SOFTWARE\\MICROSOFT\\Command Processor\\DisableUNCCheck:DWORD = 1' + '\n', 1)
    if cwd is not None and env.get('AUTO_PUSHD'):
        pushd(['-n', '-q', cwd])
        if ON_WINDOWS and _is_unc_path(d):
            d = _unc_map_temp_drive(d)
    _change_working_directory(d, follow_symlinks)
    return (None, None, 0)

def pushd_fn(dir_or_n: Annotated[tp.Optional[str], Arg(metavar='+N|-N|dir', nargs='?')]=None, cd=True, quiet=False):
    if False:
        print('Hello World!')
    'Adds a directory to the top of the directory stack, or rotates the stack,\n    making the new top of the stack the current working directory.\n\n    On Windows, if the path is a UNC path (begins with `\\\\<server>\\<share>`) and if the `DisableUNCCheck` registry\n    value is not enabled, creates a temporary mapped drive letter and sets the working directory there, emulating\n    behavior of `PUSHD` in `CMD.EXE`\n\n    Parameters\n    ----------\n    dir_or_n\n        * dir :\n            Makes dir be the top of the stack,\n            making it the new current directory as if it had been supplied as an argument to the cd builtin.\n        * +N :\n            Brings the Nth directory (counting from the left of the list printed by dirs, starting with zero)\n            to the top of the list by rotating the stack.\n        * -N :\n            Brings the Nth directory (counting from the right of the list printed by dirs, starting with zero)\n            to the top of the list by rotating the stack.\n    cd : -n, --cd\n        Suppresses the normal change of directory when adding directories to the stack,\n        so that only the stack is manipulated.\n    quiet : -q, --quiet\n        Do not call dirs, regardless of $PUSHD_SILENT\n    '
    global DIRSTACK
    env = XSH.env
    pwd = env['PWD']
    if env.get('PUSHD_MINUS', False):
        BACKWARD = '-'
        FORWARD = '+'
    else:
        BACKWARD = '+'
        FORWARD = '-'
    if dir_or_n is None:
        try:
            new_pwd: tp.Optional[str] = DIRSTACK.pop(0)
        except IndexError:
            e = 'pushd: Directory stack is empty\n'
            return (None, e, 1)
    elif os.path.isdir(dir_or_n):
        new_pwd = dir_or_n
    else:
        try:
            num = int(dir_or_n[1:])
        except ValueError:
            e = 'Invalid argument to pushd: {0}\n'
            return (None, e.format(dir_or_n), 1)
        if num < 0:
            e = 'Invalid argument to pushd: {0}\n'
            return (None, e.format(dir_or_n), 1)
        if num > len(DIRSTACK):
            e = 'Too few elements in dirstack ({0} elements)\n'
            return (None, e.format(len(DIRSTACK)), 1)
        elif dir_or_n.startswith(FORWARD):
            if num == len(DIRSTACK):
                new_pwd = None
            else:
                new_pwd = DIRSTACK.pop(len(DIRSTACK) - 1 - num)
        elif dir_or_n.startswith(BACKWARD):
            if num == 0:
                new_pwd = None
            else:
                new_pwd = DIRSTACK.pop(num - 1)
        else:
            e = 'Invalid argument to pushd: {0}\n'
            return (None, e.format(dir_or_n), 1)
    if new_pwd is not None:
        if ON_WINDOWS and _is_unc_path(new_pwd):
            new_pwd = _unc_map_temp_drive(new_pwd)
        if cd:
            DIRSTACK.insert(0, os.path.expanduser(pwd))
            _change_working_directory(new_pwd)
        else:
            DIRSTACK.insert(0, os.path.expanduser(new_pwd))
    maxsize = env.get('DIRSTACK_SIZE')
    if len(DIRSTACK) > maxsize:
        DIRSTACK = DIRSTACK[:maxsize]
    if not quiet and (not env.get('PUSHD_SILENT')):
        return dirs([], None)
    return (None, None, 0)
pushd = ArgParserAlias(func=pushd_fn, has_args=True, prog='pushd')

def popd_fn(nth: Annotated[tp.Optional[str], Arg(metavar='+N|-N', nargs='?')]=None, cd=True, quiet=False):
    if False:
        for i in range(10):
            print('nop')
    'When no arguments are given, popd removes the top directory from the stack\n    and performs a cd to the new top directory.\n    The elements are numbered from 0 starting at the first directory listed with ``dirs``;\n    that is, popd is equivalent to popd +0.\n\n    Parameters\n    ----------\n    cd : -n, --cd\n        Suppresses the normal change of directory when removing directories from the stack,\n        so that only the stack is manipulated.\n    nth\n        Removes the Nth directory (counting from the left/right of the list printed by dirs w.r.t. -/+ prefix),\n        starting with zero.\n    quiet : -q, --quiet\n        Do not call dirs, regardless of $PUSHD_SILENT\n    '
    global DIRSTACK
    env = XSH.env
    if env.get('PUSHD_MINUS'):
        BACKWARD = '-'
        FORWARD = '+'
    else:
        BACKWARD = '-'
        FORWARD = '+'
    new_pwd: tp.Optional[str] = None
    if nth is None:
        try:
            new_pwd = DIRSTACK.pop(0)
        except IndexError:
            e = 'popd: Directory stack is empty\n'
            return (None, e, 1)
    else:
        try:
            num = int(nth[1:])
        except ValueError:
            e = 'Invalid argument to popd: {0}\n'
            return (None, e.format(nth), 1)
        if num < 0:
            e = 'Invalid argument to popd: {0}\n'
            return (None, e.format(nth), 1)
        if num > len(DIRSTACK):
            e = 'Too few elements in dirstack ({0} elements)\n'
            return (None, e.format(len(DIRSTACK)), 1)
        elif nth.startswith(FORWARD):
            if num == len(DIRSTACK):
                new_pwd = DIRSTACK.pop(0)
            else:
                DIRSTACK.pop(len(DIRSTACK) - 1 - num)
        elif nth.startswith(BACKWARD):
            if num == 0:
                new_pwd = DIRSTACK.pop(0)
            else:
                DIRSTACK.pop(num - 1)
        else:
            e = 'Invalid argument to popd: {0}\n'
            return (None, e.format(nth), 1)
    if new_pwd is not None:
        if cd:
            env = XSH.env
            pwd = env['PWD']
            _change_working_directory(new_pwd)
            if ON_WINDOWS:
                (drive, rem_path) = os.path.splitdrive(pwd)
                _unc_unmap_temp_drive(drive.casefold(), new_pwd)
    if not quiet and (not env.get('PUSHD_SILENT')):
        return dirs([], None)
    return (None, None, 0)
popd = ArgParserAlias(func=popd_fn, has_args=True, prog='popd')

def dirs_fn(nth: Annotated[tp.Optional[str], Arg(metavar='N', nargs='?')]=None, clear=False, print_long=False, verbose=False, long=False):
    if False:
        i = 10
        return i + 15
    'Manage the list of currently remembered directories.\n\n    Parameters\n    ----------\n    nth\n        Displays the Nth directory (counting from the left/right according to +/x prefix respectively),\n        starting with zero\n    clear : -c\n        Clears the directory stack by deleting all of the entries.\n    print_long : -p\n        Print the directory stack with one entry per line.\n    verbose : -v\n        Print the directory stack with one entry per line,\n        prefixing each entry with its index in the stack.\n    long : -l\n        Produces a longer listing; the default listing format\n        uses a tilde to denote the home directory.\n    '
    global DIRSTACK
    env = XSH.env
    dirstack = [os.path.expanduser(env['PWD'])] + DIRSTACK
    if env.get('PUSHD_MINUS'):
        BACKWARD = '-'
        FORWARD = '+'
    else:
        BACKWARD = '-'
        FORWARD = '+'
    if clear:
        DIRSTACK = []
        return (None, None, 0)
    if long:
        o = dirstack
    else:
        d = os.path.expanduser('~')
        o = [i.replace(d, '~') for i in dirstack]
    if verbose:
        out = ''
        pad = len(str(len(o) - 1))
        for (ix, e) in enumerate(o):
            blanks = ' ' * (pad - len(str(ix)))
            out += f'\n{blanks}{ix} {e}'
        out = out[1:]
    elif print_long:
        out = '\n'.join(o)
    else:
        out = ' '.join(o)
    if nth is not None:
        try:
            num = int(nth[1:])
        except ValueError:
            e = 'Invalid argument to dirs: {0}\n'
            return (None, e.format(nth), 1)
        if num < 0:
            e = 'Invalid argument to dirs: {0}\n'
            return (None, e.format(len(o)), 1)
        if num >= len(o):
            e = 'Too few elements in dirstack ({0} elements)\n'
            return (None, e.format(len(o)), 1)
        if nth.startswith(BACKWARD):
            idx = num
        elif nth.startswith(FORWARD):
            idx = len(o) - 1 - num
        else:
            e = 'Invalid argument to dirs: {0}\n'
            return (None, e.format(nth), 1)
        out = o[idx]
    return (out + '\n', None, 0)
dirs = ArgParserAlias(prog='dirs', func=dirs_fn, has_args=True)

@contextlib.contextmanager
def with_pushd(d):
    if False:
        return 10
    'Use pushd as a context manager'
    pushd_fn(d)
    try:
        yield
    finally:
        popd_fn()