import collections.abc as cabc
import getopt
import os
import stat
import sys
from xonsh.built_ins import XSH
'Find the full path to commands.\n\nwhich(command, path=None, verbose=0, exts=None)\n    Return the full path to the first match of the given command on the\n    path.\n\nwhichall(command, path=None, verbose=0, exts=None)\n    Return a list of full paths to all matches of the given command on\n    the path.\n\nwhichgen(command, path=None, verbose=0, exts=None)\n    Return a generator which will yield full paths to all matches of the\n    given command on the path.\n\nBy default the PATH environment variable is searched (as well as, on\nWindows, the AppPaths key in the registry), but a specific \'path\' list\nto search may be specified as well.  On Windows, the PATHEXT environment\nvariable is applied as appropriate.\n\nIf "verbose" is true then a tuple of the form\n    (<fullpath>, <matched-where-description>)\nis returned for each match. The latter element is a textual description\nof where the match was found. For example:\n    from PATH element 0\n    from HKLM\\SOFTWARE\\...\\perl.exe\n'
_cmdlnUsage = "\n    Show the full path of commands.\n\n    Usage:\n        which [<options>...] [<command-name>...]\n\n    Options:\n        -h, --help      Print this help and exit.\n        -V, --version   Print the version info and exit.\n\n        -a, --all       Print *all* matching paths.\n        -v, --verbose   Print out how matches were located and\n                        show near misses on stderr.\n        -q, --quiet     Just print out matches. I.e., do not print out\n                        near misses.\n\n        -p <altpath>, --path=<altpath>\n                        An alternative path (list of directories) may\n                        be specified for searching.\n        -e <exts>, --exts=<exts>\n                        Specify a list of extensions to consider instead\n                        of the usual list (';'-separate list, Windows\n                        only).\n\n    Show the full path to the program that would be run for each given\n    command name, if any. Which, like GNU's which, returns the number of\n    failed arguments, or -1 when no <command-name> was given.\n\n    Near misses include duplicates, non-regular files and (on Un*x)\n    files without executable access.\n"
__version_info__ = (1, 2, 0)
__version__ = '.'.join(map(str, __version_info__))
__all__ = ['which', 'whichall', 'whichgen', 'WhichError']

class WhichError(Exception):
    pass

def _getRegisteredExecutable(exeName):
    if False:
        return 10
    'Windows allow application paths to be registered in the registry.'
    registered = None
    if sys.platform.startswith('win'):
        if os.path.splitext(exeName)[1].lower() != '.exe':
            exeName += '.exe'
        try:
            import winreg as _winreg
        except ImportError:
            import _winreg
        try:
            key = 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\' + exeName
            value = _winreg.QueryValue(_winreg.HKEY_LOCAL_MACHINE, key)
            registered = (value, 'from HKLM\\' + key)
        except _winreg.error:
            pass
        if registered and (not os.path.exists(registered[0])):
            registered = None
    return registered

def _samefile(fname1, fname2):
    if False:
        print('Hello World!')
    if sys.platform.startswith('win'):
        return os.path.normpath(os.path.normcase(fname1)) == os.path.normpath(os.path.normcase(fname2))
    else:
        return os.path.samefile(fname1, fname2)

def _cull(potential, matches, verbose=0):
    if False:
        while True:
            i = 10
    "Cull inappropriate matches. Possible reasons:\n        - a duplicate of a previous match\n        - not a disk file\n        - not executable (non-Windows)\n    If 'potential' is approved it is returned and added to 'matches'.\n    Otherwise, None is returned.\n    "
    for match in matches:
        if _samefile(potential[0], match[0]):
            if verbose:
                sys.stderr.write('duplicate: {} ({})\n'.format(*potential))
            return None
    else:
        if not stat.S_ISREG(os.stat(potential[0]).st_mode):
            if verbose:
                sys.stderr.write('not a regular file: {} ({})\n'.format(*potential))
        elif sys.platform != 'win32' and (not os.access(potential[0], os.X_OK)):
            if verbose:
                sys.stderr.write('no executable access: {} ({})\n'.format(*potential))
        else:
            matches.append(potential)
            return potential

def whichgen(command, path=None, verbose=0, exts=None):
    if False:
        for i in range(10):
            print('nop')
    'Return a generator of full paths to the given command.\n\n    "command" is a the name of the executable to search for.\n    "path" is an optional alternate path list to search. The default it\n        to use the PATH environment variable.\n    "verbose", if true, will cause a 2-tuple to be returned for each\n        match. The second element is a textual description of where the\n        match was found.\n    "exts" optionally allows one to specify a list of extensions to use\n        instead of the standard list for this system. This can\n        effectively be used as an optimization to, for example, avoid\n        stat\'s of "foo.vbs" when searching for "foo" and you know it is\n        not a VisualBasic script but ".vbs" is on PATHEXT. This option\n        is only supported on Windows.\n\n    This method returns a generator which yields tuples of the form (<path to\n    command>, <where path found>).\n    '
    matches = []
    if path is None:
        usingGivenPath = 0
        path = os.environ.get('PATH', '').split(os.pathsep)
        if sys.platform.startswith('win'):
            path.insert(0, os.curdir)
    else:
        usingGivenPath = 1
    if sys.platform.startswith('win'):
        if exts is None:
            exts = XSH.env['PATHEXT']
            for ext in exts:
                if ext.lower() == '.exe':
                    break
            else:
                exts = ['.COM', '.EXE', '.BAT', '.CMD']
        elif not isinstance(exts, cabc.Sequence):
            raise TypeError("'exts' argument must be a sequence or None")
    else:
        if exts is not None:
            raise WhichError("'exts' argument is not supported on platform '%s'" % sys.platform)
        exts = []
    if os.sep in command or (os.altsep and os.altsep in command):
        if os.path.exists(command):
            match = _cull((command, 'explicit path given'), matches, verbose)
            yield match
    else:
        for i in range(len(path)):
            dirName = path[i]
            if sys.platform.startswith('win') and len(dirName) >= 2 and (dirName[0] == '"') and (dirName[-1] == '"'):
                dirName = dirName[1:-1]
            for ext in [''] + exts:
                absName = os.path.abspath(os.path.normpath(os.path.join(dirName, command + ext)))
                if os.path.isfile(absName):
                    if usingGivenPath:
                        fromWhere = 'from given path element %d' % i
                    elif not sys.platform.startswith('win'):
                        fromWhere = 'from PATH element %d' % i
                    elif i == 0:
                        fromWhere = 'from current directory'
                    else:
                        fromWhere = 'from PATH element %d' % (i - 1)
                    match = _cull((absName, fromWhere), matches, verbose)
                    if match:
                        yield match
        match = _getRegisteredExecutable(command)
        if match is not None:
            match = _cull(match, matches, verbose)
            if match:
                yield match

def which(command, path=None, verbose=0, exts=None):
    if False:
        for i in range(10):
            print('nop')
    'Return the full path to the first match of the given command on\n    the path.\n\n    "command" is a the name of the executable to search for.\n    "path" is an optional alternate path list to search. The default it\n        to use the PATH environment variable.\n    "verbose", if true, will cause a 2-tuple to be returned. The second\n        element is a textual description of where the match was found.\n    "exts" optionally allows one to specify a list of extensions to use\n        instead of the standard list for this system. This can\n        effectively be used as an optimization to, for example, avoid\n        stat\'s of "foo.vbs" when searching for "foo" and you know it is\n        not a VisualBasic script but ".vbs" is on PATHEXT. This option\n        is only supported on Windows.\n\n    If no match is found for the command, a WhichError is raised.\n    '
    try:
        (absName, fromWhere) = next(whichgen(command, path, verbose, exts))
    except StopIteration as ex:
        raise WhichError("Could not find '%s' on the path." % command) from ex
    if verbose:
        return (absName, fromWhere)
    else:
        return absName

def whichall(command, path=None, verbose=0, exts=None):
    if False:
        i = 10
        return i + 15
    'Return a list of full paths to all matches of the given command\n    on the path.\n\n    "command" is a the name of the executable to search for.\n    "path" is an optional alternate path list to search. The default it\n        to use the PATH environment variable.\n    "verbose", if true, will cause a 2-tuple to be returned for each\n        match. The second element is a textual description of where the\n        match was found.\n    "exts" optionally allows one to specify a list of extensions to use\n        instead of the standard list for this system. This can\n        effectively be used as an optimization to, for example, avoid\n        stat\'s of "foo.vbs" when searching for "foo" and you know it is\n        not a VisualBasic script but ".vbs" is on PATHEXT. This option\n        is only supported on Windows.\n    '
    if verbose:
        return list(whichgen(command, path, verbose, exts))
    else:
        return list((absName for (absName, _) in whichgen(command, path, verbose, exts)))

def main(argv):
    if False:
        i = 10
        return i + 15
    all = 0
    verbose = 0
    altpath = None
    exts = None
    try:
        (optlist, args) = getopt.getopt(argv[1:], 'haVvqp:e:', ['help', 'all', 'version', 'verbose', 'quiet', 'path=', 'exts='])
    except getopt.GetoptErrsor as msg:
        sys.stderr.write(f'which: error: {msg}. Your invocation was: {argv}\n')
        sys.stderr.write("Try 'which --help'.\n")
        return 1
    for (opt, optarg) in optlist:
        if opt in ('-h', '--help'):
            print(_cmdlnUsage)
            return 0
        elif opt in ('-V', '--version'):
            print('which %s' % __version__)
            return 0
        elif opt in ('-a', '--all'):
            all = 1
        elif opt in ('-v', '--verbose'):
            verbose = 1
        elif opt in ('-q', '--quiet'):
            verbose = 0
        elif opt in ('-p', '--path'):
            if optarg:
                altpath = optarg.split(os.pathsep)
            else:
                altpath = []
        elif opt in ('-e', '--exts'):
            if optarg:
                exts = optarg.split(os.pathsep)
            else:
                exts = []
    if len(args) == 0:
        return -1
    failures = 0
    for arg in args:
        nmatches = 0
        for (absName, fromWhere) in whichgen(arg, path=altpath, verbose=verbose, exts=exts):
            if verbose:
                print(f'{absName} ({fromWhere})')
            else:
                print(absName)
            nmatches += 1
            if not all:
                break
        if not nmatches:
            failures += 1
    return failures
if __name__ == '__main__':
    sys.exit(main(sys.argv))