""" Many tools work on Nuitka sources and need to find the files.

"""
import os
from nuitka.utils.Shebang import getShebangFromFile
_default_ignore_list = ('inline_copy', 'tblib', '__pycache__')

def _addFromDirectory(path, suffixes, ignore_list):
    if False:
        i = 10
        return i + 15
    for (dirpath, dirnames, filenames) in os.walk(path):
        dirnames.sort()
        for entry in _default_ignore_list:
            if entry in dirnames:
                dirnames.remove(entry)
        filenames.sort()
        for filename in filenames:
            if filename in ignore_list:
                continue
            fullpath = os.path.join(dirpath, filename)
            if os.path.islink(fullpath):
                continue
            if filename.endswith('_flymake.py'):
                continue
            if filename.startswith('.#'):
                continue
            if filename.endswith(('.pyc', '.pyo')):
                continue
            if filename.endswith(('.exe', '.bin')):
                continue
            if '.py' in suffixes and (not filename.endswith(suffixes)):
                shebang = getShebangFromFile(fullpath)
                if shebang is None or 'python' not in shebang:
                    continue
            yield fullpath

def scanTargets(positional_args, suffixes, ignore_list=()):
    if False:
        return 10
    for positional_arg in positional_args:
        positional_arg = os.path.normpath(positional_arg)
        if os.path.isdir(positional_arg):
            for value in _addFromDirectory(positional_arg, suffixes, ignore_list):
                yield value
        else:
            yield positional_arg

def isPythonFile(filename, effective_filename=None):
    if False:
        print('Hello World!')
    if effective_filename is None:
        effective_filename = filename
    if os.path.isdir(filename):
        return False
    if effective_filename.endswith(('.py', '.pyw', '.scons')):
        return True
    else:
        shebang = getShebangFromFile(filename)
        if shebang is not None:
            shebang = shebang[2:].lstrip()
            if shebang.startswith('/usr/bin/env'):
                shebang = shebang[12:].lstrip()
            if shebang.startswith('python'):
                return True
    return False