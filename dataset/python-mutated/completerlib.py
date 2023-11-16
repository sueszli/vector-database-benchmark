"""Implementations for various useful completers.

These are all loaded by default by IPython.
"""
import glob
import inspect
import os
import re
import sys
from importlib import import_module
from importlib.machinery import all_suffixes
from time import time
from zipimport import zipimporter
from .completer import expand_user, compress_user
from .error import TryNext
from ..utils._process_common import arg_split
from IPython import get_ipython
from typing import List
_suffixes = all_suffixes()
TIMEOUT_STORAGE = 2
TIMEOUT_GIVEUP = 20
import_re = re.compile('(?P<name>[^\\W\\d]\\w*?)(?P<package>[/\\\\]__init__)?(?P<suffix>%s)$' % '|'.join((re.escape(s) for s in _suffixes)))
magic_run_re = re.compile('.*(\\.ipy|\\.ipynb|\\.py[w]?)$')

def module_list(path):
    if False:
        while True:
            i = 10
    '\n    Return the list containing the names of the modules available in the given\n    folder.\n    '
    if path == '':
        path = '.'
    pjoin = os.path.join
    if os.path.isdir(path):
        files = []
        for (root, dirs, nondirs) in os.walk(path, followlinks=True):
            subdir = root[len(path) + 1:]
            if subdir:
                files.extend((pjoin(subdir, f) for f in nondirs))
                dirs[:] = []
            else:
                files.extend(nondirs)
    else:
        try:
            files = list(zipimporter(path)._files.keys())
        except:
            files = []
    modules = []
    for f in files:
        m = import_re.match(f)
        if m:
            modules.append(m.group('name'))
    return list(set(modules))

def get_root_modules():
    if False:
        return 10
    "\n    Returns a list containing the names of all the modules available in the\n    folders of the pythonpath.\n\n    ip.db['rootmodules_cache'] maps sys.path entries to list of modules.\n    "
    ip = get_ipython()
    if ip is None:
        return list(sys.builtin_module_names)
    rootmodules_cache = ip.db.get('rootmodules_cache', {})
    rootmodules = list(sys.builtin_module_names)
    start_time = time()
    store = False
    for path in sys.path:
        try:
            modules = rootmodules_cache[path]
        except KeyError:
            modules = module_list(path)
            try:
                modules.remove('__init__')
            except ValueError:
                pass
            if path not in ('', '.'):
                rootmodules_cache[path] = modules
            if time() - start_time > TIMEOUT_STORAGE and (not store):
                store = True
                print('\nCaching the list of root modules, please wait!')
                print("(This will only be done once - type '%rehashx' to reset cache!)\n")
                sys.stdout.flush()
            if time() - start_time > TIMEOUT_GIVEUP:
                print('This is taking too long, we give up.\n')
                return []
        rootmodules.extend(modules)
    if store:
        ip.db['rootmodules_cache'] = rootmodules_cache
    rootmodules = list(set(rootmodules))
    return rootmodules

def is_importable(module, attr, only_modules):
    if False:
        return 10
    if only_modules:
        return inspect.ismodule(getattr(module, attr))
    else:
        return not (attr[:2] == '__' and attr[-2:] == '__')

def is_possible_submodule(module, attr):
    if False:
        for i in range(10):
            print('nop')
    try:
        obj = getattr(module, attr)
    except AttributeError:
        return True
    except TypeError:
        return False
    return inspect.ismodule(obj)

def try_import(mod: str, only_modules=False) -> List[str]:
    if False:
        while True:
            i = 10
    '\n    Try to import given module and return list of potential completions.\n    '
    mod = mod.rstrip('.')
    try:
        m = import_module(mod)
    except:
        return []
    m_is_init = '__init__' in (getattr(m, '__file__', '') or '')
    completions = []
    if not hasattr(m, '__file__') or not only_modules or m_is_init:
        completions.extend([attr for attr in dir(m) if is_importable(m, attr, only_modules)])
    m_all = getattr(m, '__all__', [])
    if only_modules:
        completions.extend((attr for attr in m_all if is_possible_submodule(m, attr)))
    else:
        completions.extend(m_all)
    if m_is_init:
        file_ = m.__file__
        completions.extend(module_list(os.path.dirname(file_)))
    completions_set = {c for c in completions if isinstance(c, str)}
    completions_set.discard('__init__')
    return list(completions_set)

def quick_completer(cmd, completions):
    if False:
        return 10
    " Easily create a trivial completer for a command.\n\n    Takes either a list of completions, or all completions in string (that will\n    be split on whitespace).\n\n    Example::\n\n        [d:\\ipython]|1> import ipy_completers\n        [d:\\ipython]|2> ipy_completers.quick_completer('foo', ['bar','baz'])\n        [d:\\ipython]|3> foo b<TAB>\n        bar baz\n        [d:\\ipython]|3> foo ba\n    "
    if isinstance(completions, str):
        completions = completions.split()

    def do_complete(self, event):
        if False:
            print('Hello World!')
        return completions
    get_ipython().set_hook('complete_command', do_complete, str_key=cmd)

def module_completion(line):
    if False:
        return 10
    "\n    Returns a list containing the completion possibilities for an import line.\n\n    The line looks like this :\n    'import xml.d'\n    'from xml.dom import'\n    "
    words = line.split(' ')
    nwords = len(words)
    if nwords == 3 and words[0] == 'from':
        return ['import ']
    if nwords < 3 and words[0] in {'%aimport', 'import', 'from'}:
        if nwords == 1:
            return get_root_modules()
        mod = words[1].split('.')
        if len(mod) < 2:
            return get_root_modules()
        completion_list = try_import('.'.join(mod[:-1]), True)
        return ['.'.join(mod[:-1] + [el]) for el in completion_list]
    if nwords >= 3 and words[0] == 'from':
        mod = words[1]
        return try_import(mod)

def module_completer(self, event):
    if False:
        print('Hello World!')
    "Give completions after user has typed 'import ...' or 'from ...'"
    return module_completion(event.line)

def magic_run_completer(self, event):
    if False:
        print('Hello World!')
    'Complete files that end in .py or .ipy or .ipynb for the %run command.\n    '
    comps = arg_split(event.line, strict=False)
    if len(comps) > 1 and (not event.line.endswith(' ')):
        relpath = comps[-1].strip('\'"')
    else:
        relpath = ''
    lglob = glob.glob
    isdir = os.path.isdir
    (relpath, tilde_expand, tilde_val) = expand_user(relpath)
    if any((magic_run_re.match(c) for c in comps)):
        matches = [f.replace('\\', '/') + ('/' if isdir(f) else '') for f in lglob(relpath + '*')]
    else:
        dirs = [f.replace('\\', '/') + '/' for f in lglob(relpath + '*') if isdir(f)]
        pys = [f.replace('\\', '/') for f in lglob(relpath + '*.py') + lglob(relpath + '*.ipy') + lglob(relpath + '*.ipynb') + lglob(relpath + '*.pyw')]
        matches = dirs + pys
    return [compress_user(p, tilde_expand, tilde_val) for p in matches]

def cd_completer(self, event):
    if False:
        i = 10
        return i + 15
    'Completer function for cd, which only returns directories.'
    ip = get_ipython()
    relpath = event.symbol
    if event.line.endswith('-b') or ' -b ' in event.line:
        bkms = self.db.get('bookmarks', None)
        if bkms:
            return bkms.keys()
        else:
            return []
    if event.symbol == '-':
        width_dh = str(len(str(len(ip.user_ns['_dh']) + 1)))
        fmt = '-%0' + width_dh + 'd [%s]'
        ents = [fmt % (i, s) for (i, s) in enumerate(ip.user_ns['_dh'])]
        if len(ents) > 1:
            return ents
        return []
    if event.symbol.startswith('--'):
        return ['--' + os.path.basename(d) for d in ip.user_ns['_dh']]
    (relpath, tilde_expand, tilde_val) = expand_user(relpath)
    relpath = relpath.replace('\\', '/')
    found = []
    for d in [f.replace('\\', '/') + '/' for f in glob.glob(relpath + '*') if os.path.isdir(f)]:
        if ' ' in d:
            raise TryNext
        found.append(d)
    if not found:
        if os.path.isdir(relpath):
            return [compress_user(relpath, tilde_expand, tilde_val)]
        bks = self.db.get('bookmarks', {})
        bkmatches = [s for s in bks if s.startswith(event.symbol)]
        if bkmatches:
            return bkmatches
        raise TryNext
    return [compress_user(p, tilde_expand, tilde_val) for p in found]

def reset_completer(self, event):
    if False:
        print('Hello World!')
    'A completer for %reset magic'
    return '-f -s in out array dhist'.split()