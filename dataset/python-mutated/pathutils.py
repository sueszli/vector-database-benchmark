"""
Functions for working with filesystem paths.

The :func:`expandpath` function expands the tilde to $HOME and environment
variables to their values.

The :func:`augpath` function creates variants of an existing path without
having to spend multiple lines of code splitting it up and stitching it back
together.

The :func:`shrinkuser` function replaces your home directory with a tilde.
"""
from __future__ import print_function
from os.path import expanduser, expandvars, join, normpath, split, splitext
import os
__all__ = ['augpath', 'shrinkuser', 'expandpath']

def augpath(path, suffix='', prefix='', ext=None, base=None, dpath=None, multidot=False):
    if False:
        print('Hello World!')
    "\n    Augment a path by modifying its components.\n\n    Creates a new path with a different extension, basename, directory, prefix,\n    and/or suffix.\n\n    A prefix is inserted before the basename. A suffix is inserted\n    between the basename and the extension. The basename and extension can be\n    replaced with a new one. Essentially a path is broken down into components\n    (dpath, base, ext), and then recombined as (dpath, prefix, base, suffix,\n    ext) after replacing any specified component.\n\n    Args:\n        path (str | PathLike): a path to augment\n        suffix (str, default=''): placed between the basename and extension\n        prefix (str, default=''): placed in front of the basename\n        ext (str, default=None): if specified, replaces the extension\n        base (str, default=None): if specified, replaces the basename without\n            extension\n        dpath (str | PathLike, default=None): if specified, replaces the\n            directory\n        multidot (bool, default=False): Allows extensions to contain multiple\n            dots. Specifically, if False, everything after the last dot in the\n            basename is the extension. If True, everything after the first dot\n            in the basename is the extension.\n\n    Returns:\n        str: augmented path\n\n    Example:\n        >>> path = 'foo.bar'\n        >>> suffix = '_suff'\n        >>> prefix = 'pref_'\n        >>> ext = '.baz'\n        >>> newpath = augpath(path, suffix, prefix, ext=ext, base='bar')\n        >>> print('newpath = %s' % (newpath,))\n        newpath = pref_bar_suff.baz\n\n    Example:\n        >>> augpath('foo.bar')\n        'foo.bar'\n        >>> augpath('foo.bar', ext='.BAZ')\n        'foo.BAZ'\n        >>> augpath('foo.bar', suffix='_')\n        'foo_.bar'\n        >>> augpath('foo.bar', prefix='_')\n        '_foo.bar'\n        >>> augpath('foo.bar', base='baz')\n        'baz.bar'\n        >>> augpath('foo.tar.gz', ext='.zip', multidot=True)\n        'foo.zip'\n        >>> augpath('foo.tar.gz', ext='.zip', multidot=False)\n        'foo.tar.zip'\n        >>> augpath('foo.tar.gz', suffix='_new', multidot=True)\n        'foo_new.tar.gz'\n    "
    (orig_dpath, fname) = split(path)
    if multidot:
        parts = fname.split('.', 1)
        orig_base = parts[0]
        orig_ext = '' if len(parts) == 1 else '.' + parts[1]
    else:
        (orig_base, orig_ext) = splitext(fname)
    if dpath is None:
        dpath = orig_dpath
    if ext is None:
        ext = orig_ext
    if base is None:
        base = orig_base
    new_fname = ''.join((prefix, base, suffix, ext))
    newpath = join(dpath, new_fname)
    return newpath

def shrinkuser(path, home='~'):
    if False:
        print('Hello World!')
    "\n    Inverse of :func:`os.path.expanduser`.\n\n    Args:\n        path (str | PathLike): path in system file structure\n        home (str, default='~'): symbol used to replace the home path.\n            Defaults to '~', but you might want to use '$HOME' or\n            '%USERPROFILE%' instead.\n\n    Returns:\n        str: path: shortened path replacing the home directory with a tilde\n\n    Example:\n        >>> path = expanduser('~')\n        >>> assert path != '~'\n        >>> assert shrinkuser(path) == '~'\n        >>> assert shrinkuser(path + '1') == path + '1'\n        >>> assert shrinkuser(path + '/1') == join('~', '1')\n        >>> assert shrinkuser(path + '/1', '$HOME') == join('$HOME', '1')\n    "
    path = normpath(path)
    userhome_dpath = expanduser('~')
    if path.startswith(userhome_dpath):
        if len(path) == len(userhome_dpath):
            path = home
        elif path[len(userhome_dpath)] == os.path.sep:
            path = home + path[len(userhome_dpath):]
    return path

def expandpath(path):
    if False:
        i = 10
        return i + 15
    "\n    Shell-like expansion of environment variables and tilde home directory.\n\n    Args:\n        path (str | PathLike): the path to expand\n\n    Returns:\n        str : expanded path\n\n    Example:\n        >>> import os\n        >>> os.environ['SPAM'] = 'eggs'\n        >>> assert expandpath('~/$SPAM') == expanduser('~/eggs')\n        >>> assert expandpath('foo') == 'foo'\n    "
    path = expanduser(path)
    path = expandvars(path)
    return path