"""
Methods for printing paths and other file system-related info.
"""
from __future__ import annotations
import typing
from collections import OrderedDict
from .strings import colorize
from .math import INF
RULE_CACHE = OrderedDict()
if typing.TYPE_CHECKING:
    from openage.util.fslike.abstract import FSLikeObject

def get_color_rules() -> OrderedDict[str, str]:
    if False:
        return 10
    '\n    Returns a dict of pattern : colorcode, retrieved from LS_COLORS.\n    '
    if RULE_CACHE:
        return RULE_CACHE
    from os import environ
    try:
        rules = environ['LS_COLORS']
    except KeyError:
        return {}
    for rule in rules.split(':'):
        rule = rule.strip()
        if not rule:
            continue
        try:
            (pattern, colorcode) = rule.split('=', maxsplit=1)
        except ValueError:
            continue
        RULE_CACHE[pattern] = colorcode
    return RULE_CACHE

def colorize_filename(filename: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Colorizes the filename, using the globbing rules from LS_COLORS.\n    '
    from fnmatch import fnmatch
    rules = get_color_rules()
    for (pattern, colorcode) in rules.items():
        if fnmatch(filename, pattern):
            return colorize(filename, colorcode)
    return colorize(filename, rules.get('fi'))

def colorize_dirname(dirname: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    "\n    Colorizes the dirname, using the 'di' rule from LS_COLORS.\n    "
    return colorize(dirname, get_color_rules().get('di'))

def print_tree(obj: FSLikeObject, path: str='', prefix: str='', max_entries: str=INF) -> None:
    if False:
        return 10
    '\n    Obj is a filesystem-like object; path must be a string.\n\n    Recursively descends into subdirectories using prefix.\n\n    If max_entries is given, only that number of entries per directory\n    is printed.\n    '
    entries = []
    for entry in obj.listdirs(path):
        entries.append((entry, True, False))
    for entry in obj.listfiles(path):
        entries.append((entry, False, False))
    if not entries:
        entries.append(('[empty]', False, True))
    if len(entries) > max_entries:
        omit = len(entries) - max_entries + 1
        entries = entries[:-omit] + [(f'[{omit} omitted]', False, True)]
    from .iterators import denote_last
    for ((name, isdir, is_meta), is_last) in denote_last(entries):
        if is_last:
            (treesymbol, nextindent) = ('└', '   ')
        else:
            (treesymbol, nextindent) = ('├', '│  ')
        if is_meta:
            entryindent = '╼ '
        else:
            entryindent = '─ '
        if isdir:
            print(prefix + treesymbol + entryindent + colorize_dirname(name))
            obj.print_tree(path + '/' + name, prefix + nextindent, max_entries)
            print(prefix + nextindent)
        else:
            print(prefix + treesymbol + entryindent + colorize_filename(name))