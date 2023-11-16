"""Filename matching with shell patterns.

fnmatch(FILENAME, PATTERN) matches according to the local convention.
fnmatchcase(FILENAME, PATTERN) always takes case in account.

The functions operate by translating the pattern into a regular
expression.  They cache the compiled regular expressions for speed.

The function translate(PATTERN) returns a regular expression
corresponding to PATTERN.  (It does not compile it.)
"""
import re
__all__ = ['fnmatch', 'fnmatchcase', 'translate']
_cache = {}
_MAXCACHE = 100

def _purge():
    if False:
        print('Hello World!')
    'Clear the pattern cache'
    _cache.clear()

def fnmatch(name, pat):
    if False:
        i = 10
        return i + 15
    "Test whether FILENAME matches PATTERN.\n\n    Patterns are Unix shell style:\n\n    *       matches everything\n    ?       matches any single character\n    [seq]   matches any character in seq\n    [!seq]  matches any char not in seq\n\n    An initial period in FILENAME is not special.\n    Both FILENAME and PATTERN are first case-normalized\n    if the operating system requires it.\n    If you don't want this, use fnmatchcase(FILENAME, PATTERN).\n    "
    name = name.lower()
    pat = pat.lower()
    return fnmatchcase(name, pat)

def fnmatchcase(name, pat):
    if False:
        print('Hello World!')
    "Test whether FILENAME matches PATTERN, including case.\n    This is a version of fnmatch() which doesn't case-normalize\n    its arguments.\n    "
    try:
        re_pat = _cache[pat]
    except KeyError:
        res = translate(pat)
        if len(_cache) >= _MAXCACHE:
            _cache.clear()
        _cache[pat] = re_pat = re.compile(res)
    return re_pat.match(name) is not None

def translate(pat):
    if False:
        for i in range(10):
            print('nop')
    'Translate a shell PATTERN to a regular expression.\n\n    There is no way to quote meta-characters.\n    '
    (i, n) = (0, len(pat))
    res = '^'
    while i < n:
        c = pat[i]
        i = i + 1
        if c == '*':
            if i < n and pat[i] == '*':
                i = i + 1
                if i < n and pat[i] == '/':
                    i = i + 1
                if i >= n:
                    res = f'{res}.*'
                else:
                    res = f'{res}(.*/)?'
            else:
                res = f'{res}[^/]*'
        elif c == '?':
            res = f'{res}[^/]'
        elif c == '[':
            j = i
            if j < n and pat[j] == '!':
                j = j + 1
            if j < n and pat[j] == ']':
                j = j + 1
            while j < n and pat[j] != ']':
                j = j + 1
            if j >= n:
                res = f'{res}\\['
            else:
                stuff = pat[i:j].replace('\\', '\\\\')
                i = j + 1
                if stuff[0] == '!':
                    stuff = f'^{stuff[1:]}'
                elif stuff[0] == '^':
                    stuff = f'\\{stuff}'
                res = f'{res}[{stuff}]'
        else:
            res = res + re.escape(c)
    return f'{res}$'