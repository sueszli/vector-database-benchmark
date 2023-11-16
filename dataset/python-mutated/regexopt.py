"""
    pygments.regexopt
    ~~~~~~~~~~~~~~~~~

    An algorithm that generates optimized regexes for matching long lists of
    literal strings.

    :copyright: Copyright 2006-2023 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""
import re
from re import escape
from os.path import commonprefix
from itertools import groupby
from operator import itemgetter
CS_ESCAPE = re.compile('[\\[\\^\\\\\\-\\]]')
FIRST_ELEMENT = itemgetter(0)

def make_charset(letters):
    if False:
        i = 10
        return i + 15
    return '[' + CS_ESCAPE.sub(lambda m: '\\' + m.group(), ''.join(letters)) + ']'

def regex_opt_inner(strings, open_paren):
    if False:
        return 10
    'Return a regex that matches any string in the sorted list of strings.'
    close_paren = open_paren and ')' or ''
    if not strings:
        return ''
    first = strings[0]
    if len(strings) == 1:
        return open_paren + escape(first) + close_paren
    if not first:
        return open_paren + regex_opt_inner(strings[1:], '(?:') + '?' + close_paren
    if len(first) == 1:
        oneletter = []
        rest = []
        for s in strings:
            if len(s) == 1:
                oneletter.append(s)
            else:
                rest.append(s)
        if len(oneletter) > 1:
            if rest:
                return open_paren + regex_opt_inner(rest, '') + '|' + make_charset(oneletter) + close_paren
            return open_paren + make_charset(oneletter) + close_paren
    prefix = commonprefix(strings)
    if prefix:
        plen = len(prefix)
        return open_paren + escape(prefix) + regex_opt_inner([s[plen:] for s in strings], '(?:') + close_paren
    strings_rev = [s[::-1] for s in strings]
    suffix = commonprefix(strings_rev)
    if suffix:
        slen = len(suffix)
        return open_paren + regex_opt_inner(sorted((s[:-slen] for s in strings)), '(?:') + escape(suffix[::-1]) + close_paren
    return open_paren + '|'.join((regex_opt_inner(list(group[1]), '') for group in groupby(strings, lambda s: s[0] == first[0]))) + close_paren

def regex_opt(strings, prefix='', suffix=''):
    if False:
        for i in range(10):
            print('nop')
    'Return a compiled regex that matches any string in the given list.\n\n    The strings to match must be literal strings, not regexes.  They will be\n    regex-escaped.\n\n    *prefix* and *suffix* are pre- and appended to the final regex.\n    '
    strings = sorted(strings)
    return prefix + regex_opt_inner(strings, '(') + suffix