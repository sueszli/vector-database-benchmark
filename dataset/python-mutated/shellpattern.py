import os
import re
from queue import LifoQueue

def translate(pat, match_end='\\Z'):
    if False:
        while True:
            i = 10
    'Translate a shell-style pattern to a regular expression.\n\n    The pattern may include ``**<sep>`` (<sep> stands for the platform-specific path separator; "/" on POSIX systems)\n    for matching zero or more directory levels and "*" for matching zero or more arbitrary characters except any path\n    separator. Wrap meta-characters in brackets for a literal match (i.e. "[?]" to match the literal character "?").\n\n    Using match_end=regex one can give a regular expression that is used to match after the regex that is generated from\n    the pattern. The default is to match the end of the string.\n\n    This function is derived from the "fnmatch" module distributed with the Python standard library.\n\n    :copyright: 2001-2016 Python Software Foundation. All rights reserved.\n    :license: PSFLv2\n    '
    pat = _translate_alternatives(pat)
    sep = os.path.sep
    n = len(pat)
    i = 0
    res = ''
    while i < n:
        c = pat[i]
        i += 1
        if c == '*':
            if i + 1 < n and pat[i] == '*' and (pat[i + 1] == sep):
                res += f'(?:[^\\{sep}]*\\{sep})*'
                i += 2
            else:
                res += '[^\\%s]*' % sep
        elif c == '?':
            res += '[^\\%s]' % sep
        elif c == '[':
            j = i
            if j < n and pat[j] == '!':
                j += 1
            if j < n and pat[j] == ']':
                j += 1
            while j < n and pat[j] != ']':
                j += 1
            if j >= n:
                res += '\\['
            else:
                stuff = pat[i:j].replace('\\', '\\\\')
                i = j + 1
                if stuff[0] == '!':
                    stuff = '^' + stuff[1:]
                elif stuff[0] == '^':
                    stuff = '\\' + stuff
                res += '[%s]' % stuff
        elif c in '(|)':
            if i > 0 and pat[i - 1] != '\\':
                res += c
        else:
            res += re.escape(c)
    return '(?ms)' + res + match_end

def _parse_braces(pat):
    if False:
        return 10
    'Returns the index values of paired braces in `pat` as a list of tuples.\n\n    The dict\'s keys are the indexes corresponding to opening braces. Initially,\n    they are set to a value of `None`. Once a corresponding closing brace is found,\n    the value is updated. All dict keys with a positive int value are valid pairs.\n\n    Cannot rely on re.match("[^\\(\\\\)*]?{.*[^\\(\\\\)*]}") because, while it\n    does handle unpaired braces and nested pairs of braces, it misses sequences\n    of paired braces. E.g.: "{foo,bar}{bar,baz}" would translate, incorrectly, to\n    "(foo|bar\\}\\{bar|baz)" instead of, correctly, to "(foo|bar)(bar|baz)"\n\n    So this function parses in a left-to-right fashion, tracking pairs with a LIFO\n    queue: pushing opening braces on and popping them off when finding a closing\n    brace.\n    '
    curly_q = LifoQueue()
    pairs: dict[int, int] = dict()
    for (idx, c) in enumerate(pat):
        if c == '{':
            if idx == 0 or pat[idx - 1] != '\\':
                pairs[idx] = None
                curly_q.put(idx)
        if c == '}' and curly_q.qsize():
            if idx > 0 and pat[idx - 1] != '\\':
                pairs[curly_q.get()] = idx
    return [(opening, closing) for (opening, closing) in pairs.items() if closing is not None]

def _translate_alternatives(pat):
    if False:
        print('Hello World!')
    'Translates the shell-style alternative portions of the pattern to regular expression groups.\n\n    For example: {alt1,alt2} -> (alt1|alt2)\n    '
    brace_pairs = _parse_braces(pat)
    pat_list = list(pat)
    for (opening, closing) in brace_pairs:
        commas = 0
        for i in range(opening + 1, closing):
            if pat_list[i] == ',':
                if i == opening or pat_list[i - 1] != '\\':
                    pat_list[i] = '|'
                    commas += 1
            elif pat_list[i] == '|' and (i == opening or pat_list[i - 1] != '\\'):
                commas += 1
        if commas > 0:
            pat_list[opening] = '('
            pat_list[closing] = ')'
    return ''.join(pat_list)