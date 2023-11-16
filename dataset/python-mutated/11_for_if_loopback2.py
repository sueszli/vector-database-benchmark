import os
import posixpath

def filter(names, pat):
    if False:
        i = 10
        return i + 15
    result = []
    pat = os.path.normcase(pat)
    match = _compile_pattern(pat, isinstance(pat, bytes))
    if os.path is posixpath:
        for name in names:
            if match(name):
                result.append(name)
    else:
        for name in names:
            if match(os.path.normcase(name)):
                result.append(name)