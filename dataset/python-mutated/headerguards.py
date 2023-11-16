"""
Verifies the guard macros of all C++ header files.
"""
import re
from .util import findfiles, readfile

class HeaderIssue(Exception):
    """ Some issue was detected with the Header guard. """
GUARD_RE = re.compile('^(\\n|(#|//).*\\n)*#pragma once\n')
NO_GUARD_REQUIRED_RE = re.compile('^(\\n|(#|//).*\\n)*(#|//) has no header guard:')

def find_issues(dirname):
    if False:
        i = 10
        return i + 15
    '\n    checks all headerguards in header files in the cpp folders.\n    '
    for fname in findfiles((dirname,), ('.h',)):
        try:
            data = readfile(fname)
            if NO_GUARD_REQUIRED_RE.match(data):
                continue
            match = GUARD_RE.match(data)
            if not match:
                raise HeaderIssue('No valid header guard found (e.g. #pragma once)')
        except HeaderIssue as exc:
            yield (f'header guard issue in {fname}', exc.args[0], None)