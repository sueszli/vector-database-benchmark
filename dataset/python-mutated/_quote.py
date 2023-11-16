"""Quoting helpers for Windows

This contains code to help with quoting values for use in the variable Windows
shell. Right now it should only be used in ansible.windows as the interface is
not final and could be subject to change.
"""
from __future__ import annotations
import re
from ansible.module_utils.six import text_type
_UNSAFE_C = re.compile(u'[\\s\t"]')
_UNSAFE_CMD = re.compile(u'[\\s\\(\\)\\^\\|%!"<>&]')
_UNSAFE_PWSH = re.compile(u"(['‘’‚‛])")

def quote_c(s):
    if False:
        print('Hello World!')
    'Quotes a value for the raw Win32 process command line.\n\n    Quotes a value to be safely used by anything that calls the Win32\n    CreateProcess API.\n\n    Args:\n        s: The string to quote.\n\n    Returns:\n        (text_type): The quoted string value.\n    '
    if not s:
        return u'""'
    if not _UNSAFE_C.search(s):
        return s
    s = s.replace('"', '\\"')
    s = re.sub('(\\\\+)\\\\"', '\\1\\1\\"', s)
    s = re.sub('(\\\\+)$', '\\1\\1', s)
    return u'"{0}"'.format(s)

def quote_cmd(s):
    if False:
        while True:
            i = 10
    'Quotes a value for cmd.\n\n    Quotes a value to be safely used by a command prompt call.\n\n    Args:\n        s: The string to quote.\n\n    Returns:\n        (text_type): The quoted string value.\n    '
    if not s:
        return u'""'
    if not _UNSAFE_CMD.search(s):
        return s
    for c in u'^()%!"<>&|':
        if c in s:
            s = s.replace(c, (u'\\^' if c == u'"' else u'^') + c)
    return u'^"{0}^"'.format(s)

def quote_pwsh(s):
    if False:
        return 10
    'Quotes a value for PowerShell.\n\n    Quotes a value to be safely used by a PowerShell expression. The input\n    string because something that is safely wrapped in single quotes.\n\n    Args:\n        s: The string to quote.\n\n    Returns:\n        (text_type): The quoted string value.\n    '
    if not s:
        return u"''"
    return u"'{0}'".format(_UNSAFE_PWSH.sub(u'\\1\\1', s))