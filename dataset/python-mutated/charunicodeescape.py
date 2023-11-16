"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import string
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.NORMAL

def tamper(payload, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Unicode-escapes non-encoded characters in a given payload (not processing already encoded) (e.g. SELECT -> SELECT)\n\n    Notes:\n        * Useful to bypass weak filtering and/or WAFs in JSON contexes\n\n    >>> tamper('SELECT FIELD FROM TABLE')\n    '\\\\u0053\\\\u0045\\\\u004C\\\\u0045\\\\u0043\\\\u0054\\\\u0020\\\\u0046\\\\u0049\\\\u0045\\\\u004C\\\\u0044\\\\u0020\\\\u0046\\\\u0052\\\\u004F\\\\u004D\\\\u0020\\\\u0054\\\\u0041\\\\u0042\\\\u004C\\\\u0045'\n    "
    retVal = payload
    if payload:
        retVal = ''
        i = 0
        while i < len(payload):
            if payload[i] == '%' and i < len(payload) - 2 and (payload[i + 1:i + 2] in string.hexdigits) and (payload[i + 2:i + 3] in string.hexdigits):
                retVal += '\\u00%s' % payload[i + 1:i + 3]
                i += 3
            else:
                retVal += '\\u%.4X' % ord(payload[i])
                i += 1
    return retVal