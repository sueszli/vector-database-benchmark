"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import re
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.HIGHEST

def dependencies():
    if False:
        return 10
    pass

def tamper(payload, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Replaces ORD() occurences with equivalent ASCII() calls \n\n    Requirement:\n        * MySQL\n\n    >>> tamper("ORD(\'42\')")\n    "ASCII(\'42\')"\n    '
    retVal = payload
    if payload:
        retVal = re.sub('(?i)\\bORD\\(', 'ASCII(', payload)
    return retVal