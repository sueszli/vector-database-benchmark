"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import re
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.NORMAL

def dependencies():
    if False:
        print('Hello World!')
    pass

def tamper(payload, **kwargs):
    if False:
        print('Hello World!')
    "\n    Prepends (inline) comment before parentheses (e.g. ( -> /**/()\n\n    Tested against:\n        * Microsoft SQL Server\n        * MySQL\n        * Oracle\n        * PostgreSQL\n\n    Notes:\n        * Useful to bypass web application firewalls that block usage\n          of function calls\n\n    >>> tamper('SELECT ABS(1)')\n    'SELECT ABS/**/(1)'\n    "
    retVal = payload
    if payload:
        retVal = re.sub('\\b(\\w+)\\(', '\\g<1>/**/(', retVal)
    return retVal