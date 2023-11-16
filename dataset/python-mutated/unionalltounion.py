"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.HIGHEST

def dependencies():
    if False:
        print('Hello World!')
    pass

def tamper(payload, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Replaces instances of UNION ALL SELECT with UNION SELECT counterpart\n\n    >>> tamper('-1 UNION ALL SELECT')\n    '-1 UNION SELECT'\n    "
    return payload.replace('UNION ALL SELECT', 'UNION SELECT') if payload else payload