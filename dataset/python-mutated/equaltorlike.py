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
        return 10
    "\n    Replaces all occurrences of operator equal ('=') with 'RLIKE' counterpart\n\n    Tested against:\n        * MySQL 4, 5.0 and 5.5\n\n    Notes:\n        * Useful to bypass weak and bespoke web application firewalls that\n          filter the equal character ('=')\n\n    >>> tamper('SELECT * FROM users WHERE id=1')\n    'SELECT * FROM users WHERE id RLIKE 1'\n    "
    retVal = payload
    if payload:
        retVal = re.sub('\\s*=\\s*', ' RLIKE ', retVal)
    return retVal