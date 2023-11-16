"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import re
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.HIGHEST

def dependencies():
    if False:
        print('Hello World!')
    pass

def tamper(payload, **kwargs):
    if False:
        print('Hello World!')
    "\n    Replaces greater than operator ('>') with 'GREATEST' counterpart\n\n    Tested against:\n        * MySQL 4, 5.0 and 5.5\n        * Oracle 10g\n        * PostgreSQL 8.3, 8.4, 9.0\n\n    Notes:\n        * Useful to bypass weak and bespoke web application firewalls that\n          filter the greater than character\n        * The GREATEST clause is a widespread SQL command. Hence, this\n          tamper script should work against majority of databases\n\n    >>> tamper('1 AND A > B')\n    '1 AND GREATEST(A,B+1)=A'\n    "
    retVal = payload
    if payload:
        match = re.search("(?i)(\\b(AND|OR)\\b\\s+)([^>]+?)\\s*>\\s*(\\w+|'[^']+')", payload)
        if match:
            _ = '%sGREATEST(%s,%s+1)=%s' % (match.group(1), match.group(3), match.group(4), match.group(3))
            retVal = retVal.replace(match.group(0), _)
    return retVal