"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
from lib.core.compat import xrange
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.HIGHEST

def dependencies():
    if False:
        for i in range(10):
            print('nop')
    pass

def tamper(payload, **kwargs):
    if False:
        return 10
    "\n    Replaces instances like 'IFNULL(A, B)' with 'IF(ISNULL(A), B, A)' counterpart\n\n    Requirement:\n        * MySQL\n        * SQLite (possibly)\n        * SAP MaxDB (possibly)\n\n    Tested against:\n        * MySQL 5.0 and 5.5\n\n    Notes:\n        * Useful to bypass very weak and bespoke web application firewalls\n          that filter the IFNULL() function\n\n    >>> tamper('IFNULL(1, 2)')\n    'IF(ISNULL(1),2,1)'\n    "
    if payload and payload.find('IFNULL') > -1:
        while payload.find('IFNULL(') > -1:
            index = payload.find('IFNULL(')
            depth = 1
            (comma, end) = (None, None)
            for i in xrange(index + len('IFNULL('), len(payload)):
                if depth == 1 and payload[i] == ',':
                    comma = i
                elif depth == 1 and payload[i] == ')':
                    end = i
                    break
                elif payload[i] == '(':
                    depth += 1
                elif payload[i] == ')':
                    depth -= 1
            if comma and end:
                _ = payload[index + len('IFNULL('):comma]
                __ = payload[comma + 1:end].lstrip()
                newVal = 'IF(ISNULL(%s),%s,%s)' % (_, __, _)
                payload = payload[:index] + newVal + payload[end + 1:]
            else:
                break
    return payload