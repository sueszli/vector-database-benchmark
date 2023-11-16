"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import os
import re
from lib.core.common import singleTimeWarnMessage
from lib.core.enums import DBMS
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.HIGH

def dependencies():
    if False:
        return 10
    singleTimeWarnMessage("tamper script '%s' is only meant to be run against %s" % (os.path.basename(__file__).split('.')[0], DBMS.MYSQL))

def tamper(payload, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Replaces (MySQL) instances like 'MID(A, B, C)' with 'MID(A FROM B FOR C)' counterpart\n\n    Requirement:\n        * MySQL\n\n    Tested against:\n        * MySQL 5.0 and 5.5\n\n    >>> tamper('MID(VERSION(), 1, 1)')\n    'MID(VERSION() FROM 1 FOR 1)'\n    "
    retVal = payload
    warnMsg = "you should consider usage of switch '--no-cast' along with "
    warnMsg += "tamper script '%s'" % os.path.basename(__file__).split('.')[0]
    singleTimeWarnMessage(warnMsg)
    match = re.search('(?i)MID\\((.+?)\\s*,\\s*(\\d+)\\s*\\,\\s*(\\d+)\\s*\\)', payload or '')
    if match:
        retVal = retVal.replace(match.group(0), 'MID(%s FROM %s FOR %s)' % (match.group(1), match.group(2), match.group(3)))
    return retVal