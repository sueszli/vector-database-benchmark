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
        return 10
    "\n    Replaces (MySQL) instances like 'LIMIT M, N' with 'LIMIT N OFFSET M' counterpart\n\n    Requirement:\n        * MySQL\n\n    Tested against:\n        * MySQL 5.0 and 5.5\n\n    >>> tamper('LIMIT 2, 3')\n    'LIMIT 3 OFFSET 2'\n    "
    retVal = payload
    match = re.search('(?i)LIMIT\\s*(\\d+),\\s*(\\d+)', payload or '')
    if match:
        retVal = retVal.replace(match.group(0), 'LIMIT %s OFFSET %s' % (match.group(2), match.group(1)))
    return retVal