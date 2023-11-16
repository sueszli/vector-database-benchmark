"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import os
import re
from lib.core.common import singleTimeWarnMessage
from lib.core.convert import decodeHex
from lib.core.convert import getOrds
from lib.core.enums import DBMS
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.NORMAL

def dependencies():
    if False:
        return 10
    singleTimeWarnMessage("tamper script '%s' is only meant to be run against %s" % (os.path.basename(__file__).split('.')[0], DBMS.MYSQL))

def tamper(payload, **kwargs):
    if False:
        return 10
    "\n    Replaces each (MySQL) 0x<hex> encoded string with equivalent CONCAT(CHAR(),...) counterpart\n\n    Requirement:\n        * MySQL\n\n    Tested against:\n        * MySQL 4, 5.0 and 5.5\n\n    Notes:\n        * Useful in cases when web application does the upper casing\n\n    >>> tamper('SELECT 0xdeadbeef')\n    'SELECT CONCAT(CHAR(222),CHAR(173),CHAR(190),CHAR(239))'\n    "
    retVal = payload
    if payload:
        for match in re.finditer('\\b0x([0-9a-f]+)\\b', retVal):
            if len(match.group(1)) > 2:
                result = 'CONCAT(%s)' % ','.join(('CHAR(%d)' % _ for _ in getOrds(decodeHex(match.group(1)))))
            else:
                result = 'CHAR(%d)' % ord(decodeHex(match.group(1)))
            retVal = retVal.replace(match.group(0), result)
    return retVal