"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import os
import random
from lib.core.common import singleTimeWarnMessage
from lib.core.compat import xrange
from lib.core.enums import DBMS
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.LOW

def dependencies():
    if False:
        print('Hello World!')
    singleTimeWarnMessage("tamper script '%s' is only meant to be run against %s" % (os.path.basename(__file__).split('.')[0], DBMS.MYSQL))

def tamper(payload, **kwargs):
    if False:
        return 10
    "\n    Replaces (MySQL) instances of space character (' ') with a random blank character from a valid set of alternate characters\n\n    Requirement:\n        * MySQL\n\n    Tested against:\n        * MySQL 5.1\n\n    Notes:\n        * Useful to bypass several web application firewalls\n\n    >>> random.seed(0)\n    >>> tamper('SELECT id FROM users')\n    'SELECT%A0id%0CFROM%0Dusers'\n    "
    blanks = ('%09', '%0A', '%0C', '%0D', '%0B', '%A0')
    retVal = payload
    if payload:
        retVal = ''
        (quote, doublequote, firstspace) = (False, False, False)
        for i in xrange(len(payload)):
            if not firstspace:
                if payload[i].isspace():
                    firstspace = True
                    retVal += random.choice(blanks)
                    continue
            elif payload[i] == "'":
                quote = not quote
            elif payload[i] == '"':
                doublequote = not doublequote
            elif payload[i] == ' ' and (not doublequote) and (not quote):
                retVal += random.choice(blanks)
                continue
            retVal += payload[i]
    return retVal