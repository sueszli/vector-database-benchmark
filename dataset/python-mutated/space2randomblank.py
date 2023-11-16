"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import random
from lib.core.compat import xrange
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.LOW

def dependencies():
    if False:
        print('Hello World!')
    pass

def tamper(payload, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Replaces space character (' ') with a random blank character from a valid set of alternate characters\n\n    Tested against:\n        * Microsoft SQL Server 2005\n        * MySQL 4, 5.0 and 5.5\n        * Oracle 10g\n        * PostgreSQL 8.3, 8.4, 9.0\n\n    Notes:\n        * Useful to bypass several web application firewalls\n\n    >>> random.seed(0)\n    >>> tamper('SELECT id FROM users')\n    'SELECT%0Did%0CFROM%0Ausers'\n    "
    blanks = ('%09', '%0A', '%0C', '%0D')
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