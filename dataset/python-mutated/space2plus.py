"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
from lib.core.compat import xrange
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.LOW

def dependencies():
    if False:
        i = 10
        return i + 15
    pass

def tamper(payload, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Replaces space character (' ') with plus ('+')\n\n    Notes:\n        * Is this any useful? The plus get's url-encoded by sqlmap engine invalidating the query afterwards\n        * This tamper script works against all databases\n\n    >>> tamper('SELECT id FROM users')\n    'SELECT+id+FROM+users'\n    "
    retVal = payload
    if payload:
        retVal = ''
        (quote, doublequote, firstspace) = (False, False, False)
        for i in xrange(len(payload)):
            if not firstspace:
                if payload[i].isspace():
                    firstspace = True
                    retVal += '+'
                    continue
            elif payload[i] == "'":
                quote = not quote
            elif payload[i] == '"':
                doublequote = not doublequote
            elif payload[i] == ' ' and (not doublequote) and (not quote):
                retVal += '+'
                continue
            retVal += payload[i]
    return retVal