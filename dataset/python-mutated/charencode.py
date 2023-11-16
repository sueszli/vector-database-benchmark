"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import string
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.LOWEST

def dependencies():
    if False:
        return 10
    pass

def tamper(payload, **kwargs):
    if False:
        print('Hello World!')
    "\n    URL-encodes all characters in a given payload (not processing already encoded) (e.g. SELECT -> %53%45%4C%45%43%54)\n\n    Tested against:\n        * Microsoft SQL Server 2005\n        * MySQL 4, 5.0 and 5.5\n        * Oracle 10g\n        * PostgreSQL 8.3, 8.4, 9.0\n\n    Notes:\n        * Useful to bypass very weak web application firewalls that do not url-decode the request before processing it through their ruleset\n        * The web server will anyway pass the url-decoded version behind, hence it should work against any DBMS\n\n    >>> tamper('SELECT FIELD FROM%20TABLE')\n    '%53%45%4C%45%43%54%20%46%49%45%4C%44%20%46%52%4F%4D%20%54%41%42%4C%45'\n    "
    retVal = payload
    if payload:
        retVal = ''
        i = 0
        while i < len(payload):
            if payload[i] == '%' and i < len(payload) - 2 and (payload[i + 1:i + 2] in string.hexdigits) and (payload[i + 2:i + 3] in string.hexdigits):
                retVal += payload[i:i + 3]
                i += 3
            else:
                retVal += '%%%.2X' % ord(payload[i])
                i += 1
    return retVal