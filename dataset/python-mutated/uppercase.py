"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import re
from lib.core.data import kb
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.NORMAL

def dependencies():
    if False:
        while True:
            i = 10
    pass

def tamper(payload, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Replaces each keyword character with upper case value (e.g. select -> SELECT)\n\n    Tested against:\n        * Microsoft SQL Server 2005\n        * MySQL 4, 5.0 and 5.5\n        * Oracle 10g\n        * PostgreSQL 8.3, 8.4, 9.0\n\n    Notes:\n        * Useful to bypass very weak and bespoke web application firewalls\n          that has poorly written permissive regular expressions\n        * This tamper script should work against all (?) databases\n\n    >>> tamper('insert')\n    'INSERT'\n    "
    retVal = payload
    if payload:
        for match in re.finditer('[A-Za-z_]+', retVal):
            word = match.group()
            if word.upper() in kb.keywords:
                retVal = retVal.replace(word, word.upper())
    return retVal