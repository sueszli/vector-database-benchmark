"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import re
from lib.core.common import randomRange
from lib.core.compat import xrange
from lib.core.data import kb
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.NORMAL

def dependencies():
    if False:
        for i in range(10):
            print('nop')
    pass

def tamper(payload, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Replaces each keyword character with random case value (e.g. SELECT -> SEleCt)\n\n    Tested against:\n        * Microsoft SQL Server 2005\n        * MySQL 4, 5.0 and 5.5\n        * Oracle 10g\n        * PostgreSQL 8.3, 8.4, 9.0\n        * SQLite 3\n\n    Notes:\n        * Useful to bypass very weak and bespoke web application firewalls\n          that has poorly written permissive regular expressions\n        * This tamper script should work against all (?) databases\n\n    >>> import random\n    >>> random.seed(0)\n    >>> tamper('INSERT')\n    'InSeRt'\n    >>> tamper('f()')\n    'f()'\n    >>> tamper('function()')\n    'FuNcTiOn()'\n    >>> tamper('SELECT id FROM `user`')\n    'SeLeCt id FrOm `user`'\n    "
    retVal = payload
    if payload:
        for match in re.finditer('\\b[A-Za-z_]{2,}\\b', retVal):
            word = match.group()
            if word.upper() in kb.keywords and re.search('(?i)[`\\"\'\\[]%s[`\\"\'\\]]' % word, retVal) is None or '%s(' % word in payload:
                while True:
                    _ = ''
                    for i in xrange(len(word)):
                        _ += word[i].upper() if randomRange(0, 1) else word[i].lower()
                    if len(_) > 1 and _ not in (_.lower(), _.upper()):
                        break
                retVal = retVal.replace(word, _)
    return retVal