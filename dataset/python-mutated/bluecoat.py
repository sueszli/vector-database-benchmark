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
        for i in range(10):
            print('nop')
    pass

def tamper(payload, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Replaces space character after SQL statement with a valid random blank character. Afterwards replace character '=' with operator LIKE\n\n    Requirement:\n        * Blue Coat SGOS with WAF activated as documented in\n        https://kb.bluecoat.com/index?page=content&id=FAQ2147\n\n    Tested against:\n        * MySQL 5.1, SGOS\n\n    Notes:\n        * Useful to bypass Blue Coat's recommended WAF rule configuration\n\n    >>> tamper('SELECT id FROM users WHERE id = 1')\n    'SELECT%09id FROM%09users WHERE%09id LIKE 1'\n    "

    def process(match):
        if False:
            print('Hello World!')
        word = match.group('word')
        if word.upper() in kb.keywords:
            return match.group().replace(word, '%s%%09' % word)
        else:
            return match.group()
    retVal = payload
    if payload:
        retVal = re.sub('\\b(?P<word>[A-Z_]+)(?=[^\\w(]|\\Z)', process, retVal)
        retVal = re.sub('\\s*=\\s*', ' LIKE ', retVal)
        retVal = retVal.replace('%09 ', '%09')
    return retVal