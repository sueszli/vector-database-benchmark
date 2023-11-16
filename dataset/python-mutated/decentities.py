"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.LOW

def dependencies():
    if False:
        for i in range(10):
            print('nop')
    pass

def tamper(payload, **kwargs):
    if False:
        print('Hello World!')
    '\n    HTML encode in decimal (using code points) all characters (e.g. \' -> &#39;)\n\n    >>> tamper("1\' AND SLEEP(5)#")\n    \'&#49;&#39;&#32;&#65;&#78;&#68;&#32;&#83;&#76;&#69;&#69;&#80;&#40;&#53;&#41;&#35;\'\n    '
    retVal = payload
    if payload:
        retVal = ''
        i = 0
        while i < len(payload):
            retVal += '&#%s;' % ord(payload[i])
            i += 1
    return retVal