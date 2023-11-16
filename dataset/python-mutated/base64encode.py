"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
from lib.core.convert import encodeBase64
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
    '\n    Base64-encodes all characters in a given payload\n\n    >>> tamper("1\' AND SLEEP(5)#")\n    \'MScgQU5EIFNMRUVQKDUpIw==\'\n    '
    return encodeBase64(payload, binary=False) if payload else payload