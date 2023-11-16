"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.NORMAL

def dependencies():
    if False:
        while True:
            i = 10
    pass

def tamper(payload, **kwargs):
    if False:
        return 10
    '\n    Slash escape single and double quotes (e.g. \' -> \')\n\n    >>> tamper(\'1" AND SLEEP(5)#\')\n    \'1\\\\" AND SLEEP(5)#\'\n    '
    return payload.replace("'", "\\'").replace('"', '\\"')