"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.LOWEST

def dependencies():
    if False:
        for i in range(10):
            print('nop')
    pass

def tamper(payload, **kwargs):
    if False:
        print('Hello World!')
    '\n    Replaces apostrophe character (\') with its illegal double unicode counterpart (e.g. \' -> %00%27)\n\n    >>> tamper("1 AND \'1\'=\'1")\n    \'1 AND %00%271%00%27=%00%271\'\n    '
    return payload.replace("'", '%00%27') if payload else payload