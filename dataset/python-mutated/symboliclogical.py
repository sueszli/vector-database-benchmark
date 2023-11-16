"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import re
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.LOWEST

def dependencies():
    if False:
        return 10
    pass

def tamper(payload, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Replaces AND and OR logical operators with their symbolic counterparts (&& and ||)\n\n    >>> tamper("1 AND \'1\'=\'1")\n    "1 %26%26 \'1\'=\'1"\n    '
    retVal = payload
    if payload:
        retVal = re.sub('(?i)\\bAND\\b', '%26%26', re.sub('(?i)\\bOR\\b', '%7C%7C', payload))
    return retVal