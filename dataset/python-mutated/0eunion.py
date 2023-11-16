"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import re
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.HIGHEST

def dependencies():
    if False:
        i = 10
        return i + 15
    pass

def tamper(payload, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Replaces instances of <int> UNION with <int>e0UNION\n\n    Requirement:\n        * MySQL\n        * MsSQL\n\n    Notes:\n        * Reference: https://media.blackhat.com/us-13/US-13-Salgado-SQLi-Optimization-and-Obfuscation-Techniques-Slides.pdf\n\n    >>> tamper('1 UNION ALL SELECT')\n    '1e0UNION ALL SELECT'\n    "
    return re.sub('(?i)(\\d+)\\s+(UNION )', '\\g<1>e0\\g<2>', payload) if payload else payload