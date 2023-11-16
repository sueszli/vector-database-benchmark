"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import os
import re
from lib.core.common import singleTimeWarnMessage
from lib.core.enums import DBMS
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.HIGHEST

def dependencies():
    if False:
        i = 10
        return i + 15
    singleTimeWarnMessage("tamper script '%s' is only meant to be run against %s" % (os.path.basename(__file__).split('.')[0], DBMS.ORACLE))

def tamper(payload, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Replaces instances of <int> UNION with <int>DUNION\n\n    Requirement:\n        * Oracle\n\n    Notes:\n        * Reference: https://media.blackhat.com/us-13/US-13-Salgado-SQLi-Optimization-and-Obfuscation-Techniques-Slides.pdf\n\n    >>> tamper('1 UNION ALL SELECT')\n    '1DUNION ALL SELECT'\n    "
    return re.sub('(?i)(\\d+)\\s+(UNION )', '\\g<1>D\\g<2>', payload) if payload else payload