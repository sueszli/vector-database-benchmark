"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import re
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.NORMAL

def tamper(payload, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add an inline comment (/**/) to the end of all occurrences of (MySQL) "information_schema" identifier\n\n    >>> tamper(\'SELECT table_name FROM INFORMATION_SCHEMA.TABLES\')\n    \'SELECT table_name FROM INFORMATION_SCHEMA/**/.TABLES\'\n    '
    retVal = payload
    if payload:
        retVal = re.sub('(?i)(information_schema)\\.', '\\g<1>/**/.', payload)
    return retVal