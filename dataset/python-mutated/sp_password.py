"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.HIGH

def tamper(payload, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Appends (MsSQL) function 'sp_password' to the end of the payload for automatic obfuscation from DBMS logs\n\n    Requirement:\n        * MSSQL\n\n    Notes:\n        * Appending sp_password to the end of the query will hide it from T-SQL logs as a security measure\n        * Reference: http://websec.ca/kb/sql_injection\n\n    >>> tamper('1 AND 9227=9227-- ')\n    '1 AND 9227=9227-- sp_password'\n    "
    retVal = ''
    if payload:
        retVal = '%s%ssp_password' % (payload, '-- ' if not any((_ if _ in payload else None for _ in ('#', '-- '))) else '')
    return retVal