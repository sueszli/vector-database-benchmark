"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
from lib.core.data import kb
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.HIGHEST

def dependencies():
    if False:
        for i in range(10):
            print('nop')
    pass

def tamper(payload, **kwargs):
    if False:
        return 10
    '\n    Replaces instances like \'SLEEP(5)\' with (e.g.) "GET_LOCK(\'ETgP\',5)"\n\n    Requirement:\n        * MySQL\n\n    Tested against:\n        * MySQL 5.0 and 5.5\n\n    Notes:\n        * Useful to bypass very weak and bespoke web application firewalls\n          that filter the SLEEP() and BENCHMARK() functions\n\n        * Reference: https://zhuanlan.zhihu.com/p/35245598\n\n    >>> tamper(\'SLEEP(5)\') == "GET_LOCK(\'%s\',5)" % kb.aliasName\n    True\n    '
    if payload:
        payload = payload.replace('SLEEP(', "GET_LOCK('%s'," % kb.aliasName)
    return payload