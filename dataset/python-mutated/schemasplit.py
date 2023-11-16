"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
import re
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.HIGHEST

def dependencies():
    if False:
        for i in range(10):
            print('nop')
    pass

def tamper(payload, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Splits FROM schema identifiers (e.g. 'testdb.users') with whitespace (e.g. 'testdb 9.e.users')\n\n    Requirement:\n        * MySQL\n\n    Notes:\n        * Reference: https://media.blackhat.com/us-13/US-13-Salgado-SQLi-Optimization-and-Obfuscation-Techniques-Slides.pdf\n\n    >>> tamper('SELECT id FROM testdb.users')\n    'SELECT id FROM testdb 9.e.users'\n    "
    return re.sub('(?i)( FROM \\w+)\\.(\\w+)', '\\g<1> 9.e.\\g<2>', payload) if payload else payload