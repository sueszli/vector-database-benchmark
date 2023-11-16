"""
Various OS utilities
"""
import re
from sys import platform
from .math import INF

def free_memory() -> int:
    if False:
        return 10
    '\n    Returns the amount of free bytes of memory.\n    On failure, returns +inf.\n\n    >>> free_memory() > 0\n    True\n    '
    memory = INF
    if platform.startswith('linux'):
        pattern = re.compile('^MemAvailable: +([0-9]+) kB\n$')
        with open('/proc/meminfo', encoding='utf8') as meminfo:
            for line in meminfo:
                match = pattern.match(line)
                if match:
                    memory = 1024 * int(match.group(1))
                    break
    elif platform == 'darwin':
        pass
    elif platform == 'win32':
        pass
    return memory