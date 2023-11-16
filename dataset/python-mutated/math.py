"""
Holds some math constants and helpers
"""
import math
INF = float('+inf')
TAU = 2 * math.pi
DEGSPERRAD = TAU / 360

def clamp(val: int, minval: int, maxval: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    '\n    clamps val to be at least minval, and at most maxval.\n\n    >>> clamp(9, 3, 7)\n    7\n    >>> clamp(1, 3, 7)\n    3\n    >>> clamp(5, 3, 7)\n    5\n    '
    return min(maxval, max(minval, val))