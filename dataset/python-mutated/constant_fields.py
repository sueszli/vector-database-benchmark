from builtins import _test_source
from enum import Enum

class CustomEnum(Enum):
    TRACKED_FIELD = 'A'
    UNTRACKED_field = 'B'
    untracked_field = 'C'

def tracked_index():
    if False:
        i = 10
        return i + 15
    d = {}
    d[CustomEnum.TRACKED_FIELD] = _test_source()
    return d[CustomEnum.TRACKED_FIELD]

def untracked_index_a():
    if False:
        i = 10
        return i + 15
    d = {}
    d[CustomEnum.untracked_field] = _test_source()
    return d[CustomEnum.untracked_field]

def untracked_index_b():
    if False:
        return 10
    d = {}
    d[CustomEnum.UNTRACKED_field] = _test_source()
    return d[CustomEnum.UNTRACKED_field]
CONSTANT_A = 'A'
CONSTANT_B = {'a': 'b'}
untracked_constant = '1'

def tracked_constant_A():
    if False:
        for i in range(10):
            print('nop')
    d = {}
    d[CONSTANT_A] = _test_source()
    return d[CONSTANT_A]

def tracked_constant_B():
    if False:
        return 10
    d = {}
    d[CONSTANT_B] = _test_source()
    return d[CONSTANT_B]

def test_untracked_constant():
    if False:
        while True:
            i = 10
    d = {}
    d[untracked_constant] = _test_source()
    return d[untracked_constant]