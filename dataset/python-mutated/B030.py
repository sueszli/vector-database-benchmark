"""
Should emit:
B030:
 - line 12, column 8
 - line 17, column 9
 - line 22, column 21
 - line 27, column 37
"""
try:
    pass
except 1:
    pass
try:
    pass
except (1, ValueError):
    pass
try:
    pass
except (ValueError, (RuntimeError, (KeyError, TypeError))):
    pass
try:
    pass
except (ValueError, *(RuntimeError, (KeyError, TypeError))):
    pass
try:
    pass
except (*a, *(RuntimeError, (KeyError, TypeError))):
    pass
try:
    pass
except (ValueError, *(RuntimeError, TypeError)):
    pass
try:
    pass
except (ValueError, *[RuntimeError, *(TypeError,)]):
    pass
try:
    pass
except (*a, *b):
    pass
try:
    pass
except (*a, *(RuntimeError, TypeError)):
    pass
try:
    pass
except (*a, *(b, c)):
    pass
try:
    pass
except (*a, *(*b, *c)):
    pass

def what_to_catch():
    if False:
        return 10
    return ...
try:
    pass
except what_to_catch():
    pass