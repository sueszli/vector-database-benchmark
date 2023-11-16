""" Covers Python3 meta classes with __prepare__ non-dict values. """
from enum import Enum
print('Enum class with duplicate enumeration values:')
try:

    class Color(Enum):
        red = 1
        green = 2
        blue = 3
        red = 4
        print('not allowed to get here')
except Exception as e:
    print('Occurred', e)
print('Class variable that conflicts with closure variable:')

def testClassNamespaceOverridesClosure():
    if False:
        for i in range(10):
            print('nop')
    x = 42

    class X:
        locals()['x'] = 43
        y = x
    print('should be 43:', X.y)
testClassNamespaceOverridesClosure()