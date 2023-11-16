from math import cos
from numpy import linspace
from subdir.a import bar
from ..a import MyOtherClass
from ...file2 import MyClass, foo

def baz(x):
    if False:
        while True:
            i = 10
    return x

class AnotherClass:
    E = 1

    def five(self):
        if False:
            for i in range(10):
                print('nop')
        return 5

    def six(self):
        if False:
            for i in range(10):
                print('nop')
        return 4