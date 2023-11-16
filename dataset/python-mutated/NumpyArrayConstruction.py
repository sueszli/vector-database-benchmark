from __future__ import print_function
import itertools
import numpy
g = 6

def calledRepeatedly():
    if False:
        while True:
            i = 10
    x = numpy.array([[1, 2, 3], [4, 5, 6]], numpy.int32)
    x = (numpy.array, numpy.int32)
    return x
for x in itertools.repeat(None, 20000):
    calledRepeatedly()
print('OK.')