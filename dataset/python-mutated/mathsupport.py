from __future__ import absolute_import, division, print_function, unicode_literals
import math

def average(x, bessel=False):
    if False:
        while True:
            i = 10
    '\n    Args:\n      x: iterable with len\n\n      oneless: (default ``False``) reduces the length of the array for the\n                division.\n\n    Returns:\n      A float with the average of the elements of x\n    '
    return math.fsum(x) / (len(x) - bessel)

def variance(x, avgx=None):
    if False:
        i = 10
        return i + 15
    '\n    Args:\n      x: iterable with len\n\n    Returns:\n      A list with the variance for each element of x\n    '
    if avgx is None:
        avgx = average(x)
    return [pow(y - avgx, 2.0) for y in x]

def standarddev(x, avgx=None, bessel=False):
    if False:
        print('Hello World!')
    "\n    Args:\n      x: iterable with len\n\n      bessel: (default ``False``) to be passed to the average to divide by\n      ``N - 1`` (Bessel's correction)\n\n    Returns:\n      A float with the standard deviation of the elements of x\n    "
    return math.sqrt(average(variance(x, avgx), bessel=bessel))