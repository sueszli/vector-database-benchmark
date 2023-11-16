import dlib
from math import sin, cos, pi, exp, sqrt

def holder_table(x0, x1):
    if False:
        for i in range(10):
            print('nop')
    return -abs(sin(x0) * cos(x1) * exp(abs(1 - sqrt(x0 * x0 + x1 * x1) / pi)))
(x, y) = dlib.find_min_global(holder_table, [-10, -10], [10, 10], 80)
print('optimal inputs: {}'.format(x))
print('optimal output: {}'.format(y))