import math
import sys

def gen_approx_table(f, nentries, min_x, max_x):
    if False:
        i = 10
        return i + 15
    'return a list of nentries containing tuples of the form:\n    (m, c).  min_x and max_x specify the domain\n    of the table.\n    '
    r = []
    incx = float(max_x - min_x) / nentries
    for i in range(nentries):
        a = i * incx + min_x
        b = (i + 1) * incx + min_x
        m = (f(b) - f(a)) / (b - a)
        c = f(a)
        r.append((m, c))
    return r

def scaled_sine(x):
    if False:
        print('Hello World!')
    return math.sin(x * math.pi / 2 ** 31)

def gen_sine_table():
    if False:
        print('Hello World!')
    nbits = 10
    nentries = 2 ** nbits
    min_x = 0
    max_x = 2 ** 32 - 1
    t = gen_approx_table(scaled_sine, nentries, min_x, max_x)
    for e in t:
        sys.stdout.write('  { %22.15e, %22.15e },\n' % (e[0], e[1]))
if __name__ == '__main__':
    gen_sine_table()