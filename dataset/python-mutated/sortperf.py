"""Sort performance test.

See main() for command line syntax.
See tabulate() for output format.

"""
import sys
import time
import random
import marshal
import tempfile
import os
td = tempfile.gettempdir()

def randfloats(n):
    if False:
        return 10
    'Return a list of n random floats in [0, 1).'
    fn = os.path.join(td, 'rr%06d' % n)
    try:
        fp = open(fn, 'rb')
    except OSError:
        r = random.random
        result = [r() for i in range(n)]
        try:
            try:
                fp = open(fn, 'wb')
                marshal.dump(result, fp)
                fp.close()
                fp = None
            finally:
                if fp:
                    try:
                        os.unlink(fn)
                    except OSError:
                        pass
        except OSError as msg:
            print("can't write", fn, ':', msg)
    else:
        result = marshal.load(fp)
        fp.close()
        for i in range(10):
            i = random.randrange(n)
            temp = result[:i]
            del result[:i]
            temp.reverse()
            result.extend(temp)
            del temp
    assert len(result) == n
    return result

def flush():
    if False:
        print('Hello World!')
    sys.stdout.flush()

def doit(L):
    if False:
        while True:
            i = 10
    t0 = time.perf_counter()
    L.sort()
    t1 = time.perf_counter()
    print('%6.2f' % (t1 - t0), end=' ')
    flush()

def tabulate(r):
    if False:
        return 10
    'Tabulate sort speed for lists of various sizes.\n\n    The sizes are 2**i for i in r (the argument, a list).\n\n    The output displays i, 2**i, and the time to sort arrays of 2**i\n    floating point numbers with the following properties:\n\n    *sort: random data\n    \\sort: descending data\n    /sort: ascending data\n    3sort: ascending, then 3 random exchanges\n    +sort: ascending, then 10 random at the end\n    %sort: ascending, then randomly replace 1% of the elements w/ random values\n    ~sort: many duplicates\n    =sort: all equal\n    !sort: worst case scenario\n\n    '
    cases = tuple([ch + 'sort' for ch in '*\\/3+%~=!'])
    fmt = '%2s %7s' + ' %6s' * len(cases)
    print(fmt % (('i', '2**i') + cases))
    for i in r:
        n = 1 << i
        L = randfloats(n)
        print('%2d %7d' % (i, n), end=' ')
        flush()
        doit(L)
        L.reverse()
        doit(L)
        doit(L)
        for dummy in range(3):
            i1 = random.randrange(n)
            i2 = random.randrange(n)
            (L[i1], L[i2]) = (L[i2], L[i1])
        doit(L)
        if n >= 10:
            L[-10:] = [random.random() for dummy in range(10)]
        doit(L)
        for dummy in range(n // 100):
            L[random.randrange(n)] = random.random()
        doit(L)
        if n > 4:
            del L[4:]
            L = L * (n // 4)
            L = list(map(lambda x: --x, L))
        doit(L)
        del L
        L = list(map(abs, [-0.5] * n))
        doit(L)
        del L
        half = n // 2
        L = list(range(half - 1, -1, -1))
        L.extend(range(half))
        L = list(map(float, L))
        doit(L)
        print()

def main():
    if False:
        for i in range(10):
            print('nop')
    'Main program when invoked as a script.\n\n    One argument: tabulate a single row.\n    Two arguments: tabulate a range (inclusive).\n    Extra arguments are used to seed the random generator.\n\n    '
    k1 = 15
    k2 = 20
    if sys.argv[1:]:
        k1 = k2 = int(sys.argv[1])
        if sys.argv[2:]:
            k2 = int(sys.argv[2])
            if sys.argv[3:]:
                x = 1
                for a in sys.argv[3:]:
                    x = 69069 * x + hash(a)
                random.seed(x)
    r = range(k1, k2 + 1)
    tabulate(r)
if __name__ == '__main__':
    main()