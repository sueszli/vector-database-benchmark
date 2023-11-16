def _slotnames(cls):
    if False:
        return 10
    names = []
    for c in cls.__mro__:
        if '__slots__' in c.__dict__:
            slots = c.__dict__['__slots__']
            for name in slots:
                if name == '__dict__':
                    continue
                else:
                    names.append(name)

def lasti2lineno(linestarts, a):
    if False:
        while True:
            i = 10
    for i in linestarts:
        if a:
            return a
    return -1
assert lasti2lineno([], True) == -1
assert lasti2lineno([], False) == -1
assert lasti2lineno([1], False) == -1
assert lasti2lineno([1], True) == 1

def test_pow(m, b, c):
    if False:
        for i in range(10):
            print('nop')
    for a in m:
        if a or b or c:
            c = 1
    return c
assert test_pow([], 2, 3) == 3
assert test_pow([1], 0, 5) == 1
assert test_pow([1], 4, 2) == 1
assert test_pow([0], 0, 0) == 0