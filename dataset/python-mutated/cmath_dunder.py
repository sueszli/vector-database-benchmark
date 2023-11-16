try:
    import cmath
except ImportError:
    print('SKIP')
    raise SystemExit

class TestFloat:

    def __float__(self):
        if False:
            return 10
        return 1.0

class TestComplex:

    def __complex__(self):
        if False:
            for i in range(10):
                print('nop')
        return complex(10, 1)
for clas in (TestFloat, TestComplex):
    print('%.5g' % cmath.phase(clas()))