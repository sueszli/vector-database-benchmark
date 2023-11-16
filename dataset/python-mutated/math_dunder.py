try:
    import math
except ImportError:
    print('SKIP')
    raise SystemExit

class TestFloat:

    def __float__(self):
        if False:
            while True:
                i = 10
        return 1.0
print('%.5g' % math.exp(TestFloat()))