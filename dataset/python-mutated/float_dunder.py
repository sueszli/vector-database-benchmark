class TestFloat:

    def __float__(self):
        if False:
            i = 10
            return i + 15
        return 10.0

class TestStrFloat:

    def __float__(self):
        if False:
            print('Hello World!')
        return 'a'

class TestNonFloat:

    def __float__(self):
        if False:
            while True:
                i = 10
        return 6

class Test:
    pass
print('%.1f' % float(TestFloat()))
print('%.1f' % TestFloat())
try:
    print(float(TestStrFloat()))
except TypeError:
    print('TypeError')
try:
    print(float(TestNonFloat()))
except TypeError:
    print('TypeError')
try:
    print(float(Test()))
except TypeError:
    print('TypeError')