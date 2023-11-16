class Cud:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def __repr__(self):
        if False:
            while True:
                i = 10
        print('__repr__ called')
        return ''

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        print('__lt__ called')

    def __le__(self, other):
        if False:
            print('Hello World!')
        print('__le__ called')

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        print('__eq__ called')

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        print('__ne__ called')

    def __ge__(self, other):
        if False:
            return 10
        print('__ge__ called')

    def __gt__(self, other):
        if False:
            return 10
        print('__gt__ called')

    def __abs__(self):
        if False:
            for i in range(10):
                print('nop')
        print('__abs__ called')

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        print('__add__ called')

    def __and__(self, other):
        if False:
            print('Hello World!')
        print('__and__ called')

    def __floordiv__(self, other):
        if False:
            print('Hello World!')
        print('__floordiv__ called')

    def __invert__(self):
        if False:
            for i in range(10):
                print('nop')
        print('__invert__ called')

    def __lshift__(self, val):
        if False:
            return 10
        print('__lshift__ called')

    def __mod__(self, val):
        if False:
            print('Hello World!')
        print('__mod__ called')

    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        print('__mul__ called')

    def __matmul__(self, other):
        if False:
            i = 10
            return i + 15
        print('__matmul__ called')

    def __neg__(self):
        if False:
            while True:
                i = 10
        print('__neg__ called')

    def __or__(self, other):
        if False:
            for i in range(10):
                print('nop')
        print('__or__ called')

    def __pos__(self):
        if False:
            while True:
                i = 10
        print('__pos__ called')

    def __pow__(self, val):
        if False:
            i = 10
            return i + 15
        print('__pow__ called')

    def __rshift__(self, val):
        if False:
            for i in range(10):
                print('nop')
        print('__rshift__ called')

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        print('__sub__ called')

    def __truediv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        print('__truediv__ called')

    def __div__(self, other):
        if False:
            for i in range(10):
                print('nop')
        print('__div__ called')

    def __xor__(self, other):
        if False:
            print('Hello World!')
        print('__xor__ called')

    def __iadd__(self, other):
        if False:
            i = 10
            return i + 15
        print('__iadd__ called')
        return self

    def __isub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        print('__isub__ called')
        return self

    def __dir__(self):
        if False:
            return 10
        return ['a', 'b', 'c']
cud1 = Cud()
cud2 = Cud()
try:
    +cud1
except TypeError:
    print('SKIP')
    raise SystemExit
+cud1
-cud1
~cud1
cud1 * cud2
cud1 @ cud2
cud1 / cud2
cud2 // cud1
cud1 += cud2
cud1 -= cud2
cud1 % 2
cud1 ** 2
cud1 | cud2
cud1 & cud2
cud1 ^ cud2
cud1 << 1
cud1 >> 1
print(dir(cud1))
print('a' in dir(Cud))