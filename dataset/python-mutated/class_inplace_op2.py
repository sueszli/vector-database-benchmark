class A:

    def __imul__(self, other):
        if False:
            i = 10
            return i + 15
        print('__imul__')
        return self

    def __imatmul__(self, other):
        if False:
            return 10
        print('__imatmul__')
        return self

    def __ifloordiv__(self, other):
        if False:
            i = 10
            return i + 15
        print('__ifloordiv__')
        return self

    def __itruediv__(self, other):
        if False:
            print('Hello World!')
        print('__itruediv__')
        return self

    def __imod__(self, other):
        if False:
            return 10
        print('__imod__')
        return self

    def __ipow__(self, other):
        if False:
            return 10
        print('__ipow__')
        return self

    def __ior__(self, other):
        if False:
            while True:
                i = 10
        print('__ior__')
        return self

    def __ixor__(self, other):
        if False:
            print('Hello World!')
        print('__ixor__')
        return self

    def __iand__(self, other):
        if False:
            print('Hello World!')
        print('__iand__')
        return self

    def __ilshift__(self, other):
        if False:
            print('Hello World!')
        print('__ilshift__')
        return self

    def __irshift__(self, other):
        if False:
            while True:
                i = 10
        print('__irshift__')
        return self
a = A()
try:
    a *= None
except TypeError:
    print('SKIP')
    raise SystemExit
a @= None
a //= None
a /= None
a %= None
a **= None
a |= None
a ^= None
a &= None
a <<= None
a >>= None
try:
    a * None
except TypeError:
    print('TypeError')