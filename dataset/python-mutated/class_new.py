try:
    object.__new__
except AttributeError:
    print('SKIP')
    raise SystemExit

class A:

    def __new__(cls):
        if False:
            for i in range(10):
                print('nop')
        print('A.__new__')
        return super(cls, A).__new__(cls)

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        print('A.__init__')

    def meth(self):
        if False:
            return 10
        print('A.meth')
a = A()
a.meth()
a = A.__new__(A)
a.meth()
a = a.__new__(A)
a.meth()

class B:

    def __new__(self, v1, v2):
        if False:
            for i in range(10):
                print('nop')
        print('B.__new__', v1, v2)

    def __init__(self, v1, v2):
        if False:
            for i in range(10):
                print('nop')
        print('B.__init__', v1, v2)
print('B inst:', B(1, 2))

class Dummy:
    pass

class C:

    def __new__(cls):
        if False:
            while True:
                i = 10
        print('C.__new__')
        return Dummy()

    def __init__(self):
        if False:
            i = 10
            return i + 15
        print('C.__init__')
c = C()
print(isinstance(c, Dummy))