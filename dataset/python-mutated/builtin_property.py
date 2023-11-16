try:
    property
except:
    print('SKIP')
    raise SystemExit
property()
property(1, 2, 3)
p = property()
p.getter(1)
p.setter(2)
p.deleter(3)

class A:

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self._x = x

    @property
    def x(self):
        if False:
            while True:
                i = 10
        print('x get')
        return self._x
a = A(1)
print(a.x)
try:
    a.x = 2
except AttributeError:
    print('AttributeError')

class B:

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self._x = x

    def xget(self):
        if False:
            return 10
        print('x get')
        return self._x

    def xset(self, value):
        if False:
            return 10
        print('x set')
        self._x = value

    def xdel(self):
        if False:
            print('Hello World!')
        print('x del')
    x = property(xget, xset, xdel)
b = B(3)
print(b.x)
b.x = 4
print(b.x)
del b.x

class C:

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self._x = x

    @property
    def x(self):
        if False:
            while True:
                i = 10
        print('x get')
        return self._x

    @x.setter
    def x(self, value):
        if False:
            i = 10
            return i + 15
        print('x set')
        self._x = value

    @x.deleter
    def x(self):
        if False:
            i = 10
            return i + 15
        print('x del')
c = C(5)
print(c.x)
c.x = 6
print(c.x)
del c.x

class D:
    prop = property()
d = D()
try:
    d.prop
except AttributeError:
    print('AttributeError')
try:
    d.prop = 1
except AttributeError:
    print('AttributeError')
try:
    del d.prop
except AttributeError:
    print('AttributeError')

class E:
    p = property(lambda self: 42, doc='This is truth.')
print(E().p)

class F:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.prop_member = property()
print(type(F().prop_member))