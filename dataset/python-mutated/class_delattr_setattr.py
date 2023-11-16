try:

    class Test:

        def __delattr__(self, attr):
            if False:
                i = 10
                return i + 15
            pass
    del Test().noexist
except AttributeError:
    print('SKIP')
    raise SystemExit

class A:

    def __getattr__(self, attr):
        if False:
            while True:
                i = 10
        print('get', attr)
        return 1

    def __setattr__(self, attr, val):
        if False:
            return 10
        print('set', attr, val)

    def __delattr__(self, attr):
        if False:
            return 10
        print('del', attr)
a = A()
print(getattr(a, 'foo'))
setattr(a, 'bar', 2)
delattr(a, 'baz')
getattr(a, '__getattr__')
getattr(a, '__setattr__')
getattr(a, '__delattr__')
setattr(a, '__setattr__', 1)
delattr(a, '__delattr__')

class B:

    def __init__(self, d):
        if False:
            i = 10
            return i + 15
        B.d = d

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        if attr in B.d:
            return B.d[attr]
        else:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        if False:
            print('Hello World!')
        B.d[attr] = value

    def __delattr__(self, attr):
        if False:
            for i in range(10):
                print('nop')
        del B.d[attr]
a = B({'a': 1, 'b': 2})
print(a.a, a.b)
a.a = 3
print(a.a, a.b)
del a.a
try:
    print(a.a)
except AttributeError:
    print('AttributeError')

class C:

    def __init__(self):
        if False:
            return 10
        pass

    def __setattr__(self, attr, value):
        if False:
            i = 10
            return i + 15
        print(attr, '=', value)

    def __delattr__(self, attr):
        if False:
            print('Hello World!')
        print('del', attr)
c = C()
c.a = 5
try:
    print(c.a)
except AttributeError:
    print('AttributeError')
object.__setattr__(c, 'a', 5)
super(C, c).__setattr__('b', 6)
print(c.a)
print(c.b)
try:
    object.__setattr__(c, 5, 5)
except TypeError:
    print('TypeError')
del c.a
print(c.a)
object.__delattr__(c, 'a')
try:
    print(c.a)
except AttributeError:
    print('AttributeError')
super(C, c).__delattr__('b')
try:
    print(c.b)
except AttributeError:
    print('AttributeError')
try:
    object.__delattr__(c, 'c')
except AttributeError:
    print('AttributeError')