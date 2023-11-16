try:
    object.__new__
except AttributeError:
    print('SKIP')
    raise SystemExit

class Foo:

    def __new__(cls):
        if False:
            i = 10
            return i + 15
        print('in __new__')
        raise RuntimeError

    def __init__(self):
        if False:
            while True:
                i = 10
        print('in __init__')
        self.attr = 'something'
o = object.__new__(Foo)
print('Result of __new__ has .attr:', hasattr(o, 'attr'))
print('Result of __new__ is already a Foo:', isinstance(o, Foo))
o.__init__()
print('After __init__ has .attr:', hasattr(o, 'attr'))
print('.attr:', o.attr)
try:
    object.__new__(1)
except TypeError:
    print('TypeError')
try:
    object.__new__(int)
except TypeError:
    print('TypeError')