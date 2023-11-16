def f(a, b=None, c=None):
    if False:
        while True:
            i = 10
    print(a, b, c)
f(**{'a': 1}, **{'b': 2})
f(**{'a': 1}, **{'b': 2}, c=3)
f(**{'a': 1}, b=2, **{'c': 3})
try:
    f(1, **{'b': 2}, **{'b': 3})
except TypeError:
    print('TypeError')

class A:

    def f(self, a, b=None, c=None):
        if False:
            i = 10
            return i + 15
        print(a, b, c)
a = A()
a.f(**{'a': 1}, **{'b': 2})
a.f(**{'a': 1}, **{'b': 2}, c=3)
a.f(**{'a': 1}, b=2, **{'c': 3})
try:
    a.f(1, **{'b': 2}, **{'b': 3})
except TypeError:
    print('TypeError')