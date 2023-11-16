def f(a, b):
    if False:
        i = 10
        return i + 15
    print(a, b)
f(1, **{'b': 2})
f(1, **{'b': val for val in range(1)})
try:
    f(1, **{len: 2})
except TypeError:
    print('TypeError')

class A:

    def f(self, a, b):
        if False:
            print('Hello World!')
        print(a, b)
a = A()
a.f(1, **{'b': 2})
a.f(1, **{'b': val for val in range(1)})

def f1(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    print(kwargs)
try:
    f1(**{'a': 1, 'b': 2}, **{'b': 3, 'c': 4})
except TypeError:
    print('TypeError')