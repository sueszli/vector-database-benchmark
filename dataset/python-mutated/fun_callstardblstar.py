def f(a, b, c, d):
    if False:
        while True:
            i = 10
    print(a, b, c, d)
f(*(1, 2), **{'c': 3, 'd': 4})
f(*(1, 2), **{['c', 'd'][i]: 3 + i for i in range(2)})
try:
    eval("f(**{'a': 1}, *(2, 3, 4))")
except SyntaxError:
    print('SyntaxError')

class A:

    def f(self, a, b, c, d):
        if False:
            print('Hello World!')
        print(a, b, c, d)
a = A()
a.f(*(1, 2), **{'c': 3, 'd': 4})
a.f(*(1, 2), **{['c', 'd'][i]: 3 + i for i in range(2)})
try:
    eval("a.f(**{'a': 1}, *(2, 3, 4))")
except SyntaxError:
    print('SyntaxError')

def f2(*args, **kwargs):
    if False:
        print('Hello World!')
    print(len(args), len(kwargs))
f2(*iter(range(4)), **{'a': 1})
f2(*iter(range(100)), **{str(i): i for i in range(100)})
print(1, *iter((1, 2, 3)), *iter((1, 2, 3)), 4, 5, sep=',')