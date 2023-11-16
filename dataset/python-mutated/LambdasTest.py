from __future__ import print_function

def lambdaContainer(x):
    if False:
        while True:
            i = 10
    f = lambda c: c
    g = lambda c: c if x else c * c
    y = f(x)
    z = g(4)
    print('Lambda with conditional expression gives', z)
    if 'a' <= x <= y <= 'z':
        print('Four')
    if 'a' <= x <= 'z':
        print('Yes')
    if 'a' <= x > 'z':
        print('Yes1')
    if 'a' <= ('1' if x else '2') > 'z':
        print('Yes2')
    if 'a' <= ('1' if x else '2') > 'z' > undefined_global:
        print('Yes3')
    z = lambda foo=y: foo
    print('Lambda defaulted gives', z())
lambdaContainer('b')

def lambdaGenerator():
    if False:
        while True:
            i = 10
    x = lambda : (yield 3)
    gen = x()
    print('Lambda generator gives', next(gen))
lambdaGenerator()

def lambdaDirectCall():
    if False:
        return 10
    args = range(7)
    x = (lambda *args: args)(*args)
    print('Lambda direct call gave', x)
lambdaDirectCall()