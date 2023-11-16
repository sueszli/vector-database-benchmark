from micropython import const
X = const(1)
Y = const(2)
Z = const(3)
import micropython as X
print(globals()['X'])

def X():
    if False:
        print('Hello World!')
    print('function X', X)
globals()['X']()

def f(X, *Y, **Z):
    if False:
        print('Hello World!')
    pass
f(1)

class X:

    def f(self):
        if False:
            return 10
        print('class X', X)
globals()['X']().f()

class A:
    C1 = const(4)

    def X(self):
        if False:
            while True:
                i = 10
        print('method X', Y, C1, self.C1)
A().X()