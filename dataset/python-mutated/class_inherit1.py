class A:

    def __init__(self, x):
        if False:
            print('Hello World!')
        print('A init', x)
        self.x = x

    def f(self):
        if False:
            print('Hello World!')
        print(self.x, self.y)

class B(A):

    def __init__(self, x, y):
        if False:
            while True:
                i = 10
        A.__init__(self, x)
        print('B init', x, y)
        self.y = y

    def g(self):
        if False:
            i = 10
            return i + 15
        print(self.x, self.y)
A(1)
b = B(1, 2)
b.f()
b.g()