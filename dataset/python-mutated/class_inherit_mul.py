class A:

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        print('A init', x)
        self.x = x

    def f(self):
        if False:
            for i in range(10):
                print('nop')
        print(self.x)

    def f2(self):
        if False:
            print('Hello World!')
        print(self.x)

class B:

    def __init__(self, x):
        if False:
            return 10
        print('B init', x)
        self.x = x

    def f(self):
        if False:
            while True:
                i = 10
        print(self.x)

    def f3(self):
        if False:
            return 10
        print(self.x)

class Sub(A, B):

    def __init__(self):
        if False:
            return 10
        A.__init__(self, 1)
        B.__init__(self, 2)
        print('Sub init')

    def g(self):
        if False:
            while True:
                i = 10
        print(self.x)
print(issubclass(Sub, A))
print(issubclass(Sub, B))
o = Sub()
print(o.x)
o.f()
o.f2()
o.f3()