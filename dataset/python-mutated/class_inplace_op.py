class A:

    def __init__(self, v):
        if False:
            for i in range(10):
                print('nop')
        self.v = v

    def __add__(self, o):
        if False:
            i = 10
            return i + 15
        return A(self.v + o.v)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'A({})'.format(self.v)
a = A(5)
b = a
a += A(3)
print(a)
print(b)

class L:

    def __init__(self, v):
        if False:
            print('Hello World!')
        self.v = v

    def __add__(self, o):
        if False:
            while True:
                i = 10
        print('L.__add__')
        return L(self.v + o.v)

    def __iadd__(self, o):
        if False:
            while True:
                i = 10
        self.v += o.v
        return self

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'L({})'.format(self.v)
c = L([1, 2])
d = c
c += L([3, 4])
print(c)
print(d)