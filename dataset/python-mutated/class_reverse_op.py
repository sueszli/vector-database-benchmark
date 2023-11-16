class A:

    def __init__(self, v):
        if False:
            return 10
        self.v = v

    def __add__(self, o):
        if False:
            print('Hello World!')
        if isinstance(o, A):
            return A(self.v + o.v)
        return A(self.v + o)

    def __radd__(self, o):
        if False:
            print('Hello World!')
        return A(self.v + o)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'A({})'.format(self.v)
print(A(3) + 1)
print(2 + A(5))

class B:

    def __init__(self, v):
        if False:
            for i in range(10):
                print('nop')
        self.v = v

    def __repr__(self):
        if False:
            return 10
        return 'B({})'.format(self.v)

    def __ror__(self, o):
        if False:
            print('Hello World!')
        return B(o + '|' + self.v)

    def __radd__(self, o):
        if False:
            while True:
                i = 10
        return B(o + '+' + self.v)

    def __rmul__(self, o):
        if False:
            for i in range(10):
                print('nop')
        return B(o + '*' + self.v)

    def __rtruediv__(self, o):
        if False:
            print('Hello World!')
        return B(o + '/' + self.v)
print('a' | B('b'))
print('a' + B('b'))
print('a' * B('b'))
print('a' / B('b'))
x = 'a'
x |= B('b')
print(x)
x = 'a'
x += B('b')
print(x)
x = 'a'
x *= B('b')
print(x)
x = 'a'
x /= B('b')
print(x)