try:
    NotImplemented
except NameError:
    print('SKIP')
    raise SystemExit

class C:

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.value = value

    def __str__(self):
        if False:
            print('Hello World!')
        return 'C({})'.format(self.value)

    def __add__(self, rhs):
        if False:
            while True:
                i = 10
        print(self, '+', rhs)
        return NotImplemented

    def __sub__(self, rhs):
        if False:
            print('Hello World!')
        print(self, '-', rhs)
        return NotImplemented

    def __lt__(self, rhs):
        if False:
            for i in range(10):
                print('nop')
        print(self, '<', rhs)
        return NotImplemented

    def __neg__(self):
        if False:
            while True:
                i = 10
        print('-', self)
        return NotImplemented
c = C(0)
try:
    c + 1
except TypeError:
    print('TypeError')
try:
    c - 2
except TypeError:
    print('TypeError')
try:
    c < 1
except TypeError:
    print('TypeError')
print(-c)
print(type(hash(NotImplemented)))