class foo(object):

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self.x = value

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        print('eq')
        return self.x == other.x

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        print('lt')
        return self.x < other.x

    def __gt__(self, other):
        if False:
            print('Hello World!')
        print('gt')
        return self.x > other.x

    def __le__(self, other):
        if False:
            return 10
        print('le')
        return self.x <= other.x

    def __ge__(self, other):
        if False:
            i = 10
            return i + 15
        print('ge')
        return self.x >= other.x
for i in range(3):
    for j in range(3):
        print(foo(i) == foo(j))
        print(foo(i) < foo(j))
        print(foo(i) > foo(j))
        print(foo(i) <= foo(j))
        print(foo(i) >= foo(j))