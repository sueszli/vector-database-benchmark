class C:

    @staticmethod
    def f(rhs):
        if False:
            print('Hello World!')
        print('f', rhs)

    @classmethod
    def g(self, rhs):
        if False:
            while True:
                i = 10
        print('g', rhs)

    @staticmethod
    def __sub__(rhs):
        if False:
            print('Hello World!')
        print('sub', rhs)

    @classmethod
    def __add__(self, rhs):
        if False:
            return 10
        print('add', rhs)

    @staticmethod
    def __getitem__(item):
        if False:
            while True:
                i = 10
        print('static get', item)
        return 'item'

    @staticmethod
    def __setitem__(item, value):
        if False:
            for i in range(10):
                print('nop')
        print('static set', item, value)

    @staticmethod
    def __delitem__(item):
        if False:
            print('Hello World!')
        print('static del', item)
c = C()
c.f(0)
c.g(0)
c - 1
c + 2
print(c[1])
c[1] = 2
del c[3]