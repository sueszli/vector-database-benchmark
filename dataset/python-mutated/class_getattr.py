class C:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__add__ = lambda : print('member __add__')

    def __add__(self, x):
        if False:
            i = 10
            return i + 15
        print('__add__')

    def __getattr__(self, attr):
        if False:
            while True:
                i = 10
        print('__getattr__', attr)
        return None
c = C()
c.add
c.__add__()
c + 1