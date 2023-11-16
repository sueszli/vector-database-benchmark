"""
Topic: sample
Desc : 
"""

class A:

    def spam(self):
        if False:
            print('Hello World!')
        print('A.spam')

class B(A):

    def spam(self):
        if False:
            for i in range(10):
                print('nop')
        print('B.spam')
        super().spam()

class A1:

    def __init__(self):
        if False:
            return 10
        self.x = 0

class B1(A1):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.y = 1

class Proxy:

    def __init__(self, obj):
        if False:
            return 10
        self._obj = obj

    def __getattr__(self, name):
        if False:
            return 10
        return getattr(self._obj, name)

    def __setattr__(self, name, value):
        if False:
            for i in range(10):
                print('nop')
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            setattr(self._obj, name, value)

class Base:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        print('Base.__init__')

class AA(Base):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        print('AA.__init__')

class BB(Base):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        print('BB.__init__')

class CC(AA, BB):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        print('CC.__init__')
CC()
print(CC.__mro__)

class A3:

    def spam(self):
        if False:
            while True:
                i = 10
        print('A3.spam')
        super().spam()

class B3:

    def spam(self):
        if False:
            for i in range(10):
                print('nop')
        print('B3.spam')

class C3(A3, B3):
    pass
print(C3.__mro__)
C3().spam()