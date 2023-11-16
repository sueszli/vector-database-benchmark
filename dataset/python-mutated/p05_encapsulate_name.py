"""
Topic: 隐藏私有属性
Desc : 
"""

class A:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._internal = 0
        self.public = 1

    def public_method(self):
        if False:
            print('Hello World!')
        '\n        A public method\n        '
        pass

    def _internal_method(self):
        if False:
            print('Hello World!')
        pass

class B:

    def __init__(self):
        if False:
            return 10
        self.__private = 0

    def __private_method(self):
        if False:
            while True:
                i = 10
        pass

    def public_method(self):
        if False:
            i = 10
            return i + 15
        pass
        self.__private_method()

class C(B):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.__private = 1

    def __private_method(self):
        if False:
            for i in range(10):
                print('nop')
        pass