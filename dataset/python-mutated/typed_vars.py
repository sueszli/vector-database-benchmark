attr1: str = ''
attr2: str
attr3 = ''

class _Descriptor:

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.__doc__ = f'This is {name}'

    def __get__(self):
        if False:
            i = 10
            return i + 15
        pass

class Class:
    attr1: int = 0
    attr2: int
    attr3 = 0
    descr4: int = _Descriptor('descr4')

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.attr4: int = 0
        self.attr5: int
        self.attr6 = 0
        'attr6'

class Derived(Class):
    attr7: int
Alias = Derived