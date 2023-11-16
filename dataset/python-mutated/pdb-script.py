def nop(_):
    if False:
        for i in range(10):
            print('nop')
    pass

class Foo(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        nop('__init__')
        self.bar()
        self.baz()
        self.bra()

    @classmethod
    def bar(cls):
        if False:
            for i in range(10):
                print('nop')
        nop(cls.__name__)

    @staticmethod
    def baz():
        if False:
            for i in range(10):
                print('nop')
        nop(1)

    def bra(self):
        if False:
            i = 10
            return i + 15
        nop(self.__class__.__name__)

def brah():
    if False:
        print('Hello World!')
    nop('brah')
f = Foo()
Foo.bar()
Foo.baz()
Foo.bra(f)
f.bar()
f.baz()
f.bra()
brah()