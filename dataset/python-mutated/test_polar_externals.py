class Foo:

    def __init__(self, name=''):
        if False:
            for i in range(10):
                print('nop')
        self.name = name

    def foo(self):
        if False:
            while True:
                i = 10
        return 'Foo!'

class Bar(Foo):

    def foo(self):
        if False:
            return 10
        return 'Bar!'

class Qux:
    pass

class MyClass:

    def __init__(self, x='', y=''):
        if False:
            print('Hello World!')
        self.x = x
        self.y = y

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, MyClass):
            return self.x == other.x and self.y == other.y
        return False

class YourClass:
    pass

class OurClass(MyClass, YourClass):
    pass