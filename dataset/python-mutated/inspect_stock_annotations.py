a: int = 3
b: str = 'foo'

class MyClass:
    a: int = 4
    b: str = 'bar'

    def __init__(self, a, b):
        if False:
            while True:
                i = 10
        self.a = a
        self.b = b

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, MyClass) and self.a == other.a and (self.b == other.b)

def function(a: int, b: str) -> MyClass:
    if False:
        return 10
    return MyClass(a, b)

def function2(a: int, b: 'str', c: MyClass) -> MyClass:
    if False:
        i = 10
        return i + 15
    pass

def function3(a: 'int', b: 'str', c: 'MyClass'):
    if False:
        while True:
            i = 10
    pass

class UnannotatedClass:
    pass

def unannotated_function(a, b, c):
    if False:
        while True:
            i = 10
    pass