class Person:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.name = 'monty'

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, Person) and other.name == self.name

class Language:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.name = 'python'

    def __eq__(self, other):
        if False:
            return 10
        return isinstance(other, Language) and other.name == self.name

    def __hash__(self):
        if False:
            return 10
        return hash(self.name)

class MyClass:

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return True
    __hash__ = None