"""
Topic: 扩展property的功能
Desc : 
"""

class Person:

    def __init__(self, name):
        if False:
            while True:
                i = 10
        self.name = name

    @property
    def name(self):
        if False:
            return 10
        return self._name

    @name.setter
    def name(self, value):
        if False:
            while True:
                i = 10
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._name = value

    @name.deleter
    def name(self):
        if False:
            print('Hello World!')
        raise AttributeError("Can't delete attribute")

class SubPerson(Person):

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        print('Getting name')
        return super().name

    @name.setter
    def name(self, value):
        if False:
            print('Hello World!')
        print('Setting name to', value)
        super(SubPerson, SubPerson).name.__set__(self, value)

    @name.deleter
    def name(self):
        if False:
            return 10
        print('Deleting name')
        super(SubPerson, SubPerson).name.__delete__(self)
s = SubPerson('Guido')
print(s.name)
s.name = 'Larry'

class String:

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.name = name

    def __get__(self, instance, cls):
        if False:
            i = 10
            return i + 15
        if instance is None:
            return self
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if False:
            print('Hello World!')
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        instance.__dict__[self.name] = value

class Person:
    name = String('name')

    def __init__(self, name):
        if False:
            return 10
        self.name = name

class SubPerson(Person):

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        print('Getting name')
        return super().name

    @name.setter
    def name(self, value):
        if False:
            print('Hello World!')
        print('Setting name to', value)
        super(SubPerson, SubPerson).name.__set__(self, value)

    @name.deleter
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        print('Deleting name')
        super(SubPerson, SubPerson).name.__delete__(self)