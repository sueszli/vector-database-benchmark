class BazMeta(type):
    _private_count = 1

    def __new__(mcs, name, bases, attrs):
        if False:
            for i in range(10):
                print('nop')
        if mcs._private_count <= 5:
            mcs.some_method()
        return super().__new__(mcs, name, bases, attrs)

    def some_method():
        if False:
            while True:
                i = 10
        pass

class Bar:
    _private = True

    @classmethod
    def is_private(cls):
        if False:
            while True:
                i = 10
        return cls._private

class Foo(metaclass=BazMeta):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.public_thing = 'foo'
        self._private_thing = 'bar'
        self.__really_private_thing = 'baz'
        self.bar = Bar()

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'foo'

    def get_bar():
        if False:
            print('Hello World!')
        if self.bar._private:
            return None
        if self.bar()._private:
            return None
        if Bar._private_thing:
            return None
        if Foo._private_thing:
            return None
        Foo = Bar()
        if Foo._private_thing:
            return None
        return self.bar

    def public_func(self):
        if False:
            while True:
                i = 10
        super().public_func()

    def _private_func(self):
        if False:
            return 10
        super()._private_func()

    def __really_private_func(self, arg):
        if False:
            i = 10
            return i + 15
        super().__really_private_func(arg)

    def __eq__(self, other):
        if False:
            return 10
        return self._private_thing == other._private_thing
foo = Foo()
print(foo._private_thing)
print(foo.__really_private_thing)
print(foo._private_func())
print(foo.__really_private_func(1))
print(foo.bar._private)
print(foo()._private_thing)
print(foo()._private_thing__)
print(foo.public_thing)
print(foo.public_func())
print(foo.__dict__)
print(foo.__str__())
print(foo().__class__)
print(foo._asdict())
import os
os._exit()