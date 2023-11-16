from abc import ABCMeta, abstractmethod

class Cls:

    @property
    def attribute(self, param, param1):
        if False:
            while True:
                i = 10
        return param + param1

    @property
    def attribute_keyword_only(self, *, param, param1):
        if False:
            while True:
                i = 10
        return param + param1

    @property
    def attribute_positional_only(self, param, param1, /):
        if False:
            print('Hello World!')
        return param + param1

class MyClassBase(metaclass=ABCMeta):
    """MyClassBase."""

    @property
    @abstractmethod
    def example(self):
        if False:
            print('Hello World!')
        'Getter.'

    @example.setter
    @abstractmethod
    def example(self, value):
        if False:
            i = 10
            return i + 15
        'Setter.'