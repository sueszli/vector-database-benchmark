import abc
import unittest
from inspect import isabstract

def test_abstractmethod_integration(self):
    if False:
        print('Hello World!')
    for abstractthing in [abc.abstractmethod]:

        class C(metaclass=abc.ABCMeta):

            @abstractthing
            def foo(self):
                if False:
                    print('Hello World!')
                pass

            def bar(self):
                if False:
                    i = 10
                    return i + 15
                pass
        assert C.__abstractmethods__, {'foo'}
        assert isabstract(C)
        pass
test_abstractmethod_integration(None)