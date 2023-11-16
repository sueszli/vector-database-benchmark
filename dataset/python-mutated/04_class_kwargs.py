import abc
import unittest

class TestABCWithInitSubclass(unittest.TestCase):

    def test_works_with_init_subclass(self):
        if False:
            i = 10
            return i + 15

        class ReceivesClassKwargs:

            def __init_subclass__(cls, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init_subclass__()

        class Receiver(ReceivesClassKwargs, abc.ABC, x=1, y=2, z=3):
            pass

def test_abstractmethod_integration(self):
    if False:
        return 10
    for abstractthing in [abc.abstractmethod]:

        class C(metaclass=abc.ABCMeta):

            @abstractthing
            def foo(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass