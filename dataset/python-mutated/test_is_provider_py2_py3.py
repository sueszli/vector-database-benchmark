"""Provider utils tests."""
from dependency_injector import providers

def test_with_instance():
    if False:
        print('Hello World!')
    assert providers.is_provider(providers.Provider()) is True

def test_with_class():
    if False:
        for i in range(10):
            print('nop')
    assert providers.is_provider(providers.Provider) is False

def test_with_string():
    if False:
        while True:
            i = 10
    assert providers.is_provider('some_string') is False

def test_with_object():
    if False:
        print('Hello World!')
    assert providers.is_provider(object()) is False

def test_with_subclass_instance():
    if False:
        i = 10
        return i + 15

    class SomeProvider(providers.Provider):
        pass
    assert providers.is_provider(SomeProvider()) is True

def test_with_class_with_getattr():
    if False:
        i = 10
        return i + 15

    class SomeClass(object):

        def __getattr__(self, _):
            if False:
                while True:
                    i = 10
            return False
    assert providers.is_provider(SomeClass()) is False