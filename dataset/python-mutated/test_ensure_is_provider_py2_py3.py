"""Provider utils tests."""
from dependency_injector import providers, errors
from pytest import raises

def test_with_instance():
    if False:
        i = 10
        return i + 15
    provider = providers.Provider()
    assert providers.ensure_is_provider(provider), provider

def test_with_class():
    if False:
        for i in range(10):
            print('nop')
    with raises(errors.Error):
        providers.ensure_is_provider(providers.Provider)

def test_with_string():
    if False:
        for i in range(10):
            print('nop')
    with raises(errors.Error):
        providers.ensure_is_provider('some_string')

def test_with_object():
    if False:
        for i in range(10):
            print('nop')
    with raises(errors.Error):
        providers.ensure_is_provider(object())