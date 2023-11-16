"""DelegatedFactory provider tests."""
from dependency_injector import providers
from .common import Example

def test_inheritance():
    if False:
        return 10
    assert isinstance(providers.DelegatedFactory(object), providers.Factory)

def test_is_provider():
    if False:
        return 10
    assert providers.is_provider(providers.DelegatedFactory(object)) is True

def test_is_delegated_provider():
    if False:
        print('Hello World!')
    assert providers.is_delegated(providers.DelegatedFactory(object)) is True

def test_repr():
    if False:
        while True:
            i = 10
    provider = providers.DelegatedFactory(Example)
    assert repr(provider) == '<dependency_injector.providers.DelegatedFactory({0}) at {1}>'.format(repr(Example), hex(id(provider)))