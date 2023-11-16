"""DelegatedCallable provider tests."""
from dependency_injector import providers
from .common import example

def test_inheritance():
    if False:
        print('Hello World!')
    assert isinstance(providers.DelegatedCallable(example), providers.Callable)

def test_is_provider():
    if False:
        print('Hello World!')
    assert providers.is_provider(providers.DelegatedCallable(example)) is True

def test_is_delegated_provider():
    if False:
        for i in range(10):
            print('nop')
    provider = providers.DelegatedCallable(example)
    assert providers.is_delegated(provider) is True

def test_repr():
    if False:
        while True:
            i = 10
    provider = providers.DelegatedCallable(example)
    assert repr(provider) == '<dependency_injector.providers.DelegatedCallable({0}) at {1}>'.format(repr(example), hex(id(provider)))