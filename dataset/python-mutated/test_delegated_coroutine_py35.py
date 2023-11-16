"""DelegatedCoroutine provider tests."""
from dependency_injector import providers
from .common import example

def test_inheritance():
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(providers.DelegatedCoroutine(example), providers.Coroutine)

def test_is_provider():
    if False:
        i = 10
        return i + 15
    assert providers.is_provider(providers.DelegatedCoroutine(example)) is True

def test_is_delegated_provider():
    if False:
        while True:
            i = 10
    provider = providers.DelegatedCoroutine(example)
    assert providers.is_delegated(provider) is True

def test_repr():
    if False:
        i = 10
        return i + 15
    provider = providers.DelegatedCoroutine(example)
    assert repr(provider) == '<dependency_injector.providers.DelegatedCoroutine({0}) at {1}>'.format(repr(example), hex(id(provider)))