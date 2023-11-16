"""Delegated singleton provider tests."""
from dependency_injector import providers
from pytest import fixture
from .common import Example
PROVIDER_CLASSES = [providers.DelegatedSingleton, providers.DelegatedThreadLocalSingleton, providers.DelegatedThreadSafeSingleton]

@fixture(params=PROVIDER_CLASSES)
def singleton_cls(request):
    if False:
        while True:
            i = 10
    return request.param

@fixture
def provider(singleton_cls):
    if False:
        for i in range(10):
            print('nop')
    return singleton_cls(Example)

def test_is_delegated_provider(provider):
    if False:
        return 10
    assert providers.is_delegated(provider) is True