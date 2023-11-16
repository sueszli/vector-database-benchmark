"""Factory delegate provider tests."""
from dependency_injector import providers, errors
from pytest import fixture, raises

@fixture
def factory():
    if False:
        i = 10
        return i + 15
    return providers.Factory(object)

@fixture
def delegate(factory):
    if False:
        return 10
    return providers.FactoryDelegate(factory)

def test_is_delegate(delegate):
    if False:
        return 10
    assert isinstance(delegate, providers.Delegate)

def test_init_with_not_factory():
    if False:
        while True:
            i = 10
    with raises(errors.Error):
        providers.FactoryDelegate(providers.Object(object()))