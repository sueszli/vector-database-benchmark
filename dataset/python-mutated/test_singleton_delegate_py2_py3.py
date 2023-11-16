"""SingletonDelegate provider tests."""
from dependency_injector import providers, errors
from pytest import fixture, raises

@fixture
def provider():
    if False:
        i = 10
        return i + 15
    return providers.Singleton(object)

@fixture
def delegate(provider):
    if False:
        print('Hello World!')
    return providers.SingletonDelegate(provider)

def test_is_delegate(delegate):
    if False:
        return 10
    assert isinstance(delegate, providers.Delegate)

def test_init_with_not_factory():
    if False:
        for i in range(10):
            print('nop')
    with raises(errors.Error):
        providers.SingletonDelegate(providers.Object(object()))