"""CallableDelegate provider tests."""
from dependency_injector import providers, errors
from pytest import raises
from .common import example

def test_is_delegate():
    if False:
        return 10
    provider = providers.Callable(example)
    delegate = providers.CallableDelegate(provider)
    assert isinstance(delegate, providers.Delegate)

def test_init_with_not_callable():
    if False:
        print('Hello World!')
    with raises(errors.Error):
        providers.CallableDelegate(providers.Object(object()))