"""Tests for provider async mode API."""
from dependency_injector import providers
from pytest import fixture

@fixture
def provider():
    if False:
        for i in range(10):
            print('nop')
    return providers.Provider()

def test_default_mode(provider: providers.Provider):
    if False:
        for i in range(10):
            print('nop')
    assert provider.is_async_mode_enabled() is False
    assert provider.is_async_mode_disabled() is False
    assert provider.is_async_mode_undefined() is True

def test_enable(provider: providers.Provider):
    if False:
        print('Hello World!')
    provider.enable_async_mode()
    assert provider.is_async_mode_enabled() is True
    assert provider.is_async_mode_disabled() is False
    assert provider.is_async_mode_undefined() is False

def test_disable(provider: providers.Provider):
    if False:
        print('Hello World!')
    provider.disable_async_mode()
    assert provider.is_async_mode_enabled() is False
    assert provider.is_async_mode_disabled() is True
    assert provider.is_async_mode_undefined() is False

def test_reset(provider: providers.Provider):
    if False:
        while True:
            i = 10
    provider.enable_async_mode()
    assert provider.is_async_mode_enabled() is True
    assert provider.is_async_mode_disabled() is False
    assert provider.is_async_mode_undefined() is False
    provider.reset_async_mode()
    assert provider.is_async_mode_enabled() is False
    assert provider.is_async_mode_disabled() is False
    assert provider.is_async_mode_undefined() is True