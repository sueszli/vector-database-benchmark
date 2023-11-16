"""Object provider tests."""
from dependency_injector import providers

def test_is_provider():
    if False:
        while True:
            i = 10
    assert providers.is_provider(providers.Object(object())) is True

def test_init_optional_provides():
    if False:
        for i in range(10):
            print('nop')
    instance = object()
    provider = providers.Object()
    provider.set_provides(instance)
    assert provider.provides is instance
    assert provider() is instance

def test_set_provides_returns_():
    if False:
        return 10
    provider = providers.Object()
    assert provider.set_provides(object()) is provider

def test_provided_instance_provider():
    if False:
        i = 10
        return i + 15
    provider = providers.Object(object())
    assert isinstance(provider.provided, providers.ProvidedInstance)

def test_call_object_provider():
    if False:
        for i in range(10):
            print('nop')
    obj = object()
    assert providers.Object(obj)() is obj

def test_call_overridden_object_provider():
    if False:
        i = 10
        return i + 15
    obj1 = object()
    obj2 = object()
    provider = providers.Object(obj1)
    provider.override(providers.Object(obj2))
    assert provider() is obj2

def test_deepcopy():
    if False:
        return 10
    provider = providers.Object(1)
    provider_copy = providers.deepcopy(provider)
    assert provider is not provider_copy
    assert isinstance(provider, providers.Object)

def test_deepcopy_from_memo():
    if False:
        return 10
    provider = providers.Object(1)
    provider_copy_memo = providers.Provider()
    provider_copy = providers.deepcopy(provider, memo={id(provider): provider_copy_memo})
    assert provider_copy is provider_copy_memo

def test_deepcopy_overridden():
    if False:
        return 10
    provider = providers.Object(1)
    overriding_provider = providers.Provider()
    provider.override(overriding_provider)
    provider_copy = providers.deepcopy(provider)
    overriding_provider_copy = provider_copy.overridden[0]
    assert provider is not provider_copy
    assert isinstance(provider, providers.Object)
    assert overriding_provider is not overriding_provider_copy
    assert isinstance(overriding_provider_copy, providers.Provider)

def test_deepcopy_doesnt_copy_provided_object():
    if False:
        while True:
            i = 10
    some_object = object()
    provider = providers.Object(some_object)
    provider_copy = providers.deepcopy(provider)
    assert provider() is some_object
    assert provider_copy() is some_object

def test_repr():
    if False:
        return 10
    some_object = object()
    provider = providers.Object(some_object)
    assert repr(provider) == '<dependency_injector.providers.Object({0}) at {1}>'.format(repr(some_object), hex(id(provider)))