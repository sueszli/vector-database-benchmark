"""MethodCaller provider traversal tests."""
from dependency_injector import providers

def test_traverse():
    if False:
        print('Hello World!')
    provider1 = providers.Provider()
    provided = provider1.provided
    method = provided.method
    provider = method.call()
    all_providers = list(provider.traverse())
    assert len(all_providers) == 3
    assert provider1 in all_providers
    assert provided in all_providers
    assert method in all_providers

def test_traverse_args():
    if False:
        while True:
            i = 10
    provider1 = providers.Provider()
    provided = provider1.provided
    method = provided.method
    provider2 = providers.Provider()
    provider = method.call('foo', provider2)
    all_providers = list(provider.traverse())
    assert len(all_providers) == 4
    assert provider1 in all_providers
    assert provider2 in all_providers
    assert provided in all_providers
    assert method in all_providers

def test_traverse_kwargs():
    if False:
        while True:
            i = 10
    provider1 = providers.Provider()
    provided = provider1.provided
    method = provided.method
    provider2 = providers.Provider()
    provider = method.call(foo='foo', bar=provider2)
    all_providers = list(provider.traverse())
    assert len(all_providers) == 4
    assert provider1 in all_providers
    assert provider2 in all_providers
    assert provided in all_providers
    assert method in all_providers

def test_traverse_overridden():
    if False:
        return 10
    provider1 = providers.Provider()
    provided = provider1.provided
    method = provided.method
    provider2 = providers.Provider()
    provider = method.call()
    provider.override(provider2)
    all_providers = list(provider.traverse())
    assert len(all_providers) == 4
    assert provider1 in all_providers
    assert provider2 in all_providers
    assert provided in all_providers
    assert method in all_providers