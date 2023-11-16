"""DependenciesContainer provider traversal tests."""
from dependency_injector import providers

def test_traversal():
    if False:
        for i in range(10):
            print('nop')
    provider = providers.DependenciesContainer()
    all_providers = list(provider.traverse())
    assert len(all_providers) == 0

def test_traversal_default():
    if False:
        return 10
    another_provider = providers.Provider()
    provider = providers.DependenciesContainer(default=another_provider)
    all_providers = list(provider.traverse())
    assert len(all_providers) == 1
    assert another_provider in all_providers

def test_traversal_fluent_interface():
    if False:
        while True:
            i = 10
    provider = providers.DependenciesContainer()
    provider1 = provider.provider1
    provider2 = provider.provider2
    all_providers = list(provider.traverse())
    assert len(all_providers) == 2
    assert provider1 in all_providers
    assert provider2 in all_providers

def test_traversal_overriding():
    if False:
        for i in range(10):
            print('nop')
    provider1 = providers.Provider()
    provider2 = providers.Provider()
    provider3 = providers.DependenciesContainer(provider1=provider1, provider2=provider2)
    provider = providers.DependenciesContainer()
    provider.override(provider3)
    all_providers = list(provider.traverse())
    assert len(all_providers) == 5
    assert provider1 in all_providers
    assert provider2 in all_providers
    assert provider3 in all_providers
    assert provider.provider1 in all_providers
    assert provider.provider2 in all_providers