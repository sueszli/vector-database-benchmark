"""DependencyContainer provider tests."""
from dependency_injector import containers, providers, errors
from pytest import fixture, raises

class Container(containers.DeclarativeContainer):
    dependency = providers.Provider()

@fixture
def provider():
    if False:
        i = 10
        return i + 15
    return providers.DependenciesContainer()

@fixture
def container():
    if False:
        i = 10
        return i + 15
    return Container()

def test_getattr(provider):
    if False:
        for i in range(10):
            print('nop')
    has_dependency = hasattr(provider, 'dependency')
    dependency = provider.dependency
    assert isinstance(dependency, providers.Dependency)
    assert dependency is provider.dependency
    assert has_dependency is True
    assert dependency.last_overriding is None

def test_getattr_with_container(provider, container):
    if False:
        i = 10
        return i + 15
    provider.override(container)
    dependency = provider.dependency
    assert dependency.overridden == (container.dependency,)
    assert dependency.last_overriding is container.dependency

def test_providers(provider):
    if False:
        i = 10
        return i + 15
    dependency1 = provider.dependency1
    dependency2 = provider.dependency2
    assert provider.providers == {'dependency1': dependency1, 'dependency2': dependency2}

def test_override(provider, container):
    if False:
        return 10
    dependency = provider.dependency
    provider.override(container)
    assert dependency.overridden == (container.dependency,)
    assert dependency.last_overriding is container.dependency

def test_reset_last_overriding(provider, container):
    if False:
        while True:
            i = 10
    dependency = provider.dependency
    provider.override(container)
    provider.reset_last_overriding()
    assert dependency.last_overriding is None
    assert dependency.last_overriding is None

def test_reset_override(provider, container):
    if False:
        return 10
    dependency = provider.dependency
    provider.override(container)
    provider.reset_override()
    assert dependency.overridden == tuple()
    assert not dependency.overridden

def test_assign_parent(provider):
    if False:
        i = 10
        return i + 15
    parent = providers.DependenciesContainer()
    provider.assign_parent(parent)
    assert provider.parent is parent

def test_parent_name(provider):
    if False:
        print('Hello World!')
    container = containers.DynamicContainer()
    container.name = provider
    assert provider.parent_name == 'name'

def test_parent_name_with_deep_parenting(provider):
    if False:
        for i in range(10):
            print('nop')
    container = providers.DependenciesContainer(name=provider)
    _ = providers.DependenciesContainer(container=container)
    assert provider.parent_name == 'container.name'

def test_parent_name_is_none(provider):
    if False:
        return 10
    assert provider.parent_name is None

def test_parent_deepcopy(provider, container):
    if False:
        for i in range(10):
            print('nop')
    container.name = provider
    copied = providers.deepcopy(container)
    assert container.name.parent is container
    assert copied.name.parent is copied
    assert container is not copied
    assert container.name is not copied.name
    assert container.name.parent is not copied.name.parent

def test_parent_set_on__getattr__(provider):
    if False:
        print('Hello World!')
    assert isinstance(provider.name, providers.Dependency)
    assert provider.name.parent is provider

def test_parent_set_on__init__():
    if False:
        for i in range(10):
            print('nop')
    provider = providers.Dependency()
    container = providers.DependenciesContainer(name=provider)
    assert container.name is provider
    assert container.name.parent is container

def test_resolve_provider_name(provider):
    if False:
        i = 10
        return i + 15
    assert provider.resolve_provider_name(provider.name) == 'name'

def test_resolve_provider_name_no_provider(provider):
    if False:
        return 10
    with raises(errors.Error):
        provider.resolve_provider_name(providers.Provider())