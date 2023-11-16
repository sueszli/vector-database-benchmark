"""Dependency provider tests."""
from dependency_injector import containers, providers, errors
from pytest import fixture, raises

@fixture
def provider():
    if False:
        while True:
            i = 10
    return providers.Dependency(instance_of=list)

def test_init_optional():
    if False:
        i = 10
        return i + 15
    list_provider = providers.List(1, 2, 3)
    provider = providers.Dependency()
    provider.set_instance_of(list)
    provider.set_default(list_provider)
    assert provider.instance_of is list
    assert provider.default is list_provider
    assert provider() == [1, 2, 3]

def test_set_instance_of_returns_self(provider):
    if False:
        i = 10
        return i + 15
    assert provider.set_instance_of(list) is provider

def test_set_default_returns_self(provider):
    if False:
        for i in range(10):
            print('nop')
    assert provider.set_default(providers.Provider()) is provider

def test_init_with_not_class():
    if False:
        print('Hello World!')
    with raises(TypeError):
        providers.Dependency(object())

def test_with_abc():
    if False:
        for i in range(10):
            print('nop')
    try:
        import collections.abc as collections_abc
    except ImportError:
        import collections as collections_abc
    provider = providers.Dependency(collections_abc.Mapping)
    provider.provided_by(providers.Factory(dict))
    assert isinstance(provider(), collections_abc.Mapping)
    assert isinstance(provider(), dict)

def test_is_provider(provider):
    if False:
        return 10
    assert providers.is_provider(provider) is True

def test_provided_instance_provider(provider):
    if False:
        print('Hello World!')
    assert isinstance(provider.provided, providers.ProvidedInstance)

def test_default():
    if False:
        while True:
            i = 10
    provider = providers.Dependency(instance_of=dict, default={'foo': 'bar'})
    assert provider() == {'foo': 'bar'}

def test_default_attribute():
    if False:
        for i in range(10):
            print('nop')
    provider = providers.Dependency(instance_of=dict, default={'foo': 'bar'})
    assert provider.default() == {'foo': 'bar'}

def test_default_provider():
    if False:
        print('Hello World!')
    provider = providers.Dependency(instance_of=dict, default=providers.Factory(dict, foo='bar'))
    assert provider.default() == {'foo': 'bar'}

def test_default_attribute_provider():
    if False:
        return 10
    default = providers.Factory(dict, foo='bar')
    provider = providers.Dependency(instance_of=dict, default=default)
    assert provider.default() == {'foo': 'bar'}
    assert provider.default is default

def test_default_with_empty_dict():
    if False:
        while True:
            i = 10
    default = {}
    provider = providers.Dependency(instance_of=dict, default=default)
    assert provider() == default
    assert provider.default() == default

def test_default_with_empty_string():
    if False:
        return 10
    default = ''
    provider = providers.Dependency(instance_of=str, default=default)
    assert provider() == default
    assert provider.default() == default

def test_is_defined(provider):
    if False:
        while True:
            i = 10
    assert provider.is_defined is False

def test_is_defined_when_overridden(provider):
    if False:
        i = 10
        return i + 15
    provider.override('value')
    assert provider.is_defined is True

def test_is_defined_with_default():
    if False:
        i = 10
        return i + 15
    provider = providers.Dependency(default='value')
    assert provider.is_defined is True

def test_call_overridden(provider):
    if False:
        while True:
            i = 10
    provider.provided_by(providers.Factory(list))
    assert isinstance(provider(), list)

def test_call_overridden_but_not_instance_of(provider):
    if False:
        i = 10
        return i + 15
    provider.provided_by(providers.Factory(dict))
    with raises(errors.Error):
        provider()

def test_call_undefined(provider):
    if False:
        while True:
            i = 10
    with raises(errors.Error, match='Dependency is not defined'):
        provider()

def test_call_undefined_error_message_with_container_instance_parent():
    if False:
        print('Hello World!')

    class UserService:

        def __init__(self, database):
            if False:
                for i in range(10):
                    print('nop')
            self.database = database

    class Container(containers.DeclarativeContainer):
        database = providers.Dependency()
        user_service = providers.Factory(UserService, database=database)
    container = Container()
    with raises(errors.Error, match='Dependency "Container.database" is not defined'):
        container.user_service()

def test_call_undefined_error_message_with_container_provider_parent_deep():
    if False:
        for i in range(10):
            print('nop')

    class Database:
        pass

    class UserService:

        def __init__(self, db):
            if False:
                print('Hello World!')
            self.db = db

    class Gateways(containers.DeclarativeContainer):
        database_client = providers.Singleton(Database)

    class Services(containers.DeclarativeContainer):
        gateways = providers.DependenciesContainer()
        user = providers.Factory(UserService, db=gateways.database_client)

    class Container(containers.DeclarativeContainer):
        gateways = providers.Container(Gateways)
        services = providers.Container(Services)
    container = Container()
    with raises(errors.Error, match='Dependency "Container.services.gateways.database_client" is not defined'):
        container.services().user()

def test_call_undefined_error_message_with_dependenciescontainer_provider_parent():
    if False:
        return 10

    class UserService:

        def __init__(self, db):
            if False:
                return 10
            self.db = db

    class Services(containers.DeclarativeContainer):
        gateways = providers.DependenciesContainer()
        user = providers.Factory(UserService, db=gateways.database_client)
    services = Services()
    with raises(errors.Error, match='Dependency "Services.gateways.database_client" is not defined'):
        services.user()

def test_assign_parent(provider):
    if False:
        for i in range(10):
            print('nop')
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
        while True:
            i = 10
    container = providers.DependenciesContainer(name=provider)
    _ = providers.DependenciesContainer(container=container)
    assert provider.parent_name == 'container.name'

def test_parent_name_is_none():
    if False:
        print('Hello World!')
    provider = providers.Dependency()
    assert provider.parent_name is None

def test_parent_deepcopy(provider):
    if False:
        print('Hello World!')
    container = containers.DynamicContainer()
    container.name = provider
    copied = providers.deepcopy(container)
    assert container.name.parent is container
    assert copied.name.parent is copied
    assert container is not copied
    assert container.name is not copied.name
    assert container.name.parent is not copied.name.parent

def test_forward_attr_to_default():
    if False:
        while True:
            i = 10
    default = providers.Configuration()
    provider = providers.Dependency(default=default)
    provider.from_dict({'foo': 'bar'})
    assert default() == {'foo': 'bar'}

def test_forward_attr_to_overriding(provider):
    if False:
        return 10
    overriding = providers.Configuration()
    provider.override(overriding)
    provider.from_dict({'foo': 'bar'})
    assert overriding() == {'foo': 'bar'}

def test_forward_attr_to_none(provider):
    if False:
        for i in range(10):
            print('nop')
    with raises(AttributeError):
        provider.from_dict

def test_deepcopy(provider):
    if False:
        while True:
            i = 10
    provider_copy = providers.deepcopy(provider)
    assert provider is not provider_copy
    assert isinstance(provider, providers.Dependency)

def test_deepcopy_from_memo(provider):
    if False:
        i = 10
        return i + 15
    provider_copy_memo = providers.Provider()
    provider_copy = providers.deepcopy(provider, memo={id(provider): provider_copy_memo})
    assert provider_copy is provider_copy_memo

def test_deepcopy_overridden(provider):
    if False:
        while True:
            i = 10
    overriding_provider = providers.Provider()
    provider.override(overriding_provider)
    provider_copy = providers.deepcopy(provider)
    overriding_provider_copy = provider_copy.overridden[0]
    assert provider is not provider_copy
    assert isinstance(provider, providers.Dependency)
    assert overriding_provider is not overriding_provider_copy
    assert isinstance(overriding_provider_copy, providers.Provider)

def test_deep_copy_default_object():
    if False:
        for i in range(10):
            print('nop')
    default = {'foo': 'bar'}
    provider = providers.Dependency(dict, default=default)
    provider_copy = providers.deepcopy(provider)
    assert provider_copy() is default
    assert provider_copy.default() is default

def test_deep_copy_default_provider():
    if False:
        return 10
    bar = object()
    default = providers.Factory(dict, foo=providers.Object(bar))
    provider = providers.Dependency(dict, default=default)
    provider_copy = providers.deepcopy(provider)
    assert provider_copy() == {'foo': bar}
    assert provider_copy.default() == {'foo': bar}
    assert provider_copy()['foo'] is bar

def test_with_container_default_object():
    if False:
        return 10
    default = {'foo': 'bar'}

    class Container(containers.DeclarativeContainer):
        provider = providers.Dependency(dict, default=default)
    container = Container()
    assert container.provider() is default
    assert container.provider.default() is default

def test_with_container_default_provider():
    if False:
        i = 10
        return i + 15
    bar = object()

    class Container(containers.DeclarativeContainer):
        provider = providers.Dependency(dict, default=providers.Factory(dict, foo=providers.Object(bar)))
    container = Container()
    assert container.provider() == {'foo': bar}
    assert container.provider.default() == {'foo': bar}
    assert container.provider()['foo'] is bar

def test_with_container_default_provider_with_overriding():
    if False:
        i = 10
        return i + 15
    bar = object()
    baz = object()

    class Container(containers.DeclarativeContainer):
        provider = providers.Dependency(dict, default=providers.Factory(dict, foo=providers.Object(bar)))
    container = Container(provider=providers.Factory(dict, foo=providers.Object(baz)))
    assert container.provider() == {'foo': baz}
    assert container.provider.default() == {'foo': bar}
    assert container.provider()['foo'] is baz

def test_repr(provider):
    if False:
        while True:
            i = 10
    assert repr(provider) == '<dependency_injector.providers.Dependency({0}) at {1}>'.format(repr(list), hex(id(provider)))

def test_repr_in_container():
    if False:
        i = 10
        return i + 15

    class Container(containers.DeclarativeContainer):
        dependency = providers.Dependency(instance_of=int)
    container = Container()
    assert repr(container.dependency) == '<dependency_injector.providers.Dependency({0}) at {1}, container name: "Container.dependency">'.format(repr(int), hex(id(container.dependency)))

def test_external_dependency():
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(providers.ExternalDependency(), providers.Dependency)