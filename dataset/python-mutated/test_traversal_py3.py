"""Container traversing tests."""
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    obj_factory = providers.DelegatedFactory(dict, foo=providers.Resource(dict, foo='bar'), bar=providers.Resource(dict, foo='bar'))

def test_nested_providers():
    if False:
        for i in range(10):
            print('nop')
    container = Container()
    all_providers = list(container.traverse())
    assert container.obj_factory in all_providers
    assert container.obj_factory.kwargs['foo'] in all_providers
    assert container.obj_factory.kwargs['bar'] in all_providers
    assert len(all_providers) == 3

def test_nested_providers_with_filtering():
    if False:
        while True:
            i = 10
    container = Container()
    all_providers = list(container.traverse(types=[providers.Resource]))
    assert container.obj_factory.kwargs['foo'] in all_providers
    assert container.obj_factory.kwargs['bar'] in all_providers
    assert len(all_providers) == 2

def test_container_cls_nested_providers():
    if False:
        i = 10
        return i + 15
    all_providers = list(Container.traverse())
    assert Container.obj_factory in all_providers
    assert Container.obj_factory.kwargs['foo'] in all_providers
    assert Container.obj_factory.kwargs['bar'] in all_providers
    assert len(all_providers) == 3

def test_container_cls_nested_providers_with_filtering():
    if False:
        print('Hello World!')
    all_providers = list(Container.traverse(types=[providers.Resource]))
    assert Container.obj_factory.kwargs['foo'] in all_providers
    assert Container.obj_factory.kwargs['bar'] in all_providers
    assert len(all_providers) == 2