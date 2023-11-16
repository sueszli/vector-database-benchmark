"""ProvidedInstance provider tests."""
from dependency_injector import containers, providers
from pytest import fixture

class Service:

    def __init__(self, value):
        if False:
            return 10
        self.value = value
        self.values = [self.value]

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.value

    def __getitem__(self, item):
        if False:
            return 10
        return self.values[item]

    def get_value(self):
        if False:
            while True:
                i = 10
        return self.value

    def get_closure(self):
        if False:
            return 10

        def closure():
            if False:
                while True:
                    i = 10
            return self.value
        return closure

class Client:

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self.value = value

class Container(containers.DeclarativeContainer):
    service = providers.Singleton(Service, value='foo')
    client_attribute = providers.Factory(Client, value=service.provided.value)
    client_item = providers.Factory(Client, value=service.provided[0])
    client_attribute_item = providers.Factory(Client, value=service.provided.values[0])
    client_method_call = providers.Factory(Client, value=service.provided.get_value.call())
    client_method_closure_call = providers.Factory(Client, value=service.provided.get_closure.call().call())
    client_provided_call = providers.Factory(Client, value=service.provided.call())

@fixture
def container():
    if False:
        while True:
            i = 10
    return Container()

def test_is_provider(container):
    if False:
        while True:
            i = 10
    assert providers.is_provider(container.service.provided) is True

def test_attribute(container):
    if False:
        while True:
            i = 10
    client = container.client_attribute()
    assert client.value == 'foo'

def test_item(container):
    if False:
        return 10
    client = container.client_item()
    assert client.value == 'foo'

def test_attribute_item(container):
    if False:
        while True:
            i = 10
    client = container.client_attribute_item()
    assert client.value == 'foo'

def test_method_call(container):
    if False:
        return 10
    client = container.client_method_call()
    assert client.value == 'foo'

def test_method_closure_call(container):
    if False:
        return 10
    client = container.client_method_closure_call()
    assert client.value == 'foo'

def test_provided_call(container):
    if False:
        return 10
    client = container.client_provided_call()
    assert client.value == 'foo'

def test_call_overridden(container):
    if False:
        print('Hello World!')
    value = 'bar'
    with container.service.override(Service(value)):
        assert container.client_attribute().value == value
        assert container.client_item().value == value
        assert container.client_attribute_item().value == value
        assert container.client_method_call().value == value

def test_repr_provided_instance(container):
    if False:
        while True:
            i = 10
    provider = container.service.provided
    assert repr(provider) == 'ProvidedInstance("{0}")'.format(repr(container.service))

def test_repr_attribute_getter(container):
    if False:
        while True:
            i = 10
    provider = container.service.provided.value
    assert repr(provider) == 'AttributeGetter("value")'

def test_repr_item_getter(container):
    if False:
        return 10
    provider = container.service.provided['test-test']
    assert repr(provider) == 'ItemGetter("test-test")'

def test_provided_instance():
    if False:
        while True:
            i = 10
    provides = providers.Object(object())
    provider = providers.ProvidedInstance()
    provider.set_provides(provides)
    assert provider.provides is provides
    assert provider.set_provides(providers.Provider()) is provider

def test_attribute_getter():
    if False:
        i = 10
        return i + 15
    provides = providers.Object(object())
    provider = providers.AttributeGetter()
    provider.set_provides(provides)
    provider.set_name('__dict__')
    assert provider.provides is provides
    assert provider.name == '__dict__'
    assert provider.set_provides(providers.Provider()) is provider
    assert provider.set_name('__dict__') is provider

def test_item_getter():
    if False:
        while True:
            i = 10
    provides = providers.Object({'foo': 'bar'})
    provider = providers.ItemGetter()
    provider.set_provides(provides)
    provider.set_name('foo')
    assert provider.provides is provides
    assert provider.name == 'foo'
    assert provider.set_provides(providers.Provider()) is provider
    assert provider.set_name('foo') is provider

def test_method_caller():
    if False:
        print('Hello World!')
    provides = providers.Object(lambda : 42)
    provider = providers.MethodCaller()
    provider.set_provides(provides)
    assert provider.provides is provides
    assert provider() == 42
    assert provider.set_provides(providers.Provider()) is provider

def test_puzzled():
    if False:
        print('Hello World!')
    service = providers.Singleton(Service, value='foo-bar')
    dependency = providers.Object({'a': {'b': {'c1': 10, 'c2': lambda arg: {'arg': arg}}}})
    test_list = providers.List(dependency.provided['a']['b']['c1'], dependency.provided['a']['b']['c2'].call(22)['arg'], dependency.provided['a']['b']['c2'].call(service)['arg'], dependency.provided['a']['b']['c2'].call(service)['arg'].value, dependency.provided['a']['b']['c2'].call(service)['arg'].get_value.call())
    result = test_list()
    assert result == [10, 22, service(), 'foo-bar', 'foo-bar']

def test_provided_attribute_in_base_class():
    if False:
        while True:
            i = 10
    provider = providers.Provider()
    assert isinstance(provider.provided, providers.ProvidedInstance)