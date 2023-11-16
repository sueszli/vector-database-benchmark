"""Example of the injecting of provided instance attributes and items."""
from dependency_injector import containers, providers

class Service:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.value = 'foo'
        self.values = [self.value]

    def get_value(self):
        if False:
            while True:
                i = 10
        return self.value

    def __getitem__(self, item):
        if False:
            while True:
                i = 10
        return self.values[item]

class Client:

    def __init__(self, value1, value2, value3, value4):
        if False:
            for i in range(10):
                print('nop')
        self.value1 = value1
        self.value2 = value2
        self.value3 = value3
        self.value4 = value4

class Container(containers.DeclarativeContainer):
    service = providers.Singleton(Service)
    client_factory = providers.Factory(Client, value1=service.provided[0], value2=service.provided.value, value3=service.provided.values[0], value4=service.provided.get_value.call())
if __name__ == '__main__':
    container = Container()
    client = container.client_factory()
    assert client.value1 == client.value2 == client.value3 == 'foo'