"""Declarative container provider copying with ``@copy()`` decorator."""
from dependency_injector import containers, providers

class Service:

    def __init__(self, dependency: str):
        if False:
            i = 10
            return i + 15
        self.dependency = dependency

class Base(containers.DeclarativeContainer):
    dependency = providers.Dependency(instance_of=str, default='Default value')
    service = providers.Factory(Service, dependency=dependency)

@containers.copy(Base)
class Derived1(Base):
    dependency = providers.Dependency(instance_of=str, default='Derived 1')

class Derived2(Base):
    dependency = providers.Dependency(instance_of=str, default='Derived 2')
if __name__ == '__main__':
    container1 = Derived1()
    service1 = container1.service()
    print(service1.dependency)
    container2 = Derived2()
    service2 = container2.service()
    print(service2.dependency)