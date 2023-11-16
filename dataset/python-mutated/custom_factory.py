"""Custom provider example."""
from dependency_injector import containers, providers

class CustomFactory(providers.Provider):
    __slots__ = ('_factory',)

    def __init__(self, provides, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._factory = providers.Factory(provides, *args, **kwargs)
        super().__init__()

    def __deepcopy__(self, memo):
        if False:
            for i in range(10):
                print('nop')
        copied = memo.get(id(self))
        if copied is not None:
            return copied
        copied = self.__class__(self._factory.provides, *providers.deepcopy(self._factory.args, memo), **providers.deepcopy(self._factory.kwargs, memo))
        self._copy_overridings(copied, memo)
        return copied

    @property
    def related(self):
        if False:
            while True:
                i = 10
        'Return related providers generator.'
        yield from [self._factory]
        yield from super().related

    def _provide(self, args, kwargs):
        if False:
            return 10
        return self._factory(*args, **kwargs)

class Container(containers.DeclarativeContainer):
    factory = CustomFactory(object)
if __name__ == '__main__':
    container = Container()
    object1 = container.factory()
    assert isinstance(object1, object)
    object2 = container.factory()
    assert isinstance(object1, object)
    assert object1 is not object2