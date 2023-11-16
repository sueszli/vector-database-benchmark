from __future__ import annotations
from typing import Any, Type
from sentry.plugins.providers import IntegrationRepositoryProvider, RepositoryProvider

class ProviderManager:
    type: Type[Any] | None = None

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._items = {}

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self._items)

    def add(self, item, id):
        if False:
            for i in range(10):
                print('nop')
        if self.type and (not issubclass(item, self.type)):
            raise ValueError(f'Invalid type for provider: {type(item)}')
        self._items[id] = item

    def get(self, id):
        if False:
            for i in range(10):
                print('nop')
        return self._items[id]

    def all(self):
        if False:
            while True:
                i = 10
        return self._items.items()

class RepositoryProviderManager(ProviderManager):
    type = RepositoryProvider

class IntegrationRepositoryProviderManager(ProviderManager):
    type = IntegrationRepositoryProvider

class BindingManager:
    BINDINGS = {'repository.provider': RepositoryProviderManager, 'integration-repository.provider': IntegrationRepositoryProviderManager}

    def __init__(self):
        if False:
            while True:
                i = 10
        self._bindings = {k: v() for (k, v) in self.BINDINGS.items()}

    def add(self, name, binding, **kwargs):
        if False:
            print('Hello World!')
        self._bindings[name].add(binding, **kwargs)

    def get(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self._bindings[name]