from __future__ import annotations
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class HookDescription(NonCompletableGithubObject):
    """
    This class represents HookDescriptions
    """

    def _initAttributes(self) -> None:
        if False:
            while True:
                i = 10
        self._events: Attribute[list[str]] = NotSet
        self._name: Attribute[str] = NotSet
        self._schema: Attribute[list[list[str]]] = NotSet
        self._supported_events: Attribute[list[str]] = NotSet

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.get__repr__({'name': self._name.value})

    @property
    def events(self) -> list[str]:
        if False:
            i = 10
            return i + 15
        return self._events.value

    @property
    def name(self) -> str:
        if False:
            while True:
                i = 10
        return self._name.value

    @property
    def schema(self) -> list[list[str]]:
        if False:
            while True:
                i = 10
        return self._schema.value

    @property
    def supported_events(self) -> list[str]:
        if False:
            while True:
                i = 10
        return self._supported_events.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        if 'events' in attributes:
            self._events = self._makeListOfStringsAttribute(attributes['events'])
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])
        if 'schema' in attributes:
            self._schema = self._makeListOfListOfStringsAttribute(attributes['schema'])
        if 'supported_events' in attributes:
            self._supported_events = self._makeListOfStringsAttribute(attributes['supported_events'])