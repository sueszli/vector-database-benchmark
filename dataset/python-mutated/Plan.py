from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class Plan(NonCompletableGithubObject):
    """
    This class represents Plans
    """

    def _initAttributes(self) -> None:
        if False:
            print('Hello World!')
        self._collaborators: Attribute[int] = NotSet
        self._name: Attribute[str] = NotSet
        self._private_repos: Attribute[int] = NotSet
        self._space: Attribute[int] = NotSet
        self._filled_seats: Attribute[int] = NotSet
        self._seats: Attribute[int] = NotSet

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.get__repr__({'name': self._name.value})

    @property
    def collaborators(self) -> int:
        if False:
            print('Hello World!')
        return self._collaborators.value

    @property
    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._name.value

    @property
    def private_repos(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._private_repos.value

    @property
    def space(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._space.value

    @property
    def filled_seats(self) -> int:
        if False:
            print('Hello World!')
        return self._filled_seats.value

    @property
    def seats(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._seats.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if 'collaborators' in attributes:
            self._collaborators = self._makeIntAttribute(attributes['collaborators'])
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])
        if 'private_repos' in attributes:
            self._private_repos = self._makeIntAttribute(attributes['private_repos'])
        if 'space' in attributes:
            self._space = self._makeIntAttribute(attributes['space'])
        if 'seats' in attributes:
            self._seats = self._makeIntAttribute(attributes['seats'])
        if 'filled_seats' in attributes:
            self._filled_seats = self._makeIntAttribute(attributes['filled_seats'])