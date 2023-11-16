from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class Path(NonCompletableGithubObject):
    """
    This class represents a popular Path for a GitHub repository.
    The reference can be found here https://docs.github.com/en/rest/reference/repos#traffic
    """

    def _initAttributes(self) -> None:
        if False:
            i = 10
            return i + 15
        self._path: Attribute[str] = NotSet
        self._title: Attribute[str] = NotSet
        self._count: Attribute[int] = NotSet
        self._uniques: Attribute[int] = NotSet

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.get__repr__({'path': self._path.value, 'title': self._title.value, 'count': self._count.value, 'uniques': self._uniques.value})

    @property
    def path(self) -> str:
        if False:
            print('Hello World!')
        return self._path.value

    @property
    def title(self) -> str:
        if False:
            print('Hello World!')
        return self._title.value

    @property
    def count(self) -> int:
        if False:
            while True:
                i = 10
        return self._count.value

    @property
    def uniques(self) -> int:
        if False:
            while True:
                i = 10
        return self._uniques.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if 'path' in attributes:
            self._path = self._makeStringAttribute(attributes['path'])
        if 'title' in attributes:
            self._title = self._makeStringAttribute(attributes['title'])
        if 'count' in attributes:
            self._count = self._makeIntAttribute(attributes['count'])
        if 'uniques' in attributes:
            self._uniques = self._makeIntAttribute(attributes['uniques'])