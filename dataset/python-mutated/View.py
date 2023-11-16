from datetime import datetime
from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class View(NonCompletableGithubObject):
    """
    This class represents a popular Path for a GitHub repository.
    The reference can be found here https://docs.github.com/en/rest/reference/repos#traffic
    """

    def _initAttributes(self) -> None:
        if False:
            return 10
        self._timestamp: Attribute[datetime] = NotSet
        self._count: Attribute[int] = NotSet
        self._uniques: Attribute[int] = NotSet

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.get__repr__({'timestamp': self._timestamp.value, 'count': self._count.value, 'uniques': self._uniques.value})

    @property
    def timestamp(self) -> datetime:
        if False:
            return 10
        return self._timestamp.value

    @property
    def count(self) -> int:
        if False:
            print('Hello World!')
        return self._count.value

    @property
    def uniques(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._uniques.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        if 'timestamp' in attributes:
            self._timestamp = self._makeDatetimeAttribute(attributes['timestamp'])
        if 'count' in attributes:
            self._count = self._makeIntAttribute(attributes['count'])
        if 'uniques' in attributes:
            self._uniques = self._makeIntAttribute(attributes['uniques'])