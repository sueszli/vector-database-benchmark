from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class Referrer(NonCompletableGithubObject):
    """
    This class represents a popylar Referrer for a GitHub repository.
    The reference can be found here https://docs.github.com/en/rest/reference/repos#traffic
    """

    def _initAttributes(self) -> None:
        if False:
            return 10
        self._referrer: Attribute[str] = NotSet
        self._count: Attribute[int] = NotSet
        self._uniques: Attribute[int] = NotSet

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.get__repr__({'referrer': self._referrer.value, 'count': self._count.value, 'uniques': self._uniques.value})

    @property
    def referrer(self) -> str:
        if False:
            while True:
                i = 10
        return self._referrer.value

    @property
    def count(self) -> int:
        if False:
            print('Hello World!')
        return self._count.value

    @property
    def uniques(self) -> int:
        if False:
            return 10
        return self._uniques.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if 'referrer' in attributes:
            self._referrer = self._makeStringAttribute(attributes['referrer'])
        if 'count' in attributes:
            self._count = self._makeIntAttribute(attributes['count'])
        if 'uniques' in attributes:
            self._uniques = self._makeIntAttribute(attributes['uniques'])