from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class GitObject(NonCompletableGithubObject):
    """
    This class represents GitObjects
    """

    def _initAttributes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._sha: Attribute[str] = NotSet
        self._type: Attribute[str] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            return 10
        return self.get__repr__({'sha': self._sha.value})

    @property
    def sha(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._sha.value

    @property
    def type(self) -> str:
        if False:
            print('Hello World!')
        return self._type.value

    @property
    def url(self) -> str:
        if False:
            return 10
        return self._url.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if 'sha' in attributes:
            self._sha = self._makeStringAttribute(attributes['sha'])
        if 'type' in attributes:
            self._type = self._makeStringAttribute(attributes['type'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])