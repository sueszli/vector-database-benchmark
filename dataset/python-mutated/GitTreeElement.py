from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

class GitTreeElement(NonCompletableGithubObject):
    """
    This class represents GitTreeElements
    """

    def _initAttributes(self) -> None:
        if False:
            return 10
        self._mode: Attribute[str] = NotSet
        self._path: Attribute[str] = NotSet
        self._sha: Attribute[str] = NotSet
        self._size: Attribute[int] = NotSet
        self._type: Attribute[str] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        if False:
            return 10
        return self.get__repr__({'sha': self._sha.value, 'path': self._path.value})

    @property
    def mode(self) -> str:
        if False:
            return 10
        return self._mode.value

    @property
    def path(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._path.value

    @property
    def sha(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._sha.value

    @property
    def size(self) -> int:
        if False:
            return 10
        return self._size.value

    @property
    def type(self) -> str:
        if False:
            return 10
        return self._type.value

    @property
    def url(self) -> str:
        if False:
            print('Hello World!')
        return self._url.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if False:
            return 10
        if 'mode' in attributes:
            self._mode = self._makeStringAttribute(attributes['mode'])
        if 'path' in attributes:
            self._path = self._makeStringAttribute(attributes['path'])
        if 'sha' in attributes:
            self._sha = self._makeStringAttribute(attributes['sha'])
        if 'size' in attributes:
            self._size = self._makeIntAttribute(attributes['size'])
        if 'type' in attributes:
            self._type = self._makeStringAttribute(attributes['type'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])